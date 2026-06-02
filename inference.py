#!/usr/bin/env python
"""
Inference system — single image, batch, and ONNX runtime.

Usage:
    # Single image
    python inference.py \\
        --checkpoint output/checkpoints/fold_1_best.pth \\
        --config     output/config_trained.yaml \\
        --image      path/to/xray.png

    # Batch (directory or CSV)
    python inference.py \\
        --checkpoint output/checkpoints/fold_1_best.pth \\
        --config     output/config_trained.yaml \\
        --batch_dir  path/to/images/ \\
        --output_csv predictions.csv

    # ONNX runtime
    python inference.py \\
        --onnx  output/model.onnx \\
        --config output/config_trained.yaml \\
        --image  path/to/xray.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import read_image
from src.data.transforms import get_inference_transforms
from src.models.unet import PneumothoraxModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Normalization (must match training) ────────────────────────────────────────

def _preprocess(img_uint8: np.ndarray, img_size: int) -> np.ndarray:
    """Resize + ImageNet normalize. Returns float32 (1, 1, H, W)."""
    resized = cv2.resize(img_uint8, (img_size, img_size))
    # Matches A.Normalize(mean=0.485, std=0.229, max_pixel_value=255)
    normalized = (resized.astype(np.float32) / 255.0 - 0.485) / 0.229
    return normalized[np.newaxis, np.newaxis, ...]   # (1, 1, H, W)


# ── PyTorch inference ──────────────────────────────────────────────────────────

class PTXPredictor:
    """
    PyTorch-based single/batch predictor.

    predict_single returns:
        cls_prob  : float in [0, 1]
        seg_mask  : float32 ndarray (H, W), probability map
        binary_mask: uint8 ndarray (H, W), 0 or 255
        has_ptx   : bool
    """

    def __init__(
        self,
        checkpoint: str,
        config:     dict,
        device:     Optional[torch.device] = None,
    ) -> None:
        self.cfg      = config
        self.img_size = config["data"]["img_size"]
        self.cls_t    = config["thresholds"].get("cls_threshold", 0.5)
        self.seg_t    = config["thresholds"].get("seg_threshold", 0.5)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        model = PneumothoraxModel(
            encoder_name     = config["model"]["encoder"],
            pretrained       = False,
            in_channels      = config["model"]["in_channels"],
            deep_supervision = False,
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(device)
        self.model = model
        logger.info(f"Model loaded from {checkpoint} on {device}")

    @torch.no_grad()
    def predict_single(self, image_path: str) -> dict:
        img_uint8 = read_image(image_path)
        orig_h, orig_w = img_uint8.shape[:2]

        x   = _preprocess(img_uint8, self.img_size)
        t   = torch.from_numpy(x).float().to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_pred, cls_pred = self.model(t)

        cls_prob = float(torch.sigmoid(cls_pred).item())
        seg_prob = torch.sigmoid(seg_pred).squeeze().cpu().numpy()

        # Resize seg back to original image size
        seg_resized = cv2.resize(seg_prob, (orig_w, orig_h))
        binary      = (seg_resized > self.seg_t).astype(np.uint8) * 255

        # Gatekeeper: if classifier says negative, zero the mask
        if cls_prob < self.cls_t:
            binary      = np.zeros_like(binary)
            seg_resized = np.zeros_like(seg_resized)

        mask_ratio = float(binary.sum()) / max(255 * orig_h * orig_w, 1)
        has_ptx    = (cls_prob >= self.cls_t) and (mask_ratio >= 0.005)

        return {
            "image_path":   image_path,
            "cls_prob":     round(cls_prob, 4),
            "has_ptx":      has_ptx,
            "mask_ratio":   round(mask_ratio, 5),
            "seg_prob_map": seg_resized,
            "binary_mask":  binary,
            "orig_shape":   (orig_h, orig_w),
        }

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        return [self.predict_single(p) for p in image_paths]

    def predict_with_gradcam(self, image_path: str) -> dict:
        """Adds GradCAM overlay to the standard prediction result."""
        result = self.predict_single(image_path)
        img_uint8 = read_image(image_path)

        try:
            from src.utils.gradcam import generate_gradcam_result
            overlay, _ = generate_gradcam_result(self.model, img_uint8, self.img_size)
            result["gradcam_overlay"] = overlay
        except Exception as exc:
            logger.warning(f"GradCAM failed: {exc}")
            result["gradcam_overlay"] = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        return result


# ── ONNX inference ─────────────────────────────────────────────────────────────

class ONNXPredictor:
    """ONNX Runtime predictor (no PyTorch dependency at inference time)."""

    def __init__(self, onnx_path: str, config: dict) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed: pip install onnxruntime")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.sess     = ort.InferenceSession(onnx_path, providers=providers)
        self.img_size = config["data"]["img_size"]
        self.cls_t    = config["thresholds"].get("cls_threshold", 0.5)
        self.seg_t    = config["thresholds"].get("seg_threshold", 0.5)
        logger.info(f"ONNX model loaded: {onnx_path}")

    def predict_single(self, image_path: str) -> dict:
        img_uint8 = read_image(image_path)
        orig_h, orig_w = img_uint8.shape[:2]
        x = _preprocess(img_uint8, self.img_size).astype(np.float32)

        outputs   = self.sess.run(None, {"image": x})
        seg_logit = outputs[0].squeeze()
        cls_logit = outputs[1].squeeze()

        cls_prob    = float(1 / (1 + np.exp(-cls_logit)))
        seg_prob    = 1 / (1 + np.exp(-seg_logit))
        seg_resized = cv2.resize(seg_prob, (orig_w, orig_h))
        binary      = (seg_resized > self.seg_t).astype(np.uint8) * 255

        if cls_prob < self.cls_t:
            binary      = np.zeros_like(binary)
            seg_resized = np.zeros_like(seg_resized)

        mask_ratio = float(binary.sum()) / max(255 * orig_h * orig_w, 1)
        has_ptx    = (cls_prob >= self.cls_t) and (mask_ratio >= 0.005)

        return {
            "image_path": image_path,
            "cls_prob":   round(cls_prob, 4),
            "has_ptx":    has_ptx,
            "mask_ratio": round(mask_ratio, 5),
        }


# ── Visualization helper ───────────────────────────────────────────────────────

def save_prediction_image(
    image_path:  str,
    result:      dict,
    output_path: str,
) -> None:
    img = read_image(image_path)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if result.get("binary_mask") is not None:
        mask = result["binary_mask"]
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        color_mask = np.zeros_like(bgr)
        color_mask[mask > 0] = (0, 200, 0)
        bgr = cv2.addWeighted(bgr, 0.7, color_mask, 0.3, 0)

    label = f"PTX: {result['cls_prob']:.3f} | {'POSITIVE' if result['has_ptx'] else 'NEGATIVE'}"
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(output_path, bgr)
    logger.info(f"Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="PTX inference")
    p.add_argument("--checkpoint", default=None, help="PyTorch checkpoint path")
    p.add_argument("--onnx",       default=None, help="ONNX model path")
    p.add_argument("--config",     required=True, help="config.yaml path")
    p.add_argument("--image",      default=None, help="Single image path")
    p.add_argument("--batch_dir",  default=None, help="Directory of images")
    p.add_argument("--output_csv", default=None, help="Output CSV for batch results")
    p.add_argument("--save_viz",   action="store_true", help="Save overlay images")
    p.add_argument("--viz_dir",    default="output/predictions")
    p.add_argument("--gradcam",    action="store_true", help="Include GradCAM overlay")
    args = p.parse_args()

    if not args.checkpoint and not args.onnx:
        p.error("Provide --checkpoint or --onnx")
    if not args.image and not args.batch_dir:
        p.error("Provide --image or --batch_dir")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Build predictor ───────────────────────────────────────────────────────
    if args.onnx:
        predictor = ONNXPredictor(args.onnx, cfg)
    else:
        predictor = PTXPredictor(args.checkpoint, cfg)

    # ── Collect images ────────────────────────────────────────────────────────
    if args.image:
        image_paths = [args.image]
    else:
        exts = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
        image_paths = [
            str(p) for p in sorted(Path(args.batch_dir).rglob("*"))
            if p.suffix.lower() in exts
        ]
        logger.info(f"Found {len(image_paths)} images in {args.batch_dir}")

    # ── Run inference ──────────────────────────────────────────────────────────
    viz_dir = Path(args.viz_dir)
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in image_paths:
        try:
            if args.gradcam and isinstance(predictor, PTXPredictor):
                result = predictor.predict_with_gradcam(img_path)
            else:
                result = predictor.predict_single(img_path)
            results.append(result)

            logger.info(
                f"{Path(img_path).name:<30} | "
                f"prob={result['cls_prob']:.4f} | "
                f"{'PTX' if result['has_ptx'] else 'NEG'}"
            )

            if args.save_viz:
                out = viz_dir / (Path(img_path).stem + "_pred.png")
                save_prediction_image(img_path, result, str(out))

        except Exception as exc:
            logger.error(f"Failed [{img_path}]: {exc}")
            results.append({"image_path": img_path, "error": str(exc)})

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if args.output_csv:
        csv_cols = ["image_path", "cls_prob", "has_ptx", "mask_ratio"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in csv_cols})
        logger.info(f"Predictions saved: {args.output_csv}")


if __name__ == "__main__":
    main()
