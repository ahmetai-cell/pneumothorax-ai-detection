#!/usr/bin/env python
"""
Model evaluation on a held-out test set or single fold validation set.

Usage:
    # Evaluate best checkpoint of fold 1 on its val set:
    python eval.py \\
        --checkpoint output/checkpoints/fold_1_best.pth \\
        --data_dir data/local/ptx498 \\
        --nih_dir  data/nih_negatives \\
        --config   output/config_trained.yaml

    # Evaluate on a specific CSV manifest:
    python eval.py \\
        --checkpoint output/checkpoints/fold_1_best.pth \\
        --csv data/test_manifest.csv \\
        --config output/config_trained.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_manifest, PTXDataset
from src.data.transforms import get_val_transforms
from src.metrics import (
    compute_auc,
    compute_sensitivity_specificity,
    dice_score,
    iou_score,
    find_optimal_threshold,
)
from src.models.unet import PneumothoraxModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> PneumothoraxModel:
    model = PneumothoraxModel(
        encoder_name     = cfg["model"]["encoder"],
        pretrained       = False,
        in_channels      = cfg["model"]["in_channels"],
        deep_supervision = False,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    logger.info(f"Loaded checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch','?')})")
    return model


@torch.no_grad()
def evaluate(
    model:      PneumothoraxModel,
    loader:     DataLoader,
    device:     torch.device,
    cls_t:      float = 0.5,
    seg_t:      float = 0.5,
) -> dict:

    all_cls_probs:  list[float] = []
    all_cls_labels: list[int]   = []
    all_dices:      list[float] = []
    all_ious:       list[float] = []
    all_seg_probs:  list[np.ndarray] = []
    all_masks:      list[np.ndarray] = []

    use_amp = device.type == "cuda"

    for images, masks, labels in loader:
        images = images.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            seg_pred, cls_pred = model(images)

        cls_probs  = torch.sigmoid(cls_pred).squeeze(-1).cpu().numpy()
        seg_probs_ = torch.sigmoid(seg_pred).squeeze(1).cpu().numpy()
        masks_np   = masks.squeeze(1).cpu().numpy()
        labels_np  = labels.cpu().numpy().astype(int)

        for i, lbl in enumerate(labels_np):
            cp = float(cls_probs[i]) if cls_probs.ndim > 0 else float(cls_probs)
            all_cls_probs.append(cp)
            all_cls_labels.append(lbl)
            if lbl == 1:
                d = dice_score(seg_probs_[i], masks_np[i], threshold=seg_t)
                u = iou_score(seg_probs_[i], masks_np[i], threshold=seg_t)
                all_dices.append(d)
                all_ious.append(u)
                all_seg_probs.append(seg_probs_[i])
                all_masks.append(masks_np[i])

    probs_arr  = np.array(all_cls_probs)
    labels_arr = np.array(all_cls_labels)

    auc             = compute_auc(probs_arr, labels_arr)
    sens, spec      = compute_sensitivity_specificity(probs_arr, labels_arr, cls_t)
    opt_t, opt_f1   = find_optimal_threshold(probs_arr, labels_arr, metric="f1")

    # Sensitivity at 90% specificity
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(labels_arr, probs_arr)
    spec_arr = 1 - fpr
    idx_90   = np.argmin(np.abs(spec_arr - 0.90))
    sens_at_90spec = float(tpr[idx_90])

    return {
        "n_total":          len(all_cls_labels),
        "n_positive":       int(labels_arr.sum()),
        "n_negative":       int((labels_arr == 0).sum()),
        "cls_threshold":    cls_t,
        "seg_threshold":    seg_t,
        "dice_mean":        float(np.mean(all_dices)) if all_dices else 0.0,
        "dice_std":         float(np.std(all_dices))  if all_dices else 0.0,
        "iou_mean":         float(np.mean(all_ious))  if all_ious  else 0.0,
        "auc":              float(auc),
        "sensitivity":      float(sens),
        "specificity":      float(spec),
        "sensitivity_at_90spec": float(sens_at_90spec),
        "optimal_cls_threshold": float(opt_t),
        "optimal_f1":            float(opt_f1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PTX model evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--data_dir",   default=None)
    p.add_argument("--nih_dir",    default=None)
    p.add_argument("--csv",        default=None, help="Manifest CSV (overrides data_dir)")
    p.add_argument("--output",     default="eval_results.json")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    logger.info(f"Device: {device}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if "mask_path" not in df.columns:
            df["mask_path"] = None
    elif args.data_dir:
        df = build_manifest(args.data_dir, args.nih_dir)
    else:
        raise ValueError("Provide --csv or --data_dir")

    img_size = cfg["data"]["img_size"]
    dataset  = PTXDataset(df, transform=get_val_transforms(img_size), img_size=img_size)
    loader   = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = cfg["data"].get("num_workers", 4),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, cfg, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    cls_t = cfg["thresholds"].get("cls_threshold", 0.5)
    seg_t = cfg["thresholds"].get("seg_threshold", 0.5)

    results = evaluate(model, loader, device, cls_t=cls_t, seg_t=seg_t)

    # ── Print and save ────────────────────────────────────────────────────────
    logger.info("\n=== Evaluation Results ===")
    for k, v in results.items():
        logger.info(f"  {k:<30}: {v}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
