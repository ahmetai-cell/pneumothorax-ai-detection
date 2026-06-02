"""
Model export: ONNX and TorchScript.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(
    model:      nn.Module,
    output_path: str | Path,
    img_size:   int = 512,
    opset:      int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """
    Exports model to ONNX with optional dynamic batch axis.
    Input:  float32 (B, 1, H, W)
    Outputs: seg_logits (B, 1, H, W), cls_logit (B, 1)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.zeros(1, 1, img_size, img_size, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image":      {0: "batch"},
            "seg_logits": {0: "batch"},
            "cls_logit":  {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=["seg_logits", "cls_logit"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    logger.info(f"ONNX exported: {output_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model check: PASSED")
    except ImportError:
        logger.warning("onnx package not installed; skipping verification")

    return output_path


def export_torchscript(
    model:       nn.Module,
    output_path: str | Path,
    img_size:    int = 512,
) -> Path:
    """
    Exports model to TorchScript via tracing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy  = torch.zeros(1, 1, img_size, img_size, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    traced.save(str(output_path))
    logger.info(f"TorchScript exported: {output_path}")
    return output_path
