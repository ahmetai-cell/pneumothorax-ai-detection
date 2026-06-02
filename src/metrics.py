"""
Segmentation and classification metrics for pneumothorax detection.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def dice_score(
    pred_prob: torch.Tensor | np.ndarray,
    target:    torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    smooth:    float = 1.0,
) -> float:
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_bin = (pred_prob > threshold).astype(np.float32)
    target   = target.astype(np.float32)

    pred_f   = pred_bin.reshape(-1)
    target_f = target.reshape(-1)
    intersection = (pred_f * target_f).sum()
    return float((2.0 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth))


def iou_score(
    pred_prob: torch.Tensor | np.ndarray,
    target:    torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    smooth:    float = 1.0,
) -> float:
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_bin = (pred_prob > threshold).astype(np.float32)
    target   = target.astype(np.float32)

    pred_f   = pred_bin.reshape(-1)
    target_f = target.reshape(-1)
    intersection = (pred_f * target_f).sum()
    union        = pred_f.sum() + target_f.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_auc(
    cls_probs: np.ndarray,
    labels:    np.ndarray,
) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, cls_probs))
    except ValueError:
        return float("nan")


def compute_sensitivity_specificity(
    cls_probs: np.ndarray,
    labels:    np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float]:
    preds = (cls_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return float(sensitivity), float(specificity)


def find_optimal_threshold(
    cls_probs: np.ndarray,
    labels:    np.ndarray,
    metric:    str = "f1",
) -> tuple[float, float]:
    """
    Finds classification threshold maximizing F1 on validation set.
    Returns (optimal_threshold, best_metric_value).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_val = 0.5, 0.0

    for t in thresholds:
        preds = (cls_probs >= t).astype(int)
        if metric == "f1":
            val = f1_score(labels, preds, zero_division=0)
        elif metric == "sensitivity":
            _, _, fn, tp = (
                confusion_matrix(labels, preds, labels=[0, 1]).ravel()
                if len(np.unique(preds)) > 1
                else (0, 0, 0, 0)
            )
            val = tp / max(tp + fn, 1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if val > best_val:
            best_val = val
            best_t   = t

    return float(best_t), float(best_val)


def find_optimal_seg_threshold(
    seg_probs_list: list[np.ndarray],
    mask_list:      list[np.ndarray],
) -> tuple[float, float]:
    """
    Finds segmentation threshold maximizing mean Dice over validation set.
    Returns (optimal_threshold, best_mean_dice).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_dice = 0.5, 0.0

    for t in thresholds:
        dices = [dice_score(p, m, threshold=t) for p, m in zip(seg_probs_list, mask_list)]
        mean_d = float(np.mean(dices))
        if mean_d > best_dice:
            best_dice = mean_d
            best_t    = t

    return float(best_t), float(best_dice)


class MetricAccumulator:
    """Accumulates per-batch values, returns epoch averages."""

    def __init__(self) -> None:
        self._sums:   dict[str, float] = {}
        self._counts: dict[str, int]   = {}

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0) + n

    def mean(self) -> dict[str, float]:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
