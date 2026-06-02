"""
Loss functions for pneumothorax detection.
TverskyLoss: penalizes FN more than FP (clinically safer — missing PTX is dangerous).
FocalLoss: focuses gradient on hard examples.
ProductionLoss: combines both for segmentation; BCE for classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky index loss.
    alpha: FP penalty weight
    beta:  FN penalty weight  (beta > alpha → penalize FN more)
    For PTX: alpha=0.3, beta=0.7 recommended.
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta:  float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).reshape(-1)
        target = target.reshape(-1)
        tp = (pred * target).sum()
        fp = ((1.0 - target) * pred).sum()
        fn = (target * (1.0 - pred)).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class FocalLoss(nn.Module):
    """
    Focal loss for binary segmentation.
    Reduces well-classified background weight, focuses on hard positives.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        pt   = target * prob + (1.0 - target) * (1.0 - prob)
        w    = self.alpha * (1.0 - pt) ** self.gamma
        return (w * bce).mean()


class ProductionLoss(nn.Module):
    """
    Combined loss for dual-head PTX model.

    Segmentation: Tversky + Focal, applied only to positive samples.
    Classification: BCEWithLogitsLoss, applied to all samples.
    Deep supervision: same seg loss × aux_w on each auxiliary head.

    Total = seg_loss + cls_w * cls_loss + aux_loss
    """

    def __init__(
        self,
        tversky_alpha:  float = 0.3,
        tversky_beta:   float = 0.7,
        focal_alpha:    float = 0.25,
        focal_gamma:    float = 2.0,
        seg_tversky_w:  float = 0.7,
        seg_focal_w:    float = 0.3,
        cls_w:          float = 0.3,
        aux_w:          float = 0.3,
    ) -> None:
        super().__init__()
        self.tversky      = TverskyLoss(tversky_alpha, tversky_beta)
        self.focal        = FocalLoss(focal_alpha, focal_gamma)
        self.bce          = nn.BCEWithLogitsLoss()
        self.seg_tversky_w = seg_tversky_w
        self.seg_focal_w  = seg_focal_w
        self.cls_w        = cls_w
        self.aux_w        = aux_w

    def _seg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.seg_tversky_w * self.tversky(pred, target)
            + self.seg_focal_w  * self.focal(pred, target)
        )

    def forward(
        self,
        seg_pred:   torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred:   torch.Tensor,
        cls_target: torch.Tensor,
        aux_preds:  list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        pos = cls_target.bool()

        # Segmentation loss on positive samples only
        if pos.any():
            seg_loss = self._seg_loss(seg_pred[pos], seg_target[pos])
        else:
            seg_loss = seg_pred.sum() * 0.0

        # Classification loss on all samples
        cls_loss = self.bce(cls_pred.view(-1), cls_target.float())

        # Auxiliary deep supervision
        aux_loss = seg_pred.sum() * 0.0
        if aux_preds and pos.any():
            for aux in aux_preds:
                aux_loss = aux_loss + self._seg_loss(aux[pos], seg_target[pos])
            aux_loss = aux_loss * self.aux_w

        total = seg_loss + self.cls_w * cls_loss + aux_loss

        components = {
            "seg_loss": seg_loss.item(),
            "cls_loss": cls_loss.item(),
            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0,
        }
        return total, components
