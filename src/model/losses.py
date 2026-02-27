"""
Combined Loss: Dice + BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_w = dice_weight
        self.bce_w = bce_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target,
                aux_preds: list | None = None):
        # Seg loss sadece pozitif örneklere uygula (maskesi olan vakalar).
        pos = cls_target.bool()
        if pos.any():
            seg_loss = (
                self.dice_w * self.dice(seg_pred[pos], seg_target[pos]) +
                self.bce_w  * self.bce(seg_pred[pos],  seg_target[pos])
            )
        else:
            seg_loss = seg_pred.sum() * 0.0

        cls_loss = self.bce(cls_pred.squeeze(), cls_target.float())

        # Deep Supervision: her auxiliary çıktıya 0.3 ağırlıkla ek loss
        # (sadece pozitif örnekler, ana loss'la aynı mantık)
        aux_loss = seg_pred.sum() * 0.0
        if aux_preds and pos.any():
            for aux in aux_preds:
                aux_loss = aux_loss + (
                    self.dice_w * self.dice(aux[pos], seg_target[pos]) +
                    self.bce_w  * self.bce(aux[pos],  seg_target[pos])
                )
            aux_loss = aux_loss * 0.3

        return seg_loss + 0.3 * cls_loss + aux_loss
