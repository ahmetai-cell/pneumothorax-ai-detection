"""
Evaluation Metrics: Dice, IoU, AUC-ROC
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_auc(cls_preds, cls_targets):
    probs = torch.sigmoid(cls_preds).cpu().numpy()
    targets = cls_targets.cpu().numpy()
    return roc_auc_score(targets, probs)
