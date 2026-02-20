"""
Evaluation Metrics — Segmentasyon & Sınıflandırma

Fonksiyonlar:
  dice_score()         : Dice katsayısı
  iou_score()          : Intersection over Union
  compute_auc()        : AUC-ROC (sınıflandırma başlığı)
  hausdorff_distance() : Hausdorff Mesafesi (kontur sapması, piksel)

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ── Temel segmentasyon metrikleri ─────────────────────────────────────────────

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


# ── Hausdorff Mesafesi ─────────────────────────────────────────────────────────

def hausdorff_distance(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    percentile: float = 95.0,
) -> float:
    """
    Hausdorff Mesafesi (HD) — tahmin edilen konturla gerçek kontur
    arasındaki maksimum minimum yüzey sapması (piksel cinsinden).

    Radyolojik önem:
      • HD = 0   → maskeler tam üst üste
      • HD > 10  → klinik olarak kabul edilemez sınır hatası
      • HD = ∞   → maskelerden biri tamamen boş

    Algoritma (scipy.ndimage tabanlı):
      1. İkili maskeye çevir
      2. Her maske için Distance Transform hesapla
         (her pikselin en yakın 1'e olan mesafesi)
      3. HD(A→B) = max of B_distances at A positions
         HD(B→A) = max of A_distances at B positions
      4. Simetrik HD = max(HD(A→B), HD(B→A))
         %95 Hausdorff için max yerine percentile kullanılır (outlier'a dayanıklı)

    Args:
        pred       : Model çıktısı (logit veya olasılık); Tensor veya ndarray
        target     : Ground truth binary maske; Tensor veya ndarray
        threshold  : Sigmoid sonrası eşik değeri (varsayılan 0.5)
        percentile : Kullanılacak yüzdelik dilim (95 → HD95, 100 → tam HD)

    Returns:
        HD değeri (piksel). Boş maske durumunda float('inf').
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scipy kurulu değil: pip install scipy")

    # Tensor → numpy
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Tek örnek: (1, H, W) veya (H, W)
    pred   = np.squeeze(pred)
    target = np.squeeze(target)

    pred_bin   = (pred   > threshold).astype(bool)
    target_bin = (target > 0.5).astype(bool)

    # Boş maske kontrolü
    if not pred_bin.any() and not target_bin.any():
        return 0.0          # İkisi de boş → mükemmel (Dice=1 degenerate durum)
    if not pred_bin.any() or not target_bin.any():
        return float("inf") # Biri boş → sonsuz sapma

    # Distance transform: her pikselden en yakın True'ya olan Öklid mesafesi
    # ~(X) → True olan pikseller 0, False olanlar mesafe alır
    dist_pred   = distance_transform_edt(~pred_bin)    # B'den A'ya
    dist_target = distance_transform_edt(~target_bin)  # A'dan B'ye

    # Hausdorff: A noktalarında B'nin mesafesi + B noktalarında A'nın mesafesi
    hd_pred_to_target = dist_target[pred_bin]    # pred konturundaki GT uzaklıkları
    hd_target_to_pred = dist_pred[target_bin]    # GT konturundaki pred uzaklıkları

    p = float(percentile)
    h1 = float(np.percentile(hd_pred_to_target,  p))
    h2 = float(np.percentile(hd_target_to_pred,  p))

    return max(h1, h2)


def batch_hausdorff(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    percentile: float = 95.0,
) -> float:
    """
    Batch üzerinde ortalama Hausdorff Mesafesi hesaplar.
    inf değerler (boş maske çiftleri) ortalamanın dışında tutulur.

    Args:
        preds   : (B, 1, H, W) model çıktısı
        targets : (B, 1, H, W) ground truth
    """
    hd_values = []
    for i in range(preds.shape[0]):
        hd = hausdorff_distance(preds[i], targets[i], threshold, percentile)
        if not np.isinf(hd):
            hd_values.append(hd)

    return float(np.mean(hd_values)) if hd_values else float("inf")
