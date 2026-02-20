"""
W&B (Weights & Biases) Yardımcı Fonksiyonlar
Her fold sonunda en iyi / en kötü tahminleri görsel olarak loglar.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@torch.no_grad()
def log_prediction_samples(
    model: torch.nn.Module,
    dataset,
    val_indices: list[int],
    device: torch.device,
    fold: int,
    n_samples: int = 5,
    img_size: int = 512,
) -> None:
    """
    Val seti üzerinde inference çalıştırır, en iyi ve en kötü
    n_samples tahmini W&B'ye görsel olarak loglar.

    En iyi  → Dice skoru en yüksek örnekler (True Positive)
    En kötü → Dice skoru en düşük örnekler (FP / FN ağırlıklı)
    """
    if not WANDB_AVAILABLE:
        return

    model.eval()
    val_subset = Subset(dataset, val_indices)
    loader     = DataLoader(val_subset, batch_size=8, shuffle=False)

    all_dice:   list[float] = []
    all_images: list[np.ndarray] = []
    all_gt:     list[np.ndarray] = []
    all_pred:   list[np.ndarray] = []

    for images, masks, _ in loader:
        images = images.to(device)
        seg_pred, _ = model(images)
        seg_prob     = torch.sigmoid(seg_pred)

        for i in range(len(images)):
            img_np  = images[i, 0].cpu().numpy()
            gt_np   = masks[i, 0].numpy()
            pred_np = seg_prob[i, 0].cpu().numpy()

            # Dice hesapla
            pred_bin = (pred_np > 0.5).astype(float)
            inter    = (pred_bin * gt_np).sum()
            dice_val = (2 * inter + 1e-6) / (pred_bin.sum() + gt_np.sum() + 1e-6)

            all_dice.append(float(dice_val))
            all_images.append(img_np)
            all_gt.append(gt_np)
            all_pred.append(pred_np)

    # Sırala
    sorted_idx   = np.argsort(all_dice)
    worst_idx    = sorted_idx[:n_samples]
    best_idx     = sorted_idx[-n_samples:][::-1]

    def make_panel(idx_list: list[int], title: str) -> list:
        panels = []
        for idx in idx_list:
            img  = (all_images[idx] * 255).astype(np.uint8)
            bgr  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # GT mask → yeşil overlay
            gt_overlay = bgr.copy()
            gt_overlay[all_gt[idx] > 0.5] = [0, 200, 0]
            gt_vis = cv2.addWeighted(gt_overlay, 0.4, bgr, 0.6, 0)

            # Pred mask → mavi overlay
            pred_bin = (all_pred[idx] > 0.5).astype(np.uint8)
            pred_overlay = bgr.copy()
            pred_overlay[pred_bin > 0] = [220, 60, 0]
            pred_vis = cv2.addWeighted(pred_overlay, 0.4, bgr, 0.6, 0)

            combined = np.hstack([
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB),
            ])

            panels.append(
                wandb.Image(
                    combined,
                    caption=f"{title} | Dice: {all_dice[idx]:.3f} | "
                            "[Orijinal | GT(yeşil) | Tahmin(mavi)]",
                )
            )
        return panels

    wandb.log({
        f"fold{fold}/best_predictions":  make_panel(list(best_idx),  "En İyi"),
        f"fold{fold}/worst_predictions": make_panel(list(worst_idx), "En Kötü"),
    })


def log_hnm_stats(
    fold: int,
    epoch: int,
    n_hard_negatives: int,
    total_negatives: int,
) -> None:
    """Hard Negative Mining istatistiklerini loglar."""
    if not WANDB_AVAILABLE:
        return
    wandb.log({
        "epoch": epoch,
        f"fold{fold}/hnm/hard_negatives":   n_hard_negatives,
        f"fold{fold}/hnm/fp_rate_estimate": n_hard_negatives / max(total_negatives, 1),
    })
