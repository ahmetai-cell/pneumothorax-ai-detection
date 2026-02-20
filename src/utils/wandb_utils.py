"""
W&B Yardımcı Fonksiyonlar
- Fold başına wandb.init (ayrı run)
- Görsel hata analizi: FP ve FN vakalar wandb.Table olarak
- 5-fold özet → wandb.summary

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


# ── Fold başlatma ─────────────────────────────────────────────────────────────

def init_fold_run(config: dict, fold: int, num_folds: int):
    """
    Her fold için ayrı bir W&B run başlatır.
    Tüm model ve HNM config'i wandb.config olarak kaydeder.
    """
    if not WANDB_AVAILABLE:
        return None

    run = wandb.init(
        project=config.get("wandb_project", "Pneumothorax-Detection"),
        entity=config.get("wandb_entity",  "ahmet-ai-t-bi-tak"),
        name=f"fold-{fold}-of-{num_folds}",
        group=config.get("wandb_group", "kfold-run"),  # tüm foldları grupla
        reinit=True,
        config={
            # ── Model mimarisi ─────────────────────────────────────────
            "architecture":      "U-Net++",
            "encoder":           config.get("encoder_name", "efficientnet-b0"),
            "pretrained":        "ImageNet",
            "in_channels":       1,
            "num_classes":       1,
            "loss_function":     "Dice + BCE (hybrid)",
            "dice_weight":       config.get("dice_weight",  0.5),
            "bce_weight":        1 - config.get("dice_weight", 0.5),
            # ── Eğitim hiperparametreleri ──────────────────────────────
            "learning_rate":     config["lr"],
            "batch_size":        config["batch_size"],
            "epochs":            config["epochs"],
            "optimizer":         "Adam",
            "weight_decay":      config.get("weight_decay", 1e-4),
            "lr_scheduler":      "ReduceLROnPlateau (factor=0.5, patience=5)",
            "img_size":          config.get("img_size", 512),
            # ── Cross-validation ───────────────────────────────────────
            "num_folds":         num_folds,
            "current_fold":      fold,
            "cv_strategy":       "Stratified K-Fold",
            "random_seed":       42,
            # ── Hard Negative Mining ───────────────────────────────────
            "hnm_enabled":       config.get("hard_negative_mining", True),
            "hnm_interval":      config.get("hnm_interval",  3),
            "hnm_fp_threshold":  config.get("hnm_threshold", 0.4),
            "hnm_multiplier":    config.get("hnm_multiplier", 3.0),
            # ── Augmentation ──────────────────────────────────────────
            "augmentation":      "CLAHE, HFlip, ShiftScaleRotate, ElasticTransform, GaussNoise",
            "dataset":           "SIIM-ACR + DEÜ Hastanesi",
        },
    )
    return run


# ── Per-epoch metrik log ──────────────────────────────────────────────────────

def log_epoch_metrics(
    fold: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_dice: float,
    val_iou: float,
    val_auc: float,
    val_sensitivity: float,
    val_specificity: float,
    val_precision: float,
    current_lr: float,
) -> None:
    """Her epoch sonunda tüm metrikleri W&B'ye loglar."""
    if not WANDB_AVAILABLE:
        return

    wandb.log({
        "epoch":              epoch,
        # Kayıp
        "Loss/BCE_Dice_Loss_train": train_loss,
        "Loss/BCE_Dice_Loss_val":   val_loss,
        # Segmentasyon
        "Segmentation/Dice_Score":  val_dice,
        "Segmentation/IoU":         val_iou,
        # Sınıflandırma
        "Classification/AUC_ROC":   val_auc,
        "Classification/Recall_Sensitivity":   val_sensitivity,
        "Classification/Specificity_Ozgukluk": val_specificity,
        "Classification/Precision":            val_precision,
        # Öğrenme oranı
        "LR": current_lr,
    })


def log_hnm_stats(fold: int, epoch: int, n_hard: int, n_total_neg: int) -> None:
    if not WANDB_AVAILABLE:
        return
    wandb.log({
        "epoch": epoch,
        "HNM/hard_negatives_found": n_hard,
        "HNM/fp_rate_estimate":     n_hard / max(n_total_neg, 1),
    })


# ── Görsel hata analizi ───────────────────────────────────────────────────────

@torch.no_grad()
def log_error_analysis(
    model: torch.nn.Module,
    dataset,
    val_indices: list[int],
    device: torch.device,
    fold: int,
    n_cases: int = 5,
    img_size: int = 512,
) -> None:
    """
    Val seti üzerinde inference çalıştırır.
    En zorlu FP ve FN vakalarını orijinal grafı + GT maskesi + tahmin ile
    wandb.Table formatında yükler.

    FP (False Positive): Model pnömotoraks dedi, GT negatif
    FN (False Negative): Model normal dedi, GT pozitif
    """
    if not WANDB_AVAILABLE:
        return

    model.eval()
    val_subset = Subset(dataset, val_indices)
    loader     = DataLoader(val_subset, batch_size=8, shuffle=False)

    records: list[dict] = []

    for images, masks, labels in loader:
        images = images.to(device)
        seg_pred, cls_pred = model(images)
        seg_prob = torch.sigmoid(seg_pred)
        cls_prob = torch.sigmoid(cls_pred).squeeze(-1)

        for i in range(len(images)):
            gt_label = int(labels[i].item())
            pred_prob = float(cls_prob[i].item())
            pred_label = int(pred_prob >= 0.5)

            # Sadece yanlış tahminler
            is_fp = (pred_label == 1 and gt_label == 0)
            is_fn = (pred_label == 0 and gt_label == 1)

            if not (is_fp or is_fn):
                continue

            img_np  = images[i, 0].cpu().numpy()
            gt_np   = masks[i, 0].numpy()
            pred_np = seg_prob[i, 0].cpu().numpy()

            # Dice
            pred_bin = (pred_np > 0.5).astype(float)
            inter    = (pred_bin * gt_np).sum()
            dice_val = (2 * inter + 1e-6) / (pred_bin.sum() + gt_np.sum() + 1e-6)

            records.append({
                "type":      "FP" if is_fp else "FN",
                "prob":      pred_prob,
                "dice":      float(dice_val),
                "img_np":    img_np,
                "gt_np":     gt_np,
                "pred_np":   pred_np,
                # FP'de modelin güveni ne kadar yüksek → en kötü FP'ler en üstte
                # FN'de modelin güveni ne kadar düşük → en kaçırılan FN'ler en üstte
                "sort_key":  pred_prob if is_fp else (1 - pred_prob),
            })

    # En zorlu n_cases FP ve FN
    fp_cases = sorted([r for r in records if r["type"] == "FP"],
                      key=lambda x: x["sort_key"], reverse=True)[:n_cases]
    fn_cases = sorted([r for r in records if r["type"] == "FN"],
                      key=lambda x: x["sort_key"], reverse=True)[:n_cases]

    def make_panel_image(rec: dict) -> wandb.Image:
        img   = (rec["img_np"] * 255).astype(np.uint8)
        bgr   = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # GT maskesi — yeşil
        gt_vis = bgr.copy()
        gt_vis[rec["gt_np"] > 0.5] = [0, 200, 0]
        gt_vis = cv2.addWeighted(gt_vis, 0.45, bgr, 0.55, 0)

        # Tahmin maskesi — kırmızı
        pred_bin = (rec["pred_np"] > 0.5).astype(np.uint8)
        pred_vis = bgr.copy()
        pred_vis[pred_bin > 0] = [0, 60, 220]
        pred_vis = cv2.addWeighted(pred_vis, 0.45, bgr, 0.55, 0)

        # Yan yana panel: Orijinal | GT (yeşil) | Tahmin (kırmızı)
        panel = np.hstack([
            cv2.cvtColor(bgr,      cv2.COLOR_BGR2RGB),
            cv2.cvtColor(gt_vis,   cv2.COLOR_BGR2RGB),
            cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB),
        ])

        return wandb.Image(
            panel,
            caption=(
                f"{rec['type']} | Olasılık: {rec['prob']:.3f} | "
                f"Dice: {rec['dice']:.3f} | "
                "[Orijinal | GT-Maske(yeşil) | Tahmin(mavi)]"
            ),
        )

    # W&B Table: sütunlar = Tür, Olasılık, Dice, Panel
    table = wandb.Table(columns=["Tür", "Olasılık", "Dice", "Görsel"])
    for rec in fp_cases + fn_cases:
        table.add_data(
            rec["type"],
            round(rec["prob"], 4),
            round(rec["dice"], 4),
            make_panel_image(rec),
        )

    wandb.log({
        f"fold{fold}/Hata_Analizi_Tablosu": table,
        f"fold{fold}/FP_vakalar": [make_panel_image(r) for r in fp_cases],
        f"fold{fold}/FN_vakalar": [make_panel_image(r) for r in fn_cases],
    })

    print(f"  [W&B] Fold {fold}: {len(fp_cases)} FP + {len(fn_cases)} FN vaka yüklendi.")


# ── 5-Fold Özet ───────────────────────────────────────────────────────────────

def log_kfold_summary(fold_results: list[dict]) -> None:
    """
    Tüm fold'ların ortalaması ve standart sapmasını hesaplayıp
    wandb.summary'e yazar. Yeni bir özet run açar.
    """
    if not WANDB_AVAILABLE or not fold_results:
        return

    metrics = ["best_dice", "best_iou", "best_auc", "best_sensitivity", "best_specificity"]
    summary = {}
    for m in metrics:
        vals = [r.get(m, 0.0) for r in fold_results]
        key  = m.replace("best_", "")
        summary[f"summary/{key}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"summary/{key}_std"]  = round(float(np.std(vals)),  4)

    run = wandb.init(
        project="Pneumothorax-Detection",
        entity="ahmet-ai-t-bi-tak",
        name="kfold-summary",
        group=fold_results[0].get("group", "kfold-run"),
        reinit=True,
    )

    wandb.summary.update(summary)
    wandb.log(summary)

    # Fold karşılaştırma tablosu
    cols  = ["Fold", "Dice", "IoU", "AUC", "Sensitivity", "Specificity"]
    table = wandb.Table(columns=cols)
    for r in fold_results:
        table.add_data(
            r["fold"],
            round(r.get("best_dice",        0.0), 4),
            round(r.get("best_iou",         0.0), 4),
            round(r.get("best_auc",         0.0), 4),
            round(r.get("best_sensitivity", 0.0), 4),
            round(r.get("best_specificity", 0.0), 4),
        )
    wandb.log({"5_Fold_Karsilastirma": table})

    # Konsol özeti
    print("\n" + "=" * 60)
    print("  W&B ÖZET")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<35} {v}")

    run.finish()
