"""
Eğitim Döngüsü — Tam W&B Entegrasyonlu
Stratified K-Fold + WeightedRandomSampler + ReduceLROnPlateau
+ Hard Negative Mining + Görsel Hata Analizi + Özet Tablo

TÜBİTAK 2209-A | Ahmet Demir
"""

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.model.losses import CombinedLoss
from src.model.unet import PneumothoraxModel
from src.utils.hard_negative_mining import (
    find_hard_negatives,
    update_sampler_with_hard_negatives,
)
from src.utils.metrics import compute_auc, dice_score, iou_score
from src.utils.results_table import append_fold_result, save_results_table
from src.utils.wandb_utils import (
    init_fold_run,
    log_epoch_metrics,
    log_error_analysis,
    log_hnm_stats,
    log_kfold_summary,
)


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    labels_arr    = np.array(labels)
    class_counts  = np.bincount(labels_arr)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Metrik hesaplama ──────────────────────────────────────────────────────────

def _compute_cls_metrics(
    all_preds: torch.Tensor, all_targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    """
    Sınıflandırma metriklerini hesaplar.
    Dönen metrikler: auc, sensitivity, specificity, precision
    """
    probs    = torch.sigmoid(all_preds).squeeze()
    pred_bin = (probs > threshold).float()
    targets  = all_targets.float()

    tp = ((pred_bin == 1) & (targets == 1)).sum().item()
    tn = ((pred_bin == 0) & (targets == 0)).sum().item()
    fp = ((pred_bin == 1) & (targets == 0)).sum().item()
    fn = ((pred_bin == 0) & (targets == 1)).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)          # Recall / Duyarlılık
    specificity = tn / (tn + fp + 1e-8)          # Özgüllük
    precision   = tp / (tp + fp + 1e-8)          # Kesinlik

    try:
        auc = compute_auc(all_preds, all_targets)
    except ValueError:
        auc = 0.0

    return {
        "auc":         auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
    }


# ── Epoch fonksiyonları ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss = total_dice = 0.0

    for images, masks, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        seg_pred, cls_pred = model(images)
        loss = criterion(seg_pred, masks, cls_pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(seg_pred, masks).item()

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    all_cls_preds, all_cls_targets = [], []

    for images, masks, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        seg_pred, cls_pred = model(images)
        loss = criterion(seg_pred, masks, cls_pred, labels)

        total_loss += loss.item()
        total_dice += dice_score(seg_pred, masks).item()
        total_iou  += iou_score(seg_pred, masks).item()

        all_cls_preds.append(cls_pred.cpu())
        all_cls_targets.append(labels.cpu())

    n           = len(loader)
    all_preds   = torch.cat(all_cls_preds)
    all_targets = torch.cat(all_cls_targets)
    cls_metrics = _compute_cls_metrics(all_preds, all_targets)

    return {
        "loss":        total_loss / n,
        "dice":        total_dice / n,
        "iou":         total_iou  / n,
        **cls_metrics,
    }


# ── K-Fold eğitim ─────────────────────────────────────────────────────────────

def train_kfold(dataset, config: dict) -> list[dict]:
    """
    Stratified K-Fold Cross Validation.
    Tam W&B entegrasyonu: her fold ayrı run, tüm metrikler ve görsel hata analizi.

    config anahtarları:
        epochs, batch_size, lr, weight_decay
        num_folds            (default 5)
        checkpoint_dir
        encoder_name         (default "efficientnet-b0")
        hard_negative_mining (default True)
        hnm_interval         (default 3)
        hnm_threshold        (default 0.4)
        hnm_multiplier       (default 3.0)
        dice_weight          (default 0.5)
        wandb_project        (default "Pneumothorax-Detection")
        wandb_entity         (default "ahmet-ai-t-bi-tak")
        results_csv          (default "results/kfold_results.csv")
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = config.get("num_folds", 5)
    criterion = CombinedLoss(
        dice_weight=config.get("dice_weight", 0.5),
        bce_weight=1 - config.get("dice_weight", 0.5),
    )

    # Tüm fold run'larını gruplamak için ortak bir isim
    import time
    group_name = config.get("wandb_group", f"kfold-{int(time.time())}")
    config["wandb_group"] = group_name

    # Etiketler
    labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
    skf    = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{num_folds}  —  Eğitim: {len(train_idx)}  Val: {len(val_idx)}")
        print(f"{'='*60}")

        # ── W&B: fold başına ayrı run ─────────────────────────────────────
        run = init_fold_run(config, fold, num_folds)

        # ── DataLoaders ───────────────────────────────────────────────────
        train_subset      = Subset(dataset, list(train_idx))
        val_subset        = Subset(dataset, list(val_idx))
        train_labels_fold = [labels[i] for i in train_idx]
        neg_indices_fold  = [i for i, l in enumerate(train_labels_fold) if l == 0]

        sampler      = make_weighted_sampler(train_labels_fold)
        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"],
            sampler=sampler, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True,
        )

        # ── Model ve optimizer ────────────────────────────────────────────
        model = PneumothoraxModel(
            encoder_name=config.get("encoder_name", "efficientnet-b0")
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # ── Epoch döngüsü ─────────────────────────────────────────────────
        best = {"dice": 0.0, "iou": 0.0, "auc": 0.0,
                "sensitivity": 0.0, "specificity": 0.0}
        ckpt_path    = f"{config['checkpoint_dir']}/fold{fold}_best.pth"
        hnm_enabled  = config.get("hard_negative_mining", True)
        hnm_interval = config.get("hnm_interval", 3)

        for epoch in range(1, config["epochs"] + 1):
            print(f"\n  Epoch {epoch}/{config['epochs']}")

            train_loss, train_dice = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = val_epoch(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
                f"  Val   → Loss: {val_metrics['loss']:.4f}  "
                f"Dice: {val_metrics['dice']:.4f}  IoU: {val_metrics['iou']:.4f}  "
                f"AUC: {val_metrics['auc']:.4f}  "
                f"Sens(Recall): {val_metrics['sensitivity']:.4f}  "
                f"Spec(Özgüllük): {val_metrics['specificity']:.4f}  "
                f"LR: {current_lr:.2e}"
            )

            # ── W&B: epoch metrikleri ─────────────────────────────────
            log_epoch_metrics(
                fold=fold,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_dice=val_metrics["dice"],
                val_iou=val_metrics["iou"],
                val_auc=val_metrics["auc"],
                val_sensitivity=val_metrics["sensitivity"],
                val_specificity=val_metrics["specificity"],
                val_precision=val_metrics["precision"],
                current_lr=current_lr,
            )

            # ── Hard Negative Mining ──────────────────────────────────
            if hnm_enabled and epoch % hnm_interval == 0:
                print("  [HNM] Zor negatifler taranıyor…")
                hard_negs = find_hard_negatives(
                    model, train_subset, neg_indices_fold, device,
                    fp_threshold=config.get("hnm_threshold", 0.4),
                )
                if hard_negs:
                    print(f"  [HNM] {len(hard_negs)} hard negative → sampler güncellendi.")
                    new_sampler  = update_sampler_with_hard_negatives(
                        train_labels_fold, hard_negs,
                        hard_neg_multiplier=config.get("hnm_multiplier", 3.0),
                    )
                    train_loader = DataLoader(
                        train_subset, batch_size=config["batch_size"],
                        sampler=new_sampler, num_workers=4, pin_memory=True,
                    )
                    log_hnm_stats(fold, epoch, len(hard_negs), len(neg_indices_fold))

            # ── Checkpoint ───────────────────────────────────────────
            if val_metrics["dice"] > best["dice"]:
                best.update({
                    "dice":        val_metrics["dice"],
                    "iou":         val_metrics["iou"],
                    "auc":         val_metrics["auc"],
                    "sensitivity": val_metrics["sensitivity"],
                    "specificity": val_metrics["specificity"],
                })
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Checkpoint: {ckpt_path}  (Dice: {best['dice']:.4f})")

        # ── Fold sonu: görsel hata analizi ────────────────────────────────
        print(f"\n  [W&B] Fold {fold} hata analizi yükleniyor…")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log_error_analysis(
            model, dataset, list(val_idx), device, fold=fold, n_cases=5
        )

        # Fold sonuçlarını kaydet
        append_fold_result(
            fold_results, fold,
            best_dice=best["dice"],        best_auc=best["auc"],
            best_iou=best["iou"],          best_sensitivity=best["sensitivity"],
        )
        fold_results[-1]["best_specificity"] = best["specificity"]
        fold_results[-1]["group"]            = group_name

        if run:
            run.finish()

    # ── Tüm fold'lar bitti ────────────────────────────────────────────────────

    # Results CSV
    save_results_table(
        fold_results,
        output_path=config.get("results_csv", "results/kfold_results.csv"),
    )

    # W&B özet run
    log_kfold_summary(fold_results)

    return fold_results


# ── Tek split (hızlı deney) ───────────────────────────────────────────────────

def train(config: dict) -> None:
    """Tek train/val split. Hızlı deney ve W&B sweep amaçlı."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = init_fold_run(config, fold=1, num_folds=1)

    model     = PneumothoraxModel(encoder_name=config.get("encoder_name", "efficientnet-b0")).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_dice = 0.0
    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_loss, train_dice = train_epoch(
            model, config["train_loader"], optimizer, criterion, device
        )
        val_metrics = val_epoch(model, config["val_loader"], criterion, device)
        scheduler.step(val_metrics["loss"])

        print(
            f"Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
            f"Val   → Loss: {val_metrics['loss']:.4f}  "
            f"Dice: {val_metrics['dice']:.4f}  AUC: {val_metrics['auc']:.4f}  "
            f"Sens: {val_metrics['sensitivity']:.4f}  Spec: {val_metrics['specificity']:.4f}"
        )

        log_epoch_metrics(
            fold=1, epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            val_dice=val_metrics["dice"],
            val_iou=val_metrics["iou"],
            val_auc=val_metrics["auc"],
            val_sensitivity=val_metrics["sensitivity"],
            val_specificity=val_metrics["specificity"],
            val_precision=val_metrics["precision"],
            current_lr=optimizer.param_groups[0]["lr"],
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            ckpt = config.get("checkpoint_path", "results/checkpoints/best_model.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"✓ Best model kaydedildi (Dice: {best_dice:.4f})")

    if run:
        run.finish()
    print(f"\nEğitim tamamlandı. En iyi Dice: {best_dice:.4f}")
