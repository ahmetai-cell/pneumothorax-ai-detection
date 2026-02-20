"""
Eğitim Döngüsü
K-Fold CV + WeightedRandomSampler + ReduceLROnPlateau
+ Hard Negative Mining + W&B + Results Table

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
from src.utils.wandb_utils import log_hnm_stats, log_prediction_samples

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Sınıf dengesizliğini gidermek için WeightedRandomSampler.
    Pnömotoraks nadir (~%20 pozitif) → pozitif örnekler daha sık seçilir.
    """
    labels_arr    = np.array(labels)
    class_counts  = np.bincount(labels_arr)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Epoch fonksiyonları ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
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
def val_epoch(model, loader, criterion, device):
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

    try:
        auc = compute_auc(all_preds, all_targets)
    except ValueError:
        auc = 0.0

    # Sensitivity (Recall for positive class)
    probs    = torch.sigmoid(all_preds).squeeze()
    pred_bin = (probs > 0.5).float()
    tp = ((pred_bin == 1) & (all_targets == 1)).sum().item()
    fn = ((pred_bin == 0) & (all_targets == 1)).sum().item()
    sensitivity = tp / (tp + fn + 1e-8)

    return total_loss / n, total_dice / n, total_iou / n, auc, sensitivity


# ── K-Fold eğitim ─────────────────────────────────────────────────────────────

def train_kfold(dataset, config: dict) -> list[dict]:
    """
    Stratified K-Fold Cross Validation.
    W&B logging, HNM, WeightedRandomSampler ve Results Table entegreli.

    Config anahtarları:
        epochs, batch_size, lr, weight_decay
        num_folds          (default 5)
        checkpoint_dir
        encoder_name       (default "efficientnet-b0")
        hard_negative_mining (default True)
        hnm_interval       (default 3)
        hnm_threshold      (default 0.4)
        hnm_multiplier     (default 3.0)
        wandb_project      (default "pneumothorax-tubitak")
        results_csv        (default "results/kfold_results.csv")
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CombinedLoss(
        dice_weight=config.get("dice_weight", 0.5),
        bce_weight=1 - config.get("dice_weight", 0.5),
    )

    # W&B başlat
    use_wandb = WANDB_AVAILABLE and config.get("wandb_project")
    if use_wandb:
        wandb.init(
            project=config["wandb_project"],
            config=config,
            reinit=True,
        )

    # Etiketler
    labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
    skf    = StratifiedKFold(
        n_splits=config.get("num_folds", 5), shuffle=True, random_state=42
    )
    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold}/{config.get('num_folds', 5)}")
        print(f"{'='*55}")

        train_subset = Subset(dataset, list(train_idx))
        val_subset   = Subset(dataset, list(val_idx))

        train_labels_fold = [labels[i] for i in train_idx]
        neg_indices_fold  = [i for i, l in enumerate(train_labels_fold) if l == 0]
        sampler           = make_weighted_sampler(train_labels_fold)

        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"],
            sampler=sampler, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True,
        )

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

        best_dice = best_auc = best_iou = best_sens = 0.0
        ckpt_path = f"{config['checkpoint_dir']}/fold{fold}_best.pth"

        hnm_enabled  = config.get("hard_negative_mining", True)
        hnm_interval = config.get("hnm_interval", 3)

        for epoch in range(1, config["epochs"] + 1):
            print(f"\n  Epoch {epoch}/{config['epochs']}")

            train_loss, train_dice = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_dice, val_iou, val_auc, val_sens = val_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
                f"  Val   → Loss: {val_loss:.4f}   Dice: {val_dice:.4f}  "
                f"IoU: {val_iou:.4f}  AUC: {val_auc:.4f}  "
                f"Sens: {val_sens:.4f}  LR: {current_lr:.2e}"
            )

            # ── W&B per-epoch log ─────────────────────────────────────────────
            if use_wandb:
                wandb.log({
                    "epoch":              epoch,
                    "fold":               fold,
                    "train/loss":         train_loss,
                    "train/dice":         train_dice,
                    "val/loss":           val_loss,
                    "val/dice":           val_dice,
                    "val/iou":            val_iou,
                    "val/auc":            val_auc,
                    "val/sensitivity":    val_sens,
                    "lr":                 current_lr,
                })

            # ── Hard Negative Mining ──────────────────────────────────────────
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

            # ── Checkpoint ───────────────────────────────────────────────────
            if val_dice > best_dice:
                best_dice = val_dice
                best_auc  = val_auc
                best_iou  = val_iou
                best_sens = val_sens
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Checkpoint: {ckpt_path}")

        # ── Fold sonu: görsel tahmin log'u ───────────────────────────────────
        if use_wandb:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            log_prediction_samples(
                model, dataset, list(val_idx), device, fold=fold
            )

        append_fold_result(
            fold_results, fold,
            best_dice=best_dice, best_auc=best_auc,
            best_iou=best_iou,  best_sensitivity=best_sens,
        )

    # ── Tüm fold'lar bitti: tablo kaydet ────────────────────────────────────
    results_df = save_results_table(
        fold_results,
        output_path=config.get("results_csv", "results/kfold_results.csv"),
    )

    # W&B özet tablosu
    if use_wandb:
        wandb.log({"kfold_results": wandb.Table(dataframe=results_df)})
        wandb.finish()

    return fold_results


# ── Tek split (hızlı deney) ───────────────────────────────────────────────────

def train(config: dict) -> None:
    """Tek train/val split. Hızlı deney amaçlı, W&B opsiyonel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_wandb = WANDB_AVAILABLE and config.get("wandb_project")
    if use_wandb:
        wandb.init(project=config["wandb_project"], config=config)

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
        val_loss, val_dice, val_iou, val_auc, val_sens = val_epoch(
            model, config["val_loader"], criterion, device
        )
        scheduler.step(val_loss)

        print(
            f"Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
            f"Val   → Loss: {val_loss:.4f}   Dice: {val_dice:.4f}  "
            f"IoU: {val_iou:.4f}  AUC: {val_auc:.4f}  Sens: {val_sens:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/dice": train_dice,
                "val/loss": val_loss,     "val/dice": val_dice,
                "val/iou": val_iou,       "val/auc": val_auc,
                "val/sensitivity": val_sens,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt = config.get("checkpoint_path", "results/checkpoints/best_model.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"✓ Best model kaydedildi (Dice: {best_dice:.4f})")

    if use_wandb:
        wandb.finish()
    print(f"\nEğitim tamamlandı. En iyi Dice: {best_dice:.4f}")
