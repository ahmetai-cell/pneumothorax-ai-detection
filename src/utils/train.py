"""
Eğitim Döngüsü — K-Fold CV + WeightedRandomSampler + ReduceLROnPlateau

Değişiklikler:
  - CosineAnnealingLR → ReduceLROnPlateau (val loss'a tepkisel LR düşürme)
  - Sabit train/test split → Stratified K-Fold (5 fold)
  - Sınıf dengesizliği → WeightedRandomSampler

TÜBİTAK 2209-A | Ahmet Demir
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np

from src.model.unet import PneumothoraxModel
from src.model.losses import CombinedLoss
from src.utils.metrics import dice_score, iou_score, compute_auc


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Sınıf dengesizliğini gidermek için WeightedRandomSampler oluşturur.

    Pnömotoraks nadir bir durum (örn. %20 pozitif).
    Bu sampler her epoch'ta pozitif/negatif örnekleri dengeli örnekler.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    # Her sınıfın ağırlığı: az olan sınıf daha sık seçilir
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

    n = len(loader)
    all_preds   = torch.cat(all_cls_preds)
    all_targets = torch.cat(all_cls_targets)

    try:
        auc = compute_auc(all_preds, all_targets)
    except ValueError:
        auc = 0.0  # tek sınıf bulunuyorsa (küçük val set)

    return total_loss / n, total_dice / n, total_iou / n, auc


# ── K-Fold eğitim ────────────────────────────────────────────────────────────

def train_kfold(dataset, config: dict) -> list[dict]:
    """
    Stratified 5-Fold Cross Validation.

    Args:
        dataset : PneumothoraxDataset veya DicomSlicerDataset
        config  : {
            epochs, batch_size, lr, weight_decay,
            num_folds, checkpoint_dir, encoder_name
          }

    Returns:
        fold_results : Her fold için {"fold", "best_dice", "best_auc"} listesi
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CombinedLoss()

    # Etiketleri topla (stratify için)
    labels = [int(dataset[i][2].item()) for i in range(len(dataset))]

    skf = StratifiedKFold(
        n_splits=config.get("num_folds", 5), shuffle=True, random_state=42
    )
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold}/{config.get('num_folds', 5)}")
        print(f"{'='*50}")

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        # WeightedRandomSampler sadece train set için
        train_labels = [labels[i] for i in train_idx]
        sampler      = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = PneumothoraxModel(
            encoder_name=config.get("encoder_name", "efficientnet-b0")
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
        # ReduceLROnPlateau: val loss iyileşmezse LR'yi yarıya indir
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        best_dice = best_auc = 0.0
        ckpt_path = f"{config['checkpoint_dir']}/fold{fold}_best.pth"

        for epoch in range(1, config["epochs"] + 1):
            print(f"\n  Epoch {epoch}/{config['epochs']}")

            train_loss, train_dice = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_dice, val_iou, val_auc = val_epoch(
                model, val_loader, criterion, device
            )

            scheduler.step(val_loss)

            print(
                f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
                f"  Val   → Loss: {val_loss:.4f}   Dice: {val_dice:.4f}  "
                f"IoU: {val_iou:.4f}  AUC: {val_auc:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                best_auc  = val_auc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Checkpoint kaydedildi: {ckpt_path}")

        fold_results.append(
            {"fold": fold, "best_dice": best_dice, "best_auc": best_auc}
        )

    # Özet
    print("\n" + "=" * 50)
    print("K-FOLD SONUÇLARI")
    print("=" * 50)
    mean_dice = np.mean([r["best_dice"] for r in fold_results])
    mean_auc  = np.mean([r["best_auc"]  for r in fold_results])
    for r in fold_results:
        print(f"  Fold {r['fold']}: Dice={r['best_dice']:.4f}  AUC={r['best_auc']:.4f}")
    print(f"\n  Ortalama Dice: {mean_dice:.4f}")
    print(f"  Ortalama AUC : {mean_auc:.4f}")

    return fold_results


# ── Tek split eğitim (hızlı deney için) ──────────────────────────────────────

def train(config: dict) -> None:
    """Tek train/val split ile klasik eğitim. Hızlı deney amaçlı."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = PneumothoraxModel(
        encoder_name=config.get("encoder_name", "efficientnet-b0")
    ).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_dice = 0.0
    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss, train_dice = train_epoch(
            model, config["train_loader"], optimizer, criterion, device
        )
        val_loss, val_dice, val_iou, val_auc = val_epoch(
            model, config["val_loader"], criterion, device
        )
        scheduler.step(val_loss)

        print(
            f"Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
            f"Val   → Loss: {val_loss:.4f}   Dice: {val_dice:.4f}  "
            f"IoU: {val_iou:.4f}  AUC: {val_auc:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config.get("checkpoint_path", "results/checkpoints/best_model.pth"))
            print(f"✓ Best model kaydedildi (Dice: {best_dice:.4f})")

    print(f"\nEğitim tamamlandı. En iyi Dice: {best_dice:.4f}")
