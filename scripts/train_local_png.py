"""
Training Script — PTX-498 (pozitif) + NIH Negatif
===================================================

v2 değişiklikleri (negatif vaka desteği):
  - PTXDataset (image, mask, label) 3-tuple döndürür
  - Cls head aktif: 498 pozitif + 550 NIH negatif ile birlikte eğitilir
  - Loss: DiceLoss (segmentasyon) + BCE (sınıflandırma, ağırlık 0.5)
  - WeightedRandomSampler: pozitif/negatif dengeli batch
  - Val metrikleri: Dice/IoU/HD95 (pozitifler) + AUC/Sensitivity/Specificity (tümü)
  - Site-stratified 5-fold + per-site metrikler korunur

Kullanım:
    python scripts/train_local_png.py \
        --data_root /path/to/PTX-498-v2-fix \
        --nih_root  /path/to/NIH
    python scripts/train_local_png.py ... --resume
    python scripts/train_local_png.py ... --quick_test

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.losses import DiceLoss
from src.model.unet import PneumothoraxModel
from src.preprocessing.augmentation import get_local_train_transforms, get_val_transforms
from src.preprocessing.ptx_dataset import (
    SITE_NAMES,
    PTXDataset,
    build_combined_folds,
    build_nih_negatives,
    build_ptx_manifest,
)
from src.utils.metrics import batch_hausdorff, dice_score, iou_score
from src.utils.results_table import append_fold_result, save_results_table
from src.utils.wandb_utils import init_fold_run, log_epoch_metrics, log_kfold_summary

try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


# ── Sabitler ──────────────────────────────────────────────────────────────────

AUX_DICE_WEIGHT : float = 0.3   # Deep supervision aux head Dice ağırlığı
SEG_THRESHOLD   : float = 0.5   # Sigmoid sonrası segmentasyon binary eşiği
CLS_BCE_WEIGHT  : float = 0.5   # BCE sınıflandırma loss ağırlığı
N_NEGATIVES     : int   = 550   # NIH negatif vaka sayısı


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_root:      Path       = field(default_factory=lambda: ROOT / "data" / "local" / "ptx498")
    nih_root:       Path | None = None
    checkpoint_dir: Path       = field(default_factory=lambda: ROOT / "results" / "checkpoints")
    results_csv:    Path       = field(default_factory=lambda: ROOT / "results" / "ptx_local_kfold.csv")
    encoder_name:   str        = "efficientnet-b0"
    img_size:       int        = 512
    epochs:         int        = 50
    batch_size:     int        = 8
    lr:             float      = 1e-4
    weight_decay:   float      = 1e-4
    num_folds:      int        = 5
    num_workers:    int        = 4
    seed:           int        = 42
    resume:         bool       = False
    no_wandb:       bool       = False
    quick_test:     bool       = False
    wandb_project:  str        = "Pneumothorax-Detection"
    wandb_entity:   str        = "ahmet-ai-t-bi-tak"
    wandb_group:    str        = ""

    def as_wandb_dict(self) -> dict:
        return {
            "encoder_name":         self.encoder_name,
            "img_size":             self.img_size,
            "epochs":               self.epochs,
            "batch_size":           self.batch_size,
            "lr":                   self.lr,
            "weight_decay":         self.weight_decay,
            "num_folds":            self.num_folds,
            "dice_weight":          1.0,
            "cls_bce_weight":       CLS_BCE_WEIGHT,
            "n_negatives":          N_NEGATIVES,
            "wandb_project":        self.wandb_project,
            "wandb_entity":         self.wandb_entity,
            "wandb_group":          self.wandb_group,
            "cv_strategy":          "Site-Stratified K-Fold + NIH Negatives",
            "dataset":              "PTX-498 + NIH No-Finding",
        }


# ── Cihaz ─────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(cfg: TrainConfig, device: torch.device) -> PneumothoraxModel:
    """
    EfficientNet-B0 + UNet++ yükler.
    Cls head aktif: pozitif/negatif ayrımını öğrenir.
    """
    model = PneumothoraxModel(
        encoder_name=cfg.encoder_name,
        pretrained=True,
        in_channels=1,
        deep_supervision=True,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Model: {cfg.encoder_name} + UNet++  |  "
        f"Eğitilecek: {trainable:,}  |  Toplam: {total:,}"
    )
    return model


# ── Balanced sampler ──────────────────────────────────────────────────────────

def _make_sampler(train_df) -> WeightedRandomSampler:
    """Pozitif/negatif dengesini koruyacak şekilde WeightedRandomSampler üretir."""
    labels = train_df["label"].tolist()
    n_pos  = sum(1 for l in labels if l == 1)
    n_neg  = sum(1 for l in labels if l == 0)
    w_pos  = 1.0 / n_pos if n_pos > 0 else 0.0
    w_neg  = 1.0 / n_neg if n_neg > 0 else 0.0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Epoch fonksiyonları ───────────────────────────────────────────────────────

def _train_epoch(
    model:          PneumothoraxModel,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    dice_criterion: DiceLoss,
    bce_criterion:  nn.BCEWithLogitsLoss,
    device:         torch.device,
) -> tuple[float, float]:
    """
    3-tuple (image, mask, label) batch döngüsü.
    Loss = DiceLoss(seg) + AUX * DiceLoss(aux) + CLS_BCE_WEIGHT * BCE(cls)
    """
    model.train()
    total_loss = total_dice = 0.0

    for images, masks, labels in tqdm(loader, desc="  Train", leave=False):
        images   = images.to(device)
        masks_d  = masks.to(device)
        labels_d = labels.to(device)

        optimizer.zero_grad()

        out = model(images)
        if len(out) == 3:               # deep supervision
            seg_pred, cls_pred, aux_preds = out
            seg_loss = dice_criterion(seg_pred, masks_d)
            for aux in aux_preds:
                seg_loss = seg_loss + AUX_DICE_WEIGHT * dice_criterion(aux, masks_d)
        else:
            seg_pred, cls_pred = out
            seg_loss = dice_criterion(seg_pred, masks_d)

        cls_loss = bce_criterion(cls_pred.squeeze(-1), labels_d)
        loss = seg_loss + CLS_BCE_WEIGHT * cls_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(seg_pred.detach(), masks_d).item()

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def _val_epoch(
    model:          PneumothoraxModel,
    loader:         DataLoader,
    dice_criterion: DiceLoss,
    device:         torch.device,
    val_df,
) -> dict:
    """
    Validation:
      - Dice / IoU / HD95 : sadece pozitif vakalar (label==1)
      - AUC / Sensitivity / Specificity : tüm vakalar
      - Per-site : SiteA/B/C pozitif vakalar
    """
    model.eval()
    total_loss = 0.0

    all_preds:  list[torch.Tensor] = []
    all_masks:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, masks, labels in tqdm(loader, desc="  Val  ", leave=False):
        images   = images.to(device)
        masks_d  = masks.to(device)
        labels_d = labels.to(device)

        seg_pred, cls_pred = model(images)
        total_loss += dice_criterion(seg_pred, masks_d).item()

        all_preds.append(seg_pred.cpu())
        all_masks.append(masks.cpu())
        all_labels.append(labels.cpu())

    n          = max(len(loader), 1)
    preds_cat  = torch.cat(all_preds)    # [N, 1, H, W]
    masks_cat  = torch.cat(all_masks)
    labels_cat = torch.cat(all_labels)   # [N]

    # ── Segmentasyon metrikleri — sadece pozitifler ───────────────────────────
    pos_idx = (labels_cat == 1).nonzero(as_tuple=True)[0]
    if len(pos_idx) > 0:
        pos_preds = preds_cat[pos_idx]
        pos_masks = masks_cat[pos_idx]
        val_dice = dice_score(pos_preds, pos_masks).item()
        val_iou  = iou_score(pos_preds,  pos_masks).item()
        hd95     = batch_hausdorff(pos_preds, pos_masks, threshold=SEG_THRESHOLD)
    else:
        val_dice = val_iou = 0.0
        hd95 = float("inf")

    # ── Sınıflandırma metrikleri — tüm vakalar ────────────────────────────────
    cls_probs = torch.sigmoid(
        torch.cat([
            model(imgs.to(device))[1]
            for imgs, _, _ in loader
        ])
    ).squeeze(-1).cpu().numpy()

    # Yukarıdaki re-inference yerine cls logit'leri topla
    # (zaten all_preds'ten ayrı tutmak gerekir — yeniden hesaplayalım)
    # Not: cls_pred ayrı tutulmadığı için ikinci pass gerekiyor; bu yüzden
    # bir sonraki refactoring'de cls_pred'i de toplamak daha verimli olur.
    # Şimdilik auc/sens/spec hesabı için yeniden geçiyoruz (val set küçük).
    model.eval()
    cls_logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            _, cls_p = model(imgs.to(device))
            cls_logits_list.append(cls_p.squeeze(-1).cpu())
    cls_probs_np = torch.sigmoid(torch.cat(cls_logits_list)).numpy()
    labels_np    = labels_cat.numpy()

    if len(np.unique(labels_np)) > 1:
        val_auc = roc_auc_score(labels_np, cls_probs_np)
        preds_bin    = (cls_probs_np >= 0.5).astype(int)
        tp = int(((preds_bin == 1) & (labels_np == 1)).sum())
        tn = int(((preds_bin == 0) & (labels_np == 0)).sum())
        fp = int(((preds_bin == 1) & (labels_np == 0)).sum())
        fn = int(((preds_bin == 0) & (labels_np == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        val_auc = sensitivity = specificity = 0.0

    # ── Per-site segmentasyon metrikleri ──────────────────────────────────────
    val_df_r = val_df.reset_index(drop=True)
    sites     = val_df_r["site"].tolist()
    lbl_list  = val_df_r["label"].tolist()
    per_site: dict[str, dict] = {}

    for site in SITE_NAMES:
        idx = [
            i for i, (s, l) in enumerate(zip(sites, lbl_list))
            if s == site and l == 1
        ]
        if not idx:
            continue
        sp = preds_cat[idx]
        sm = masks_cat[idx]
        per_site[site] = {
            "dice": dice_score(sp, sm).item(),
            "iou":  iou_score(sp,  sm).item(),
            "hd95": batch_hausdorff(sp, sm, threshold=SEG_THRESHOLD),
            "n":    len(idx),
        }

    return {
        "loss":        total_loss / n,
        "dice":        val_dice,
        "iou":         val_iou,
        "hd95":        hd95,
        "auc":         val_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "per_site":    per_site,
    }


# ── W&B ve konsol yardımcıları ────────────────────────────────────────────────

def _log_per_site(fold: int, epoch: int, per_site: dict) -> None:
    if not _WANDB_OK:
        return
    payload: dict = {"epoch": epoch}
    for site, m in per_site.items():
        payload[f"PerSite/{site}/Dice"] = m["dice"]
        payload[f"PerSite/{site}/IoU"]  = m["iou"]
        if not np.isinf(m["hd95"]):
            payload[f"PerSite/{site}/HD95_px"] = m["hd95"]
    _wandb.log(payload)


def _print_per_site(per_site: dict) -> None:
    for site, m in per_site.items():
        hd_str = f"{m['hd95']:.1f}" if not np.isinf(m["hd95"]) else "∞"
        print(
            f"    {site:6s} (n={m['n']:>3}): "
            f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  HD95={hd_str}px"
        )


# ── Ana eğitim döngüsü ────────────────────────────────────────────────────────

def train_kfold_local(cfg: TrainConfig) -> list[dict]:
    device = _get_device()
    print(f"\n  Cihaz: {device}")
    pin_memory = device.type == "cuda"

    if cfg.quick_test:
        cfg.epochs     = 3
        cfg.batch_size = 4
        print("  [QUICK TEST] 3 epoch, batch=4")

    if cfg.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # ── Veri ─────────────────────────────────────────────────────────────────
    print(f"\n  Veri kökü: {cfg.data_root}")
    pos_df = build_ptx_manifest(cfg.data_root)

    if cfg.nih_root is not None:
        print(f"  NIH kökü:  {cfg.nih_root}")
        n_neg = 20 if cfg.quick_test else N_NEGATIVES
        neg_df = build_nih_negatives(cfg.nih_root, n=n_neg, seed=cfg.seed)
    else:
        print("  [UYARI] --nih_root verilmedi, sadece pozitif vakalar kullanılıyor.")
        neg_df = None

    if cfg.quick_test:
        pos_df = pos_df.sample(n=min(40, len(pos_df)), random_state=cfg.seed).reset_index(drop=True)
        if neg_df is not None:
            neg_df = neg_df.sample(n=min(20, len(neg_df)), random_state=cfg.seed).reset_index(drop=True)

    # ── Fold oluştur ──────────────────────────────────────────────────────────
    print()
    if neg_df is not None:
        folds = build_combined_folds(pos_df, neg_df, n_folds=cfg.num_folds, seed=cfg.seed)
    else:
        from src.preprocessing.ptx_dataset import make_site_stratified_folds
        folds = make_site_stratified_folds(pos_df, n_folds=cfg.num_folds, seed=cfg.seed)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_csv.parent.mkdir(parents=True, exist_ok=True)

    group_name = cfg.wandb_group or f"ptx-local-{int(time.time())}"
    wandb_cfg  = cfg.as_wandb_dict()
    wandb_cfg["wandb_group"] = group_name

    progress_file    = cfg.checkpoint_dir / "ptx_fold_progress.json"
    completed_folds: set[int] = set()
    fold_results:    list[dict] = []

    if cfg.resume and progress_file.exists():
        state = json.loads(progress_file.read_text())
        completed_folds = set(state.get("completed_folds", []))
        fold_results    = state.get("fold_results", [])
        if state.get("group_name"):
            group_name = state["group_name"]
            wandb_cfg["wandb_group"] = group_name
        print(f"\n  [RESUME] Tamamlanan fold'lar: {sorted(completed_folds)}")

    dice_criterion = DiceLoss()
    bce_criterion  = nn.BCEWithLogitsLoss()

    # ── Fold döngüsü ──────────────────────────────────────────────────────────
    for fold_i, (train_df, val_df) in enumerate(folds, start=1):
        n_pos_train = int((train_df["label"] == 1).sum())
        n_neg_train = int((train_df["label"] == 0).sum())
        n_pos_val   = int((val_df["label"]   == 1).sum())
        n_neg_val   = int((val_df["label"]   == 0).sum())

        print(f"\n{'='*65}")
        print(
            f"  FOLD {fold_i}/{cfg.num_folds}  —  "
            f"train={len(train_df)} (pos={n_pos_train}, neg={n_neg_train})  "
            f"val={len(val_df)} (pos={n_pos_val}, neg={n_neg_val})"
        )
        print(f"{'='*65}")

        if fold_i in completed_folds:
            print(f"  [RESUME] Fold {fold_i} tamamlandı, atlanıyor.")
            continue

        run = init_fold_run(wandb_cfg, fold_i, cfg.num_folds)

        # ── DataLoaders ───────────────────────────────────────────────────────
        train_ds = PTXDataset(
            train_df,
            transform=get_local_train_transforms(cfg.img_size),
            img_size=cfg.img_size,
        )
        val_ds = PTXDataset(
            val_df,
            transform=get_val_transforms(cfg.img_size),
            img_size=cfg.img_size,
        )

        sampler = _make_sampler(train_df)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=pin_memory,
        )

        # ── Model ve optimizer ────────────────────────────────────────────────
        model = _build_model(cfg, device)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6,
        )

        best_ckpt   = cfg.checkpoint_dir / f"fold_{fold_i}_best.pth"
        resume_ckpt = cfg.checkpoint_dir / f"fold_{fold_i}_resume.pth"

        best_dice:     float = 0.0
        best_iou:      float = 0.0
        best_auc:      float = 0.0
        best_sens:     float = 0.0
        best_spec:     float = 0.0
        best_per_site: dict  = {}
        start_epoch:   int   = 1

        if resume_ckpt.exists():
            try:
                state = torch.load(resume_ckpt, map_location=device)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                best_dice      = state["best_dice"]
                best_iou       = state.get("best_iou",      0.0)
                best_auc       = state.get("best_auc",      0.0)
                best_sens      = state.get("best_sens",     0.0)
                best_spec      = state.get("best_spec",     0.0)
                best_per_site  = state.get("best_per_site", {})
                start_epoch    = state["epoch"] + 1
                print(
                    f"  [RESUME] Fold {fold_i} epoch {start_epoch}'den devam "
                    f"(best Dice: {best_dice:.4f}, AUC: {best_auc:.4f})"
                )
            except Exception as exc:
                print(f"  [RESUME] Checkpoint okunamadı ({exc}), sıfırdan başlanıyor.")
                start_epoch = 1

        # ── Epoch döngüsü ─────────────────────────────────────────────────────
        for epoch in range(start_epoch, cfg.epochs + 1):
            print(f"\n  Epoch {epoch}/{cfg.epochs}")

            train_loss, train_dice = _train_epoch(
                model, train_loader, optimizer,
                dice_criterion, bce_criterion, device
            )
            val_m = _val_epoch(model, val_loader, dice_criterion, device, val_df)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            hd_str = f"{val_m['hd95']:.1f}" if not np.isinf(val_m["hd95"]) else "∞"
            print(
                f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
                f"  Val   → Loss: {val_m['loss']:.4f}  Dice: {val_m['dice']:.4f}  "
                f"IoU: {val_m['iou']:.4f}  HD95: {hd_str}px  LR: {current_lr:.2e}\n"
                f"          AUC: {val_m['auc']:.4f}  Sens: {val_m['sensitivity']:.4f}  "
                f"Spec: {val_m['specificity']:.4f}"
            )
            _print_per_site(val_m["per_site"])

            log_epoch_metrics(
                fold=fold_i, epoch=epoch,
                train_loss=train_loss,
                val_loss=val_m["loss"],
                val_dice=val_m["dice"],
                val_iou=val_m["iou"],
                val_auc=val_m["auc"],
                val_sensitivity=val_m["sensitivity"],
                val_specificity=val_m["specificity"],
                val_precision=0.0,
                current_lr=current_lr,
                val_hausdorff=val_m["hd95"],
            )
            _log_per_site(fold_i, epoch, val_m["per_site"])

            if val_m["dice"] > best_dice:
                best_dice      = val_m["dice"]
                best_iou       = val_m["iou"]
                best_auc       = val_m["auc"]
                best_sens      = val_m["sensitivity"]
                best_spec      = val_m["specificity"]
                best_per_site  = val_m["per_site"]
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ✓ Checkpoint: {best_ckpt.name}  (Dice: {best_dice:.4f}  AUC: {best_auc:.4f})")

            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_dice":     best_dice,
                "best_iou":      best_iou,
                "best_auc":      best_auc,
                "best_sens":     best_sens,
                "best_spec":     best_spec,
                "best_per_site": best_per_site,
            }, resume_ckpt)

        # ── Fold sonu ─────────────────────────────────────────────────────────
        if resume_ckpt.exists():
            resume_ckpt.unlink()

        append_fold_result(
            fold_results, fold_i,
            best_dice=best_dice,      best_auc=best_auc,
            best_iou=best_iou,        best_sensitivity=best_sens,
            per_site=best_per_site,
        )

        completed_folds.add(fold_i)
        progress_file.write_text(json.dumps({
            "group_name":      group_name,
            "completed_folds": sorted(completed_folds),
            "fold_results":    fold_results,
        }, indent=2, ensure_ascii=False))

        if run:
            run.finish()

    # ── Tüm fold'lar tamamlandı ───────────────────────────────────────────────
    save_results_table(fold_results, output_path=str(cfg.results_csv))
    log_kfold_summary(fold_results)

    if progress_file.exists():
        progress_file.unlink()

    return fold_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="PTX-498 + NIH Negatif — Site-stratified 5-fold CV"
    )
    p.add_argument("--data_root", required=True,
                   help="PTX-498-v2-fix kök dizini (SiteA/, SiteB/, SiteC/)")
    p.add_argument("--nih_root", default=None,
                   help="NIH ChestX-ray14 kök dizini (Data_Entry_2017.csv + images_*/images/)")
    p.add_argument("--checkpoint_dir", default=str(ROOT / "results" / "checkpoints"))
    p.add_argument("--results_csv",    default=str(ROOT / "results" / "ptx_local_kfold.csv"))
    p.add_argument("--encoder",        default="efficientnet-b0")
    p.add_argument("--img_size",   type=int,   default=512)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_folds",  type=int,   default=5)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--no_wandb",   action="store_true")
    p.add_argument("--quick_test", action="store_true",
                   help="3 epoch, 40+20 sample — altyapı testi")
    p.add_argument("--wandb_project", default="Pneumothorax-Detection")
    p.add_argument("--wandb_entity",  default="ahmet-ai-t-bi-tak")
    p.add_argument("--wandb_group",   default="")

    args = p.parse_args()
    return TrainConfig(
        data_root      = Path(args.data_root),
        nih_root       = Path(args.nih_root) if args.nih_root else None,
        checkpoint_dir = Path(args.checkpoint_dir),
        results_csv    = Path(args.results_csv),
        encoder_name   = args.encoder,
        img_size       = args.img_size,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        num_folds      = args.num_folds,
        num_workers    = args.num_workers,
        seed           = args.seed,
        resume         = args.resume,
        no_wandb       = args.no_wandb,
        quick_test     = args.quick_test,
        wandb_project  = args.wandb_project,
        wandb_entity   = args.wandb_entity,
        wandb_group    = args.wandb_group,
    )


if __name__ == "__main__":
    cfg = _parse_args()

    print("\n" + "═" * 65)
    print("  PTX-498 + NIH NEGATİF EĞİTİMİ  —  TÜBİTAK 2209-A")
    print("═" * 65)
    print(f"  Veri kökü     : {cfg.data_root}")
    print(f"  NIH kökü      : {cfg.nih_root or 'yok (sadece pozitif)'}")
    print(f"  Encoder       : {cfg.encoder_name}")
    print(f"  Epochs        : {cfg.epochs}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Fold sayısı   : {cfg.num_folds}")
    print(f"  Img size      : {cfg.img_size}×{cfg.img_size}")
    print(f"  Loss          : Dice + {CLS_BCE_WEIGHT}×BCE (cls aktif)")
    print(f"  W&B           : {'hayır' if cfg.no_wandb else 'evet'}")
    print(f"  Quick test    : {'evet' if cfg.quick_test else 'hayır'}")
    print("═" * 65)

    results = train_kfold_local(cfg)
