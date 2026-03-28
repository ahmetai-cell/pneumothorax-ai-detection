"""
Training Script — 498 Yerel PNG Vakası (PTX-498)
================================================

train.py'den NEDEN farklı:
  - PTXDataset (image, mask) döndürür — label yok (hepsi pozitif)
  - Cls head frozen: parametreler dondurulmuş, loss'a dahil edilmez
  - Sadece DiceLoss (+ deep supervision aux * 0.3)
  - Fold yönetimi: make_site_stratified_folds() — StratifiedKFold değil
  - Hard Negative Mining yok (negatif vaka sıfır)
  - Per-site Dice / IoU / HD95 her epoch sonunda raporlanır

train.py'den yeniden kullanılan:
  - DiceLoss (src/model/losses.py)
  - dice_score, iou_score, batch_hausdorff (src/utils/metrics.py)
  - init_fold_run, log_epoch_metrics, log_kfold_summary (src/utils/wandb_utils.py)
  - append_fold_result, save_results_table (src/utils/results_table.py)
  - PneumothoraxModel (src/model/unet.py)
  - get_local_train_transforms, get_val_transforms (src/preprocessing/augmentation.py)

Kullanım:
    python scripts/train_local_png.py --data_root /path/to/PTX-498-v2-fix
    python scripts/train_local_png.py --data_root /path/to/PTX-498-v2-fix --quick_test
    python scripts/train_local_png.py --data_root /path/to/PTX-498-v2-fix --resume

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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.losses import DiceLoss
from src.model.unet import PneumothoraxModel
from src.preprocessing.augmentation import get_local_train_transforms, get_val_transforms
from src.preprocessing.ptx_dataset import (
    SITE_NAMES,
    PTXDataset,
    build_ptx_manifest,
    make_site_stratified_folds,
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

AUX_DICE_WEIGHT: float = 0.3   # Deep supervision aux head Dice ağırlığı
SEG_THRESHOLD:   float = 0.5   # Sigmoid sonrası binary eşik


# ── Config (argparse dicts değil, @dataclass) ─────────────────────────────────

@dataclass
class TrainConfig:
    data_root:      Path  = field(default_factory=lambda: ROOT / "data" / "local" / "ptx498")
    checkpoint_dir: Path  = field(default_factory=lambda: ROOT / "results" / "checkpoints")
    results_csv:    Path  = field(default_factory=lambda: ROOT / "results" / "ptx_local_kfold.csv")
    encoder_name:   str   = "efficientnet-b0"
    img_size:       int   = 512
    epochs:         int   = 50
    batch_size:     int   = 8
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    num_folds:      int   = 5
    num_workers:    int   = 4
    seed:           int   = 42
    resume:         bool  = False
    no_wandb:       bool  = False
    quick_test:     bool  = False
    wandb_project:  str   = "Pneumothorax-Detection"
    wandb_entity:   str   = "ahmet-ai-t-bi-tak"
    wandb_group:    str   = ""

    def as_wandb_dict(self) -> dict:
        """init_fold_run() tarafından beklenen dict formatı."""
        return {
            "encoder_name":  self.encoder_name,
            "img_size":      self.img_size,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "lr":            self.lr,
            "weight_decay":  self.weight_decay,
            "num_folds":     self.num_folds,
            "dice_weight":   1.0,   # Sadece Dice loss
            "wandb_project": self.wandb_project,
            "wandb_entity":  self.wandb_entity,
            "wandb_group":   self.wandb_group,
            # cls head kapalı — HNM yok
            "hard_negative_mining": False,
            "cv_strategy":          "Site-Stratified K-Fold",
            "dataset":              "PTX-498 (SiteA/B/C, all positive)",
        }


# ── Cihaz seçimi ──────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model kurulumu ────────────────────────────────────────────────────────────

def _build_model(cfg: TrainConfig, device: torch.device) -> PneumothoraxModel:
    """
    EfficientNet-B0 + UNet++ yükler, cls head'i dondurur.
    deep_supervision=True korunur: aux head'ler segmentasyona katkı sağlar.
    """
    model = PneumothoraxModel(
        encoder_name=cfg.encoder_name,
        pretrained=True,
        in_channels=1,
        deep_supervision=True,
    ).to(device)

    # Cls head dondur: tüm vakalar pozitif, label anlamsız
    frozen = 0
    for param in model.classifier.parameters():
        param.requires_grad = False
        frozen += param.numel()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Model: {cfg.encoder_name} + UNet++  |  "
        f"Eğitilecek: {trainable:,}  |  Frozen (cls head): {frozen:,}  |  "
        f"Toplam: {total:,}"
    )
    return model


# ── Epoch fonksiyonları ───────────────────────────────────────────────────────

def _train_epoch(
    model:     PneumothoraxModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DiceLoss,
    device:    torch.device,
) -> tuple[float, float]:
    """
    PTXDataset 2-tuple döndürür: (image, mask) — labels yok.
    Deep supervision aktif (model.train() → 3-çıktı): aux head'ler için ek Dice.
    """
    model.train()
    total_loss = total_dice = 0.0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images  = images.to(device)
        masks_d = masks.to(device)

        optimizer.zero_grad()

        out = model(images)
        if len(out) == 3:               # deep supervision: (seg, cls, [aux1, aux2])
            seg_pred, _, aux_preds = out
            loss = criterion(seg_pred, masks_d)
            for aux in aux_preds:
                loss = loss + AUX_DICE_WEIGHT * criterion(aux, masks_d)
        else:                           # model.eval() veya deep_supervision=False
            seg_pred, _ = out
            loss = criterion(seg_pred, masks_d)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(seg_pred.detach(), masks_d).item()

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def _val_epoch(
    model:     PneumothoraxModel,
    loader:    DataLoader,
    criterion: DiceLoss,
    device:    torch.device,
    val_df,                    # pd.DataFrame — 'site' sütunu per-site metrik için
) -> dict:
    """
    Validation: Dice, IoU, HD95 (global + per-site).
    model.eval() → deep supervision kapalı, (seg_pred, cls_pred) döner.
    """
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    all_preds: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images  = images.to(device)
        masks_d = masks.to(device)

        seg_pred, _ = model(images)    # cls_pred yok sayılır
        total_loss += criterion(seg_pred, masks_d).item()
        total_dice += dice_score(seg_pred, masks_d).item()
        total_iou  += iou_score(seg_pred,  masks_d).item()

        all_preds.append(seg_pred.cpu())
        all_masks.append(masks.cpu())

    n          = max(len(loader), 1)
    preds_cat  = torch.cat(all_preds)   # [N, 1, H, W]
    masks_cat  = torch.cat(all_masks)

    hd95 = batch_hausdorff(preds_cat, masks_cat, threshold=SEG_THRESHOLD)

    # ── Per-site metrikler ────────────────────────────────────────────────────
    sites = val_df["site"].tolist()
    per_site: dict[str, dict] = {}
    for site in SITE_NAMES:
        idx = [i for i, s in enumerate(sites) if s == site]
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
        "loss":     total_loss / n,
        "dice":     total_dice / n,
        "iou":      total_iou  / n,
        "hd95":     hd95,
        "per_site": per_site,
    }


# ── W&B per-site log ──────────────────────────────────────────────────────────

def _log_per_site(fold: int, epoch: int, per_site: dict) -> None:
    """Per-site metrikleri W&B'ye gönderir (wandb_utils.py'ye eklenmedi — lokal)."""
    if not _WANDB_OK:
        return
    payload: dict = {"epoch": epoch}
    for site, m in per_site.items():
        payload[f"PerSite/{site}/Dice"] = m["dice"]
        payload[f"PerSite/{site}/IoU"]  = m["iou"]
        if not np.isinf(m["hd95"]):
            payload[f"PerSite/{site}/HD95_px"] = m["hd95"]
    _wandb.log(payload)


# ── Per-site konsol çıktısı ───────────────────────────────────────────────────

def _print_per_site(per_site: dict) -> None:
    for site, m in per_site.items():
        hd_str = f"{m['hd95']:.1f}" if not np.isinf(m["hd95"]) else "∞"
        print(
            f"    {site:6s} (n={m['n']:>3}): "
            f"Dice={m['dice']:.4f}  IoU={m['iou']:.4f}  HD95={hd_str}px"
        )


# ── Ana eğitim döngüsü ────────────────────────────────────────────────────────

def train_kfold_local(cfg: TrainConfig) -> list[dict]:
    """
    Site-stratified 5-fold CV.

    train.py'deki train_kfold()'dan farklı:
      - Dataset: PTXDataset (2-tuple, no labels)
      - Loss: DiceLoss only (no CombinedLoss, no cls branch)
      - Fold split: make_site_stratified_folds() — site orantısını korur
      - HNM: yok (negatif vaka sıfır)
    """
    device = _get_device()
    print(f"\n  Cihaz: {device}")

    # pin_memory sadece CUDA'da desteklenir
    pin_memory = device.type == "cuda"

    # quick_test: 3 epoch, küçük loader
    if cfg.quick_test:
        cfg.epochs     = 3
        cfg.batch_size = 4
        print("  [QUICK TEST] 3 epoch, batch=4")

    # W&B devre dışı bırakma
    if cfg.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # ── Veri ─────────────────────────────────────────────────────────────────
    print(f"\n  Veri kökü: {cfg.data_root}")
    df = build_ptx_manifest(cfg.data_root)

    if cfg.quick_test:
        df = df.sample(n=min(40, len(df)), random_state=cfg.seed).reset_index(drop=True)

    print()
    folds = make_site_stratified_folds(df, n_folds=cfg.num_folds, seed=cfg.seed)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── W&B grup adı ──────────────────────────────────────────────────────────
    group_name = cfg.wandb_group or f"ptx-local-{int(time.time())}"
    wandb_cfg  = cfg.as_wandb_dict()
    wandb_cfg["wandb_group"] = group_name

    # ── Resume: tamamlanan fold'ları yükle ────────────────────────────────────
    progress_file   = cfg.checkpoint_dir / "ptx_fold_progress.json"
    completed_folds: set[int] = set()
    fold_results:    list[dict] = []

    if cfg.resume and progress_file.exists():
        state = json.loads(progress_file.read_text())
        completed_folds = set(state.get("completed_folds", []))
        fold_results    = state.get("fold_results", [])
        if state.get("group_name"):
            group_name            = state["group_name"]
            wandb_cfg["wandb_group"] = group_name
        print(f"\n  [RESUME] Tamamlanan fold'lar: {sorted(completed_folds)}")

    criterion = DiceLoss()

    # ── Fold döngüsü ──────────────────────────────────────────────────────────
    for fold_i, (train_df, val_df) in enumerate(folds, start=1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_i}/{cfg.num_folds}  —  train={len(train_df)}  val={len(val_df)}")
        site_dist = train_df["site"].value_counts().to_dict()
        print(f"  Train site: {site_dist}")
        print(f"{'='*60}")

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
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
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

        # ── Checkpoint yolları ────────────────────────────────────────────────
        best_ckpt   = cfg.checkpoint_dir / f"fold_{fold_i}_best.pth"
        resume_ckpt = cfg.checkpoint_dir / f"fold_{fold_i}_resume.pth"

        best_dice:     float = 0.0
        best_iou:      float = 0.0
        best_per_site: dict  = {}
        start_epoch:   int   = 1

        # Within-fold resume
        if resume_ckpt.exists():
            try:
                state = torch.load(resume_ckpt, map_location=device)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                best_dice      = state["best_dice"]
                best_iou       = state.get("best_iou",      0.0)
                best_per_site  = state.get("best_per_site", {})
                start_epoch    = state["epoch"] + 1
                print(f"  [RESUME] Fold {fold_i} epoch {start_epoch}'den devam "
                      f"(best Dice: {best_dice:.4f})")
            except Exception as exc:
                print(f"  [RESUME] Checkpoint okunamadı ({exc}), sıfırdan başlanıyor.")
                start_epoch = 1

        # ── Epoch döngüsü ─────────────────────────────────────────────────────
        for epoch in range(start_epoch, cfg.epochs + 1):
            print(f"\n  Epoch {epoch}/{cfg.epochs}")

            train_loss, train_dice = _train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_m = _val_epoch(model, val_loader, criterion, device, val_df)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            hd_str = f"{val_m['hd95']:.1f}" if not np.isinf(val_m["hd95"]) else "∞"
            print(
                f"  Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
                f"  Val   → Loss: {val_m['loss']:.4f}  Dice: {val_m['dice']:.4f}  "
                f"IoU: {val_m['iou']:.4f}  HD95: {hd_str}px  LR: {current_lr:.2e}"
            )
            _print_per_site(val_m["per_site"])

            # W&B global + per-site
            log_epoch_metrics(
                fold=fold_i, epoch=epoch,
                train_loss=train_loss,
                val_loss=val_m["loss"],
                val_dice=val_m["dice"],
                val_iou=val_m["iou"],
                val_auc=0.0,           # cls head kapalı
                val_sensitivity=0.0,
                val_specificity=0.0,
                val_precision=0.0,
                current_lr=current_lr,
                val_hausdorff=val_m["hd95"],
            )
            _log_per_site(fold_i, epoch, val_m["per_site"])

            # ── Best checkpoint ───────────────────────────────────────────────
            if val_m["dice"] > best_dice:
                best_dice      = val_m["dice"]
                best_iou       = val_m["iou"]
                best_per_site  = val_m["per_site"]
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ✓ Checkpoint: {best_ckpt.name}  (Dice: {best_dice:.4f})")

            # ── Within-fold resume checkpoint ─────────────────────────────────
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "best_dice":      best_dice,
                "best_iou":       best_iou,
                "best_per_site":  best_per_site,
            }, resume_ckpt)

        # ── Fold sonu ─────────────────────────────────────────────────────────
        if resume_ckpt.exists():
            resume_ckpt.unlink()

        append_fold_result(
            fold_results, fold_i,
            best_dice=best_dice, best_auc=0.0,
            best_iou=best_iou,   best_sensitivity=0.0,
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
        description="PTX-498 yerel PNG eğitimi — Site-stratified 5-fold CV"
    )
    p.add_argument("--data_root",    required=True,
                   help="PTX-498-v2-fix kök dizini (SiteA/, SiteB/, SiteC/ içerir)")
    p.add_argument("--checkpoint_dir", default=str(ROOT / "results" / "checkpoints"))
    p.add_argument("--results_csv",    default=str(ROOT / "results" / "ptx_local_kfold.csv"))
    p.add_argument("--encoder",        default="efficientnet-b0")
    p.add_argument("--img_size",  type=int,   default=512)
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch_size",type=int,   default=8)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_folds", type=int,   default=5)
    p.add_argument("--num_workers",type=int,  default=4)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--no_wandb",  action="store_true")
    p.add_argument("--quick_test",action="store_true",
                   help="3 epoch, 40 sample — altyapı testi")
    p.add_argument("--wandb_project", default="Pneumothorax-Detection")
    p.add_argument("--wandb_entity",  default="ahmet-ai-t-bi-tak")
    p.add_argument("--wandb_group",   default="")

    args = p.parse_args()
    return TrainConfig(
        data_root      = Path(args.data_root),
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

    print("\n" + "═" * 60)
    print("  PTX-498 YERELEŞTİRME EĞİTİMİ  —  TÜBİTAK 2209-A")
    print("═" * 60)
    print(f"  Veri kökü     : {cfg.data_root}")
    print(f"  Encoder       : {cfg.encoder_name}")
    print(f"  Epochs        : {cfg.epochs}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Fold sayısı   : {cfg.num_folds}")
    print(f"  Img size      : {cfg.img_size}×{cfg.img_size}")
    print(f"  Loss          : Dice only (cls head frozen)")
    print(f"  W&B           : {'hayır' if cfg.no_wandb else 'evet'}")
    print(f"  Quick test    : {'evet' if cfg.quick_test else 'hayır'}")
    print("═" * 60)

    results = train_kfold_local(cfg)
