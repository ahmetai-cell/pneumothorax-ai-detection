"""
Fine-tuning Script  —  DEU Local Data (DICOM + NRRD)
=====================================================

Global base modeli (train_global.py çıktısı) DEU DICOM+NRRD verisiyle
fine-tune eder.

Strateji:
  - Encoder: DEU verisine uyum için çok düşük LR (5e-5)
  - Decoder + segmentation head: normal LR
  - --freeze_encoder ile encoder tamamen dondurulabilir (transfer learning saf mod)
  - Hizalama kontrolü aktif (AlignmentError dışa aktarımı tespit eder)

Kullanım:
    # Temel fine-tuning (encoder de eğitilir, düşük LR)
    python scripts/fine_tune_local.py

    # Encoder dondurulmuş (sadece decoder + head — çok hızlı, ~30 dk)
    python scripts/fine_tune_local.py --freeze_encoder

    # Özel base model yolu
    python scripts/fine_tune_local.py --base_model checkpoints/global_base_model.pth

    # Veri gelmeden altyapıyı test et (dummy dosyalar varsa çalışır)
    python scripts/fine_tune_local.py --dry_run

    # Özel parametreler
    python scripts/fine_tune_local.py --epochs 30 --lr 1e-5 --batch_size 8

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Proje kökü path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.losses import CombinedLoss
from src.model.unet import PneumothoraxModel
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.preprocessing.dicom_dataset import DicomSlicerDataset, pair_dicom_nrrd
from src.utils.metrics import batch_hausdorff, dice_score, iou_score
from src.utils.wandb_utils import init_fold_run, log_epoch_metrics


# ── Sabitler ──────────────────────────────────────────────────────────────────

LOCAL_DICOM_DIR  = ROOT / "data" / "local" / "dicom"
LOCAL_NRRD_DIR   = ROOT / "data" / "local" / "nrrd"
CHECKPOINT_DIR   = ROOT / "checkpoints"
BASE_MODEL_PATH  = CHECKPOINT_DIR / "global_base_model.pth"
BASE_MODEL_META  = CHECKPOINT_DIR / "global_base_model_meta.json"
FINE_TUNED_OUT   = CHECKPOINT_DIR / "deu_fine_tuned_model.pth"


# ── Argümanlar ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DEU DICOM+NRRD fine-tuning  (TÜBİTAK 2209-A)"
    )
    p.add_argument("--base_model",  default=str(BASE_MODEL_PATH),
                   help="Pre-trained base model checkpoint yolu")
    p.add_argument("--dicom_dir",   default=str(LOCAL_DICOM_DIR),
                   help="DEU DICOM dizini (varsayılan: data/local/dicom)")
    p.add_argument("--nrrd_dir",    default=str(LOCAL_NRRD_DIR),
                   help="DEU NRRD maske dizini (varsayılan: data/local/nrrd)")
    p.add_argument("--encoder",     default=None,
                   help="smp encoder adı (varsayılan: base_model meta'dan okunur)")
    p.add_argument("--img_size",    type=int, default=None,
                   help="Görüntü boyutu (varsayılan: base_model meta'dan okunur)")
    p.add_argument("--epochs",      type=int, default=20,
                   help="Fine-tuning epoch sayısı (varsayılan: 20)")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Öğrenme oranı (pre-training'den ~10x daha düşük)")
    p.add_argument("--encoder_lr",  type=float, default=None,
                   help="Encoder için ayrı LR (varsayılan: --lr / 5)")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Encoder ağırlıklarını dondur (sadece decoder + head eğitilir)")
    p.add_argument("--val_split",   type=float, default=0.2,
                   help="Validation oranı (varsayılan: 0.20)")
    p.add_argument("--align_tolerance", type=int, default=0,
                   help="NRRD-DICOM boyut toleransı (piksel). Varsayılan: 0 = tam eşitlik")
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--dry_run",     action="store_true",
                   help="Sadece altyapıyı test et, eğitim yapma")
    return p.parse_args()


# ── Metadata okuyucu ──────────────────────────────────────────────────────────

def load_base_meta(meta_path: Path) -> dict:
    """global_base_model_meta.json'dan encoder ve img_size bilgisini okur."""
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"encoder": "efficientnet-b0", "img_size": 512}


# ── Eğitim döngüsü ────────────────────────────────────────────────────────────

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
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    all_seg_preds, all_seg_gt = [], []
    all_cls_preds, all_labels = [], []

    for images, masks, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)
        seg_pred, cls_pred = model(images)
        loss = criterion(seg_pred, masks, cls_pred, labels)
        total_loss += loss.item()
        total_dice += dice_score(seg_pred, masks).item()
        total_iou  += iou_score(seg_pred, masks).item()
        all_seg_preds.append(seg_pred.cpu())
        all_seg_gt.append(masks.cpu())
        all_cls_preds.append(cls_pred.cpu())
        all_labels.append(labels.cpu())

    n = max(len(loader), 1)

    # Hausdorff HD95 — CPU'da hesapla
    seg_p = torch.cat(all_seg_preds)
    seg_g = torch.cat(all_seg_gt)
    hd95  = batch_hausdorff(seg_p, seg_g)

    # Sınıflandırma metrikleri
    cls_p = torch.sigmoid(torch.cat(all_cls_preds)).squeeze()
    lbl   = torch.cat(all_labels).float()
    pred_bin = (cls_p > 0.5).float()
    tp = ((pred_bin == 1) & (lbl == 1)).sum().item()
    tn = ((pred_bin == 0) & (lbl == 0)).sum().item()
    fp = ((pred_bin == 1) & (lbl == 0)).sum().item()
    fn = ((pred_bin == 0) & (lbl == 1)).sum().item()

    return {
        "loss":        total_loss / n,
        "dice":        total_dice / n,
        "iou":         total_iou  / n,
        "hd95":        hd95,
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "precision":   tp / (tp + fp + 1e-8),
    }


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Base model metadata
    meta = load_base_meta(BASE_MODEL_META)
    encoder  = args.encoder  or meta.get("encoder",  "efficientnet-b0")
    img_size = args.img_size or meta.get("img_size", 512)

    print("\n" + "═" * 60)
    print("  DEU FINE-TUNING  —  TÜBİTAK 2209-A")
    print("═" * 60)
    print(f"  Base Model     : {args.base_model}")
    print(f"  Encoder        : {encoder}")
    print(f"  Img Size       : {img_size}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  LR             : {args.lr}")
    print(f"  Freeze Encoder : {args.freeze_encoder}")
    print(f"  DICOM dizini   : {args.dicom_dir}")
    print(f"  NRRD dizini    : {args.nrrd_dir}")
    print(f"  Dry Run        : {args.dry_run}")
    print("═" * 60)

    # W&B
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # ── Veri ──────────────────────────────────────────────────────────────────
    dicom_dir = Path(args.dicom_dir)
    nrrd_dir  = Path(args.nrrd_dir)

    # Dizinler yok veya boşsa bilgilendir
    dcm_files = list(dicom_dir.glob("**/*.dcm")) if dicom_dir.exists() else []
    if not dcm_files:
        print(f"\n  [!] DICOM dosyası bulunamadı: {dicom_dir}")
        print("  Beklenen yapı:")
        print("    data/local/dicom/patient_001.dcm")
        print("    data/local/dicom/patient_002.dcm  ...")
        print("    data/local/nrrd/patient_001.nrrd")
        print("    data/local/nrrd/patient_002.nrrd  ...")
        if args.dry_run:
            print("\n  [DRY RUN] Altyapı testi: Veri olmadan model yükleme kontrol edilecek.")
        else:
            print("\n  Gerçek veriler gelene kadar --dry_run kullanabilirsin.")
            sys.exit(0)

    if not args.dry_run:
        df = pair_dicom_nrrd(
            args.dicom_dir, args.nrrd_dir,
            strict=False,
        )
        print(f"\n  DICOM-NRRD çifti : {len(df)}")
        print(f"  Pozitif (NRRD var): {df['is_pneumo'].sum()}")
        print(f"  Negatif (NRRD yok): {(df['is_pneumo'] == 0).sum()}")
    else:
        # Dry run: boş DataFrame ile yalnızca model yükleme test edilir
        import pandas as pd
        df = pd.DataFrame(columns=["img_path", "mask_path", "is_pneumo", "source"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_tf = get_train_transforms(img_size=img_size)
    val_tf   = get_val_transforms(img_size=img_size)

    if not args.dry_run and len(df) > 0:
        # Val split
        n_val   = max(1, int(len(df) * args.val_split))
        n_train = len(df) - n_val
        indices = list(range(len(df)))
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]

        full_ds = DicomSlicerDataset(
            df,
            img_size=img_size,
            lung_window=True,
            force_lung=True,              # DEU: her zaman klinik pencere
            align_tolerance=args.align_tolerance,
            skip_on_align_error=True,
        )
        train_ds = Subset(DicomSlicerDataset(
            df.iloc[train_idx].reset_index(drop=True),
            transform=train_tf, img_size=img_size,
            lung_window=True, force_lung=True,
            align_tolerance=args.align_tolerance, skip_on_align_error=True,
        ), list(range(n_train)))
        val_ds = Subset(DicomSlicerDataset(
            df.iloc[val_idx].reset_index(drop=True),
            transform=val_tf, img_size=img_size,
            lung_window=True, force_lung=True,
            align_tolerance=args.align_tolerance, skip_on_align_error=True,
        ), list(range(n_val)))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
    else:
        train_loader = val_loader = None

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = PneumothoraxModel(encoder_name=encoder).to(device)

    # Base model ağırlıklarını yükle
    base_path = Path(args.base_model)
    if base_path.exists():
        state = torch.load(str(base_path), map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Base model yüklendi: {base_path.name}")
        if meta.get("best_dice"):
            print(f"    (Global pre-training Dice: {meta['best_dice']:.4f})")
    else:
        print(f"  [!] Base model bulunamadı: {base_path}")
        print("      ImageNet ağırlıklarıyla sıfırdan başlanacak.")

    # ── Encoder dondurma ──────────────────────────────────────────────────────
    if args.freeze_encoder:
        for param in model.unet.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(
            p.numel() for p in model.unet.encoder.parameters()
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\n  Encoder donduruldu: {frozen_params:,} parametre pasif\n"
            f"  Eğitilecek       : {trainable:,} / {total_params:,} parametre"
        )
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  Tüm parametreler eğitilecek: {total:,}")

    if args.dry_run:
        print("\n  [DRY RUN] Model yükleme başarılı. Eğitim atlandı.")
        print(f"  Fine-tuning için çalıştır:")
        print(f"    python scripts/fine_tune_local.py  (veriler hazır olduğunda)")
        return

    # ── Optimizer — differential LR ──────────────────────────────────────────
    encoder_lr = args.encoder_lr if args.encoder_lr else args.lr / 5

    if args.freeze_encoder:
        # Sadece decoder + head
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    else:
        # Encoder çok daha düşük LR, decoder normal LR (differential learning)
        optimizer = optim.AdamW([
            {"params": model.unet.encoder.parameters(), "lr": encoder_lr},
            {"params": model.unet.decoder.parameters(), "lr": args.lr},
            {"params": model.unet.segmentation_head.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    criterion = CombinedLoss(dice_weight=0.6, bce_weight=0.4)

    # ── W&B ───────────────────────────────────────────────────────────────────
    ft_config = {
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "encoder_name": encoder, "img_size": img_size,
        "wandb_project": "Pneumothorax-Detection",
        "wandb_entity": "ahmet-ai-t-bi-tak",
        "wandb_group": "deu-fine-tune",
        "fine_tuning": True,
        "freeze_encoder": args.freeze_encoder,
        "base_model": str(base_path.name),
    }
    run = init_fold_run(ft_config, fold=1, num_folds=1)

    # ── Eğitim döngüsü ────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    print(f"\n  Fine-tuning başlıyor: {args.epochs} epoch...\n")

    for epoch in range(1, args.epochs + 1):
        print(f"  Epoch {epoch}/{args.epochs}")

        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_m = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[-1]["lr"]

        hd_str = f"{val_m['hd95']:.1f}" if not np.isinf(val_m["hd95"]) else "∞"
        print(
            f"    Train → Loss: {train_loss:.4f}  Dice: {train_dice:.4f}\n"
            f"    Val   → Loss: {val_m['loss']:.4f}  Dice: {val_m['dice']:.4f}  "
            f"IoU: {val_m['iou']:.4f}  HD95: {hd_str}px  "
            f"Sens: {val_m['sensitivity']:.4f}  Spec: {val_m['specificity']:.4f}"
        )

        log_epoch_metrics(
            fold=1, epoch=epoch,
            train_loss=train_loss,
            val_loss=val_m["loss"],
            val_dice=val_m["dice"],
            val_iou=val_m["iou"],
            val_auc=0.0,           # AUC için sklearn gerekir, hafif veri setinde skip
            val_sensitivity=val_m["sensitivity"],
            val_specificity=val_m["specificity"],
            val_precision=val_m["precision"],
            current_lr=current_lr,
            val_hausdorff=val_m["hd95"],
        )

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save(model.state_dict(), str(FINE_TUNED_OUT))
            print(f"    ✓ Checkpoint: deu_fine_tuned_model.pth  (Dice: {best_dice:.4f})")

    # ── Özet ──────────────────────────────────────────────────────────────────
    if run:
        run.finish()

    print("\n" + "═" * 60)
    print("  FINE-TUNING TAMAMLANDI")
    print("═" * 60)
    print(f"  En iyi Dice : {best_dice:.4f}")
    print(f"  Model       : {FINE_TUNED_OUT}")
    print("\n  Streamlit arayüzü için:")
    print(f"    streamlit run app/streamlit_app.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
