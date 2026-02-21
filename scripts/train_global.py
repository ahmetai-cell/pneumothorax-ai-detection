"""
Pre-training Script  —  Global Data (SIIM + NIH)
================================================

unified_manifest.csv'deki SIIM ve NIH verilerini kullanarak base modeli eğitir.
Eğitim bittiğinde en iyi fold'un checkpoint'i global_base_model.pth olarak
kaydedilir; fine_tune_local.py bu dosyayı --base_model parametresiyle alır.

Strateji:
  - SIIM (~12 k görüntü) + NIH (~112 k görüntü, pozitif/negatif dengesi korunur)
  - U-Net++ (EfficientNet-B0 encoder, ImageNet pre-trained)
  - Stratified K-Fold (5 fold)
  - WeightedRandomSampler + Hard Negative Mining
  - W&B entegrasyonu

Kullanım:
    # Önerilen: tam K-Fold pre-training
    python scripts/train_global.py

    # Yalnızca SIIM (NIH indirme bitmemişse)
    python scripts/train_global.py --sources SIIM

    # Hızlı sözdizimi/altyapı testi (3 epoch, 200 sample)
    python scripts/train_global.py --quick_test

    # Özel parametreler
    python scripts/train_global.py --epochs 80 --encoder resnet34 --batch_size 32

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

# Proje kökü path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.dicom_dataset import DicomSlicerDataset
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.utils.train import train_kfold


# ── Sabitler ──────────────────────────────────────────────────────────────────

MANIFEST_PATH  = ROOT / "data" / "processed" / "master_manifest.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR    = ROOT / "results"
BASE_MODEL_OUT = CHECKPOINT_DIR / "global_base_model.pth"
META_OUT       = CHECKPOINT_DIR / "global_base_model_meta.json"


# ── Argümanlar ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIIM+NIH global pre-training  (TÜBİTAK 2209-A)"
    )
    p.add_argument("--manifest",    default=str(MANIFEST_PATH),
                   help="unified_manifest.csv yolu")
    p.add_argument("--sources",     default="SIIM,NIH",
                   help="Kullanılacak kaynaklar, virgülle ayrılmış (varsayılan: SIIM,NIH)")
    p.add_argument("--encoder",     default="efficientnet-b0",
                   help="smp encoder adı (varsayılan: efficientnet-b0)")
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_folds",   type=int, default=5)
    p.add_argument("--dice_weight", type=float, default=0.5,
                   help="Kombine kayıpta Dice ağırlığı (0-1)")
    p.add_argument("--no_wandb",    action="store_true",
                   help="W&B loglamayı devre dışı bırak")
    p.add_argument("--quick_test",  action="store_true",
                   help="Sözdizimi/altyapı testi: 3 epoch, max 200 örnek")
    p.add_argument("--split",       choices=["kfold", "single"], default="kfold",
                   help="kfold (önerilen) veya single 80/20 split")
    p.add_argument("--positive_only", action="store_true",
                   help="Yalnızca pnömotoraks pozitif vakaları al (sınırsız negatif)")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Checkpoint kayıt dizini (varsayılan: proje içi checkpoints/)")
    return p.parse_args()


# ── Veri yükleme ──────────────────────────────────────────────────────────────

def load_dataframe(manifest_path: str, sources: list[str], positive_only: bool) -> pd.DataFrame:
    """
    Unified manifest'i filtreler.
    Boş veya var olmayan mask_path satırları negatif vaka olarak tutulur.
    """
    if not Path(manifest_path).exists():
        print(f"\n[!] Manifest bulunamadı: {manifest_path}")
        print("    Önce çalıştır:")
        print("      python scripts/data_manager.py --convert_rle --build_manifest")
        print("      python scripts/unify_annotations.py")
        sys.exit(1)

    df = pd.read_csv(manifest_path)

    # Kaynak filtresi
    src_upper = [s.strip().upper() for s in sources]
    df = df[df["source"].str.upper().isin(src_upper)].copy()

    if positive_only:
        # Negatiflerden bazılarını yine de tut (1:3 oranı)
        pos = df[df["is_pneumo"] == 1]
        neg = df[df["is_pneumo"] == 0].sample(
            min(len(pos) * 3, len(df[df["is_pneumo"] == 0])),
            random_state=42,
        )
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    # img_path sütun adı uyumu (eski CSV'lerde image_path olabilir)
    if "image_path" in df.columns and "img_path" not in df.columns:
        df = df.rename(columns={"image_path": "img_path"})

    print(f"\n  Kayıtlar : {len(df):,}")
    print(f"  Pozitif  : {(df['is_pneumo'] == 1).sum():,}")
    print(f"  Negatif  : {(df['is_pneumo'] == 0).sum():,}")
    print(f"  Kaynaklar: {df['source'].value_counts().to_dict()}")

    return df.reset_index(drop=True)


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  GLOBAL PRE-TRAINING  —  TÜBİTAK 2209-A")
    print("═" * 60)
    print(f"  Kaynaklar  : {args.sources}")
    print(f"  Encoder    : {args.encoder}")
    print(f"  Img Size   : {args.img_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Folds      : {args.num_folds}")
    print(f"  Quick Test : {args.quick_test}")
    print("═" * 60)

    # W&B env kapatma
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # ── Dizinler ──────────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Veri ──────────────────────────────────────────────────────────────────
    sources = [s.strip() for s in args.sources.split(",")]
    df = load_dataframe(args.manifest, sources, args.positive_only)

    if args.quick_test:
        df = df.sample(min(200, len(df)), random_state=42).reset_index(drop=True)
        args.epochs     = 3
        args.num_folds  = 2
        args.batch_size = 8
        print(f"\n  [QUICK TEST] {len(df)} örnek, {args.epochs} epoch, {args.num_folds} fold")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # align_tolerance=-1: SIIM/NIH için hizalama kontrolü atlanır
    train_tf = get_train_transforms(img_size=args.img_size)
    val_tf   = get_val_transforms(img_size=args.img_size)

    dataset = DicomSlicerDataset(
        df,
        transform=train_tf,
        img_size=args.img_size,
        lung_window=True,        # DICOM varsa HU normalizasyonu
        align_tolerance=-1,      # Global veri karışık format — hizalama atlanır
        skip_on_align_error=True,
    )

    print(f"\n  Dataset boyutu: {len(dataset):,}")

    # ── Config ────────────────────────────────────────────────────────────────
    ckpt_dir = str(Path(args.checkpoint_dir) / "global_folds") if args.checkpoint_dir else str(CHECKPOINT_DIR / "global_folds")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    config = {
        "encoder_name":         args.encoder,
        "epochs":               args.epochs,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        "weight_decay":         args.weight_decay,
        "num_folds":            args.num_folds,
        "dice_weight":          args.dice_weight,
        "checkpoint_dir":       ckpt_dir,
        "results_csv":          str(RESULTS_DIR / "global_kfold_results.csv"),
        "wandb_project":        "Pneumothorax-Detection",
        "wandb_entity":         "ahmet-ai-t-bi-tak",
        "wandb_group":          f"global-pretrain-{args.encoder}",
        "hard_negative_mining": True,
        "hnm_interval":         3,
        "hnm_threshold":        0.4,
        "hnm_multiplier":       3.0,
        "img_size":             args.img_size,
        "sources":              args.sources,
    }

    # ── Eğitim ────────────────────────────────────────────────────────────────
    if args.split == "kfold":
        fold_results = train_kfold(dataset, config)
    else:
        # Tek split — hızlı test için
        from torch.utils.data import random_split
        from src.utils.train import train
        n_val = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        from torch.utils.data import DataLoader
        single_config = {
            **config,
            "train_loader": DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=True,  num_workers=4, pin_memory=True),
            "val_loader":   DataLoader(val_ds,   batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True),
            "checkpoint_path": str(CHECKPOINT_DIR / "global_single_best.pth"),
        }
        train(single_config)
        fold_results = []

    # ── En iyi modeli global_base_model.pth olarak kaydet ─────────────────────
    if fold_results:
        best_fold = max(fold_results, key=lambda r: r.get("best_dice", 0.0))
        best_ckpt = Path(ckpt_dir) / f"fold{best_fold['fold']}_best.pth"

        if best_ckpt.exists():
            shutil.copy(best_ckpt, BASE_MODEL_OUT)
            meta = {
                "encoder":     args.encoder,
                "img_size":    args.img_size,
                "sources":     args.sources,
                "best_fold":   best_fold["fold"],
                "best_dice":   round(best_fold["best_dice"], 4),
                "best_auc":    round(best_fold.get("best_auc", 0.0), 4),
                "epochs":      args.epochs,
            }
            META_OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f"\n  ✓ Base model kaydedildi   → {BASE_MODEL_OUT}")
            print(f"  ✓ Metadata               → {META_OUT}")
            print(f"  ✓ En iyi Fold {best_fold['fold']}  Dice: {best_fold['best_dice']:.4f}")
        else:
            print(f"  [!] Checkpoint bulunamadı: {best_ckpt}")

    print("\n  Sıradaki adım:")
    print("    python scripts/fine_tune_local.py --base_model", BASE_MODEL_OUT)
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
