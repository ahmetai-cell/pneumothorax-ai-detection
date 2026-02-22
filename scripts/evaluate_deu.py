"""
DEU Klinik Veri Değerlendirme Scripti
======================================

Global pre-training sonrası modeli DEU (Dokuz Eylül Üniversitesi) DICOM+NRRD
verisiyle test eder. Bu veri eğitimde kullanılmaz — sadece klinik evaluation içindir.

Veri formatı:
  data/local/dicom/  →  DICOM (.dcm)
  data/local/nrrd/   →  NRRD segmentasyon maskesi (.nrrd, 3D Slicer çıktısı)

Çıktılar:
  results/deu_evaluation.csv        — Her vaka için metrikler
  results/deu_evaluation_summary.txt — Özet rapor

Kullanım:
  python scripts/evaluate_deu.py
  python scripts/evaluate_deu.py --model checkpoints/global_base_model.pth
  python scripts/evaluate_deu.py --dicom_dir /path/to/dicom --nrrd_dir /path/to/nrrd
  python scripts/evaluate_deu.py --threshold 0.4   # Segmentasyon eşiği
  python scripts/evaluate_deu.py --no_wandb

TÜBİTAK 2209-A | Ahmet Demir | Dokuz Eylül Üniversitesi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model.unet import PneumothoraxModel
from src.preprocessing.augmentation import get_val_transforms
from src.preprocessing.dicom_dataset import DicomSlicerDataset, pair_dicom_nrrd
from src.utils.metrics import hausdorff_distance, iou_score


# ── Sabitler ──────────────────────────────────────────────────────────────────

LOCAL_DICOM_DIR = ROOT / "data" / "local" / "dicom"
LOCAL_NRRD_DIR  = ROOT / "data" / "local" / "nrrd"
CHECKPOINT_DIR  = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"


# ── Argümanlar ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DEU DICOM+NRRD klinik değerlendirme (TÜBİTAK 2209-A)"
    )
    p.add_argument("--model",      default=str(CHECKPOINT_DIR / "global_base_model.pth"),
                   help="Değerlendirilecek model checkpoint yolu")
    p.add_argument("--meta",       default=str(CHECKPOINT_DIR / "global_base_model_meta.json"),
                   help="Model metadata JSON (encoder, img_size)")
    p.add_argument("--dicom_dir",  default=str(LOCAL_DICOM_DIR))
    p.add_argument("--nrrd_dir",   default=str(LOCAL_NRRD_DIR))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Segmentasyon eşiği (varsayılan: 0.5)")
    p.add_argument("--align_tolerance", type=int, default=0,
                   help="NRRD-DICOM boyut toleransı (piksel)")
    p.add_argument("--no_wandb",   action="store_true")
    p.add_argument("--out_dir",    default=str(RESULTS_DIR))
    return p.parse_args()


# ── Metrik hesaplama ──────────────────────────────────────────────────────────

def compute_case_metrics(
    seg_logit: torch.Tensor,
    mask: torch.Tensor,
    cls_logit: torch.Tensor,
    label: torch.Tensor,
    threshold: float,
) -> dict:
    """Tek bir görüntü için tüm metrikleri hesaplar."""
    pred_prob = torch.sigmoid(seg_logit)
    pred_bin  = (pred_prob > threshold).float()
    gt        = mask.float()

    # Flatten
    p = pred_bin.view(-1)
    g = gt.view(-1)

    tp = (p * g).sum().item()
    fp = (p * (1 - g)).sum().item()
    fn = ((1 - p) * g).sum().item()
    tn = ((1 - p) * (1 - g)).sum().item()

    smooth = 1e-6
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    sens = (tp + smooth) / (tp + fn + smooth)   # Recall / Sensitivity
    spec = (tn + smooth) / (tn + fp + smooth)   # Specificity
    prec = (tp + smooth) / (tp + fp + smooth)   # Precision

    hd95 = hausdorff_distance(seg_logit, mask, threshold=threshold, percentile=95.0)

    cls_prob   = torch.sigmoid(cls_logit).item()
    cls_pred   = int(cls_prob > 0.5)
    cls_label  = int(label.item())
    cls_correct = int(cls_pred == cls_label)

    return {
        "dice":        round(dice, 4),
        "iou":         round(iou, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision":   round(prec, 4),
        "hd95":        round(hd95, 2) if not np.isinf(hd95) else None,
        "cls_prob":    round(cls_prob, 4),
        "cls_pred":    cls_pred,
        "cls_label":   cls_label,
        "cls_correct": cls_correct,
    }


# ── Özet rapor ────────────────────────────────────────────────────────────────

def print_and_save_summary(df: pd.DataFrame, out_path: str) -> None:
    pos = df[df["cls_label"] == 1]
    neg = df[df["cls_label"] == 0]

    hd_vals = df["hd95"].dropna()

    lines = [
        "=" * 65,
        "  DEU KLİNİK DEĞERLENDİRME RAPORU  —  TÜBİTAK 2209-A",
        "=" * 65,
        f"  Toplam vaka        : {len(df)}",
        f"  Pozitif (pnömotoraks): {len(pos)}",
        f"  Negatif (normal)    : {len(neg)}",
        "",
        "  — Segmentasyon (tüm vakalar) —",
        f"  Dice         : {df['dice'].mean():.4f} ± {df['dice'].std():.4f}",
        f"  IoU          : {df['iou'].mean():.4f} ± {df['iou'].std():.4f}",
        f"  Sensitivity  : {df['sensitivity'].mean():.4f} ± {df['sensitivity'].std():.4f}",
        f"  Specificity  : {df['specificity'].mean():.4f} ± {df['specificity'].std():.4f}",
        f"  Precision    : {df['precision'].mean():.4f} ± {df['precision'].std():.4f}",
        f"  HD95         : {hd_vals.mean():.2f} ± {hd_vals.std():.2f} px"
        if len(hd_vals) > 0 else "  HD95         : N/A (boş maske)",
        "",
        "  — Sınıflandırma —",
        f"  Doğruluk     : {df['cls_correct'].mean():.4f}  ({df['cls_correct'].sum()}/{len(df)})",
    ]

    if len(pos) > 0:
        lines += [
            "",
            "  — Sadece Pozitif Vakalar (pnömotoraks) —",
            f"  Dice         : {pos['dice'].mean():.4f} ± {pos['dice'].std():.4f}",
            f"  Sensitivity  : {pos['sensitivity'].mean():.4f}",
            f"  HD95         : {pos['hd95'].dropna().mean():.2f} px"
            if pos['hd95'].notna().any() else "  HD95         : N/A",
        ]

    lines.append("=" * 65)
    report = "\n".join(lines)
    print("\n" + report + "\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Rapor kaydedildi: {out_path}")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # Model metadata
    meta_path = Path(args.meta)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"encoder": "efficientnet-b0", "img_size": 512}

    encoder  = meta.get("encoder",  "efficientnet-b0")
    img_size = meta.get("img_size", 512)

    print("\n" + "═" * 65)
    print("  DEU EVALUATION  —  TÜBİTAK 2209-A")
    print("═" * 65)
    print(f"  Model      : {args.model}")
    print(f"  Encoder    : {encoder}  |  Img size: {img_size}")
    print(f"  DICOM dizini: {args.dicom_dir}")
    print(f"  NRRD dizini : {args.nrrd_dir}")
    print(f"  Eşik        : {args.threshold}")
    print("═" * 65)

    # ── Veri ──────────────────────────────────────────────────────────────────
    dicom_dir = Path(args.dicom_dir)
    nrrd_dir  = Path(args.nrrd_dir)

    if not dicom_dir.exists() or not list(dicom_dir.glob("**/*.dcm")):
        print(f"\n  [!] DICOM dosyası bulunamadı: {dicom_dir}")
        print("  Beklenen yapı:")
        print("    data/local/dicom/hasta_001.dcm")
        print("    data/local/nrrd/hasta_001.nrrd  ← (sadece pozitif vakalar)")
        sys.exit(1)

    df = pair_dicom_nrrd(args.dicom_dir, args.nrrd_dir, strict=False)
    print(f"\n  DICOM-NRRD çifti yüklendi: {len(df)} vaka")
    print(f"  Pozitif (NRRD var) : {df['is_pneumo'].sum()}")
    print(f"  Negatif (NRRD yok) : {(df['is_pneumo'] == 0).sum()}")

    transform = get_val_transforms(img_size=img_size)
    dataset = DicomSlicerDataset(
        df,
        transform=transform,
        img_size=img_size,
        lung_window=True,
        force_lung=True,
        align_tolerance=args.align_tolerance,
        skip_on_align_error=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = PneumothoraxModel(encoder_name=encoder).to(device)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  [!] Model bulunamadı: {model_path}")
        sys.exit(1)

    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  ✓ Model yüklendi: {model_path.name}")
    if meta.get("best_dice"):
        print(f"    (Pre-training Dice: {meta['best_dice']:.4f})")

    # ── Evaluation döngüsü ────────────────────────────────────────────────────
    records = []
    img_paths = df["img_path"].tolist()
    idx = 0

    print(f"\n  Değerlendirme başlıyor ({len(dataset)} vaka)...\n")

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating"):
            images, masks, labels = batch
            images = images.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            seg_logits, cls_logits = model(images)

            for i in range(images.shape[0]):
                metrics = compute_case_metrics(
                    seg_logits[i], masks[i], cls_logits[i], labels[i],
                    threshold=args.threshold,
                )
                metrics["img_path"]   = img_paths[idx]
                metrics["is_pneumo"]  = int(labels[i].item())
                records.append(metrics)
                idx += 1

    # ── Sonuçlar ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)

    # Sütun sırası
    col_order = [
        "img_path", "is_pneumo",
        "dice", "iou", "sensitivity", "specificity", "precision", "hd95",
        "cls_prob", "cls_pred", "cls_label", "cls_correct",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "deu_evaluation.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  Vaka başına sonuçlar: {csv_path}")

    summary_path = out_dir / "deu_evaluation_summary.txt"
    print_and_save_summary(results_df, str(summary_path))


if __name__ == "__main__":
    main()
