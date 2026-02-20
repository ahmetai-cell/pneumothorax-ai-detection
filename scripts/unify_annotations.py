"""
Annotation Birleştirme — Multi-Source Mask Standardization

Tüm kaynaklardaki maskeleri tek formata indirir:
  DEU  (yeşil boyama / NRRD)  →  PNG binary (0 / 255)
  SIIM (NRRD ← RLE)           →  PNG binary (0 / 255)
  NIH  (maske yok)            →  atlanır, mask_path="N/A" kalır

Çıktı:
  data/processed/unified_masks/<SOURCE>/<id>.png
  data/processed/unified_manifest.csv   ← güncel mask_path sütunuyla

Kullanım:
  python scripts/unify_annotations.py
  python scripts/unify_annotations.py --verify   # sadece kontrol, yazma

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Proje kökünü path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.green_mask_extractor import extract_green_mask

# ── Sabitler ──────────────────────────────────────────────────────────────────

MANIFEST_IN  = ROOT / "data" / "processed" / "master_manifest.csv"
MASK_OUT_DIR = ROOT / "data" / "processed" / "unified_masks"
MANIFEST_OUT = ROOT / "data" / "processed" / "unified_manifest.csv"


# ── Format dönüştürücüler ─────────────────────────────────────────────────────

def _from_nrrd(mask_path: Path) -> np.ndarray | None:
    """NRRD maskesini okur → uint8 (0/255)."""
    try:
        import nrrd
        data, _ = nrrd.read(str(mask_path))
        arr = np.array(data, dtype=np.uint8)
        # Bazı NRRD dosyaları 0/1, bazıları 0/255
        if arr.max() == 1:
            arr = arr * 255
        return arr
    except Exception as e:
        print(f"  [!] NRRD okunamadı ({mask_path.name}): {e}")
        return None


def _from_png(mask_path: Path) -> np.ndarray | None:
    """PNG maskesini okur → uint8 (0/255)."""
    arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        print(f"  [!] PNG okunamadı: {mask_path.name}")
        return None
    # Threshold: her türlü gri değeri 0 veya 255'e sabitle
    _, arr = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    return arr


def _from_green_image(img_path: Path) -> np.ndarray | None:
    """
    DEU yeşil boyalı görüntüden maske çıkarır.
    green_mask_extractor modülünü kullanır.
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        mask = extract_green_mask(img)          # 0/1 → 0/255'e çevir
        return (mask * 255).astype(np.uint8)
    except Exception as e:
        print(f"  [!] Yeşil maske çıkarılamadı ({img_path.name}): {e}")
        return None


def load_mask(row: pd.Series) -> np.ndarray | None:
    """
    Satırdaki kaynak ve yol bilgisine göre doğru okuyucuyu seçer.

    Öncelik sırası:
      1. mask_path varsa → formatına göre oku (NRRD / PNG)
      2. mask_path yoksa ve source=DEU → img_path'ten yeşil maske çıkar
      3. Hiçbiri yoksa → None (NIH gibi)
    """
    mask_path = str(row.get("mask_path", "N/A")).strip()
    img_path  = str(row.get("img_path",  "N/A")).strip()
    source    = str(row.get("source",    "")).upper()

    if mask_path not in ("N/A", "", "nan"):
        p = Path(mask_path)
        if not p.exists():
            return None
        if p.suffix.lower() == ".nrrd":
            return _from_nrrd(p)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return _from_png(p)

    # DEU: mask_path yoksa görüntüdeki yeşil bölgeden çıkar
    if source == "DEU" and img_path not in ("N/A", "", "nan"):
        p = Path(img_path)
        if p.exists():
            return _from_green_image(p)

    return None   # NIH veya bulunamayan


def verify_mask(mask: np.ndarray, row: pd.Series) -> list[str]:
    """
    Maskeyi doğrular. Sorun listesi döner (boşsa temiz).
    """
    issues = []

    if mask.ndim != 2:
        issues.append(f"boyut_hatası:{mask.shape}")

    unique = set(np.unique(mask))
    if not unique.issubset({0, 255}):
        issues.append(f"değer_hatası:{unique}")

    if int(row.get("is_pneumo", 0)) == 1 and mask.sum() == 0:
        issues.append("pozitif_ama_boş_maske")

    return issues


# ── Ana dönüşüm ───────────────────────────────────────────────────────────────

def unify(verify_only: bool = False) -> None:
    """
    master_manifest.csv'deki tüm maskeleri unified PNG formatına çevirir.
    verify_only=True ise sadece sorunları raporlar, dosya yazmaz.
    """
    if not MANIFEST_IN.exists():
        print(f"[!] Manifest bulunamadı: {MANIFEST_IN}")
        print("    Önce: python scripts/data_manager.py --build_manifest")
        return

    df = pd.read_csv(MANIFEST_IN)
    print(f"\n  Kayıt sayısı : {len(df)}")
    print(f"  Mod          : {'Sadece Doğrulama' if verify_only else 'Dönüştür + Kaydet'}")
    print()

    if not verify_only:
        MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    new_mask_paths = []
    stats = {"converted": 0, "skipped_no_mask": 0, "error": 0, "already_done": 0}
    all_issues: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Unify"):
        source   = str(row.get("source", "UNK")).upper()
        img_path = str(row.get("img_path", ""))

        # Çıktı yolu: unified_masks/SIIM/abc123.png
        img_stem = Path(img_path).stem if img_path not in ("N/A", "") else "unknown"
        out_path = MASK_OUT_DIR / source / f"{img_stem}.png"

        # is_pneumo=0 ve mask yoksa atla
        is_pneumo = int(row.get("is_pneumo", 0))
        if is_pneumo == 0 and str(row.get("mask_path", "N/A")).strip() in ("N/A", "", "nan"):
            new_mask_paths.append("N/A")
            stats["skipped_no_mask"] += 1
            continue

        # Zaten dönüştürülmüşse atla
        if not verify_only and out_path.exists():
            new_mask_paths.append(str(out_path))
            stats["already_done"] += 1
            continue

        # Maskeyi yükle
        mask = load_mask(row)
        if mask is None:
            new_mask_paths.append("N/A")
            stats["skipped_no_mask"] += 1
            continue

        # Doğrula
        issues = verify_mask(mask, row)
        if issues:
            all_issues.append({
                "source":   source,
                "img_path": img_path,
                "issues":   "; ".join(issues),
            })
            stats["error"] += 1

        # Kaydet
        if not verify_only:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), mask)
            new_mask_paths.append(str(out_path))
            stats["converted"] += 1
        else:
            new_mask_paths.append(str(out_path) if not issues else "ERROR")

    # Unified manifest kaydet
    if not verify_only:
        df_out = df.copy()
        df_out["mask_path"] = new_mask_paths
        df_out.to_csv(MANIFEST_OUT, index=False, encoding="utf-8-sig")

    # Özet
    print("\n" + "═" * 55)
    print("  ANNOTATION BİRLEŞTİRME SONUCU")
    print("═" * 55)
    print(f"  Dönüştürüldü      : {stats['converted']:>6,}")
    print(f"  Zaten vardı       : {stats['already_done']:>6,}")
    print(f"  Maske yok (NIH)   : {stats['skipped_no_mask']:>6,}")
    print(f"  Hatalı            : {stats['error']:>6,}")

    if all_issues:
        issues_df = pd.DataFrame(all_issues)
        issues_path = ROOT / "data" / "processed" / "annotation_issues.csv"
        issues_df.to_csv(issues_path, index=False)
        print(f"\n  ⚠  {len(all_issues)} sorun tespit edildi → {issues_path.name}")
        # Kısa özet
        for issue_row in all_issues[:5]:
            print(f"    [{issue_row['source']}] {Path(issue_row['img_path']).name}: {issue_row['issues']}")
        if len(all_issues) > 5:
            print(f"    ... ve {len(all_issues) - 5} sorun daha")
    else:
        print("  ✓ Tüm maskeler temiz!")

    if not verify_only:
        print(f"\n  ✓ Unified manifest → {MANIFEST_OUT}")
        print(f"  ✓ Maskeler        → {MASK_OUT_DIR}/")
        print()
        print("  Sıradaki adım:")
        print("    python scripts/prepare_splits.py \\")
        print("      --opensource data/processed/unified_manifest.csv")

    print("═" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tüm annotation formatlarını standart binary PNG'ye çevirir."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Sadece doğrula, dosya yazma.",
    )
    args = parser.parse_args()
    unify(verify_only=args.verify)
