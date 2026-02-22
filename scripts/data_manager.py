"""
Global Veri Entegrasyon ve Standardizasyon Yöneticisi

Desteklenen Kaynaklar:
  - SIIM-ACR Pneumothorax Segmentation (Kaggle) — RLE maskeli DICOM
  - ChestX-ray14 (NIH / Kaggle)             — PNG, sınıflandırma etiketi
  - DEU Yerel                                — DICOM + NRRD maske

Komutlar:
  python scripts/data_manager.py --sync_all           # Tam pipeline
  python scripts/data_manager.py --download_siim      # Sadece SIIM indir
  python scripts/data_manager.py --download_nih       # Sadece NIH indir
  python scripts/data_manager.py --convert_rle        # RLE → NRRD dönüştür
  python scripts/data_manager.py --build_manifest     # master_manifest.csv oluştur
  python scripts/data_manager.py --quality_check      # Sanity check çalıştır
  python scripts/data_manager.py --positive_only      # Sadece pozitif vakalar (disk tasarrufu)

master_manifest.csv Sütunları:
  source, img_path, mask_path, is_pneumo, split,
  patient_id, view_position, pixel_spacing, img_rows, img_cols

TÜBİTAK 2209-A | Ahmet Demir
Dokuz Eylül Üniversitesi Tıp Fakültesi
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_manager")

# ── Dizin sabitleri ───────────────────────────────────────────────────────────

ROOT            = Path(__file__).parent.parent
RAW_DIR         = ROOT / "data" / "raw"
PROCESSED_DIR   = ROOT / "data" / "processed"
MASKS_DIR       = ROOT / "data" / "masks"

SIIM_DIR        = RAW_DIR / "global" / "siim"
NIH_DIR         = RAW_DIR / "global" / "nih"
DEU_DIR         = RAW_DIR / "local" / "deu"

MASTER_MANIFEST = PROCESSED_DIR / "master_manifest.csv"
QC_LOG          = PROCESSED_DIR / "quality_check.csv"

# SIIM-ACR görüntü boyutu (orijinal: 1024×1024)
SIIM_IMG_SIZE   = 1024

# ── Yardımcı: dizin oluşturma ─────────────────────────────────────────────────

def _makedirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. KAGGLE ENTEGRASYONU
# ══════════════════════════════════════════════════════════════════════════════

def download_siim(positive_only: bool = False) -> None:
    """
    SIIM-ACR Pneumothorax Segmentation veri setini Kaggle API ile indirir.

    Gereksinim:
        pip install kaggle
        ~/.kaggle/kaggle.json  ←  Kaggle hesabından API token

    Disk ihtiyacı:
        Tam veri seti : ~12 GB  (DICOM)
        positive_only : ~2-3 GB (sadece pnömotoraks vakalar)
    """
    _makedirs(SIIM_DIR)
    log.info("SIIM-ACR indiriliyor → %s", SIIM_DIR)

    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "siim-acr-pneumothorax-segmentation",
                "-p", str(SIIM_DIR),
            ],
            check=True,
        )
    except FileNotFoundError:
        log.error(
            "kaggle komutu bulunamadı. Kurulum:\n"
            "  pip install kaggle\n"
            "  ~/.kaggle/kaggle.json dosyanıza API token'ınızı ekleyin."
        )
        return
    except subprocess.CalledProcessError as exc:
        log.error("Kaggle indirme başarısız: %s", exc)
        return

    # Zip dosyalarını aç
    for zf_path in SIIM_DIR.glob("*.zip"):
        log.info("Arşiv açılıyor: %s", zf_path.name)
        with zipfile.ZipFile(zf_path) as zf:
            if positive_only:
                # Sadece pnömotoraks vakalarını çıkar (RLE CSV'den belirle)
                rle_csv = SIIM_DIR / "train-rle.csv"
                if rle_csv.exists():
                    pos_ids = _get_positive_ids(rle_csv)
                    members = [
                        m for m in zf.namelist()
                        if any(pid in m for pid in pos_ids) or m.endswith(".csv")
                    ]
                    zf.extractall(SIIM_DIR, members=members)
                    log.info("Pozitif vaka filtresi: %d / %d dosya", len(members), len(zf.namelist()))
                else:
                    zf.extractall(SIIM_DIR)
            else:
                zf.extractall(SIIM_DIR)
        zf_path.unlink()

    log.info("✓ SIIM-ACR indirme tamamlandı → %s", SIIM_DIR)


def download_nih(positive_only: bool = False) -> None:
    """
    NIH ChestX-ray14 veri setini Kaggle üzerinden indirir.
    (Kaggle dataset: 'nih-chest-xrays/data')

    Disk ihtiyacı:
        Tam veri seti : ~45 GB
        positive_only : ~5 GB  (sadece pnömotoraks içeren görüntüler)
    """
    _makedirs(NIH_DIR)
    log.info("NIH ChestX-ray14 indiriliyor → %s", NIH_DIR)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "nih-chest-xrays/data",
                "-p", str(NIH_DIR),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.error("NIH indirme başarısız: %s", exc)
        return

    for zf_path in NIH_DIR.glob("*.zip"):
        log.info("Arşiv açılıyor: %s", zf_path.name)
        with zipfile.ZipFile(zf_path) as zf:
            zf.extractall(NIH_DIR)
        zf_path.unlink()

    # positive_only: sadece pnömotoraks etiketli satırları tut, diğer PNG'leri sil
    if positive_only:
        _filter_nih_positive_only()

    log.info("✓ NIH indirme tamamlandı → %s", NIH_DIR)


def _get_positive_ids(rle_csv: Path) -> set[str]:
    """SIIM train-rle.csv'den pnömotoraks olan ImageID'leri döner."""
    df  = pd.read_csv(rle_csv)
    col = "EncodedPixels" if "EncodedPixels" in df.columns else df.columns[-1]
    pos = df[df[col].astype(str).str.strip() != "-1"]["ImageId"].unique()
    return set(pos.astype(str))


def _filter_nih_positive_only() -> None:
    """NIH dizininde sadece pnömotoraks etiketli PNG'leri bırakır."""
    label_csv = NIH_DIR / "Data_Entry_2017.csv"
    if not label_csv.exists():
        log.warning("NIH label CSV bulunamadı, filtreleme atlandı.")
        return

    df       = pd.read_csv(label_csv)
    ptx_imgs = set(df[df["Finding Labels"].str.contains("Pneumothorax", na=False)]["Image Index"])
    deleted  = 0

    for png in NIH_DIR.rglob("*.png"):
        if png.name not in ptx_imgs:
            png.unlink()
            deleted += 1

    log.info("NIH filtresi: %d görüntü silindi, %d pozitif kaldı.", deleted, len(ptx_imgs))


# ══════════════════════════════════════════════════════════════════════════════
# 2. RLE → NRRD DÖNÜŞÜMÜ
# ══════════════════════════════════════════════════════════════════════════════

def rle_to_mask(rle_str: str, height: int = SIIM_IMG_SIZE, width: int = SIIM_IMG_SIZE) -> np.ndarray:
    """
    SIIM-ACR RLE (Run-Length Encoding) formatını 2D binary mask'e çevirir.

    SIIM RLE formatı column-major (Fortran) sıralı.
    Birden fazla pnömotoraks bölgesi varsa OR ile birleştirilir.
    """
    rle_str = str(rle_str).strip()
    if rle_str in ("-1", "", "nan", "None"):
        return np.zeros((height, width), dtype=np.uint8)

    flat   = np.zeros(height * width, dtype=np.uint8)
    tokens = list(map(int, rle_str.split()))
    starts  = tokens[0::2]
    lengths = tokens[1::2]

    for start, length in zip(starts, lengths):
        end = start + length
        if end <= len(flat):
            flat[start:end] = 1
        else:
            flat[start:] = 1   # kenar koruma

    # column-major → row-major
    return flat.reshape(height, width, order="F")


def convert_rle_to_nrrd(
    siim_dir: Path | None = None,
    mask_out_dir: Path | None = None,
) -> None:
    """
    SIIM-ACR train-rle.csv dosyasındaki tüm RLE maskelerini NRRD olarak kaydeder.

    Çıktı: data/masks/siim/<ImageId>.nrrd
    Not  : Aynı görüntü için birden fazla RLE satırı varsa OR birleştirilir.
    """
    try:
        import nrrd
    except ImportError:
        log.error("pynrrd kurulu değil: pip install pynrrd")
        return

    siim_dir     = siim_dir or SIIM_DIR
    mask_out_dir = mask_out_dir or MASKS_DIR / "siim"
    _makedirs(mask_out_dir)

    rle_csv = next(siim_dir.glob("*train-rle*.csv"), None) or (siim_dir / "train-rle.csv")
    if not rle_csv.exists():
        log.error("train-rle.csv bulunamadı: %s", rle_csv)
        return

    df = pd.read_csv(rle_csv)
    # Sütun adlarını normalize et
    df.columns = [c.strip() for c in df.columns]
    id_col  = "ImageId"   if "ImageId"       in df.columns else df.columns[0]
    rle_col = "EncodedPixels" if "EncodedPixels" in df.columns else df.columns[1]

    # Aynı görüntü için birden fazla RLE → OR birleştirme
    grouped = df.groupby(id_col)[rle_col].apply(list)
    converted = skipped = 0

    for image_id, rle_list in tqdm(grouped.items(), total=len(grouped), desc="RLE→NRRD"):
        out_path = mask_out_dir / f"{image_id}.nrrd"
        if out_path.exists():
            skipped += 1
            continue

        # Birden fazla maske → OR
        combined = np.zeros((SIIM_IMG_SIZE, SIIM_IMG_SIZE), dtype=np.uint8)
        for rle in rle_list:
            combined |= rle_to_mask(str(rle))

        header = {
            "type":      "uint8",
            "dimension": 2,
            "sizes":     [SIIM_IMG_SIZE, SIIM_IMG_SIZE],
            "encoding":  "gzip",
            "space":     "left-posterior-superior",
        }
        nrrd.write(str(out_path), combined, header)
        converted += 1

    log.info("✓ RLE → NRRD: %d dönüştürüldü, %d zaten vardı.", converted, skipped)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DICOM METADATA ÇIKARMA
# ══════════════════════════════════════════════════════════════════════════════

def extract_dicom_meta(dcm_path: Path) -> dict:
    """
    DICOM dosyasından temel metadata çıkarır.
    pixel_array yüklenmez (stop_before_pixels=True → hızlı).
    """
    try:
        import pydicom
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    except Exception as exc:
        log.debug("DICOM meta okunamadı (%s): %s", dcm_path.name, exc)
        return {"patient_id": "N/A", "view_position": "N/A",
                "pixel_spacing": "N/A", "img_rows": 0, "img_cols": 0}

    raw_ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    ps_str = f"{float(raw_ps[0]):.4f}" if raw_ps else "N/A"

    return {
        "patient_id":    str(getattr(ds, "PatientID",    "N/A")).strip(),
        "view_position": str(getattr(ds, "ViewPosition", "N/A")).strip().upper(),
        "pixel_spacing": ps_str,
        "img_rows":      int(getattr(ds, "Rows",    0)),
        "img_cols":      int(getattr(ds, "Columns", 0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. KAYNAK MANIFEST'LERİ OLUŞTURMA
# ══════════════════════════════════════════════════════════════════════════════

def _build_siim_records(positive_only: bool = False) -> list[dict]:
    """SIIM-ACR dizininden kayıt listesi oluşturur."""
    rle_csv = next(SIIM_DIR.glob("*train-rle*.csv"), None)
    if not rle_csv:
        log.warning("SIIM train-rle.csv bulunamadı, atlanıyor.")
        return []

    df = pd.read_csv(rle_csv)
    df.columns = [c.strip() for c in df.columns]
    id_col  = "ImageId"       if "ImageId"       in df.columns else df.columns[0]
    rle_col = "EncodedPixels" if "EncodedPixels"  in df.columns else df.columns[1]

    # Görüntü başına is_pneumo
    img_labels = (
        df.groupby(id_col)[rle_col]
        .apply(lambda x: int(any(str(v).strip() != "-1" for v in x)))
        .reset_index()
        .rename(columns={rle_col: "is_pneumo"})
    )

    if positive_only:
        img_labels = img_labels[img_labels["is_pneumo"] == 1]

    # DICOM dosyalarını bul
    dcm_files: dict[str, Path] = {}
    for dcm in SIIM_DIR.rglob("*.dcm"):
        dcm_files[dcm.stem] = dcm

    mask_dir = MASKS_DIR / "siim"
    records  = []

    for _, row in tqdm(img_labels.iterrows(), total=len(img_labels), desc="SIIM manifest"):
        img_id = str(row[id_col])
        dcm    = dcm_files.get(img_id)
        if dcm is None:
            log.debug("DICOM bulunamadı: %s", img_id)
            continue

        mask_path = mask_dir / f"{img_id}.nrrd"
        meta      = extract_dicom_meta(dcm)

        records.append({
            "source":        "SIIM",
            "img_path":      str(dcm),
            "mask_path":     str(mask_path) if mask_path.exists() else "N/A",
            "is_pneumo":     int(row["is_pneumo"]),
            "split":         "",
            **meta,
        })

    log.info("SIIM: %d kayıt (pozitif: %d)", len(records), sum(r["is_pneumo"] for r in records))
    return records


def _build_nih_records(positive_only: bool = False) -> list[dict]:
    """NIH ChestX-ray14 dizininden kayıt listesi oluşturur."""
    # Kaggle dataset'i v2020 versiyonunu kullanabilir
    label_csv = NIH_DIR / "Data_Entry_2017.csv"
    if not label_csv.exists():
        for alt in [
            NIH_DIR / "Data_Entry_2017_v2020.csv",
            NIH_DIR / "Data_Entry_2017_v2020updated.csv",
        ]:
            if alt.exists():
                label_csv = alt
                log.info("NIH label CSV (alternatif): %s", alt.name)
                break
        else:
            log.warning("NIH Data_Entry_2017.csv bulunamadı, atlanıyor.")
            return []

    df = pd.read_csv(label_csv)
    df["is_pneumo"] = df["Finding Labels"].str.contains("Pneumothorax", na=False).astype(int)

    if positive_only:
        df = df[df["is_pneumo"] == 1]

    # PNG dosyalarını bul
    png_files: dict[str, Path] = {}
    for png in NIH_DIR.rglob("*.png"):
        png_files[png.name] = png

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="NIH manifest"):
        img_name = str(row["Image Index"])
        png      = png_files.get(img_name)
        if png is None:
            continue

        # NIH: PixelSpacing sütunu "x y" formatında
        raw_ps = str(row.get("OriginalImagePixelSpacing[x]", "N/A"))
        try:
            ps_float = float(raw_ps) / 10.0   # mm → cm
            ps_str   = f"{ps_float:.4f}"
        except (ValueError, TypeError):
            ps_str = "N/A"

        records.append({
            "source":        "NIH",
            "img_path":      str(png),
            "mask_path":     "N/A",      # NIH'de piksel maskesi yok
            "is_pneumo":     int(row["is_pneumo"]),
            "split":         "",
            "patient_id":    str(row.get("Patient ID", "N/A")),
            "view_position": str(row.get("View Position", "N/A")).upper(),
            "pixel_spacing": ps_str,
            "img_rows":      int(row.get("OriginalImage[Height]", 0)),
            "img_cols":      int(row.get("OriginalImage[Width]", 0)),
        })

    log.info("NIH: %d kayıt (pozitif: %d)", len(records), sum(r["is_pneumo"] for r in records))
    return records


def _build_deu_records() -> list[dict]:
    """
    DEU yerel veri manifest'ini yükler.
    data/hospital_manifest.csv varsa kullanır;
    yoksa DEU DICOM dizinini tarar.
    """
    hospital_csv = ROOT / "data" / "hospital_manifest.csv"

    if hospital_csv.exists():
        df = pd.read_csv(hospital_csv)
        records = []
        for _, row in df.iterrows():
            meta = {}
            if Path(str(row.get("image_path", ""))).suffix.lower() in (".dcm", ".dicom"):
                meta = extract_dicom_meta(Path(row["image_path"]))
            records.append({
                "source":        "DEU",
                "img_path":      str(row.get("image_path",      "N/A")),
                "mask_path":     str(row.get("mask_path",       "N/A")),
                "is_pneumo":     int(row.get("has_pneumothorax", 0)),
                "split":         str(row.get("split", "")),
                "patient_id":    str(row.get("patient_id",    meta.get("patient_id",    "N/A"))),
                "view_position": str(row.get("view_position", meta.get("view_position", "N/A"))),
                "pixel_spacing": str(row.get("pixel_spacing", meta.get("pixel_spacing", "N/A"))),
                "img_rows":      int(row.get("img_rows",      meta.get("img_rows", 0))),
                "img_cols":      int(row.get("img_cols",      meta.get("img_cols", 0))),
            })
        log.info("DEU: %d kayıt hospital_manifest.csv'den yüklendi.", len(records))
        return records

    # CSV yoksa DICOM dizinini tara
    if not DEU_DIR.exists():
        log.warning("DEU dizini bulunamadı (%s), atlanıyor.", DEU_DIR)
        return []

    records = []
    for dcm in tqdm(list(DEU_DIR.rglob("*.dcm")), desc="DEU tarama"):
        # Eşleşen maskeyi bul: aynı stem + .nrrd
        nrrd_mask = dcm.with_suffix(".nrrd")
        if not nrrd_mask.exists():
            nrrd_mask = MASKS_DIR / "deu" / f"{dcm.stem}.nrrd"

        has_mask  = nrrd_mask.exists()
        is_pneumo = 1 if has_mask else 0
        meta      = extract_dicom_meta(dcm)

        records.append({
            "source":    "DEU",
            "img_path":  str(dcm),
            "mask_path": str(nrrd_mask) if has_mask else "N/A",
            "is_pneumo": is_pneumo,
            "split":     "",
            **meta,
        })

    log.info("DEU: %d DICOM tarandı.", len(records))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 5. MASTER MANIFEST BİRLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def build_manifest(positive_only: bool = False) -> pd.DataFrame:
    """
    DEU + SIIM + NIH kayıtlarını birleştirip master_manifest.csv kaydeder.

    Özellikler:
      - Tekrar eden görüntüler md5 hash ile tespit edilir
      - Sütun sırası standartlaştırılır
      - is_pneumo, img_rows, img_cols integer'a cast edilir
    """
    _makedirs(PROCESSED_DIR)

    all_records: list[dict] = []
    all_records.extend(_build_deu_records())
    all_records.extend(_build_siim_records(positive_only))
    all_records.extend(_build_nih_records(positive_only))

    if not all_records:
        log.error("Hiç kayıt bulunamadı. Veri dizinlerini kontrol edin.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Sütun sırası
    cols = [
        "source", "img_path", "mask_path", "is_pneumo", "split",
        "patient_id", "view_position", "pixel_spacing", "img_rows", "img_cols",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "N/A"
    df = df[cols]

    # Tip düzeltmeleri
    df["is_pneumo"] = pd.to_numeric(df["is_pneumo"], errors="coerce").fillna(0).astype(int)
    df["img_rows"]  = pd.to_numeric(df["img_rows"],  errors="coerce").fillna(0).astype(int)
    df["img_cols"]  = pd.to_numeric(df["img_cols"],  errors="coerce").fillna(0).astype(int)

    # Tekrar kontrolü: img_path hash
    before = len(df)
    df = df.drop_duplicates(subset=["img_path"])
    after  = len(df)
    if before != after:
        log.info("Tekrar eden %d kayıt kaldırıldı.", before - after)

    df.to_csv(MASTER_MANIFEST, index=False, encoding="utf-8-sig")

    # Özet
    log.info("\n" + "=" * 55)
    log.info("  MASTER MANIFEST ÖZET")
    log.info("=" * 55)
    for src, grp in df.groupby("source"):
        n     = len(grp)
        n_pos = grp["is_pneumo"].sum()
        log.info("  %-6s → %5d görüntü  (+: %4d  -: %4d)", src, n, n_pos, n - n_pos)
    log.info("  TOPLAM → %5d görüntü  (+: %4d  -: %4d)",
             len(df), df["is_pneumo"].sum(), (df["is_pneumo"] == 0).sum())
    log.info("=" * 55)
    log.info("✓ Kaydedildi: %s", MASTER_MANIFEST)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. SANITY CHECK (Kalite Kontrol)
# ══════════════════════════════════════════════════════════════════════════════

def _check_image(img_path: str) -> str | None:
    """
    Görüntüyü kontrol eder. Sorun yoksa None, varsa açıklama döner.
    DICOM ve PNG/JPEG desteklenir.
    """
    p = Path(img_path)
    if not p.exists():
        return "dosya_yok"

    try:
        if p.suffix.lower() in (".dcm", ".dicom"):
            import pydicom
            ds  = pydicom.dcmread(str(p))
            arr = ds.pixel_array.astype(np.float32)
        else:
            import cv2
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if arr is None:
                return "okunamadı"

        if arr.size == 0:
            return "boş_görüntü"
        if np.isnan(arr).any():
            return "nan_değer"
        if arr.max() == arr.min():
            return "sabit_piksel"   # tamamen siyah/beyaz görüntü

    except Exception as exc:
        return f"hata:{str(exc)[:60]}"

    return None


def _check_mask(mask_path: str, img_rows: int, img_cols: int) -> str | None:
    """Maske dosyasını kontrol eder."""
    if mask_path in ("N/A", "", None):
        return None    # Maske yok ama is_pneumo=0 ise sorun değil

    p = Path(mask_path)
    if not p.exists():
        return "maske_yok"

    try:
        if p.suffix.lower() == ".nrrd":
            import nrrd
            data, _ = nrrd.read(str(p))
            arr      = np.array(data)
        else:
            import cv2
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return "maske_okunamadı"

        if arr.sum() == 0:
            return "boş_maske"   # is_pneumo=1 ama maske boş

        # Boyut uyumu (varsa)
        if img_rows > 0 and img_cols > 0:
            if arr.shape[0] != img_rows or arr.shape[1] != img_cols:
                return f"boyut_uyumsuz:{arr.shape}≠({img_rows},{img_cols})"

    except Exception as exc:
        return f"maske_hata:{str(exc)[:60]}"

    return None


def run_quality_check(manifest_path: Path | None = None) -> pd.DataFrame:
    """
    master_manifest.csv üzerinde kapsamlı kalite kontrolü çalıştırır.

    Tespit edilen sorunlar:
      - Eksik / bozuk görüntü dosyaları
      - NaN piksel değerleri
      - Sabit (boş) görüntüler
      - Boş maskeler (is_pneumo=1 ama maske sıfır)
      - Görüntü-maske boyut uyumsuzluğu

    Çıktı: data/processed/quality_check.csv
    """
    manifest_path = manifest_path or MASTER_MANIFEST
    if not manifest_path.exists():
        log.error("Manifest bulunamadı: %s — Önce --build_manifest çalıştırın.", manifest_path)
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    issues: list[dict] = []

    log.info("Sanity Check başlıyor — %d kayıt…", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Kalite Kontrol"):
        img_issue  = _check_image(str(row["img_path"]))
        mask_issue = None

        if row["is_pneumo"] == 1:
            mask_issue = _check_mask(
                str(row["mask_path"]),
                int(row.get("img_rows", 0)),
                int(row.get("img_cols", 0)),
            )

        if img_issue or mask_issue:
            issues.append({
                "source":     row["source"],
                "img_path":   row["img_path"],
                "mask_path":  row["mask_path"],
                "is_pneumo":  row["is_pneumo"],
                "img_issue":  img_issue  or "",
                "mask_issue": mask_issue or "",
            })

    issues_df = pd.DataFrame(issues)
    _makedirs(PROCESSED_DIR)
    issues_df.to_csv(QC_LOG, index=False, encoding="utf-8-sig")

    log.info("\n" + "=" * 55)
    log.info("  KALİTE KONTROL SONUÇLARI")
    log.info("=" * 55)
    log.info("  Toplam kayıt     : %d", len(df))
    log.info("  Sorunlu kayıt    : %d  (%.1f%%)", len(issues_df), len(issues_df) / max(len(df), 1) * 100)

    if not issues_df.empty:
        for src, grp in issues_df.groupby("source"):
            log.info("    %-6s  %d sorun", src, len(grp))
        log.info("  Detay: %s", QC_LOG)
    else:
        log.info("  ✓ Hiç sorun bulunamadı!")
    log.info("=" * 55)

    return issues_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. TEMİZLEME: Sorunlu Kayıtları Manifestten Çıkar
# ══════════════════════════════════════════════════════════════════════════════

def remove_bad_records() -> None:
    """
    quality_check.csv'de listelenen sorunlu kayıtları manifest'ten kaldırır.
    Temizlenmiş manifest → data/processed/master_manifest_clean.csv
    """
    if not MASTER_MANIFEST.exists() or not QC_LOG.exists():
        log.error("Önce --build_manifest ve --quality_check çalıştırın.")
        return

    manifest  = pd.read_csv(MASTER_MANIFEST)
    bad_df    = pd.read_csv(QC_LOG)
    bad_paths = set(bad_df["img_path"].astype(str))

    clean = manifest[~manifest["img_path"].astype(str).isin(bad_paths)].copy()
    out   = PROCESSED_DIR / "master_manifest_clean.csv"
    clean.to_csv(out, index=False, encoding="utf-8-sig")

    removed = len(manifest) - len(clean)
    log.info("✓ %d sorunlu kayıt kaldırıldı. Temiz manifest: %s (%d kayıt)", removed, out, len(clean))


# ══════════════════════════════════════════════════════════════════════════════
# 8. EĞİTİM STRATEJİSİ: Pre-training config üreteci
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_configs() -> None:
    """
    İki eğitim stratejisi için hazır config dosyaları üretir:

    A) Joint Training  — Tek aşamalı, source-aware sampler (kolay, hızlı)
    B) Pre-train + Fine-tune — İki aşamalı, daha yüksek AUC beklentisi

    Literatür: Domain adaptation for medical imaging
    (Raghu et al. 2019, Chen et al. 2020 ChestX-ray)
    """
    import json

    configs_dir = ROOT / "configs"
    _makedirs(configs_dir)

    # A) Joint Training
    joint = {
        "strategy":       "joint_training",
        "description":    "Tüm kaynaklar birlikte, source-aware sampler",
        "encoder_name":   "efficientnet-b0",
        "epochs":         50,
        "batch_size":     16,
        "lr":             1e-3,
        "weight_decay":   1e-4,
        "dice_weight":    0.5,
        "num_folds":      5,
        "sampler": {
            "strategy":       "custom_ratio",
            "target_pos_ratio": 0.35,
            "local_boost":    2.0,    # DEU verisi 2x ağırlık
        },
        "hard_negative_mining": True,
        "hnm_interval":   3,
        "hnm_threshold":  0.4,
        "hnm_multiplier": 3.0,
        "checkpoint_dir": "results/checkpoints/joint",
        "wandb_project":  "Pneumothorax-Detection",
        "wandb_entity":   "ahmet-ai-t-bi-tak",
        "note": (
            "Önerilen başlangıç stratejisi. DEU local_boost=2.0 ile "
            "yerel veri ağırlıklandırılır. SIIM+NIH genel özellik öğretir."
        ),
    }

    # B) Pre-train + Fine-tune
    pretrain = {
        "strategy":       "pretrain_finetune",
        "description":    "Aşama 1: SIIM+NIH global, Aşama 2: DEU fine-tune",
        "stage_1": {
            "data_sources":   ["SIIM", "NIH"],
            "encoder_name":   "efficientnet-b0",
            "epochs":         20,
            "batch_size":     16,
            "lr":             1e-3,
            "weight_decay":   1e-4,
            "dice_weight":    0.5,
            "checkpoint_dir": "results/checkpoints/pretrain",
            "note":           "Global veriyle encoder öğretme aşaması.",
        },
        "stage_2": {
            "data_sources":   ["DEU", "SIIM"],
            "encoder_name":   "efficientnet-b0",
            "freeze_encoder": True,    # Encoder dondurulur, sadece decoder fine-tune
            "epochs":         30,
            "batch_size":     8,       # Küçük DEU seti için
            "lr":             1e-4,    # Fine-tune için düşük LR
            "weight_decay":   1e-5,
            "dice_weight":    0.6,
            "checkpoint_dir": "results/checkpoints/finetune",
            "pretrain_ckpt":  "results/checkpoints/pretrain/best_model.pth",
            "local_boost":    2.5,
            "note": (
                "Encoder dondurulup decoder + classifier fine-tune edilir. "
                "DEU lokal scanner karakteristiklerine adaptasyon sağlanır. "
                "AUC beklentisi: joint'den %2-4 daha yüksek."
            ),
        },
    }

    for name, cfg in [("joint_training", joint), ("pretrain_finetune", pretrain)]:
        out = configs_dir / f"{name}_config.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        log.info("✓ Config üretildi: %s", out)

    log.info("\n  Strateji Karşılaştırması:")
    log.info("  %-25s  Kolay | Hızlı | AUC +0    → Başlangıç için önerilir", "A) Joint Training")
    log.info("  %-25s  Zor   | Yavaş | AUC +2-4%% → Yayın / TÜBİTAK için", "B) Pre-train+Fine-tune")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ÖZET RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    """Mevcut veri durumunu terminal'e yazar."""
    if not MASTER_MANIFEST.exists():
        log.info("Manifest yok — önce --build_manifest çalıştırın.")
        return

    df = pd.read_csv(MASTER_MANIFEST)

    print("\n" + "═" * 60)
    print("  MASTER MANIFEST ÖZET")
    print("═" * 60)
    for src in ["DEU", "SIIM", "NIH"]:
        grp  = df[df["source"] == src]
        n    = len(grp)
        if n == 0:
            continue
        npos = grp["is_pneumo"].sum()
        print(f"  {src:<6} → {n:>6,} görüntü   (+): {npos:>5,}  ({npos/n:.0%})")

    total = len(df)
    tpos  = df["is_pneumo"].sum()
    print("─" * 60)
    print(f"  TOPLAM → {total:>6,} görüntü   (+): {tpos:>5,}  ({tpos/total:.0%})")

    vp_counts = df["view_position"].value_counts()
    print("\n  Çekim Pozisyonu:")
    for vp, cnt in vp_counts.head(5).items():
        print(f"    {vp:<8} {cnt:>6,}")

    if QC_LOG.exists():
        bad = pd.read_csv(QC_LOG)
        print(f"\n  ⚠  Kalite sorunu: {len(bad)} kayıt ({QC_LOG.name})")

    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global Veri Entegrasyon ve Standardizasyon Yöneticisi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sync_all",      action="store_true", help="Tam pipeline (indir + dönüştür + manifest + QC)")
    parser.add_argument("--download_siim", action="store_true", help="SIIM-ACR Kaggle'dan indir")
    parser.add_argument("--download_nih",  action="store_true", help="NIH ChestX-ray14 indir")
    parser.add_argument("--convert_rle",   action="store_true", help="RLE → NRRD dönüştür")
    parser.add_argument("--build_manifest",action="store_true", help="master_manifest.csv oluştur")
    parser.add_argument("--quality_check", action="store_true", help="Sanity check çalıştır")
    parser.add_argument("--remove_bad",    action="store_true", help="Sorunlu kayıtları manifestten kaldır")
    parser.add_argument("--gen_configs",   action="store_true", help="Eğitim stratejisi config'leri üret")
    parser.add_argument("--summary",       action="store_true", help="Özet raporu göster")
    parser.add_argument("--positive_only", action="store_true",
                        help="Sadece pnömotoraks vakalarını indir (~80 pct disk tasarrufu)")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.sync_all or args.download_siim:
        download_siim(positive_only=args.positive_only)

    if args.sync_all or args.download_nih:
        download_nih(positive_only=args.positive_only)

    if args.sync_all or args.convert_rle:
        convert_rle_to_nrrd()

    if args.sync_all or args.build_manifest:
        build_manifest(positive_only=args.positive_only)

    if args.sync_all or args.quality_check:
        run_quality_check()

    if args.remove_bad:
        remove_bad_records()

    if args.gen_configs:
        generate_training_configs()

    if args.summary or args.sync_all:
        print_summary()

    if args.sync_all:
        log.info("\n✓ Tam pipeline tamamlandı. Sıradaki adım:")
        log.info("  python scripts/prepare_splits.py \\")
        log.info("    --hospital data/hospital_manifest.csv \\")
        log.info("    --opensource data/processed/master_manifest.csv")


if __name__ == "__main__":
    main()
