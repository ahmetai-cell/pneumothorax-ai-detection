"""
DICOM + NRRD Pipeline  —  DEU Hastane Formatı
==============================================

DEU PACS arşivinden gelen çiftleri işler:
  • Görüntü : DICOM  (.dcm)
  • Etiket  : NRRD   (.nrrd, 3D Slicer çıktısı)

Üç temel yenilik (TÜBİTAK 2209-A):
  1. pair_dicom_nrrd()     : Dizin tarama → dosya adına göre otomatik eşleştirme
  2. apply_lung_window()   : DICOM tag'larından akciğer penceresi normalizasyonu
                             (WindowCenter / WindowWidth + RescaleSlope/Intercept)
  3. _check_alignment()    : NRRD maskesi DICOM çözünürlüğüyle tam çakışıyor mu?
                             Uyumsuzlukta AlignmentError fırlatır.

Kullanım — Dizin modu (DEU):
    pairs = pair_dicom_nrrd("data/raw/deu/dicom", "data/raw/deu/nrrd")
    ds = DicomSlicerDataset(pairs, img_size=512, lung_window=True)

Kullanım — CSV modu (unified_manifest.csv):
    df = pd.read_csv("data/processed/unified_manifest.csv")
    ds = DicomSlicerDataset(df, img_size=512, lung_window=True)

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import cv2
import nrrd
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    _NIBABEL_OK = True
except ImportError:
    _NIBABEL_OK = False

log = logging.getLogger(__name__)


# ── Özel Hata ─────────────────────────────────────────────────────────────────

class AlignmentError(ValueError):
    """NRRD maskesi ile DICOM görüntüsü boyutları uyuşmadığında fırlatılır."""


# ── 1. DOSYA ADI EŞLEŞTİRME ──────────────────────────────────────────────────

def pair_dicom_nrrd(
    dicom_dir: str | Path,
    nrrd_dir: str | Path,
    *,
    dicom_glob: str = "**/*.dcm",
    nrrd_glob: str = "**/*.nrrd",
    strict: bool = False,
) -> pd.DataFrame:
    """
    İki dizini tarar ve dosya adlarını (stem) eşleştirerek DataFrame döner.

    Args:
        dicom_dir  : DICOM dosyalarının bulunduğu kök dizin.
        nrrd_dir   : NRRD maske dosyalarının bulunduğu kök dizin.
        dicom_glob : DICOM için glob örüntüsü (varsayılan: **/*.dcm).
        nrrd_glob  : NRRD için glob örüntüsü (varsayılan: **/*.nrrd).
        strict     : True ise eşleşmeyenler ValueError fırlatır.

    Returns:
        DataFrame sütunları: img_path, mask_path, is_pneumo, source
            is_pneumo = 1  → eşleşen NRRD var (pozitif vaka)
            is_pneumo = 0  → NRRD yok (negatif vaka)
    """
    dicom_dir = Path(dicom_dir)
    nrrd_dir  = Path(nrrd_dir)

    # Stem → path sözlükleri
    dicom_map: dict[str, Path] = {p.stem: p for p in dicom_dir.glob(dicom_glob)}
    nrrd_map:  dict[str, Path] = {p.stem: p for p in nrrd_dir.glob(nrrd_glob)}

    if not dicom_map:
        raise FileNotFoundError(f"DICOM bulunamadı: {dicom_dir} / {dicom_glob}")

    # Eşleşmeyen NRRD'leri raporla
    unmatched_nrrd = set(nrrd_map) - set(dicom_map)
    if unmatched_nrrd:
        msg = f"{len(unmatched_nrrd)} NRRD dosyasının DICOM eşi yok: {sorted(unmatched_nrrd)[:5]} ..."
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    # Eşleşmeyen DICOM'ları raporla
    unmatched_dcm = set(dicom_map) - set(nrrd_map)
    if unmatched_dcm:
        log.info("%d DICOM için NRRD yok (negatif vaka olarak işlenecek)", len(unmatched_dcm))

    rows = []
    for stem, dcm_path in sorted(dicom_map.items()):
        nrrd_path = nrrd_map.get(stem)
        rows.append({
            "img_path":  str(dcm_path),
            "mask_path": str(nrrd_path) if nrrd_path else "N/A",
            "is_pneumo": 1 if nrrd_path else 0,
            "source":    "DEU",
        })

    df = pd.DataFrame(rows)
    log.info(
        "pair_dicom_nrrd: %d DICOM | %d eşleşti | %d negatif",
        len(dicom_map), df["is_pneumo"].sum(), (df["is_pneumo"] == 0).sum()
    )
    return df


# ── 2. AKCİĞER PENCERESİ NORMALİZASYON ──────────────────────────────────────

#  Akciğer parankimi için standart klinik pencere değerleri (Hounsfield Unit)
_LUNG_WC = -600   # Window Center
_LUNG_WW = 1600   # Window Width


def apply_lung_window(
    ds: pydicom.Dataset,
    *,
    force_lung: bool = False,
    wc: int = _LUNG_WC,
    ww: int = _LUNG_WW,
) -> np.ndarray:
    """
    DICOM pixel array'ini akciğer penceresine göre normalize eder.

    Adımlar:
      1. Ham piksel → Hounsfield Unit (HU):
             HU = pixel × RescaleSlope + RescaleIntercept
         (tag yoksa identity: slope=1, intercept=0)
      2. DICOM kendi WindowCenter/WindowWidth tag'larına bak:
         - Eğer bu değerler akciğer penceresine uygunsa (WC ∈ [-800,-400]) kullan
         - force_lung=True veya tag yoksa → sabit klinik değer kullan
      3. Klinik pencere formülü:
             lower = WC - WW/2
             upper = WC + WW/2
             normalize → [0.0, 1.0]
      4. float32 [0,1] döner.

    Args:
        ds         : pydicom.Dataset (dcmread çıktısı)
        force_lung : True → DICOM tag'larını yok say, daima klinik pencere kullan
        wc         : Varsayılan Window Center (-600)
        ww         : Varsayılan Window Width  (1600)
    """
    arr = ds.pixel_array.astype(np.float32)

    # Çok dilimli DICOM → ilk dilim
    if arr.ndim == 3:
        arr = arr[0]

    # Adım 1 — HU dönüşümü
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = arr * slope + intercept

    # Adım 2 — Pencere seçimi
    if not force_lung:
        raw_wc = getattr(ds, "WindowCenter", None)
        raw_ww = getattr(ds, "WindowWidth",  None)
        if raw_wc is not None and raw_ww is not None:
            # DICOM tag'ları bazen liste olarak gelir
            _wc = float(raw_wc[0] if hasattr(raw_wc, "__len__") else raw_wc)
            _ww = float(raw_ww[0] if hasattr(raw_ww, "__len__") else raw_ww)
            # Akciğer penceresine uygun mu? (WC ∈ [-900, -300])
            if -900 <= _wc <= -300:
                wc, ww = int(_wc), int(_ww)
                log.debug("DICOM kendi pencere değerleri kullanıldı: WC=%d WW=%d", wc, ww)
            else:
                log.debug(
                    "DICOM pencere akciğer aralığı dışında (WC=%.0f), klinik değer kullanılıyor.", _wc
                )

    # Adım 3 — Klinik pencere uygulaması
    lower = wc - ww / 2
    upper = wc + ww / 2
    hu_clipped = np.clip(hu, lower, upper)
    normalized = (hu_clipped - lower) / (upper - lower)   # [0, 1]

    return normalized.astype(np.float32)


# ── 3. BOYUT UYUM KONTROLÜ ────────────────────────────────────────────────────

def _check_alignment(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    img_path: str = "",
    mask_path: str = "",
    tolerance: int = 0,
) -> None:
    """
    NRRD maskesinin DICOM görüntüsüyle piksel çözünürlüğü eşleşiyor mu?

    Args:
        img_arr    : DICOM'dan elde edilen 2D numpy dizisi (H, W)
        mask_arr   : NRRD'den elde edilen 2D numpy dizisi (H, W)
        img_path   : Hata mesajı için görüntü yolu
        mask_path  : Hata mesajı için maske yolu
        tolerance  : İzin verilen maksimum piksel farkı (her boyut için).
                     0 = tam eşitlik zorunlu

    Raises:
        AlignmentError: Boyutlar tolerance'ı aşacak kadar farklıysa.
    """
    ih, iw = img_arr.shape[:2]
    mh, mw = mask_arr.shape[:2]

    dh = abs(ih - mh)
    dw = abs(iw - mw)

    if dh > tolerance or dw > tolerance:
        raise AlignmentError(
            f"Boyut uyumsuzluğu!\n"
            f"  DICOM : {ih}×{iw}  ({Path(img_path).name})\n"
            f"  NRRD  : {mh}×{mw}  ({Path(mask_path).name})\n"
            f"  Fark  : Δh={dh}, Δw={dw} (izin={tolerance}px)\n"
            f"  Çözüm : NRRD'yi 3D Slicer'da DICOM alanıyla referanslanarak "
            f"yeniden dışa aktarın."
        )

    if dh > 0 or dw > 0:
        log.warning(
            "Küçük boyut farkı tolere edildi: DICOM %dx%d, NRRD %dx%d (Δh=%d, Δw=%d)",
            ih, iw, mh, mw, dh, dw
        )


# ── Alt seviye okuyucular ──────────────────────────────────────────────────────

def _read_dicom_arr(
    path: str,
    lung_window: bool = True,
    force_lung: bool = False,
) -> tuple[np.ndarray, pydicom.Dataset]:
    """
    DICOM dosyasını okur.

    Returns:
        (arr, ds) — arr: float32 [0,1] 2D array; ds: pydicom Dataset
    """
    ds = pydicom.dcmread(str(path))

    if lung_window:
        arr = apply_lung_window(ds, force_lung=force_lung)
    else:
        # Basit min-max (eski davranış)
        raw = ds.pixel_array.astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        arr = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    return arr, ds


def _read_nrrd(path: str) -> np.ndarray:
    """3D Slicer NRRD segmentasyon → binary uint8 mask (H, W)."""
    data, header = nrrd.read(str(path))
    arr = np.asarray(data, dtype=np.float32)

    # 3D Slicer çıktıları genellikle (1, H, W) veya (H, W, 1) şeklindedir
    arr = arr.squeeze()

    if arr.ndim != 2:
        # (D, H, W) → ilk dilim al (tek-dilimli PA grafisi)
        if arr.ndim == 3:
            arr = arr[0]
        else:
            raise ValueError(f"Beklenmedik NRRD boyutu: {arr.shape} ({path})")

    return (arr > 0).astype(np.uint8) * 255


def _read_nifti(path: str) -> np.ndarray:
    """NIfTI segmentasyon → binary uint8 mask (H, W)."""
    if not _NIBABEL_OK:
        raise ImportError("nibabel kurulu değil: pip install nibabel")
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = np.rot90(arr, k=1)
    return (arr > 0).astype(np.uint8) * 255


def _read_mask(path: str) -> np.ndarray:
    """Uzantıya göre doğru okuyucuyu seçer → uint8 (0/255)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".nrrd":
        return _read_nrrd(path)
    elif suffix in (".nii", ".gz"):
        return _read_nifti(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Maske okunamadı: {path}")
        return (mask > 0).astype(np.uint8) * 255


# ── Dataset sınıfı ────────────────────────────────────────────────────────────

class DicomSlicerDataset(Dataset):
    """
    DICOM görüntüsü + 3D Slicer NRRD maskesi çiftlerinden oluşan dataset.

    İki çalışma modu:

    A) CSV / DataFrame modu — unified_manifest.csv:
        df = pd.read_csv("data/processed/unified_manifest.csv")
        ds = DicomSlicerDataset(df, lung_window=True)

    B) Dizin modu — DEU ham dizinleri:
        pairs = pair_dicom_nrrd("data/raw/deu/dicom", "data/raw/deu/nrrd")
        ds = DicomSlicerDataset(pairs, lung_window=True)

    CSV beklenen sütunlar:
        img_path   : DICOM veya PNG görüntü yolu
        mask_path  : NRRD / NIfTI / PNG maske yolu  (N/A = yok)
        is_pneumo  : 0 veya 1

    Args:
        df              : Manifest DataFrame (pair_dicom_nrrd() çıktısı veya CSV)
        transform       : Albumentations Compose (image + mask birlikte)
        img_size        : Çıktı piksel boyutu (kare), örn. 512
        lung_window     : True → DICOM'lara akciğer pencereleme uygula
        force_lung      : True → DICOM tag'larını yok say, daima klinik WC/WW kullan
        align_tolerance : Boyut uyum kontrolü toleransı (piksel). 0 = tam eşitlik.
        skip_on_align_error : True → hizalama hatası olan örnekleri atla (uyar)
                              False → AlignmentError fırlat (varsayılan)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        img_size: int = 512,
        lung_window: bool = True,
        force_lung: bool = False,
        align_tolerance: int = 0,
        skip_on_align_error: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.lung_window = lung_window
        self.force_lung = force_lung
        self.align_tolerance = align_tolerance
        self.skip_on_align_error = skip_on_align_error

        # Atlanacak index'leri önceden tespit et (hizalama ön kontrolü)
        self._valid_idx = self._prevalidate()

    # ── Ön hizalama taraması ──────────────────────────────────────────────────

    def _prevalidate(self) -> list[int]:
        """
        Tüm çiftleri ön taramadan geçirir, hizalama hatalarını raporlar.
        skip_on_align_error=True ise hatalı satırları dışarıda bırakır.
        """
        if self.align_tolerance < 0:
            # Hizalama kontrolü devre dışı
            return list(range(len(self.df)))

        valid: list[int] = []
        skipped = 0

        for i, row in self.df.iterrows():
            img_path  = str(row.get("img_path",  row.get("image_path", "")))
            mask_path = str(row.get("mask_path", "N/A")).strip()

            # Maske yoksa kontrol gerekmez
            if mask_path in ("N/A", "", "nan") or not Path(mask_path).exists():
                valid.append(int(i))
                continue

            # Sadece DICOM çiftleri kontrol edilir
            if Path(img_path).suffix.lower() not in (".dcm", ".dicom"):
                valid.append(int(i))
                continue

            try:
                img_arr, _ = _read_dicom_arr(img_path, lung_window=False)
                mask_arr   = _read_mask(mask_path)
                _check_alignment(
                    img_arr, mask_arr,
                    img_path=img_path, mask_path=mask_path,
                    tolerance=self.align_tolerance,
                )
                valid.append(int(i))
            except AlignmentError as e:
                skipped += 1
                if self.skip_on_align_error:
                    log.warning("[Atlandı] %s", e)
                else:
                    raise

        if skipped:
            log.warning(
                "Ön tarama: %d / %d çift hizalama hatası nedeniyle atlandı.",
                skipped, len(self.df)
            )
        return valid

    # ── PyTorch Dataset arayüzü ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._valid_idx)

    def get_dicom_meta(self, idx: int) -> dict:
        """
        idx'e karşılık gelen DICOM metadata sözlüğü döner.
        pydicom stop_before_pixels=True ile hızlı header okuma yapar.

        Dönen anahtarlar (de-identified uyumlu, klinik anlam taşıyan):
            img_path       : Dosya yolu
            view_position  : PA / AP / LL ...
            manufacturer   : Cihaz üreticisi (Siemens, GE ...)
            kvp            : Tüp voltajı (kVp)
            pixel_spacing  : mm/piksel (tek değer, karesel piksel varsayımı)
            rows, cols     : Orijinal çözünürlük
        """
        real_idx = self._valid_idx[idx]
        row = self.df.iloc[real_idx]
        img_path = str(row.get("img_path", row.get("image_path", "")))

        meta: dict = {
            "img_path":      img_path,
            "view_position": "N/A",
            "manufacturer":  "N/A",
            "kvp":           "N/A",
            "pixel_spacing": "N/A",
            "rows":          "N/A",
            "cols":          "N/A",
        }

        if Path(img_path).suffix.lower() in (".dcm", ".dicom") and Path(img_path).exists():
            try:
                ds = pydicom.dcmread(str(img_path), stop_before_pixels=True)
                meta["view_position"] = str(getattr(ds, "ViewPosition", "N/A"))
                meta["manufacturer"]  = str(getattr(ds, "Manufacturer",  "N/A"))
                meta["kvp"]           = str(getattr(ds, "KVP",           "N/A"))
                meta["rows"]          = int(getattr(ds, "Rows",    0))
                meta["cols"]          = int(getattr(ds, "Columns", 0))
                ps = getattr(ds, "PixelSpacing", None)
                if ps:
                    meta["pixel_spacing"] = f"{float(ps[0]):.4f}"
            except Exception as exc:
                log.debug("get_dicom_meta okuma hatası (%s): %s", img_path, exc)

        return meta

    def __getitem__(self, idx: int):
        real_idx = self._valid_idx[idx]
        row = self.df.iloc[real_idx]

        img_path  = str(row.get("img_path",  row.get("image_path", "")))
        mask_path = str(row.get("mask_path", "N/A")).strip()

        # ── Görüntü yükleme ───────────────────────────────────────────────────
        suffix = Path(img_path).suffix.lower()
        if suffix in (".dcm", ".dicom"):
            img_arr, _ = _read_dicom_arr(
                img_path,
                lung_window=self.lung_window,
                force_lung=self.force_lung,
            )
        else:
            raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")
            img_arr = raw.astype(np.float32) / 255.0

        # ── Maske yükleme ─────────────────────────────────────────────────────
        has_mask = mask_path not in ("N/A", "", "nan") and Path(mask_path).exists()
        if has_mask:
            mask_arr = _read_mask(mask_path)
            # Yeniden hizalama kontrolü (prevalidate bypass edilmişse)
            if suffix in (".dcm", ".dicom") and self.align_tolerance >= 0:
                try:
                    _check_alignment(
                        img_arr, mask_arr,
                        img_path=img_path, mask_path=mask_path,
                        tolerance=self.align_tolerance,
                    )
                except AlignmentError:
                    if self.skip_on_align_error:
                        mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
                    else:
                        raise
        else:
            mask_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)

        # ── Yeniden boyutlandırma ─────────────────────────────────────────────
        img_rs  = cv2.resize(img_arr,  (self.img_size, self.img_size))
        mask_rs = cv2.resize(mask_arr, (self.img_size, self.img_size))

        img_f32  = img_rs.astype(np.float32)           # [0, 1]
        mask_f32 = (mask_rs > 0).astype(np.float32)   # 0 / 1

        label = float(mask_f32.sum() > 0)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.transform:
            augmented = self.transform(image=img_f32, mask=mask_f32)
            img_f32  = augmented["image"]
            mask_f32 = augmented["mask"]

        image_t = torch.tensor(img_f32).unsqueeze(0)    # (1, H, W)
        mask_t  = torch.tensor(mask_f32).unsqueeze(0)   # (1, H, W)
        label_t = torch.tensor(label)

        return image_t, mask_t, label_t


# ── Klinik alan hesabı ─────────────────────────────────────────────────────────

def compute_area_mm2(
    mask: np.ndarray,
    pixel_spacing_mm: float | None = None,
    dicom_path: str | None = None,
) -> float:
    """
    Binary maske üzerindeki pozitif piksel sayısını fiziksel alana (mm²) çevirir.

    Öncelik:
      1. pixel_spacing_mm  → doğrudan kullan
      2. dicom_path        → DICOM PixelSpacing tag'ından oku
      3. İkisi de yoksa   → ValueError fırlat

    Args:
        mask             : 2D binary numpy dizisi (0/1 veya 0/255)
        pixel_spacing_mm : Piksel kenar uzunluğu (mm); karesel piksel varsayılır
        dicom_path       : PixelSpacing'i okumak için DICOM dosyası

    Returns:
        Pnömotoraks alanı (mm²)

    Örnek:
        area = compute_area_mm2(mask_arr, dicom_path="patient01.dcm")
        print(f"Alan: {area:.1f} mm²  ({area/100:.2f} cm²)")
    """
    # Piksel boyutu
    if pixel_spacing_mm is None:
        if dicom_path is None:
            raise ValueError("pixel_spacing_mm veya dicom_path belirtilmeli.")
        ds  = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        ps  = getattr(ds, "PixelSpacing", None)
        if ps is None:
            raise ValueError(f"DICOM PixelSpacing tag'ı bulunamadı: {dicom_path}")
        pixel_spacing_mm = float(ps[0])

    # Piksel sayısı (0/255 veya 0/1 — her ikisini de destekle)
    binary = (mask > 0).astype(np.uint8)
    n_pixels = int(binary.sum())

    return n_pixels * (pixel_spacing_mm ** 2)
