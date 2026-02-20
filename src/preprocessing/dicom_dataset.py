"""
DICOM + 3D Slicer Mask Dataset
3D Slicer'dan gelen NRRD/NIfTI segmentasyon dosyalarını DICOM görüntülerle
eşleştiren PyTorch Dataset sınıfı.

Desteklenen maske formatları:
  - NRRD  (.nrrd)            → 3D Slicer'ın yerli formatı
  - NIfTI (.nii, .nii.gz)    → araştırma standartı
  - PNG   (.png)             → manuel export / green_mask_extractor çıktısı

Kullanım:
    df = pd.read_csv("data/hospital_manifest.csv")
    ds = DicomSlicerDataset(df, dicom_dir="data/raw/dicom",
                            mask_dir="data/raw/masks")

TÜBİTAK 2209-A | Ahmet Demir
"""

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import nrrd
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset


# ── Yardımcı okuyucular ───────────────────────────────────────────────────────

def _read_dicom(path: str) -> np.ndarray:
    """DICOM → uint8 grayscale ndarray."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[0]  # çok dilimli DICOM'da ilk dilimi al
    return arr


def _read_nrrd(path: str) -> np.ndarray:
    """3D Slicer NRRD segmentasyon → binary uint8 mask."""
    data, _ = nrrd.read(path)
    arr = data.astype(np.float32)
    if arr.ndim == 3:
        # Çoğu 3D Slicer NRRD'si (1, H, W) veya (H, W, 1) şeklinde gelir
        arr = arr.squeeze()
    return (arr > 0).astype(np.uint8) * 255


def _read_nifti(path: str) -> np.ndarray:
    """NIfTI segmentasyon → binary uint8 mask."""
    img = nib.load(path)
    arr = np.asarray(img.dataobj).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # ilk slice
    # NIfTI eksenleri: NIfTI (H, W) iken görüntü (W, H) olabilir
    arr = np.rot90(arr, k=1)
    return (arr > 0).astype(np.uint8) * 255


def _read_mask(path: str) -> np.ndarray:
    """Uzantıya göre doğru okuyucuyu seçer."""
    suffix = Path(path).suffix.lower()
    if suffix == ".nrrd":
        return _read_nrrd(path)
    elif suffix in (".nii", ".gz"):
        return _read_nifti(path)
    else:
        # PNG / BMP — green_mask_extractor veya 3D Slicer PNG export
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Maske okunamadı: {path}")
        return (mask > 0).astype(np.uint8) * 255


# ── Dataset sınıfı ────────────────────────────────────────────────────────────

class DicomSlicerDataset(Dataset):
    """
    DICOM görüntüsü + 3D Slicer maskesi çiftlerinden oluşan dataset.

    CSV manifest beklenen sütunlar:
        image_id        : dosya adı (uzantısız)
        image_path      : DICOM veya PNG görüntü yolu
        mask_path       : NRRD / NIfTI / PNG maske yolu
        has_pneumothorax: 0 veya 1

    Args:
        df          : Manifest DataFrame
        transform   : Albumentations Compose (image + mask birlikte)
        img_size    : Çıktı piksel boyutu (kare)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        img_size: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path  = row["image_path"]
        mask_path = row.get("mask_path", None)

        # ── Görüntü ──────────────────────────────────────────────────────────
        suffix = Path(img_path).suffix.lower()
        if suffix in (".dcm", ".dicom"):
            image = _read_dicom(img_path)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0

        # ── Maske ────────────────────────────────────────────────────────────
        if mask_path and Path(mask_path).exists():
            mask = _read_mask(str(mask_path))
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 0).astype(np.float32)

        label = float(mask.sum() > 0)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        image = torch.tensor(image).unsqueeze(0)   # (1, H, W)
        mask  = torch.tensor(mask).unsqueeze(0)    # (1, H, W)
        label = torch.tensor(label)

        return image, mask, label
