"""
PTXDataset — Çoklu Format Segmentasyon Dataset
===============================================

498 pnömotoraks (pozitif) + N NIH negatif vaka için PyTorch Dataset
ve site-stratified K-Fold splitter.

Desteklenen formatlar:
  - Görüntü : PNG (.png), DICOM (.dcm / .dicom)
  - Etiket  : PNG (.png), NRRD (.nrrd), NIfTI (.nii / .nii.gz)
  Format dosya uzantısından otomatik algılanır.

Veri yapısı (PTX-498-v2-fix):
    {data_root}/
        SiteA/  {id}.1.img.png   {id}.2.mask.png
        SiteB/  ...
        SiteC/  ...

DICOM + NRRD veri seti:
    {dcm_root}/
        {site}/{case_id}.dcm       (veya herhangi bir uzantı)
        {site}/{case_id}.nrrd      (segmentasyon maskesi)

Negatif vakalar (NIH ChestX-ray14):
    {nih_root}/images_*/images/*.png   — Data_Entry_2017.csv ile filtrelenir

Tasarım kararları:
  - PTXDataset (image, mask, label) 3-tuple döndürür.
    label=1: pnömotoraks  |  label=0: normal
  - Negatif vakalar için mask_path=None → sıfır maske döner.
  - DICOM: pydicom ile okunur, HU pencere normalizasyonu uygulanır.
  - NRRD: pynrrd ile okunur, ilk kanal/dilim alınır.
  - Maskeler her formattan float32 0/1'e dönüştürülür.

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

import albumentations as A


# ── Sabitler ──────────────────────────────────────────────────────────────────

SITE_NAMES    : tuple[str, ...] = ("SiteA", "SiteB", "SiteC")
IMG_SUFFIX    : str             = ".1.img.png"
MASK_SUFFIX   : str             = ".2.mask.png"
MASK_THRESHOLD: int             = 127   # uint8 → binary: piksel > 127 → 1.0


# ── Format-agnostik okuma ─────────────────────────────────────────────────────

_DICOM_EXT  = {".dcm", ".dicom"}
_NRRD_EXT   = {".nrrd"}
_NIFTI_EXT  = {".nii", ".gz"}   # .nii.gz → suffix = .gz
_PNG_EXT    = {".png", ".jpg", ".jpeg"}


def _read_image(path: str) -> np.ndarray:
    """
    Görüntü dosyasını uint8 grayscale ndarray olarak okur.
    Desteklenen formatlar: PNG/JPEG, DICOM.

    Returns:
        uint8 ndarray (H, W)
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _DICOM_EXT:
        import pydicom
        ds  = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype(np.float32)
        # HU pencere normalizasyonu (Rescale Slope/Intercept varsa)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            arr = np.clip(arr, -1500.0, 500.0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr

    # PNG / JPEG
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {p}")
    return img


def _read_mask(path: str) -> np.ndarray:
    """
    Maske dosyasını float32 binary (0/1) ndarray olarak okur.
    Desteklenen formatlar: PNG/JPEG, NRRD, NIfTI (.nii / .nii.gz).

    Returns:
        float32 ndarray (H, W)
    """
    p = Path(path)
    name = p.name.lower()
    ext  = p.suffix.lower()

    # NRRD
    if ext in _NRRD_EXT:
        import nrrd
        data, _ = nrrd.read(str(p))
        # (H, W) veya (H, W, D) veya (D, H, W)
        if data.ndim == 3:
            # En büyük dilimi al (pnömotoraks genellikle tek dilim)
            data = data[..., 0] if data.shape[-1] < data.shape[0] else data[0]
        arr = (data > 0).astype(np.float32)
        return arr

    # NIfTI (.nii veya .nii.gz)
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import nibabel as nib
            vol = nib.load(str(p)).get_fdata()
        except ImportError:
            import SimpleITK as sitk
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
        if vol.ndim == 3:
            # Maksimum projeksiyon: tüm dilimlerde pnömotoraks varsa birleştir
            vol = vol.max(axis=-1) if vol.shape[-1] <= vol.shape[0] else vol[0]
        return (vol > 0).astype(np.float32)

    # PNG / JPEG
    mask_raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Maske okunamadı: {p}")
    return (mask_raw > MASK_THRESHOLD).astype(np.float32)


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_ptx_manifest(data_root: Path | str) -> pd.DataFrame:
    """
    PTX-498 dizin yapısını tarar; (case_id, site, img_path, mask_path, label) DataFrame döndürür.

    Args:
        data_root: SiteA/, SiteB/, SiteC/ alt klasörlerini içeren kök dizin.

    Returns:
        Sütunlar: case_id (str), site (str), img_path (str), mask_path (str), label (int=1).

    Raises:
        FileNotFoundError: Hiç img+mask çifti bulunamazsa.
    """
    data_root = Path(data_root)
    records: list[dict] = []

    for site in SITE_NAMES:
        site_dir = data_root / site
        if not site_dir.exists():
            continue
        for img_path in sorted(site_dir.glob(f"*{IMG_SUFFIX}")):
            case_id   = img_path.name[: -len(IMG_SUFFIX)]
            mask_path = site_dir / f"{case_id}{MASK_SUFFIX}"
            if not mask_path.exists():
                continue
            records.append({
                "case_id":   case_id,
                "site":      site,
                "img_path":  str(img_path),
                "mask_path": str(mask_path),
                "label":     1,
            })

    if not records:
        raise FileNotFoundError(
            f"PTX vakası bulunamadı: {data_root}\n"
            f"Beklenen yapı: {data_root}/SiteA/{{id}}{IMG_SUFFIX}"
        )

    df = pd.DataFrame(records)
    site_counts = "  ".join(
        f"{s}={len(df[df['site'] == s])}"
        for s in SITE_NAMES
        if s in df["site"].values
    )
    print(f"  PTX manifest: {len(df)} pozitif vaka  [{site_counts}]")
    return df


def build_dicom_nrrd_manifest(
    data_root: Path | str,
    img_glob:  str = "**/*.dcm",
    mask_glob: str = "**/*.nrrd",
    label:     int = 1,
    site_name: str = "External",
) -> pd.DataFrame:
    """
    DICOM görüntü + NRRD etiket çiftlerinden manifest üretir.

    Eşleştirme kuralı: aynı klasördeki aynı stem (uzantısız ad) çiftleri.
    Örnek:
        SiteA/001.dcm  ↔  SiteA/001.nrrd

    Args:
        data_root : DICOM ve NRRD dosyalarının bulunduğu kök dizin.
        img_glob  : Görüntü dosyası glob deseni (varsayılan "**/*.dcm").
        mask_glob : Maske dosyası glob deseni (varsayılan "**/*.nrrd").
        label     : 1 = pnömotoraks, 0 = normal.
        site_name : DataFrame'deki site etiketi.

    Returns:
        Sütunlar: case_id, site, img_path, mask_path (None: negatif), label.
    """
    data_root = Path(data_root)

    # Maskeleri stem → path sözlüğüne al
    mask_by_stem: dict[str, Path] = {}
    for mp in data_root.glob(mask_glob):
        # Çift uzantıyı temizle: foo.nrrd → foo, foo.nii.gz → foo
        stem = mp.name
        for ext in (".nrrd", ".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        mask_by_stem[stem] = mp

    records: list[dict] = []
    for ip in sorted(data_root.glob(img_glob)):
        stem = ip.stem   # .dcm → stem
        mask_path = mask_by_stem.get(stem)
        if label == 1 and mask_path is None:
            continue     # pozitif vaka için maske zorunlu
        records.append({
            "case_id":   stem,
            "site":      site_name,
            "img_path":  str(ip),
            "mask_path": str(mask_path) if mask_path else None,
            "label":     label,
        })

    if not records:
        raise FileNotFoundError(
            f"DICOM/NRRD çifti bulunamadı: {data_root}\n"
            f"  img_glob={img_glob!r}  mask_glob={mask_glob!r}"
        )

    df = pd.DataFrame(records)
    print(
        f"  DICOM+NRRD manifest ({site_name}): {len(df)} vaka  "
        f"[label={'pozitif' if label==1 else 'negatif'}]"
    )
    return df


def build_nih_negatives(
    nih_root: Path | str,
    n: int = 550,
    seed: int = 42,
) -> pd.DataFrame:
    """
    NIH ChestX-ray14'ten 'No Finding' vakalarını seçer.

    Args:
        nih_root: Data_Entry_2017.csv ve images_*/images/ klasörlerini içeren dizin.
        n       : Seçilecek negatif vaka sayısı (varsayılan 550).
        seed    : Tekrarlanabilirlik.

    Returns:
        Sütunlar: case_id, site='NIH', img_path, mask_path=None, label=0.

    Raises:
        FileNotFoundError: CSV veya görüntüler bulunamazsa.
    """
    nih_root = Path(nih_root)
    csv_path = nih_root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data_Entry_2017.csv bulunamadı: {csv_path}")

    # 'No Finding' filtresi
    df_csv = pd.read_csv(csv_path)
    no_finding = df_csv[df_csv["Finding Labels"] == "No Finding"].copy()

    # Mevcut PNG dosyalarını bul (images_*/images/ alt dizini)
    available: dict[str, str] = {}
    for img_dir in nih_root.glob("images_*/images"):
        for f in img_dir.glob("*.png"):
            available[f.name] = str(f)

    if not available:
        raise FileNotFoundError(
            f"NIH PNG bulunamadı: {nih_root}/images_*/images/\n"
            "Colab'da nih_root'u Drive'daki NIH klasörüne işaret etmelisiniz."
        )

    no_finding = no_finding[no_finding["Image Index"].isin(available)].copy()
    if len(no_finding) == 0:
        raise ValueError("Eşleşen No Finding vakası yok — CSV ve PNG dizini kontrol edin.")

    sampled = no_finding.sample(n=min(n, len(no_finding)), random_state=seed)

    records = []
    for _, row in sampled.iterrows():
        records.append({
            "case_id":   row["Image Index"],
            "site":      "NIH",
            "img_path":  available[row["Image Index"]],
            "mask_path": None,
            "label":     0,
        })

    df_neg = pd.DataFrame(records).reset_index(drop=True)
    print(f"  NIH negatif: {len(df_neg)} vaka  (No Finding @ seed={seed})")
    return df_neg


# ── Dataset ───────────────────────────────────────────────────────────────────

class PTXDataset(Dataset):
    """
    PTX-498 pozitif + NIH negatif vakalar için PyTorch Dataset.

    Returns:
        image : float32 Tensor [1, H, W]
        mask  : float32 Tensor [1, H, W] — pozitifler için binary [0,1],
                negatifler için sıfır maske
        label : float32 Tensor [] — 1.0 (pnömotoraks) veya 0.0 (normal)

    Args:
        df       : build_ptx_manifest / build_nih_negatives / birleşik DataFrame.
                   Zorunlu sütunlar: img_path, mask_path (None olabilir), label.
        transform: albumentations Compose (opsiyonel).
        img_size : Resize hedefi (transform verilmezse).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose | None = None,
        img_size: int = 512,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.img_size  = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        label = int(row.get("label", 1))

        # ── Görüntü (PNG veya DICOM) ──────────────────────────────────────────
        image = _read_image(str(row["img_path"])).astype(np.float32) / 255.0

        # ── Maske (PNG, NRRD veya NIfTI) ──────────────────────────────────────
        mask_path = row.get("mask_path")
        if mask_path is not None and label == 1:
            mask = _read_mask(str(mask_path))
            # Boyut uyumsuzluğunu gider (NRRD/NIfTI farklı rezolüsyonda olabilir)
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            # Negatif vaka: sıfır maske
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # ── Augmentation ───────────────────────────────────────────────────────
        if self.transform is not None:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask  = cv2.resize(
                mask, (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # ── [H, W] → [1, H, W] Tensor ────────────────────────────────────────
        image_t = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask_t  = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        label_t = torch.tensor(label, dtype=torch.float32)

        return image_t, mask_t, label_t

    def get_site(self, idx: int) -> str:
        """İndex'e göre site adını döndürür (per-site metrik hesabı için)."""
        return str(self.df.iloc[idx]["site"])


# ── Site-stratified K-Fold (PTX pozitifler) ───────────────────────────────────

def make_site_stratified_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Her sitenin vaka sayısını orantılı biçimde koruyarak K-Fold üretir.
    Sadece PTX pozitif vakalar için (label=1).

    Returns:
        n_folds uzunluğunda liste; her eleman (train_df, val_df).
    """
    required_cols = {"site", "img_path", "mask_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame'de eksik sütunlar: {missing}")

    site_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for site in SITE_NAMES:
        site_df = df[df["site"] == site]
        if len(site_df) == 0:
            continue
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        site_splits[site] = list(kf.split(site_df))

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold_i in range(n_folds):
        train_parts: list[pd.DataFrame] = []
        val_parts:   list[pd.DataFrame] = []

        for site, splits in site_splits.items():
            site_df = df[df["site"] == site].reset_index(drop=True)
            train_idx, val_idx = splits[fold_i]
            train_parts.append(site_df.iloc[train_idx])
            val_parts.append(site_df.iloc[val_idx])

        train_df = (
            pd.concat(train_parts, ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        val_df = (
            pd.concat(val_parts, ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        val_sites = val_df["site"].value_counts().to_dict()
        print(
            f"  Fold {fold_i + 1}: train={len(train_df):>3}  val={len(val_df):>3}  "
            + "  ".join(
                f"val_{s}={val_sites.get(s, 0)}"
                for s in SITE_NAMES
            )
        )

        folds.append((train_df, val_df))

    return folds


# ── Combined K-Fold (pozitif + negatif) ───────────────────────────────────────

def build_combined_folds(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Pozitif vakalar için site-stratified, negatif vakalar için random K-Fold
    üretir; her fold'da ikisini birleştirir.

    Args:
        pos_df : build_ptx_manifest() çıktısı (label=1, site=SiteA/B/C).
        neg_df : build_nih_negatives() çıktısı (label=0, site=NIH).
        n_folds: Fold sayısı.
        seed   : Tekrarlanabilirlik.

    Returns:
        n_folds uzunluğunda liste; her eleman (train_df, val_df).
        train_df/val_df hem pozitif hem negatif içerir, karıştırılmış.
    """
    print("\n  [Site-stratified pozitif fold'lar]")
    pos_folds = make_site_stratified_folds(pos_df, n_folds, seed)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    neg_splits = list(kf.split(neg_df))
    neg_df = neg_df.reset_index(drop=True)

    combined: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    print()

    for i, (pos_train, pos_val) in enumerate(pos_folds):
        neg_train_idx, neg_val_idx = neg_splits[i]
        neg_train = neg_df.iloc[neg_train_idx].reset_index(drop=True)
        neg_val   = neg_df.iloc[neg_val_idx].reset_index(drop=True)

        train_df = (
            pd.concat([pos_train, neg_train], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        val_df = (
            pd.concat([pos_val, neg_val], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        print(
            f"  Combined Fold {i+1}: "
            f"train={len(train_df)} (pos={len(pos_train)}, neg={len(neg_train)})  "
            f"val={len(val_df)} (pos={len(pos_val)}, neg={len(neg_val)})"
        )
        combined.append((train_df, val_df))

    return combined
