"""
PTXDataset — Yerel PNG Segmentasyon Dataset
============================================

498 pnömotoraks vakası için PyTorch Dataset ve site-stratified K-Fold splitter.

Veri yapısı (PTX-498-v2-fix):
    {data_root}/
        SiteA/  {id}.1.img.png   {id}.2.mask.png
        SiteB/  ...
        SiteC/  ...

Tasarım kararları:
  - Sınıflandırma başlığı KAPALI: tüm vakalar pozitif, label döndürülmez.
    Dönen tuple: (image, mask) — sadece segmentasyon.
  - Maskeler uint8 0/255 → float32 0/1 olarak normalleştirilir (A-3 bug fix).
  - Site bilgisi DataFrame'de 'site' sütunu olarak tutulur;
    per-site metrik hesabı için get_site(idx) metodu mevcuttur.
  - make_site_stratified_folds(): her site kendi içinde KFold ile bölünür,
    fold başına orantılı site dağılımı garanti edilir.

TÜBİTAK 2209-A | Ahmet Demir
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


# ── Manifest builder ──────────────────────────────────────────────────────────

def build_ptx_manifest(data_root: Path | str) -> pd.DataFrame:
    """
    PTX-498 dizin yapısını tarar; (case_id, site, img_path, mask_path) DataFrame döndürür.

    Args:
        data_root: SiteA/, SiteB/, SiteC/ alt klasörlerini içeren kök dizin.

    Returns:
        Sütunlar: case_id (str), site (str), img_path (str), mask_path (str).

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
                continue  # eşsiz dosyaları atla
            records.append({
                "case_id":   case_id,
                "site":      site,
                "img_path":  str(img_path),
                "mask_path": str(mask_path),
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
    print(f"  PTX manifest: {len(df)} vaka  [{site_counts}]")
    return df


# ── Dataset ───────────────────────────────────────────────────────────────────

class PTXDataset(Dataset):
    """
    498 yerel PNG vakası için PyTorch Dataset.

    Returns:
        image : float32 Tensor [1, H, W] — ImageNet mean/std ile normalize edilmiş
        mask  : float32 Tensor [1, H, W] — binary [0.0, 1.0]

    Sınıflandırma başlığı devre dışı; label döndürülmez.
    Augmentation beklentisi: albumentations >=2.0, additional_targets={"mask":"mask"}.
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Görüntü: uint8 [0,255] → float32 [0,1] ───────────────────────────
        image = cv2.imread(str(row["img_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {row['img_path']}")
        image = image.astype(np.float32) / 255.0

        # ── Maske: uint8 0/255 → float32 0/1  (A-3 bug fix) ──────────────────
        mask_raw = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Maske okunamadı: {row['mask_path']}")
        mask = (mask_raw > MASK_THRESHOLD).astype(np.float32)

        # ── Augmentation ───────────────────────────────────────────────────────
        if self.transform is not None:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]   # float32 [H, W] — A.Normalize uygulandı
            mask  = aug["mask"]    # float32 [H, W] — Normalize maskeye uygulanmaz
        else:
            # Transform verilmezse manuel resize (inference veya test için)
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask  = cv2.resize(
                mask, (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,  # maske interpolasyonu: NN
            )

        # ── [H, W] → [1, H, W] Tensor ────────────────────────────────────────
        # ascontiguousarray: albumentations bazen non-contiguous array döndürür
        image_t = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask_t  = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)

        return image_t, mask_t

    def get_site(self, idx: int) -> str:
        """İndex'e göre site adını döndürür (per-site metrik hesabı için)."""
        return str(self.df.iloc[idx]["site"])


# ── Site-stratified K-Fold ────────────────────────────────────────────────────

def make_site_stratified_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Her sitenin vaka sayısını orantılı biçimde koruyarak K-Fold üretir.

    Strateji:
      1. Her site kendi içinde KFold ile bölünür.
      2. fold_i için: her sitenin val kısmı birleştirilir → o fold'un val seti.
      3. Geri kalanlar train seti.

    Bu yöntem SiteC (72 vaka) gibi küçük sitelerin fold'lardan düşmesini engeller.
    Her fold'un val setinde ~50 SiteA, ~36 SiteB, ~14 SiteC vakası beklenir.

    Args:
        df     : build_ptx_manifest() çıktısı; 'site' sütunu zorunlu.
        n_folds: Fold sayısı (varsayılan 5).
        seed   : Tekrarlanabilirlik (varsayılan 42).

    Returns:
        n_folds uzunluğunda liste; her eleman (train_df, val_df).
    """
    required_cols = {"site", "img_path", "mask_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame'de eksik sütunlar: {missing}")

    # Her site için KFold index'lerini hesapla
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
