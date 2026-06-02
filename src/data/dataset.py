"""
PTX Dataset — multi-format, patient-level split support
Supports: PNG/JPG images, optional DICOM, NRRD/NIfTI masks
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold


# ── Constants ──────────────────────────────────────────────────────────────────
SITE_NAMES  = ("SiteA", "SiteB", "SiteC")
IMG_SUFFIX  = ".1.img.png"
MASK_SUFFIX = ".2.mask.png"


# ── Low-level readers ──────────────────────────────────────────────────────────

def read_image(path: str) -> np.ndarray:
    """
    Returns uint8 grayscale ndarray (H, W).
    Supports PNG/JPG and DICOM.
    Raises FileNotFoundError or RuntimeError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    if p.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom
            ds  = pydicom.dcmread(str(p))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                arr = np.clip(arr, -1500.0, 500.0)
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr
        except Exception as exc:
            raise RuntimeError(f"DICOM read failed [{p}]: {exc}") from exc

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2.imread failed (corrupt or unsupported): {p}")
    return img


def read_mask(path: Optional[str]) -> np.ndarray:
    """
    Returns float32 binary ndarray (H, W) with values 0.0 or 1.0.
    Returns zeros array (1, 1) placeholder when path is None.
    Supports PNG/JPG and NRRD.
    """
    if path is None:
        return np.zeros((1, 1), dtype=np.float32)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mask not found: {p}")

    if p.suffix.lower() == ".nrrd":
        try:
            import nrrd
            data, _ = nrrd.read(str(p))
            if data.ndim == 3:
                data = data[..., 0] if data.shape[-1] < data.shape[0] else data[0]
            return (data > 0).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"NRRD read failed [{p}]: {exc}") from exc

    if p.name.lower().endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
            vol = nib.load(str(p)).get_fdata()
            if vol.ndim == 3:
                vol = vol.max(axis=-1) if vol.shape[-1] <= vol.shape[0] else vol[0]
            return (vol > 0).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"NIfTI read failed [{p}]: {exc}") from exc

    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Mask read failed (corrupt): {p}")
    return (mask > 127).astype(np.float32)


# ── Patient ID derivation ──────────────────────────────────────────────────────

def derive_patient_id(case_id: str) -> str:
    """
    Strips trailing numeric suffix from case_id to approximate patient grouping.
    Examples:
        "PTX_0042_001" → "PTX_0042"
        "case042"      → "case042"  (no trailing digits after underscore)
        "00000038_002" → "00000038" (NIH style)
    """
    match = re.match(r"^(.+?)(?:_\d+)?$", case_id)
    return match.group(1) if match else case_id


# ── Manifest builders ──────────────────────────────────────────────────────────

def build_ptx_manifest(data_root: str | Path) -> pd.DataFrame:
    """
    Scans PTX-498 directory structure (SiteA/B/C with .1.img.png + .2.mask.png pairs).
    Returns DataFrame: case_id, site, img_path, mask_path, label, patient_id
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
                "case_id":    case_id,
                "site":       site,
                "img_path":   str(img_path),
                "mask_path":  str(mask_path),
                "label":      1,
                "patient_id": derive_patient_id(case_id),
            })

    if not records:
        raise FileNotFoundError(
            f"No PTX cases found in {data_root}. "
            f"Expected: {data_root}/SiteA/{{id}}{IMG_SUFFIX}"
        )

    df = pd.DataFrame(records)
    counts = " ".join(f"{s}={len(df[df.site==s])}" for s in SITE_NAMES if s in df.site.values)
    print(f"  PTX manifest: {len(df)} positives  [{counts}]")
    return df


def build_nih_manifest(nih_root: str | Path, n: int = 550, seed: int = 42) -> pd.DataFrame:
    """
    Selects N 'No Finding' cases from NIH ChestX-ray14.
    Expects: {nih_root}/Data_Entry_2017.csv and flat PNG folder.
    """
    nih_root = Path(nih_root)

    # Support flat folder (just PNGs, no CSV)
    png_files = sorted(nih_root.glob("*.png"))
    if png_files and not (nih_root / "Data_Entry_2017.csv").exists():
        rng = np.random.default_rng(seed)
        selected = rng.choice(png_files, size=min(n, len(png_files)), replace=False)
        records = [{
            "case_id":    f.stem,
            "site":       "NIH",
            "img_path":   str(f),
            "mask_path":  None,
            "label":      0,
            "patient_id": derive_patient_id(f.stem),
        } for f in selected]
        df = pd.DataFrame(records).reset_index(drop=True)
        print(f"  NIH manifest (flat): {len(df)} negatives")
        return df

    csv_path = nih_root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data_Entry_2017.csv not found: {csv_path}")

    df_csv = pd.read_csv(csv_path)
    no_finding = df_csv[df_csv["Finding Labels"] == "No Finding"].copy()

    available: dict[str, str] = {}
    for img_dir in nih_root.glob("images_*/images"):
        for f in img_dir.glob("*.png"):
            available[f.name] = str(f)

    no_finding = no_finding[no_finding["Image Index"].isin(available)]
    if len(no_finding) == 0:
        raise ValueError("No matching No Finding cases found.")

    sampled = no_finding.sample(n=min(n, len(no_finding)), random_state=seed)
    records = [{
        "case_id":    row["Image Index"],
        "site":       "NIH",
        "img_path":   available[row["Image Index"]],
        "mask_path":  None,
        "label":      0,
        "patient_id": derive_patient_id(row["Image Index"]),
    } for _, row in sampled.iterrows()]

    df = pd.DataFrame(records).reset_index(drop=True)
    print(f"  NIH manifest: {len(df)} negatives")
    return df


def build_manifest(
    data_root: str | Path,
    nih_root:  Optional[str | Path] = None,
    n_negatives: int = 550,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds combined PTX + NIH manifest.
    Falls back to positives-only if nih_root not given.
    """
    pos_df = build_ptx_manifest(data_root)

    if nih_root is not None:
        neg_df = build_nih_manifest(nih_root, n=n_negatives, seed=seed)
        df = pd.concat([pos_df, neg_df], ignore_index=True)
    else:
        print("  Warning: no nih_root given, training on positives only.")
        df = pos_df

    print(f"  Total: {len(df)}  pos={df.label.sum()}  neg={(df.label==0).sum()}")
    return df


# ── Fold splitter ──────────────────────────────────────────────────────────────

def build_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Patient-level StratifiedGroupKFold split.
    Stratifies by label, groups by patient_id — same patient never in both splits.
    Returns list of (train_df, val_df) tuples.
    """
    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = df["case_id"].apply(derive_patient_id)

    sgkf   = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds  = []
    indices = np.arange(len(df))

    for fold_i, (train_idx, val_idx) in enumerate(
        sgkf.split(indices, y=df["label"].values, groups=df["patient_id"].values)
    ):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        n_pos_t = int(train_df.label.sum())
        n_pos_v = int(val_df.label.sum())
        print(
            f"  Fold {fold_i+1}: "
            f"train={len(train_df)} (pos={n_pos_t} neg={len(train_df)-n_pos_t})  "
            f"val={len(val_df)} (pos={n_pos_v} neg={len(val_df)-n_pos_v})"
        )
        folds.append((train_df, val_df))

    return folds


# ── Dataset ────────────────────────────────────────────────────────────────────

class PTXDataset(Dataset):
    """
    Pneumothorax segmentation dataset.

    Returns: (image_tensor, mask_tensor, label_tensor)
        image_tensor : float32 (1, H, W) — normalized
        mask_tensor  : float32 (1, H, W) — binary 0/1
        label_tensor : float32 scalar   — 1.0 positive, 0.0 negative
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        img_size: int = 512,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.img_size  = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row.get("label", 1))

        try:
            image = read_image(str(row["img_path"]))
        except Exception as exc:
            raise RuntimeError(f"[idx={idx}] Image load failed: {exc}") from exc

        mask_path = row.get("mask_path")
        try:
            mask = read_mask(mask_path if (label == 1 and mask_path is not None) else None)
        except Exception as exc:
            raise RuntimeError(f"[idx={idx}] Mask load failed: {exc}") from exc

        # Resize mask to match image if needed (NRRD resolution mismatch)
        if mask.shape != image.shape:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.transform is not None:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]   # float32 normalized by A.Normalize
            mask  = aug["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask  = cv2.resize(
                mask, (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            image = image.astype(np.float32) / 255.0

        # Handle ToTensorV2: if transform includes it, image is already a tensor
        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(np.ascontiguousarray(image))
            if image_t.ndim == 2:
                image_t = image_t.unsqueeze(0)
        else:
            image_t = image
            if image_t.ndim == 2:
                image_t = image_t.unsqueeze(0)

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
        else:
            mask_t = mask.float()
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)

        # Negative case: zero mask regardless
        if label == 0:
            mask_t = torch.zeros_like(mask_t)

        label_t = torch.tensor(float(label), dtype=torch.float32)
        return image_t, mask_t, label_t

    def get_labels(self) -> list[int]:
        return self.df["label"].tolist()


# ── Weighted sampler helper ────────────────────────────────────────────────────

def make_weighted_sampler(dataset: PTXDataset):
    from torch.utils.data import WeightedRandomSampler
    labels  = dataset.get_labels()
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    n_total = len(labels)
    w_pos   = n_total / (2 * max(n_pos, 1))
    w_neg   = n_total / (2 * max(n_neg, 1))
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )
