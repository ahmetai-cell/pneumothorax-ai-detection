#!/usr/bin/env python
"""
Main training entry point — 5-fold patient-level cross-validation.

Usage:
    python train.py \\
        --config configs/config.yaml \\
        --data_dir data/local/ptx498 \\
        --nih_dir  data/nih_negatives \\
        --output_dir output/

    # Resume interrupted training:
    python train.py --config configs/config.yaml --data_dir data/local/ptx498 --resume

    # Single fold (for testing):
    python train.py --config configs/config.yaml --data_dir data/local/ptx498 --fold 1

    # Debug (2 epochs, 40 samples):
    python train.py --config configs/config.yaml --data_dir data/local/ptx498 --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    build_manifest,
    build_folds,
    PTXDataset,
    make_weighted_sampler,
)
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import PneumothoraxModel
from src.trainer import Trainer
from src.metrics import (
    find_optimal_threshold,
    find_optimal_seg_threshold,
    compute_auc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Device selection ───────────────────────────────────────────────────────────

def get_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        logger.info("Device: CPU")
    return dev


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── Fold training ──────────────────────────────────────────────────────────────

def train_fold(
    fold:       int,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    cfg:        dict,
    device:     torch.device,
    output_dir: Path,
    resume:     bool = False,
) -> dict:

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    img_size = d_cfg["img_size"]

    train_tf = get_train_transforms(img_size)
    val_tf   = get_val_transforms(img_size)

    train_ds = PTXDataset(train_df, transform=train_tf, img_size=img_size)
    val_ds   = PTXDataset(val_df,   transform=val_tf,   img_size=img_size)

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size  = t_cfg["batch_size"],
        sampler     = sampler,
        num_workers = d_cfg.get("num_workers", 4),
        pin_memory  = d_cfg.get("pin_memory", True),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = t_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = d_cfg.get("num_workers", 4),
        pin_memory  = d_cfg.get("pin_memory", True),
    )

    model = PneumothoraxModel(
        encoder_name    = cfg["model"]["encoder"],
        pretrained      = cfg["model"]["pretrained"],
        in_channels     = cfg["model"]["in_channels"],
        deep_supervision = cfg["model"]["deep_supervision"],
    )

    params = model.count_parameters()
    logger.info(
        f"\n  Model: UNet++ / {cfg['model']['encoder']}\n"
        f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable"
    )

    trainer = Trainer(model, cfg, device, output_dir, fold=fold)
    best_metrics = trainer.fit(train_loader, val_loader, resume=resume)

    return best_metrics


# ── Threshold optimization (run after all folds trained) ──────────────────────

@torch.no_grad()
def optimize_thresholds(
    folds:      list[tuple[pd.DataFrame, pd.DataFrame]],
    cfg:        dict,
    device:     torch.device,
    output_dir: Path,
) -> tuple[float, float]:
    """
    Loads each fold's best checkpoint, runs inference on val set,
    finds optimal cls and seg thresholds over pooled val predictions.
    """
    logger.info("\n=== Threshold Optimization ===")

    from src.data.transforms import get_val_transforms

    all_cls_probs:    list[float] = []
    all_cls_labels:   list[int]   = []
    all_seg_probs:    list[np.ndarray] = []
    all_masks:        list[np.ndarray] = []

    img_size   = cfg["data"]["img_size"]
    val_tf     = get_val_transforms(img_size)
    ckpt_dir   = output_dir / cfg["checkpoint"]["dir"]
    n_folds    = cfg["training"]["n_folds"]

    for fold_i, (_, val_df) in enumerate(folds):
        fold = fold_i + 1
        ckpt_path = ckpt_dir / f"fold_{fold}_best.pth"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
            continue

        model = PneumothoraxModel(
            encoder_name     = cfg["model"]["encoder"],
            pretrained       = False,
            in_channels      = cfg["model"]["in_channels"],
            deep_supervision = False,
        )
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(device)

        val_ds = PTXDataset(val_df, transform=val_tf, img_size=img_size)
        loader = DataLoader(
            val_ds, batch_size=8, shuffle=False,
            num_workers=cfg["data"].get("num_workers", 4),
        )

        for images, masks, labels in loader:
            images = images.to(device)
            seg_pred, cls_pred = model(images)
            cls_probs = torch.sigmoid(cls_pred).squeeze(-1).cpu().numpy()
            seg_probs_ = torch.sigmoid(seg_pred).squeeze(1).cpu().numpy()
            masks_np   = masks.squeeze(1).cpu().numpy()
            labels_np  = labels.cpu().numpy().astype(int)

            for i, lbl in enumerate(labels_np):
                cp = float(cls_probs[i]) if cls_probs.ndim > 0 else float(cls_probs)
                all_cls_probs.append(cp)
                all_cls_labels.append(lbl)
                if lbl == 1:
                    all_seg_probs.append(seg_probs_[i])
                    all_masks.append(masks_np[i])

        del model

    if not all_cls_probs:
        logger.error("No predictions collected — check checkpoints.")
        return 0.5, 0.5

    cls_arr    = np.array(all_cls_probs)
    label_arr  = np.array(all_cls_labels)

    opt_cls_t, best_f1 = find_optimal_threshold(cls_arr, label_arr, metric="f1")
    logger.info(f"  Optimal cls threshold: {opt_cls_t:.3f}  (F1={best_f1:.4f})")

    opt_seg_t, best_dice = find_optimal_seg_threshold(all_seg_probs, all_masks)
    logger.info(f"  Optimal seg threshold: {opt_seg_t:.3f}  (Dice={best_dice:.4f})")

    return opt_cls_t, opt_seg_t


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTX segmentation training")
    p.add_argument("--config",     required=True,     help="Path to config.yaml")
    p.add_argument("--data_dir",   default=None,      help="PTX-498 root (SiteA/B/C); not needed if --csv given")
    p.add_argument("--nih_dir",    default=None,       help="NIH negatives directory")
    p.add_argument("--output_dir", default="output/",  help="Output directory")
    p.add_argument("--fold",       type=int, default=None,
                   help="Train only this fold (1-5); None = all folds")
    p.add_argument("--resume",     action="store_true", help="Resume from last checkpoint")
    p.add_argument("--device",     default=None,
                   help="Force device: cuda | cpu | mps")
    p.add_argument("--debug",      action="store_true",
                   help="Quick test: 40 samples, 2 epochs")
    p.add_argument("--csv",        default=None,
                   help="Pre-built manifest CSV (skips directory scan)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device     = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build manifest ────────────────────────────────────────────────────────
    if args.csv:
        logger.info(f"Loading manifest: {args.csv}")
        df = pd.read_csv(args.csv)
        for col in ("img_path", "label"):
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")
        if "mask_path" not in df.columns:
            df["mask_path"] = None
        if "patient_id" not in df.columns:
            from src.data.dataset import derive_patient_id
            df["patient_id"] = df.get("case_id", df.index.astype(str)).apply(derive_patient_id)
    else:
        if not args.data_dir:
            raise ValueError("Provide --data_dir or --csv")
        df = build_manifest(
            data_root    = args.data_dir,
            nih_root     = args.nih_dir,
            n_negatives  = 550,
            seed         = seed,
        )

    if args.debug:
        logger.info("DEBUG mode: subsetting to 40 samples, 2 epochs")
        df = df.sample(n=min(40, len(df)), random_state=seed).reset_index(drop=True)
        cfg["training"]["epochs"] = 2
        cfg["training"]["early_stopping"]["patience"] = 9999

    n_folds = cfg["training"]["n_folds"]
    folds   = build_folds(df, n_folds=n_folds, seed=seed)

    # ── Optionally train a single fold ────────────────────────────────────────
    fold_range = [args.fold] if args.fold is not None else list(range(1, n_folds + 1))

    all_results: dict[int, dict] = {}

    for fold in fold_range:
        logger.info(f"\n{'#'*60}\n  FOLD {fold}/{n_folds}\n{'#'*60}")
        train_df, val_df = folds[fold - 1]

        result = train_fold(
            fold       = fold,
            train_df   = train_df,
            val_df     = val_df,
            cfg        = cfg,
            device     = device,
            output_dir = output_dir,
            resume     = args.resume,
        )
        all_results[fold] = result
        logger.info(
            f"  Fold {fold} best: "
            f"dice={result.get('dice', float('nan')):.4f}  "
            f"auc={result.get('auc', float('nan')):.4f}  "
            f"sens={result.get('sensitivity', float('nan')):.4f}"
        )

    # ── Aggregate results ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        logger.info("\n=== Cross-Validation Summary ===")
        metric_keys = ["dice", "iou", "auc", "sensitivity", "specificity"]
        for key in metric_keys:
            vals = [v.get(key, float("nan")) for v in all_results.values()]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                logger.info(f"  {key:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Threshold optimization ────────────────────────────────────────────────
    if not args.debug:
        opt_cls_t, opt_seg_t = optimize_thresholds(folds, cfg, device, output_dir)
        cfg["thresholds"]["cls_threshold"] = round(float(opt_cls_t), 3)
        cfg["thresholds"]["seg_threshold"] = round(float(opt_seg_t), 3)
        save_config(cfg, output_dir / "config_trained.yaml")
        logger.info(f"\nTrained config saved: {output_dir / 'config_trained.yaml'}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_path = output_dir / "cv_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {str(k): {mk: round(mv, 4) if isinstance(mv, float) else mv
                      for mk, mv in v.items()}
             for k, v in all_results.items()},
            f, indent=2,
        )
    logger.info(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()
