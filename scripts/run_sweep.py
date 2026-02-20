"""
W&B Sweep Çalıştırıcı
Her sweep agent çağrısında bu script çalışır.

Kullanım:
    1. wandb sweep configs/wandb_sweep.yaml   → SWEEP_ID alırsın
    2. wandb agent <SWEEP_ID>                 → sweep başlar

TÜBİTAK 2209-A | Ahmet Demir
"""

import os
import pandas as pd
import wandb

from src.utils.train import train_kfold
from src.preprocessing.dicom_dataset import DicomSlicerDataset
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms


def sweep_train():
    with wandb.init() as run:
        cfg = wandb.config

        # Dataset yükle
        df = pd.read_csv(os.getenv("MANIFEST_CSV", "data/hospital_manifest.csv"))
        dataset = DicomSlicerDataset(df, transform=get_train_transforms())

        config = {
            "epochs":               20,          # sweep'te kısa tut
            "batch_size":           cfg.batch_size,
            "lr":                   cfg.lr,
            "weight_decay":         1e-4,
            "num_folds":            3,            # sweep'te 3-fold yeterli
            "checkpoint_dir":       "results/checkpoints",
            "encoder_name":         cfg.encoder_name,
            "hard_negative_mining": True,
            "hnm_interval":         3,
            "hnm_threshold":        cfg.hnm_threshold,
            "hnm_multiplier":       cfg.hnm_multiplier,
            "dice_weight":          cfg.dice_weight,
            "wandb_project":        "pneumothorax-tubitak",
        }

        train_kfold(dataset, config)


if __name__ == "__main__":
    sweep_train()
