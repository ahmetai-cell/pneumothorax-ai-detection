"""
Albumentations Augmentation Pipeline
TÜBİTAK 2209-A | Ahmet Demir
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=0.485, std=0.229),
    ])


def get_val_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=0.485, std=0.229),
    ])
