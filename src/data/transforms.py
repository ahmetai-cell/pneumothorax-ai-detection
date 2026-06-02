"""
Albumentations v2 transforms — X-ray safe augmentation pipeline.

Design decisions:
  - No VerticalFlip: anatomically invalid for erect chest X-rays.
  - HorizontalFlip kept: bilateral symmetry is valid.
  - No extreme rotations: >15° unrealistic for standard PA/AP views.
  - CLAHE: standard for low-contrast X-ray enhancement.
  - CoarseDropout: simulates tubes/electrodes/superimposed structures.
  - ElasticTransform: simulates soft-tissue deformation (small alpha).
  - A.Normalize(mean=0.485, std=0.229, max_pixel_value=255): correct
    ImageNet normalization applied to uint8 input.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

_NORM = A.Normalize(mean=0.485, std=0.229, max_pixel_value=255)
_EXTRA = {"additional_targets": {"mask": "mask"}}


def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.07, 0.07), "y": (-0.05, 0.05)},
            scale=(0.85, 1.15),
            rotate=(-12, 12),
            shear=(-4, 4),
            p=0.6,
        ),
        A.ElasticTransform(alpha=80, sigma=8, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.10, p=0.25),
        A.OpticalDistortion(distort_limit=0.08, p=0.2),

        # Intensity — exposure/scanner variability
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2),
        A.GaussNoise(std_range=(0.03, 0.15), p=0.25),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),

        # Occlusion — tubes, electrodes, ribs superimposition
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill=0,
            p=0.35,
        ),

        _NORM,
        ToTensorV2(),
    ], **_EXTRA)


def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        _NORM,
        ToTensorV2(),
    ], **_EXTRA)


def get_inference_transforms(img_size: int = 512) -> A.Compose:
    """Identical to val; kept separate for clarity."""
    return get_val_transforms(img_size)
