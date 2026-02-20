"""
Medical Augmentation Pipeline — Albumentations
ElasticTransform eklendi: plevral çizgilerin doğal deformasyonunu simüle eder

TÜBİTAK 2209-A | Ahmet Demir
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),

            # ── Kontrast ve parlaklık ─────────────────────────────────────────
            # CLAHE: portatif grafi düşük kontrastını düzeltir
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),

            # ── Geometrik dönüşümler ─────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            # ElasticTransform: akciğer/plevra sınırlarının doğal
            # deformasyonunu taklit eder, küçük pnömotoraks tespitini güçlendirir
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, p=0.3
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

            # ── Gürültü ──────────────────────────────────────────────────────
            A.GaussNoise(var_limit=(10, 50), p=0.2),

            # ── Normalizasyon ─────────────────────────────────────────────────
            A.Normalize(mean=0.485, std=0.229),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.485, std=0.229),
        ],
        additional_targets={"mask": "mask"},
    )
