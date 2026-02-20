"""
Gelişmiş Tıbbi Augmentation Pipeline
Yerel veri (1500) için agresif, açık kaynak veri (3500) için standart pipeline.

Yeni eklenenler:
  - CoarseDropout   : ICU tüpleri, elektrod artefaktlarını simüle eder
  - OpticalDistortion: Portatif grafi lens bozulmalarını simüle eder
  - RandomShadow    : Göğüs kafesi gölgelerini simüle eder

TÜBİTAK 2209-A | Ahmet Demir
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Ortak ek dönüşümler ───────────────────────────────────────────────────────

_GEOMETRIC = [
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    # Plevra ve akciğer sınırlarının doğal deformasyonu
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    # Portatif grafi lens bozulmaları
    A.OpticalDistortion(distort_limit=0.08, p=0.25),
    A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.2),
]

_INTENSITY = [
    # Kontrast artırma — düşük kaliteli grafi için kritik
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    # albumentations 2.x: std_range (normalleştirilmiş, [0,1] aralığında)
    A.GaussNoise(std_range=(0.04, 0.20), p=0.2),
]

_DROPOUT = [
    # ICU tüpleri, elektrodlar, vücut parçaları tarafından örtülen akciğer bölgelerini simüle eder
    # albumentations 2.x: num_holes_range, hole_height_range, hole_width_range
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(8, 32),
        hole_width_range=(8, 32),
        fill=0,
        p=0.3,
    ),
]

_NORMALIZE = [A.Normalize(mean=0.485, std=0.229)]
_ADDITIONAL = {"additional_targets": {"mask": "mask"}}


# ── Pipeline'lar ──────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 512) -> A.Compose:
    """Standart eğitim pipeline'ı — açık kaynak veri için."""
    return A.Compose(
        [A.Resize(img_size, img_size)]
        + _GEOMETRIC
        + _INTENSITY
        + _DROPOUT
        + _NORMALIZE,
        **_ADDITIONAL,
    )


def get_local_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Agresif augmentation — yerel 1500 vaka için.
    Daha az veriyle daha fazla çeşitlilik üretir.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),

            # Geometrik (daha geniş aralıklar)
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.07, scale=(0.85, 1.15), rotate=(-15, 15), p=0.6),
            A.ElasticTransform(alpha=150, sigma=150 * 0.05, p=0.4),
            A.OpticalDistortion(distort_limit=0.12, p=0.35),
            A.GridDistortion(num_steps=5, distort_limit=0.12, p=0.3),

            # Yoğunluk (daha agresif CLAHE)
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.25, p=0.5),
            A.GaussNoise(std_range=(0.06, 0.27), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),

            # Dropout (daha sık ve daha büyük)
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(12, 48),
                hole_width_range=(12, 48),
                fill=0,
                p=0.4,
            ),
        ]
        + _NORMALIZE,
        **_ADDITIONAL,
    )


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Doğrulama ve test pipeline'ı — augmentation yok."""
    return A.Compose(
        [A.Resize(img_size, img_size)] + _NORMALIZE,
        **_ADDITIONAL,
    )


def get_test_transforms(img_size: int = 512) -> A.Compose:
    """Test pipeline'ı — val ile aynı, isim ayrımı için."""
    return get_val_transforms(img_size)
