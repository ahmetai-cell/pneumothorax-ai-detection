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
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    # Plevra ve akciğer sınırlarının doğal deformasyonu
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
    # Portatif grafi lens bozulmaları (yeni)
    A.OpticalDistortion(distort_limit=0.08, shift_limit=0.05, p=0.25),
    A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.2),
]

_INTENSITY = [
    # Kontrast artırma — düşük kaliteli grafi için kritik
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
]

_DROPOUT = [
    # ICU tüpleri, elektrodlar, vücut parçaları tarafından örtülen akciğer bölgelerini simüle eder
    # Modelin kısmi görüntülerden de pnömotoraks tanımasını sağlar
    A.CoarseDropout(
        max_holes=6,
        max_height=32,
        max_width=32,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=0,
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
            A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15,
                               rotate_limit=15, p=0.6),
            A.ElasticTransform(alpha=150, sigma=150 * 0.05, p=0.4),
            A.OpticalDistortion(distort_limit=0.12, shift_limit=0.07, p=0.35),
            A.GridDistortion(num_steps=5, distort_limit=0.12, p=0.3),

            # Yoğunluk (daha agresif CLAHE)
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.25, p=0.5),
            A.GaussNoise(var_limit=(15, 70), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),

            # Dropout (daha sık ve daha büyük)
            A.CoarseDropout(
                max_holes=8, max_height=48, max_width=48,
                min_holes=2, min_height=12, min_width=12,
                fill_value=0, p=0.4,
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
