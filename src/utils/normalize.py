"""
Merkezi Normalizasyon Yardımcısı
=================================

Eğitim sırasında albumentations pipeline'ı şu işlemi yapar:
  PTXDataset.__getitem__  → image / 255.0          → img_01 ∈ [0, 1]
  A.Normalize(mean=0.485, std=0.229,
              max_pixel_value=255)  → (img_01 / 255 - 0.485) / 0.229

Bu formül basitleştirildiğinde:
  (img_01 - 0.485 * 255) / (0.229 * 255)
  ≈ (img_01 - 123.675) / 58.395

Tipik ortalama piksel (0.5) için sonuç ≈ -2.11.
Model BN istatistikleri bu aralığa kalibre edilmiştir.

Tüm inference yollarının (api/main.py, tta.py, predict.py, gradcam.py)
bu tek fonksiyonu kullanması zorunludur. Formülü buradan başka hiçbir
yerde tekrar yazmayın.

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

from __future__ import annotations

import numpy as np
import torch

# Eğitim sırasındaki "çift bölme" etkisinin ürettiği sabitler.
# max_pixel_value=255 olan A.Normalize'ın [0,1] girdisine uyguladığı
# (input/255 - mean) / std formülünün basitleştirilmiş karşılığıdır.
_MEAN: float = 0.485 * 255.0   # 123.675
_STD:  float = 0.229 * 255.0   # 58.395


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Görüntüyü eğitim normalizasyonuyla eşleşen aralığa taşır.

    Args:
        img: uint8 [0, 255] veya float32 [0, 1] grayscale ndarray (H, W).

    Returns:
        float32 ndarray (H, W), tipik aralık ≈ [-2.12, -2.10].
    """
    if img.dtype == np.uint8:
        img_01 = img.astype(np.float32) / 255.0
    else:
        img_01 = img.astype(np.float32)
    return (img_01 - _MEAN) / _STD


def normalize_to_tensor(
    img: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Görüntüyü normalize edip model girdisi formatına (1, 1, H, W) getirir.

    Args:
        img   : uint8 veya float32 grayscale ndarray (H, W).
        device: Hedef cihaz; None ise CPU.

    Returns:
        torch.FloatTensor (1, 1, H, W).
    """
    arr = normalize_image(img)
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    if device is not None:
        t = t.to(device)
    return t
