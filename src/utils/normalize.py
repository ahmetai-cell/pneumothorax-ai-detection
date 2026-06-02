"""
Merkezi Normalizasyon Yardımcısı
=================================

Yeni eğitim (src/data/transforms.py) şunu kullanır:
  A.Normalize(mean=0.485, std=0.229, max_pixel_value=255)
  uint8 [0,255] girdi → (pixel/255 - 0.485) / 0.229
  Çıktı aralığı: [-2.12, +2.25]

Bu standart ImageNet normalizasyonudur.

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

from __future__ import annotations

import numpy as np
import torch

_MEAN: float = 0.485
_STD:  float = 0.229


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    uint8 [0,255] veya float32 [0,1] → ImageNet normalize float32.
    Eğitim pipeline'ındaki A.Normalize ile birebir aynı sonucu verir.

    Returns:
        float32 ndarray, aralık ≈ [-2.12, +2.25]
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
    normalize_image + (1, 1, H, W) tensor formatı.

    Args:
        img   : uint8 veya float32 grayscale ndarray (H, W).
        device: Hedef cihaz; None ise CPU.

    Returns:
        torch.FloatTensor (1, 1, H, W)
    """
    arr = normalize_image(img)
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    if device is not None:
        t = t.to(device)
    return t
