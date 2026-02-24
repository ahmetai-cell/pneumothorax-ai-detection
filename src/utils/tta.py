"""
Test-Time Augmentation (TTA)
Inference aşamasında aynı görüntüyü N farklı transformasyonla analiz eder,
tahminlerin ortalamasını alarak daha güvenilir sonuç üretir.

Özellikle şu durumlarda kritik:
  - Apikal küçük pnömotorakslar (tek açıdan gözden kaçabilir)
  - Sınırda olasılıklar (0.45–0.55 arası belirsiz vakalar)
  - Portatif grafi kalite değişkenlikleri

Augmentation seti (5 varyant):
  1. Orijinal (flip yok, değişiklik yok)
  2. Yatay flip
  3. Hafif parlaklık artışı  (+0.1)
  4. Hafif parlaklık azalması (−0.1)
  5. Hafif kontrast artışı

FastAPI ve Streamlit ile entegre çalışır.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── TTA dönüşüm tanımları ─────────────────────────────────────────────────────

def _build_tta_variants(gray: np.ndarray, img_size: int = 512) -> list[np.ndarray]:
    """
    Tek bir gri görüntüden 5 TTA varyantı oluşturur.

    Returns:
        variants : Normalize edilmiş float32 görüntü listesi, her biri (img_size, img_size)
    """
    resized = cv2.resize(gray, (img_size, img_size)).astype(np.float32) / 255.0

    def clip(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    variants = [
        resized,                                    # 1. Orijinal
        np.fliplr(resized).copy(),                  # 2. Yatay flip
        clip(resized + 0.10),                       # 3. Parlaklık +
        clip(resized - 0.10),                       # 4. Parlaklık -
        clip((resized - 0.5) * 1.15 + 0.5),        # 5. Kontrast +
    ]
    return variants


def _variants_to_batch(variants: list[np.ndarray]) -> torch.Tensor:
    """
    (N, H, W) float32 listesi → (N, 1, H, W) tensor

    Returns:
        batch : torch.FloatTensor (N, 1, H, W)
    """
    stack = np.stack(variants, axis=0)               # (N, H, W)
    return torch.tensor(stack).unsqueeze(1).float()  # (N, 1, H, W)


# ── TTA inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    gray_image: np.ndarray,
    img_size: int = 512,
    seg_threshold: float = 0.5,
) -> dict:
    """
    TTA ile tek görüntü üzerinde inference yapar.

    Args:
        model         : PneumothoraxModel (eval mode)
        gray_image    : uint8 grayscale X-ray (herhangi boyut)
        img_size      : Model giriş boyutu
        seg_threshold : Segmentasyon binary eşiği

    Returns:
        dict ile:
          prob_mean    : 5 tahmin ortalaması (güvenilir olasılık)
          prob_std     : 5 tahmin standart sapması (belirsizlik göstergesi)
          prob_votes   : Her varyantın tahmini
          seg_mean     : Ortalama segmentasyon maskesi (float, 0-1)
          seg_binary   : Binary maske (uint8, 0/255), orijinal boyuta yeniden ölçekli
          is_uncertain : Std > 0.15 ise tahmin belirsiz kabul edilir
    """
    device = next(model.parameters()).device
    model.eval()

    variants = _build_tta_variants(gray_image, img_size)
    batch    = _variants_to_batch(variants).to(device)  # (5, 1, H, W)

    seg_preds, cls_preds = model(batch)

    probs     = torch.sigmoid(cls_preds).squeeze(-1).cpu().numpy()   # (5,)
    seg_probs = torch.sigmoid(seg_preds).cpu().numpy()               # (5, 1, H, W)

    # Flip varyantı için segmentasyon maskesini geri çevir
    seg_probs[1, 0] = np.fliplr(seg_probs[1, 0])

    prob_mean = float(probs.mean())
    prob_std  = float(probs.std())

    # Segmentasyon ortalaması → orijinal boyuta yeniden ölçekli
    seg_mean = seg_probs.mean(axis=0)[0]   # (5,1,H,W) → (1,H,W) → (H,W)
    seg_resized = cv2.resize(
        seg_mean, (gray_image.shape[1], gray_image.shape[0])
    )
    seg_binary = (seg_resized > seg_threshold).astype(np.uint8) * 255

    return {
        "prob_mean":    prob_mean,
        "prob_std":     prob_std,
        "prob_votes":   probs.tolist(),
        "seg_mean":     seg_mean,
        "seg_binary":   seg_binary,
        "is_uncertain": prob_std > 0.15,
    }


# ── Belirsizlik etiketi ───────────────────────────────────────────────────────

def uncertainty_label(result: dict) -> str:
    """
    TTA sonucuna göre klinik karar etiketi üretir.

    Std < 0.05  → Yüksek güven
    Std 0.05-0.15 → Orta güven
    Std > 0.15  → Belirsiz, radyolog incelemesi önerilir
    """
    std = result["prob_std"]
    if std < 0.05:
        return "✅ Yüksek güven"
    elif std < 0.15:
        return "⚠️  Orta güven — ikinci okuma önerilir"
    else:
        return "🔴 Belirsiz — Radyolog incelemesi gerekli"
