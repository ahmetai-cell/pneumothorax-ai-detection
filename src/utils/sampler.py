"""
Optimize WeightedRandomSampler
1500 yerel + 3500 açık kaynak = 5000 görüntü için tasarlandı.

Stratejiler:
  - inverse_freq  : Nadir sınıfa ters orantılı ağırlık (standart)
  - sqrt          : Kare kök ölçekleme (aşırı oversampling'i önler)
  - custom_ratio  : Hedef pos:neg oranını elle ayarla (önerilen: 1:2)
  - source_aware  : Yerel veriyi açık kaynağa göre ağırlıklandır

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def make_sampler(
    labels: list[int],
    strategy: str = "custom_ratio",
    target_pos_ratio: float = 0.35,
    source_flags: list[int] | None = None,
    local_boost: float = 1.5,
) -> WeightedRandomSampler:
    """
    Sınıf dengesizliği ve veri kaynağı için optimize edilmiş sampler.

    Args:
        labels           : 0/1 etiket listesi (tüm dataset)
        strategy         : "inverse_freq" | "sqrt" | "custom_ratio"
        target_pos_ratio : custom_ratio stratejisinde hedef pozitif oran
                           (0.35 → her batch'in %35'i pnömotoraks)
        source_flags     : 1=yerel, 0=açık kaynak (her örnek için)
        local_boost      : Yerel örneklerin ek ağırlık çarpanı

    Returns:
        WeightedRandomSampler
    """
    labels_arr = np.array(labels, dtype=int)
    n          = len(labels_arr)
    n_pos      = labels_arr.sum()
    n_neg      = n - n_pos

    print(f"\n[Sampler] Toplam: {n}  |  Pozitif: {n_pos} ({n_pos/n:.1%})  "
          f"|  Negatif: {n_neg} ({n_neg/n:.1%})")
    print(f"[Sampler] Strateji: {strategy}  "
          + (f"|  Hedef pozitif oran: {target_pos_ratio:.0%}" if strategy == "custom_ratio" else ""))

    if strategy == "inverse_freq":
        weights = _inverse_freq(labels_arr, n_pos, n_neg)

    elif strategy == "sqrt":
        weights = _sqrt_scale(labels_arr, n_pos, n_neg)

    elif strategy == "custom_ratio":
        weights = _custom_ratio(labels_arr, n_pos, n_neg, target_pos_ratio)

    else:
        raise ValueError(f"Bilinmeyen strateji: {strategy}. "
                         f"Seçenekler: inverse_freq, sqrt, custom_ratio")

    # Yerel veri boost'u
    if source_flags is not None:
        flags = np.array(source_flags, dtype=float)
        # 1=yerel → local_boost çarpanı, 0=açık kaynak → 1.0
        boost = np.where(flags == 1, local_boost, 1.0)
        weights = weights * boost
        n_local = int(flags.sum())
        print(f"[Sampler] Yerel veri boost ({local_boost}x): {n_local} örnek")

    weights_t = torch.tensor(weights, dtype=torch.float)
    sampler   = WeightedRandomSampler(
        weights=weights_t,
        num_samples=n,
        replacement=True,
    )

    # Beklenen dağılımı hesapla
    expected_pos = float((weights_t[labels_arr == 1].sum() / weights_t.sum()).item())
    print(f"[Sampler] Beklenen pozitif oran (batch'te): {expected_pos:.1%}\n")

    return sampler


# ── Strateji implementasyonları ───────────────────────────────────────────────

def _inverse_freq(labels_arr, n_pos, n_neg) -> np.ndarray:
    """Standart ters frekans: w_i = 1 / class_count."""
    w_pos = 1.0 / n_pos if n_pos > 0 else 0.0
    w_neg = 1.0 / n_neg if n_neg > 0 else 0.0
    return np.where(labels_arr == 1, w_pos, w_neg)


def _sqrt_scale(labels_arr, n_pos, n_neg) -> np.ndarray:
    """
    Kare kök ölçekleme: inverse_freq'den daha az agresif.
    Aşırı oversampling'i önler, küçük veri setlerinde daha kararlı.
    """
    w_pos = 1.0 / np.sqrt(n_pos) if n_pos > 0 else 0.0
    w_neg = 1.0 / np.sqrt(n_neg) if n_neg > 0 else 0.0
    return np.where(labels_arr == 1, w_pos, w_neg)


def _custom_ratio(labels_arr, n_pos, n_neg, target_pos_ratio) -> np.ndarray:
    """
    Hedef pozitif oranı elle ayarlar.
    target_pos_ratio=0.35 → batch'lerin %35'i pnömotoraks olur.

    Formül:
      w_pos * n_pos / (w_pos * n_pos + w_neg * n_neg) = target_pos_ratio
      → w_pos / w_neg = target_pos_ratio * n_neg / ((1 - target_pos_ratio) * n_pos)
    """
    if n_pos == 0 or n_neg == 0:
        return _inverse_freq(labels_arr, n_pos, n_neg)

    ratio    = target_pos_ratio * n_neg / ((1 - target_pos_ratio) * n_pos)
    w_pos    = ratio          # negatife göre normalize
    w_neg    = 1.0
    return np.where(labels_arr == 1, w_pos, w_neg)


# ── Veri seti istatistikleri ─────────────────────────────────────────────────

def print_dataset_stats(labels: list[int], source_flags: list[int] | None = None) -> None:
    """5000 görüntülük veri setinin dağılımını raporlar."""
    arr = np.array(labels)
    n   = len(arr)

    print("=" * 55)
    print("  VERİ SETİ İSTATİSTİKLERİ")
    print("=" * 55)
    print(f"  Toplam görüntü    : {n:>6,}")
    print(f"  Pnömotoraks (+)   : {arr.sum():>6,}  ({arr.sum()/n:.1%})")
    print(f"  Normal      (-)   : {(n - arr.sum()):>6,}  ({(n - arr.sum())/n:.1%})")

    if source_flags is not None:
        flags = np.array(source_flags)
        n_local  = int(flags.sum())
        n_open   = int((flags == 0).sum())
        local_pos = int(arr[flags == 1].sum())
        open_pos  = int(arr[flags == 0].sum())
        print(f"\n  Yerel (DEÜ)       : {n_local:>6,}  "
              f"(+: {local_pos}, -: {n_local - local_pos})")
        print(f"  Açık kaynak       : {n_open:>6,}  "
              f"(+: {open_pos}, -: {n_open - open_pos})")

    print("=" * 55 + "\n")
