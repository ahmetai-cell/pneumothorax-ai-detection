"""
Hard Negative Mining (HNM)
Modelin cilt kıvrımı / skapula kenarı / apikal gölge gibi
yapıları pnömotoraks sanmasını (False Positive) önler.

Yaklaşım:
  1. Her epoch sonunda negatif örnekleri modele gönder
  2. Yüksek güvenle yanlış pozitif tahmin edilenleri bul
  3. Bu "zor negatifler"in örnekleme ağırlığını artır
  4. Sonraki epoch WeightedRandomSampler bu örnekleri daha sık seçer

Bu döngü, modelin kolay negatifleri değil zor olanları öğrenmesini sağlar.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


@torch.no_grad()
def find_hard_negatives(
    model: torch.nn.Module,
    dataset,
    negative_indices: list[int],
    device: torch.device,
    fp_threshold: float = 0.4,
    batch_size: int = 16,
) -> list[int]:
    """
    Negatif örnekler arasında yüksek güvenle yanlış pozitif
    tahmin edilenleri (False Positive) bulur.

    Args:
        model            : Güncel epoch sonu modeli
        dataset          : Tam dataset
        negative_indices : Negatif (pnömotoraks yok) örnek indeksleri
        device           : CPU veya CUDA
        fp_threshold     : Bu eşiğin üzerindeki negatif tahminler "hard negative"
        batch_size       : Inference batch boyutu

    Returns:
        hard_neg_indices : Zor negatif indeksleri listesi
    """
    model.eval()

    # Sadece negatif örnekler üzerinde mini DataLoader
    neg_subset  = torch.utils.data.Subset(dataset, negative_indices)
    neg_loader  = DataLoader(neg_subset, batch_size=batch_size, shuffle=False)

    hard_neg_indices: list[int] = []

    for batch_pos, (images, _, _) in enumerate(
        tqdm(neg_loader, desc="  HNM tarama", leave=False)
    ):
        images = images.to(device)
        _, cls_pred = model(images)
        probs = torch.sigmoid(cls_pred).squeeze(-1)  # (B,)

        # Bu batch içinde fp_threshold üzerinde olan örnekler
        false_pos_mask = probs > fp_threshold
        for local_idx, is_fp in enumerate(false_pos_mask):
            if is_fp:
                global_idx = negative_indices[batch_pos * batch_size + local_idx]
                hard_neg_indices.append(global_idx)

    return hard_neg_indices


def update_sampler_with_hard_negatives(
    all_labels: list[int],
    hard_neg_indices: list[int],
    hard_neg_multiplier: float = 3.0,
) -> WeightedRandomSampler:
    """
    Hard negativelerin örnekleme ağırlığını artıran güncel sampler döndürür.

    Args:
        all_labels          : Tüm dataset için etiket listesi (0/1)
        hard_neg_indices    : find_hard_negatives() çıktısı
        hard_neg_multiplier : Hard negativeler bu katsayı kadar daha sık seçilir

    Returns:
        WeightedRandomSampler
    """
    labels_arr    = np.array(all_labels)
    class_counts  = np.bincount(labels_arr)
    class_weights = 1.0 / (class_counts + 1e-8)

    # Temel ağırlık: sınıf dengelemesi
    sample_weights = class_weights[labels_arr].copy()

    # Hard negativeler için ekstra ağırlık
    hard_neg_set = set(hard_neg_indices)
    for idx in hard_neg_set:
        sample_weights[idx] *= hard_neg_multiplier

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_negative_indices(dataset) -> list[int]:
    """Dataset içindeki negatif (pnömotoraks yok) örnek indekslerini döndürür."""
    neg_indices = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if label.item() == 0:
            neg_indices.append(i)
    return neg_indices


# ── Eğitim döngüsüne entegrasyon örneği ──────────────────────────────────────
#
# from src.utils.hard_negative_mining import (
#     find_hard_negatives, update_sampler_with_hard_negatives, get_negative_indices
# )
#
# neg_indices = get_negative_indices(train_dataset)
# all_labels  = [int(train_dataset[i][2].item()) for i in range(len(train_dataset))]
#
# for epoch in range(epochs):
#     train_loss, train_dice = train_epoch(model, train_loader, ...)
#
#     # Her 3 epoch'ta bir HNM uygula (her epoch çok maliyetli)
#     if epoch % 3 == 0:
#         hard_negs = find_hard_negatives(model, train_dataset, neg_indices, device)
#         print(f"  Hard negative: {len(hard_negs)} örnek güncellendi")
#         new_sampler = update_sampler_with_hard_negatives(all_labels, hard_negs)
#         train_loader = DataLoader(train_dataset, batch_size=BS, sampler=new_sampler)
