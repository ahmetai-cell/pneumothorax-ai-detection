"""
Result Table Generator
K-Fold CV sonuçlarını CSV'ye kaydeder, ortalama ± std hesaplar.
TÜBİTAK raporuna doğrudan kopyalanabilir tablo üretir.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_results_table(
    fold_results: list[dict],
    output_path: str = "results/kfold_results.csv",
) -> pd.DataFrame:
    """
    fold_results formatı:
        [{"fold": 1, "best_dice": 0.87, "best_auc": 0.94,
          "best_iou": 0.79, "best_sensitivity": 0.91}, ...]

    Dönen DataFrame (CSV'ye de kaydedilir):

        Fold | Dice  | IoU   | AUC   | Sensitivity
        ---- | ----- | ----- | ----- | -----------
        1    | 0.872 | 0.814 | 0.961 | 0.934
        ...
        Mean | 0.865 | 0.808 | 0.955 | 0.929
        Std  | 0.012 | 0.015 | 0.008 | 0.011
    """
    metrics = ["dice", "iou", "auc", "sensitivity"]
    # Eksik metrikler için 0.0 varsayılan
    rows = []
    for r in fold_results:
        rows.append({
            "Fold":        r["fold"],
            "Dice":        round(r.get("best_dice", 0.0), 4),
            "IoU":         round(r.get("best_iou",  0.0), 4),
            "AUC":         round(r.get("best_auc",  0.0), 4),
            "Sensitivity": round(r.get("best_sensitivity", 0.0), 4),
        })

    df = pd.DataFrame(rows)

    # Ortalama ve std satırları
    num_cols = ["Dice", "IoU", "AUC", "Sensitivity"]
    mean_row = {"Fold": "Ortalama"}
    std_row  = {"Fold": "Std"}
    for col in num_cols:
        mean_row[col] = round(df[col].mean(), 4)
        std_row[col]  = round(df[col].std(),  4)

    summary_df = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Konsol çıktısı
    print("\n" + "=" * 58)
    print("  K-FOLD SONUÇ TABLOSU")
    print("=" * 58)
    print(summary_df.to_string(index=False))
    print("=" * 58)
    print(f"\n  CSV kaydedildi: {output_path}")

    # TÜBİTAK raporu için hazır metin
    mean = mean_row
    std  = std_row
    print(f"""
  ── TÜBİTAK Raporu İçin Özet (kopyala-yapıştır) ──────────────
  5-Fold Çapraz Doğrulama sonuçları:
    Dice Katsayısı : {mean['Dice']} ± {std['Dice']}
    IoU (Jaccard)  : {mean['IoU']}  ± {std['IoU']}
    AUC-ROC        : {mean['AUC']}  ± {std['AUC']}
    Duyarlılık     : {mean['Sensitivity']} ± {std['Sensitivity']}
  ──────────────────────────────────────────────────────────────
""")

    return summary_df


def append_fold_result(
    results: list[dict],
    fold: int,
    best_dice: float,
    best_auc: float,
    best_iou: float = 0.0,
    best_sensitivity: float = 0.0,
) -> None:
    """Fold sonucunu listeye ekler. train_kfold() içinde kullanılır."""
    results.append({
        "fold":             fold,
        "best_dice":        best_dice,
        "best_iou":         best_iou,
        "best_auc":         best_auc,
        "best_sensitivity": best_sensitivity,
    })
