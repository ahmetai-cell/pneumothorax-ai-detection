"""
Result Table Generator
K-Fold CV sonuçlarını CSV'ye kaydeder, ortalama ± std hesaplar.
TÜBİTAK raporuna doğrudan kopyalanabilir tablo üretir.

Per-site desteği (PTX-498 için):
  append_fold_result(..., per_site={"SiteA": {"dice":…, "iou":…, "hd95":…}, …})
  save_results_table() per-site sütunları otomatik olarak CSV'ye ve konsola ekler.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _fmt_hd95(val: float) -> str | float:
    """inf → '∞', diğerleri → yuvarlama."""
    return "∞" if np.isinf(val) else round(val, 1)


# ── Ana fonksiyonlar ──────────────────────────────────────────────────────────

def save_results_table(
    fold_results: list[dict],
    output_path: str = "results/kfold_results.csv",
) -> pd.DataFrame:
    """
    fold_results formatı (per_site opsiyonel):
        [{"fold": 1, "best_dice": 0.87, "best_auc": 0.94,
          "best_iou": 0.79, "best_sensitivity": 0.91,
          "per_site": {"SiteA": {"dice": 0.89, "iou": 0.82, "hd95": 12.3},
                       "SiteB": {"dice": 0.84, "iou": 0.73, "hd95": 15.1},
                       "SiteC": {"dice": 0.81, "iou": 0.69, "hd95": 18.4}}}, ...]

    CSV çıktısı (per_site varsa sütunlar otomatik eklenir):
        Fold | Dice | IoU | AUC | Sensitivity | SiteA_Dice | SiteA_IoU | SiteA_HD95 | …
        1    | …    | …   | …   | …           | …          | …         | …          | …
        …
        Mean | …
        Std  | …
    """
    # ── Global metrik satırları ───────────────────────────────────────────────
    rows = []
    for r in fold_results:
        row: dict = {
            "Fold":        r["fold"],
            "Dice":        round(r.get("best_dice", 0.0), 4),
            "IoU":         round(r.get("best_iou",  0.0), 4),
            "AUC":         round(r.get("best_auc",  0.0), 4),
            "Sensitivity": round(r.get("best_sensitivity", 0.0), 4),
        }

        # ── Per-site sütunlar (varsa) ─────────────────────────────────────────
        ps = r.get("per_site") or {}
        for site, m in sorted(ps.items()):
            row[f"{site}_Dice"] = round(m.get("dice", 0.0), 4)
            row[f"{site}_IoU"]  = round(m.get("iou",  0.0), 4)
            row[f"{site}_HD95"] = _fmt_hd95(m.get("hd95", float("inf")))

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Ortalama ve std satırları ─────────────────────────────────────────────
    # Sadece sayısal sütunlar üzerinde hesapla (∞ string içeren HD95 hariç)
    num_cols = [c for c in df.columns if c != "Fold" and df[c].dtype != object]

    mean_row: dict = {"Fold": "Ortalama"}
    std_row:  dict = {"Fold": "Std"}
    for col in num_cols:
        mean_row[col] = round(float(df[col].mean()), 4)
        std_row[col]  = round(float(df[col].std()),  4)

    # HD95 sütunları: string "∞" içerebilir; bunlar için ayrı hesap
    hd95_cols = [c for c in df.columns if c.endswith("_HD95")]
    for col in hd95_cols:
        finite = [v for v in df[col] if v != "∞"]
        mean_row[col] = round(float(np.mean(finite)),  1) if finite else "∞"
        std_row[col]  = round(float(np.std(finite)),   1) if finite else "∞"

    summary_df = pd.concat(
        [df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ── Konsol: global tablo ──────────────────────────────────────────────────
    global_cols = ["Fold", "Dice", "IoU", "AUC", "Sensitivity"]
    print("\n" + "=" * 58)
    print("  K-FOLD SONUÇ TABLOSU")
    print("=" * 58)
    print(summary_df[global_cols].to_string(index=False))
    print("=" * 58)

    # ── Konsol: per-site tablo (varsa) ────────────────────────────────────────
    site_cols = [c for c in summary_df.columns if "_Dice" in c or "_IoU" in c or "_HD95" in c]
    if site_cols:
        sites = sorted({c.rsplit("_", 1)[0] for c in site_cols})
        print("\n  PER-SITE METRİKLER (en iyi fold epoch'ta)")
        print("-" * 58)
        header = f"  {'Site':<8}  {'Fold':<6}  {'Dice':>6}  {'IoU':>6}  {'HD95':>8}"
        print(header)
        print("  " + "-" * 54)
        for _, row in summary_df.iterrows():
            for site in sites:
                d = row.get(f"{site}_Dice", "-")
                i = row.get(f"{site}_IoU",  "-")
                h = row.get(f"{site}_HD95", "-")
                print(f"  {site:<8}  {str(row['Fold']):<6}  {str(d):>6}  {str(i):>6}  {str(h):>8}")
        print("-" * 58)

    print(f"\n  CSV kaydedildi: {output_path}")

    # ── TÜBİTAK özet metni ───────────────────────────────────────────────────
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
    per_site: dict | None = None,
) -> None:
    """
    Fold sonucunu listeye ekler. train_kfold() ve train_kfold_local() içinde kullanılır.

    per_site (opsiyonel):
        {"SiteA": {"dice": 0.89, "iou": 0.82, "hd95": 12.3}, …}
        Geçilirse save_results_table() per-site sütunlarını CSV'ye ekler.
    """
    results.append({
        "fold":             fold,
        "best_dice":        best_dice,
        "best_iou":         best_iou,
        "best_auc":         best_auc,
        "best_sensitivity": best_sensitivity,
        "per_site":         per_site or {},
    })
