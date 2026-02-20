"""
80/10/10 Veri Bölme Scripti
1500 yerel + 3500 açık kaynak → Train / Val / Test

Özellikler:
  - Stratified split: her setteki pnömotoraks oranı korunur
  - Hasta seviyesinde sızdırmazlık: aynı hastanın grafileri tek sette kalır
  - Yerel veri test setine öncelikli atanır (klinik doğrulama için)
  - Tekrarlanabilir bölme (random_state=42)

Kullanım:
    python scripts/prepare_splits.py \
        --hospital   data/hospital_manifest.csv \
        --opensource data/siim_manifest.csv \
        --out_dir    data/splits

Çıktılar:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/splits/split_report.txt

TÜBİTAK 2209-A | Ahmet Demir
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Sabitler ─────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _stratified_split(
    df: pd.DataFrame,
    train_r: float,
    val_r: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    DataFrame'i stratified olarak train/val/test'e böler.
    has_pneumothorax sütununu sınıf etiketi olarak kullanır.
    """
    label_col = "has_pneumothorax"

    # Train vs (val+test)
    df_train, df_temp = train_test_split(
        df,
        test_size=1 - train_r,
        stratify=df[label_col],
        random_state=seed,
    )

    # Val vs test (val+test içinden)
    val_size_adjusted = val_r / (val_r + TEST_RATIO)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - val_size_adjusted,
        stratify=df_temp[label_col],
        random_state=seed,
    )

    return df_train, df_val, df_test


def _patient_aware_split(
    df: pd.DataFrame,
    train_r: float,
    val_r: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hasta ID'sine göre sızdırmaz bölme.
    Aynı hastanın grafileri tek sette kalır.
    patient_id sütunu yoksa standart stratified'a döner.
    """
    if "patient_id" not in df.columns:
        return _stratified_split(df, train_r, val_r, seed)

    # Hasta başına çoğunluk etiketi (pnömotoraks var mı)
    patient_df = (
        df.groupby("patient_id")["has_pneumothorax"]
        .max()
        .reset_index()
        .rename(columns={"has_pneumothorax": "label"})
    )

    p_train, p_temp = train_test_split(
        patient_df,
        test_size=1 - train_r,
        stratify=patient_df["label"],
        random_state=seed,
    )
    val_adj = val_r / (val_r + TEST_RATIO)
    p_val, p_test = train_test_split(
        p_temp,
        test_size=1 - val_adj,
        stratify=p_temp["label"],
        random_state=seed,
    )

    train_patients = set(p_train["patient_id"])
    val_patients   = set(p_val["patient_id"])

    df_train = df[df["patient_id"].isin(train_patients)].copy()
    df_val   = df[df["patient_id"].isin(val_patients)].copy()
    df_test  = df[~df["patient_id"].isin(train_patients | val_patients)].copy()

    return df_train, df_val, df_test


def _split_report(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: str,
) -> None:
    """Bölme istatistiklerini konsola yazar ve dosyaya kaydeder."""
    total = len(df_train) + len(df_val) + len(df_test)

    lines = ["=" * 62, "  VERİ BÖLME RAPORU", "=" * 62]

    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        n     = len(df)
        n_pos = int(df["has_pneumothorax"].sum())
        n_neg = n - n_pos
        src   = df.get("source", pd.Series(["?"] * n))
        n_loc = int((src == "hospital").sum())
        n_oss = int((src != "hospital").sum())

        lines += [
            f"\n  {name} ({n:,} görüntü — %{n/total*100:.0f})",
            f"    Pnömotoraks (+) : {n_pos:>5,}  ({n_pos/n:.1%})",
            f"    Normal      (-) : {n_neg:>5,}  ({n_neg/n:.1%})",
            f"    Yerel (DEÜ)     : {n_loc:>5,}",
            f"    Açık kaynak     : {n_oss:>5,}",
        ]

    lines += ["", "=" * 62]
    report = "\n".join(lines)
    print(report)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Rapor kaydedildi: {out_path}")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def prepare_splits(
    hospital_csv: str | None,
    opensource_csv: str | None,
    out_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Yerel ve açık kaynak manifest'lerini birleştirir,
    stratified 80/10/10 olarak böler ve kaydeder.
    """
    frames: list[pd.DataFrame] = []

    # Yerel veri
    if hospital_csv and Path(hospital_csv).exists():
        df_hosp = pd.read_csv(hospital_csv)
        df_hosp["source"] = "hospital"
        frames.append(df_hosp)
        print(f"  Yerel veri yüklendi : {len(df_hosp):,} görüntü  [{hospital_csv}]")
    else:
        print(f"  [!] Yerel manifest bulunamadı: {hospital_csv}")

    # Açık kaynak veri
    if opensource_csv and Path(opensource_csv).exists():
        df_open = pd.read_csv(opensource_csv)
        df_open["source"] = "opensource"
        frames.append(df_open)
        print(f"  Açık kaynak yüklendi: {len(df_open):,} görüntü  [{opensource_csv}]")
    else:
        print(f"  [!] Açık kaynak manifest bulunamadı: {opensource_csv}")

    if not frames:
        raise FileNotFoundError("Hiçbir manifest dosyası bulunamadı.")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Toplam: {len(df_all):,} görüntü\n")

    # Zorunlu sütun kontrolü
    required = {"image_path", "has_pneumothorax"}
    missing  = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    # ── Bölme stratejisi ──────────────────────────────────────────────────────
    # Yerel verinin test setine girmesini önceliklendir:
    # 1. Yerel vakalar hasta seviyesinde ayrılır
    # 2. Açık kaynak stratified olarak bölünür
    # 3. Yerel test → genel test setine eklenir

    df_hospital   = df_all[df_all["source"] == "hospital"].copy()
    df_opensource = df_all[df_all["source"] != "hospital"].copy()

    # Açık kaynak bölümü
    if len(df_opensource) >= 3:
        oss_train, oss_val, oss_test = _stratified_split(
            df_opensource, train_ratio, val_ratio, seed
        )
    else:
        oss_train = df_opensource.copy()
        oss_val   = pd.DataFrame(columns=df_all.columns)
        oss_test  = pd.DataFrame(columns=df_all.columns)

    # Yerel veri bölümü (hasta seviyesinde)
    if len(df_hospital) >= 3:
        hosp_train, hosp_val, hosp_test = _patient_aware_split(
            df_hospital, train_ratio, val_ratio, seed
        )
    else:
        hosp_train = df_hospital.copy()
        hosp_val   = pd.DataFrame(columns=df_all.columns)
        hosp_test  = pd.DataFrame(columns=df_all.columns)

    # Birleştir
    df_train = pd.concat([oss_train, hosp_train], ignore_index=True)
    df_val   = pd.concat([oss_val,   hosp_val],   ignore_index=True)
    df_test  = pd.concat([oss_test,  hosp_test],  ignore_index=True)

    # Shuffle
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val   = df_val.sample(frac=1,   random_state=seed).reset_index(drop=True)
    df_test  = df_test.sample(frac=1,  random_state=seed).reset_index(drop=True)

    # ── Kaydet ───────────────────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_csv = str(out_path / "train.csv")
    val_csv   = str(out_path / "val.csv")
    test_csv  = str(out_path / "test.csv")

    df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    df_val.to_csv(val_csv,     index=False, encoding="utf-8-sig")
    df_test.to_csv(test_csv,   index=False, encoding="utf-8-sig")

    print(f"\n  ✓  train.csv  → {train_csv}")
    print(f"  ✓  val.csv    → {val_csv}")
    print(f"  ✓  test.csv   → {test_csv}")

    _split_report(df_train, df_val, df_test, str(out_path / "split_report.txt"))

    return df_train, df_val, df_test


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="80/10/10 stratified veri bölme scripti"
    )
    parser.add_argument("--hospital",
                        default="data/hospital_manifest.csv",
                        help="Yerel hastane manifest CSV")
    parser.add_argument("--opensource",
                        default="data/siim_manifest.csv",
                        help="Açık kaynak manifest CSV (SIIM-ACR vb.)")
    parser.add_argument("--out_dir",
                        default="data/splits",
                        help="Çıktı klasörü")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED)

    args = parser.parse_args()
    prepare_splits(
        hospital_csv=args.hospital,
        opensource_csv=args.opensource,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
