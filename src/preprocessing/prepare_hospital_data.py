"""
Hastane Verisi Hazırlık Scripti

Tıp Fakültesi'nden gelen yeşil boyalı görüntüleri toplu işler:
  - Yeşil maskeler çıkarılır → data/processed/hospital/masks/
  - Orijinal gri görüntüler kurtarılır → data/processed/hospital/images/
  - Eğitime hazır CSV manifest oluşturulur

Kullanım:
    python -m src.preprocessing.prepare_hospital_data \
        --input  data/raw/hospital_annotated \
        --output data/processed/hospital \
        --csv    data/hospital_manifest.csv

TÜBİTAK 2209-A | Ahmet Demir
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from src.preprocessing.green_mask_extractor import process_annotated_image

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".dicom"}


def prepare_dataset(input_dir: str, output_dir: str, csv_path: str) -> None:
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    images_dir  = output_path / "images"
    masks_dir   = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    )

    if not image_files:
        print(f"[!] '{input_dir}' klasöründe desteklenen görüntü bulunamadı.")
        print(f"    Desteklenen uzantılar: {SUPPORTED_EXT}")
        return

    print(f"Toplam {len(image_files)} görüntü bulundu.\n")

    rows = []
    ok = skipped = 0

    for img_file in tqdm(image_files, desc="İşleniyor", unit="görüntü"):
        try:
            gray, mask, has_ptx = process_annotated_image(str(img_file))
        except Exception as e:
            print(f"\n  [HATA] {img_file.name}: {e}")
            skipped += 1
            continue

        stem = img_file.stem
        out_img  = images_dir / f"{stem}.png"
        out_mask = masks_dir  / f"{stem}_mask.png"

        cv2.imwrite(str(out_img),  gray)
        cv2.imwrite(str(out_mask), mask)

        rows.append({
            "image_id":         stem,
            "image_path":       str(out_img.relative_to(output_path)),
            "mask_path":        str(out_mask.relative_to(output_path)),
            "has_pneumothorax": int(has_ptx),
            "source":           "hospital",
        })
        ok += 1

    if not rows:
        print("\n[!] Hiçbir görüntü işlenemedi.")
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    pos = sum(r["has_pneumothorax"] for r in rows)
    neg = len(rows) - pos

    print(f"""
Tamamlandı!
  Başarıyla işlenen : {ok}
  Atlanan (hata)    : {skipped}
  Pnömotoraks (+)   : {pos}
  Normal       (-)  : {neg}
  Görüntüler        : {images_dir}
  Maskeler          : {masks_dir}
  Manifest CSV      : {csv_path}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tıp Fakültesi yeşil boyalı X-ray'lerini eğitime hazırlar."
    )
    parser.add_argument(
        "--input", required=True,
        help="Giriş klasörü (yeşil boyalı görüntüler)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Çıkış klasörü (işlenmiş görüntüler ve maskeler burada saklanır)"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Oluşturulacak manifest CSV dosyasının yolu"
    )
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.csv)
