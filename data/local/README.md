# DEU Yerel Veri Dizini

Bu dizin, Dokuz Eylül Üniversitesi Hastanesi'nden gelecek
de-identified DICOM ve NRRD dosyaları için ayrılmıştır.

## Beklenen Yapı

```
data/local/
├── dicom/
│   ├── patient_001.dcm
│   ├── patient_002.dcm
│   └── ...
└── nrrd/
    ├── patient_001.nrrd   ← sadece pozitif vakalar (3D Slicer çıktısı)
    ├── patient_002.nrrd
    └── ...
```

## Eşleştirme Kuralı

- Dosya adları (uzantısız, **stem**) eşleşmelidir:
  `patient_001.dcm` ↔ `patient_001.nrrd`
- Negatif vakalar (pnömotoraks yok) için NRRD dosyası olmayabilir.
  `pair_dicom_nrrd()` otomatik olarak `is_pneumo=0` etiketler.

## Fine-tuning Komutu

```bash
# Encoder dondurulmuş mod (~30 dk, önerilen ilk deneme)
python scripts/fine_tune_local.py --freeze_encoder

# Tam fine-tuning (~2-3 saat)
python scripts/fine_tune_local.py
```

TÜBİTAK 2209-A | Ahmet Demir
