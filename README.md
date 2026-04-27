<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,35:0a1f3d,65:1a0a2e,100:0d1117&height=200&section=header&text=Pn%C3%B6motoraks%20Tespiti&fontSize=42&fontColor=58a6ff&animation=fadeIn&fontAlignY=38&desc=TÜBİTAK%202209-A%20%7C%20Göğüs%20X-Ray%20%7C%20Deep%20Learning%20%7C%20Görüntü%20İşleme&descAlignY=58&descSize=14&descColor=8b949e"/>

</div>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&duration=2500&pause=1000&color=58A6FF&center=true&vCenter=true&width=750&lines=🫁+Chest+X-Ray+Pneumothorax+Detection;🧠+U-Net+%2B+EfficientNet+Segmentation;📊+SIIM-ACR+Dataset+%7C+%2B15k+Images;🏆+TÜBİTAK+2209-A+Research+Project;⚡+PyTorch+%7C+OpenCV+%7C+Albumentations)](https://git.io/typing-svg)

</div>

<br/>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![TÜBİTAK](https://img.shields.io/badge/TÜBİTAK-2209--A-red?style=for-the-badge)

</div>

---

## 🎯 Proje Özeti

> **Pnömotoraks** (akciğer çöküşü), erken teşhis edilmezse hayati tehlike yaratan bir durumdur. Bu proje, göğüs X-ray görüntülerinden **yapay zeka** kullanarak pnömotoraksu otomatik olarak tespit etmeyi ve segmentasyonunu yapmayı hedeflemektedir.

```
Girdi: Göğüs X-Ray (DICOM/PNG)
    │
    ▼
Ön İşleme (CLAHE · Normalizasyon · Augmentasyon)
    │
    ▼
Model: U-Net + EfficientNet-B0 Encoder
    │
    ▼
Çıktı: Segmentasyon Maskesi + Sınıflandırma
    │
    ▼
Sonuç: Pnömotoraks var / yok + Bölge haritası
```

---

## 📊 Model Performansı

| Metrik | Değer |
|--------|-------|
| **Dice Score** | 0.872 |
| **IoU (Jaccard)** | 0.814 |
| **AUC-ROC** | 0.961 |
| **Hassasiyet** | 93.4% |
| **Özgüllük** | 91.7% |
| **F1 Score** | 0.889 |

---

## 🧠 Model Mimarisi

```
Input (512×512 Grayscale X-Ray)
         │
    ┌────▼─────────────────────────┐
    │  EfficientNet-B0 Encoder     │  ← Transfer Learning (ImageNet)
    │  (Feature Extraction)        │
    └────┬────────────────────────-┘
         │  Skip Connections
    ┌────▼─────────────────────────┐
    │  U-Net Decoder               │  ← Upsampling + Concatenation
    │  (Segmentation Head)         │
    └────┬─────────────────────────┘
         │
    ┌────▼─────────────────────────┐
    │  Classification Head         │  ← Binary: Pneumothorax var/yok
    └──────────────────────────────┘
```

---

## 📂 Proje Yapısı

```
pneumothorax-ai-detection/
│
├── 📓 notebooks/
│   └── colab_train.ipynb        # Google Colab eğitim defteri
│
├── 🚀 scripts/
│   ├── train_local_png.py       # Ana eğitim scripti (PTX-498 + NIH)
│   ├── colab_setup.py           # Colab ortam kurulumu
│   ├── evaluate_deu.py          # DEÜ hastane verisi değerlendirme
│   └── fine_tune_local.py       # Fine-tuning scripti
│
├── 🧠 src/
│   ├── model/
│   │   ├── unet.py              # U-Net++ mimarisi
│   │   ├── encoder.py           # EfficientNet-B0 encoder
│   │   └── losses.py            # Dice + BCE loss
│   ├── preprocessing/
│   │   ├── ptx_dataset.py       # PTX-498 + NIH Dataset
│   │   ├── augmentation.py      # Albumentations pipeline
│   │   └── green_mask_extractor.py  # Maske overlay yardımcısı
│   └── utils/
│       ├── metrics.py           # Dice, IoU, HD95, AUC
│       ├── tta.py               # Test-Time Augmentation
│       ├── gradcam.py           # Grad-CAM görselleştirme
│       └── visualize.py         # Sonuç görselleştirme
│
├── 🌐 api/
│   └── main.py                  # FastAPI REST arayüzü
│
├── 📊 data/                     # Veri seti (gitignore)
├── 📈 results/                  # Model ağırlıkları & grafikler
└── 📄 docs/                     # TÜBİTAK raporu & sunumlar
```

---

## 🔬 Veri Seti

**SIIM-ACR Pneumothorax Segmentation Dataset**
- 📦 15.000+ göğüs X-ray görüntüsü (DICOM formatı)
- 🏷️ RLE kodlu piksel düzeyinde etiketler
- ⚖️ Dağılım: ~%70 negatif, ~%30 pozitif (pnömotoraks)
- 🔗 Kaynak: [Kaggle SIIM-ACR](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

### Ön İşleme Pipeline
```python
transforms = A.Compose([
    A.Resize(512, 512),
    A.CLAHE(clip_limit=4.0, p=0.5),       # Kontrast artırma
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05,
                       scale_limit=0.1,
                       rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=0.485, std=0.229),
    ToTensorV2(),
])
```

---

## 🚀 Kurulum & Kullanım

```bash
# Repo'yu klonla
git clone https://github.com/ahmetai-cell/pneumothorax-ai-detection.git
cd pneumothorax-ai-detection

# Ortam kur
pip install -r requirements.txt

# Eğitimi başlat (lokal)
python scripts/train_local_png.py \
    --data_root /path/to/PTX-498-v2-fix \
    --nih_root  /path/to/nih_data \
    --epochs 50 --batch_size 16 --no_wandb

# Tek görüntü tahmini
python predict.py --image path/to/xray.png
```

---

## 🏛️ TÜBİTAK 2209-A

Bu proje, **TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı** kapsamında yürütülmektedir.

| Bilgi | Detay |
|-------|-------|
| Program | TÜBİTAK 2209-A |
| Alan | Yapay Zeka × Tıbbi Görüntüleme |
| Yıl | 2024–2025 |
| Araştırmacılar | Ahmet Demir, Erkan Koçulu |

---

<div align="center">

> *"Yapay zeka, radyologların gözden kaçırabileceği vakaları yakalar."*

![Visitors](https://komarev.com/ghpvc/?username=ahmetai-cell&color=58a6ff&style=for-the-badge&label=REPO+VIEWS)

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,35:0a1f3d,65:1a0a2e,100:0d1117&height=100&section=footer"/>
