# TÜBİTAK 2209-A — "Yöntem" Bölümü Taslağı

> Bu taslak doğrudan başvuru formundaki **B. Araştırma Önerisinin Yöntemi** bölümüne yapıştırılmak
> üzere akademik Türkçe ile hazırlanmıştır. Sayfa sınırına göre bölümler kısaltılabilir.

---

## 3. YÖNTEM

### 3.1 Araştırma Tasarımı

Bu çalışma, retrospektif ve gözlemsel bir tasarıma sahip olup bilgisayar destekli tanı (CAD)
sistemi geliştirmeyi amaçlamaktadır. Araştırma; veri toplama ve ön işleme, model geliştirme,
performans değerlendirme ve klinik doğrulama olmak üzere dört ana aşamadan oluşmaktadır.

---

### 3.2 Veri Toplama ve Ön İşleme

**Veri Kaynakları**

Çalışmada iki tür veri kaynağı kullanılacaktır:

1. *Yerel Klinik Veri:* Dokuz Eylül Üniversitesi Hastanesi PACS arşivinden etik kurul onayı
   alınarak retrospektif olarak toplanacak AP ve PA akciğer grafileri. Görüntüler DICOM
   formatında elde edilecek; hasta kimlik bilgileri KVKK kapsamında anonimleştirilecektir.
2. *Açık Erişimli Veri Setleri:* ChestX-ray14 (Wang ve ark., 2017), CheXpert (Irvin ve ark.,
   2019) ve SIIM-ACR Pneumothorax Segmentation veri seti (15.000+ görüntü, piksel düzeyinde
   RLE kodlu etiketler) modelin dış geçerliliğini güçlendirmek amacıyla kullanılacaktır.

Hedef veri büyüklüğü en az 5.000 akciğer grafisidir (%30 pnömotoraks pozitif).

**Etiketleme**

Pnömotoraks alanlarının manuel segmentasyonu 3D Slicer (v5.x) Segment Editor modülü
kullanılarak uzman radyologlar tarafından gerçekleştirilecektir. Etiketleme çıktıları
NRRD formatında alınacak; geliştirilen `DicomSlicerDataset` sınıfı bu dosyaları
otomatik olarak binary maskelere dönüştürecektir.

**Görüntü Ön İşleme**

Tüm görüntülere aşağıdaki standardizasyon adımları uygulanacaktır:

| İşlem | Yöntem | Amaç |
|-------|--------|------|
| Kontrast artırma | CLAHE (clip_limit=4.0) | Portatif grafi düşük kontrastını iyileştirme |
| Yeniden boyutlandırma | 512 × 512 piksel | Model giriş standardizasyonu |
| Normalizasyon | μ=0.485, σ=0.229 | ImageNet ön eğitim ağırlıklarıyla uyum |
| Geometrik augmentation | ElasticTransform, HFlip, ShiftScaleRotate | Veri çeşitlendirme |
| Gürültü augmentation | GaussNoise | Portatif grafi artefaktlarını simüle etme |

---

### 3.3 Model Mimarisi

**Hibrit Encoder-Decoder Yapısı**

Modelin temel mimarisi, standart U-Net'e kıyasla daha zengin atlama (skip) bağlantıları
içeren **U-Net++** (Zhou ve ark., 2018) olarak belirlenmiştir. Bu tercih, küçük boyutlu
pnömotoraks odaklarının lokalizasyonundaki üstün performansına dayanmaktadır.

Encoder olarak ImageNet üzerinde ön eğitilmiş **EfficientNet-B0** kullanılmıştır
(Tan & Le, 2019). EfficientNet'in bileşik ölçeklendirme (compound scaling) yaklaşımı,
hesaplama maliyetini minimize ederken özellik çıkarım kapasitesini maksimize etmektedir.

Model iki paralel çıkış başlığı içermektedir:

```
Giriş X-Ray (1 × 512 × 512)
        │
   EfficientNet-B0 Encoder
   (Transfer Learning — ImageNet)
        │
   U-Net++ Decoder
   (Zenginleştirilmiş skip connections)
        ├──→ Segmentasyon Başlığı → Piksel düzeyinde maske (1 × 512 × 512)
        └──→ Sınıflandırma Başlığı → Pnömotoraks olasılığı (skalar)
```

**Kayıp Fonksiyonu**

Segmentasyon ve sınıflandırma çıkışları için hibrit bir kayıp fonksiyonu tanımlanmıştır:

$$\mathcal{L}_{toplam} = \lambda_1 \mathcal{L}_{Dice} + \lambda_2 \mathcal{L}_{BCE}^{seg} + \lambda_3 \mathcal{L}_{BCE}^{cls}$$

Burada λ₁ = λ₂ = 0.5, λ₃ = 0.3'tür. Dice kaybı, özellikle küçük pnömotoraks alanlarında
piksel sınıf dengesizliğinin yol açtığı eğitim sorunlarını gidermektedir.

---

### 3.4 Eğitim Stratejisi

**Sınıf Dengesizliğiyle Başa Çıkma**

Pnömotoraks nadir bir klinik durum olduğundan veri setinde sınıf dengesizliği beklenmektedir
(yaklaşık %70 negatif, %30 pozitif). Bu sorunu gidermek için `WeightedRandomSampler`
kullanılmıştır; her sınıfın örnekleme ağırlığı frekansının tersiyle orantılı olarak
hesaplanmaktadır.

**Hard Negative Mining (HNM)**

Cilt kıvrımları ve skapula kenarı gibi anatomik yapıların False Positive (FP) olarak
sınıflandırılması riskini azaltmak amacıyla Hard Negative Mining uygulanmıştır.
Her üç epoch'ta bir, model negatif örnekler üzerinde çalıştırılmakta; yüksek güvenle
yanlış pozitif tahmin edilen örnekler tespit edilmekte ve bu örneklerin örnekleme ağırlığı
3 katına çıkarılmaktadır. Bu döngüsel yaklaşım modelin zor örnekler üzerinden öğrenmesini sağlar.

**Optimizasyon**

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam (weight_decay=1e-4) |
| Başlangıç LR | 1 × 10⁻⁴ |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch boyutu | 16 |
| Epoch sayısı | 50 |

**Çapraz Doğrulama**

Modelin genellenebilirliğini kanıtlamak ve varyansı azaltmak amacıyla **Stratified 5-Fold
Cross Validation** uygulanmıştır. Pozitif/negatif dağılımı her fold'da korunmaktadır.
Hiperparametre optimizasyonu Weights & Biases (W&B) Bayesian sweep altyapısı ile
öğrenme oranı [10⁻⁵, 10⁻³] ve HNM çarpanı {2.0, 3.0, 5.0} aralıklarında gerçekleştirilmiştir.

---

### 3.5 Performans Değerlendirmesi

Model performansı aşağıdaki metriklerle değerlendirilecektir:

| Metrik | Formül | Hedef |
|--------|--------|-------|
| Dice Katsayısı | 2\|P∩G\| / (\|P\|+\|G\|) | ≥ 0.85 |
| IoU (Jaccard) | \|P∩G\| / \|P∪G\| | ≥ 0.80 |
| AUC-ROC | — | ≥ 0.90 |
| Duyarlılık | TP / (TP+FN) | ≥ 0.85 |
| Özgüllük | TN / (TN+FP) | ≥ 0.85 |

Sonuçlar **ortalama ± standart sapma** formatında raporlanacak olup literatürdeki referans
sistemlerle (Thian ve ark., 2021; Malhotra ve ark., 2022) karşılaştırılacaktır.

---

### 3.6 Açıklanabilirlik (Explainable AI)

Modelin karar mekanizması **Captum** kütüphanesi (Kokhlikyan ve ark., 2020) kullanılarak
görselleştirilecektir:

- **Grad-CAM (LayerGradCam):** EfficientNet encoder'ının son katmanındaki gradyanlar
  kullanılarak ısı haritası üretilmekte; pnömotoraks kararının tetiklendiği anatomik
  bölge renk skalasıyla gösterilmektedir.
- **Integrated Gradients:** Giriş piksellerinin sınıflandırma kararına bireysel katkısı
  hesaplanmakta; küçük apikal pnömotoraksların tanımlanmasında Grad-CAM'e tamamlayıcı
  bir granülarite sağlamaktadır.

Her iki yöntemde de ısı haritası 3D renkli maske olarak orijinal grafi üzerine bindirilmekte;
böylece sistem açıklanabilir yapay zekâ (XAI) ilkelerine uygun hale getirilmektedir.

---

### 3.7 Test-Time Augmentation (TTA)

Inference aşamasında aynı görüntü beş farklı dönüşümle (orijinal, yatay flip, parlaklık ±,
kontrast +) analiz edilmekte; tahminlerin ortalaması alınmaktadır. Bu yaklaşım özellikle
apikal küçük pnömotoraks olgularında ve sınırda olasılıklarda (0.45–0.55) daha güvenilir
tahminler üretmektedir. Beş tahminin standart sapması belirsizlik ölçüsü olarak sunulmakta;
std > 0.15 olan vakalar radyolog incelemesine yönlendirilmektedir.

---

### 3.8 Sistem Entegrasyonu ve Klinik Uygulama Prototipi

Geliştirilen model iki arayüz üzerinden test edilecektir:

1. **FastAPI Backend:** `POST /predict` (standart) ve `POST /predict/tta` endpointleri
   aracılığıyla DICOM veya PNG formatında görüntü alıp JSON formatında olasılık,
   Grad-CAM ve segmentasyon maskesi döndürmektedir.
2. **Streamlit Demo:** Üç panel görselleştirme (orijinal X-ray / segmentasyon maskesi /
   Grad-CAM ısı haritası), TTA toggle ve belirsizlik göstergesi içeren kullanıcı arayüzü.

Model çıktıları DICOM uyumlu formata dönüştürülerek mevcut PACS altyapısıyla
entegre edilmesi planlanmaktadır.

---

### 3.9 Çalışma Takvimi ile İlişkilendirme

| Aşama | Zaman | Bu Yöntem Bölümüyle İlgisi |
|-------|-------|---------------------------|
| Veri toplama & etiketleme | Oca–May 2025 | §3.2 |
| Ön işleme & augmentation | Haz–Ağu 2025 | §3.2 |
| Model geliştirme & eğitim | Eyl–Ara 2025 | §3.3, §3.4 |
| Performans değerlendirme | Oca–Şub 2026 | §3.5 |
| XAI & TTA modülleri | Mar–Nis 2026 | §3.6, §3.7 |
| Klinik test & entegrasyon | May–Haz 2026 | §3.8 |

---

> **Not:** W&B eğitim grafiklerini (Loss/Dice eğrileri, HNM FP oranı düşüşü, fold
> karşılaştırma tablosu) rapor ekine eklemek için `wandb.ai` profilinizdeki "Export"
> özelliğini kullanın. Bu grafikler §3.5 sonuçlarını görsel olarak destekleyecektir.
