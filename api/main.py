"""
FastAPI Arayüzü — Pnömotoraks Tespit Sistemi
Akış: Dosya Yükle → Analiz Et → Sonucu Göster

Başlatma:
    uvicorn api.main:app --reload --port 8000

Endpointler:
    POST /predict       — görüntü yükle, analiz et, sonuç döndür
    POST /predict/tta   — TTA ile daha güvenilir analiz
    GET  /health        — servis sağlık kontrolü
    GET  /results       — K-fold metrik sonuçları

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

import base64
import csv
import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.model.unet import PneumothoraxModel
from src.preprocessing.green_mask_extractor import load_image, overlay_mask_on_image
from src.utils.gradcam import generate_gradcam_result
from src.utils.normalize import normalize_to_tensor
from src.utils.tta import predict_tta, uncertainty_label, CLS_THRESHOLD, SEG_THRESHOLD

logger = logging.getLogger("ptx-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Uygulama ──────────────────────────────────────────────────────────────────

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Pnömotoraks Tespit API",
    description=(
        "Akciğer grafilerinde pnömotoraksı otomatik tespit eden "
        "derin öğrenme tabanlı karar destek sistemi.\n\n"
        "TÜBİTAK 2209-A Projesi"
    ),
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Model yükleme ──────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "results/checkpoints"))
N_FOLDS = 5
IMG_SIZE = 512

if torch.cuda.is_available():
    _device = torch.device("cuda")
elif torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")
_ensemble: torch.nn.Module | None = None
_gradcam_model: PneumothoraxModel | None = None  # Fold 5 (en iyi) — sadece Grad-CAM için


class _EnsembleModel(torch.nn.Module):
    """5 fold modelinin çıktılarını ortalar. predict_tta ile uyumlu."""

    def __init__(self, models: list[PneumothoraxModel]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor):
        seg_acc = cls_acc = None
        for m in self.models:
            out = m(x)
            seg, cls = out[0], out[1]
            seg_acc = seg if seg_acc is None else seg_acc + seg
            cls_acc = cls if cls_acc is None else cls_acc + cls
        n = len(self.models)
        return seg_acc / n, cls_acc / n


def _load_fold(fold_i: int) -> PneumothoraxModel:
    path = CHECKPOINT_DIR / f"fold_{fold_i}_best.pth"
    if not path.exists():
        raise RuntimeError(
            f"Checkpoint bulunamadı: {path}\n"
            f"CHECKPOINT_DIR={CHECKPOINT_DIR} — env var veya path'i kontrol edin."
        )
    m = PneumothoraxModel()
    m.load_state_dict(torch.load(path, map_location=_device, weights_only=False))
    return m.eval().to(_device)


def get_ensemble() -> torch.nn.Module:
    """5 fold ensemble singleton."""
    global _ensemble
    if _ensemble is None:
        models = [_load_fold(i) for i in range(1, N_FOLDS + 1)]
        _ensemble = _EnsembleModel(models).to(_device)
        _ensemble.eval()
    return _ensemble


def get_gradcam_model() -> PneumothoraxModel:
    """Grad-CAM için en iyi fold (fold_5) singleton."""
    global _gradcam_model
    if _gradcam_model is None:
        _gradcam_model = _load_fold(N_FOLDS)
    return _gradcam_model


# Geriye dönük uyumluluk — eski kod get_model() çağırıyorsa ensemble döner
def get_model() -> torch.nn.Module:
    return get_ensemble()


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_ALLOWED_EXT      = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
_DICOM_EXT        = {".dcm", ".dicom"}


def _validate_upload(upload: UploadFile) -> str:
    """
    Dosya uzantısını doğrular; DICOM veya raster olduğunu belirler.
    Geçersizse HTTPException(400) fırlatır.
    Döner: 'dicom' veya 'raster'
    """
    filename = upload.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen uzantı '{ext}'. İzin verilenler: {sorted(_ALLOWED_EXT)}",
        )
    return "dicom" if ext in _DICOM_EXT else "raster"


def read_upload_as_gray(upload: UploadFile) -> np.ndarray:
    """
    UploadFile'ı uint8 grayscale numpy dizisine dönüştürür.
    PNG, JPEG ve DICOM formatlarını destekler.
    50 MB boyut limiti uygulanır.
    """
    file_type = _validate_upload(upload)

    content = upload.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya çok büyük. Maksimum: {_MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")

    if file_type == "dicom":
        import pydicom
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            try:
                ds = pydicom.dcmread(tmp_path)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Geçersiz DICOM dosyası: {exc}")
            if not hasattr(ds, "pixel_array"):
                raise HTTPException(status_code=422, detail="DICOM piksel verisi bulunamadı.")
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                arr = np.clip(arr, -1500, 500)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            logger.info(f"DICOM okundu: shape={arr.shape} file={upload.filename}")
            return arr
        finally:
            os.unlink(tmp_path)

    # PNG / JPEG
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Geçersiz veya bozuk görüntü dosyası.")
    arr = np.array(pil_img, dtype=np.uint8)
    if arr.size == 0:
        raise HTTPException(status_code=422, detail="Boş görüntü içeriği.")
    logger.info(f"Görüntü okundu: shape={arr.shape} file={upload.filename}")
    return arr


def ndarray_to_base64(img_bgr: np.ndarray) -> str:
    """BGR görüntüyü PNG base64 string'e çevirir."""
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def run_inference(gray: np.ndarray):
    """
    Ensemble (5 fold) ile inference yapar.

    Returns:
        seg_mask : float32 ndarray (IMG_SIZE x IMG_SIZE), ortalama sigmoid çıktısı
        cls_prob : float, 5 fold ortalaması pnömotoraks olasılığı
    """
    ensemble = get_ensemble()

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor  = normalize_to_tensor(resized, device=_device)

    with torch.no_grad():
        seg_pred, cls_pred = ensemble(tensor)

    cls_prob = torch.sigmoid(cls_pred).item()
    seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
    return seg_mask, cls_prob


# ── Endpointler ───────────────────────────────────────────────────────────────

@app.get("/health", summary="Servis sağlık kontrolü")
async def health():
    """API ve ensemble model durumunu döndürür."""
    fold_status = {
        f"fold_{i}": (CHECKPOINT_DIR / f"fold_{i}_best.pth").exists()
        for i in range(1, N_FOLDS + 1)
    }
    all_ready = all(fold_status.values())
    return {
        "status": "ok",
        "ensemble_ready": all_ready,
        "folds": fold_status,
        "device": str(_device),
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }


@app.post("/predict", summary="Akciğer grafisi analiz et (standart)")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    **Akış:**
    1. Dosyayı yükle (PNG, JPEG veya DICOM)
    2. Modeli çalıştır (segmentasyon + sınıflandırma)
    3. Grad-CAM ısı haritası üret
    4. Sonucu JSON olarak döndür

    **Dönen değerler:**
    - `has_pneumothorax` : Tespit var mı (bool)
    - `probability`      : Pnömotoraks olasılığı (0.0 – 1.0)
    - `diagnosis`        : Türkçe tanı metni
    - `gradcam_image`    : Base64 PNG (Grad-CAM overlay)
    - `segmentation_image` : Base64 PNG (segmentasyon maskesi overlay)
    """
    # Model hazır mı?
    try:
        get_ensemble()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Görüntüyü oku
    try:
        gray = read_upload_as_gray(file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Görüntü okunamadı: {e}")

    orig_h, orig_w = gray.shape[:2]

    # Segmentasyon + sınıflandırma (ensemble)
    try:
        seg_mask, cls_prob = run_inference(gray)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model hatası: {e}")

    # Grad-CAM — fold_5 (en iyi tek model) üzerinden
    try:
        gradcam_bgr, _ = generate_gradcam_result(get_gradcam_model(), gray, img_size=IMG_SIZE)
    except Exception as e:
        logger.warning(f"GradCAM başarısız, orijinal görüntü kullanılıyor: {e}")
        gradcam_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Segmentasyon maskesini orijinal boyuta getir
    seg_resized = cv2.resize(seg_mask, (orig_w, orig_h))
    binary_mask = (seg_resized > SEG_THRESHOLD).astype(np.uint8) * 255
    seg_overlay = overlay_mask_on_image(gray, binary_mask, alpha=0.4, color_bgr=(0, 200, 0))

    # Orijinal görüntü (grayscale → BGR)
    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Karar — hem cls_prob hem maske alanı kontrol edilir
    mask_ratio = float(binary_mask.sum()) / (255 * orig_h * orig_w)
    has_ptx = (cls_prob >= CLS_THRESHOLD) and (mask_ratio >= 0.005)
    logger.info(
        f"/predict | file={file.filename} | prob={cls_prob:.4f} "
        f"mask_ratio={mask_ratio:.4f} has_ptx={has_ptx} device={_device}"
    )
    if has_ptx:
        diagnosis = f"PNÖMOTORAKS TESPİT EDİLDİ (olasılık: {cls_prob:.1%})"
    else:
        diagnosis = f"Normal — Pnömotoraks saptanmadı (olasılık: {cls_prob:.1%})"

    return JSONResponse({
        "has_pneumothorax":    has_ptx,
        "probability":         round(cls_prob, 4),
        "diagnosis":           diagnosis,
        "original_image":      ndarray_to_base64(original_bgr),
        "gradcam_image":       ndarray_to_base64(gradcam_bgr),
        "segmentation_image":  ndarray_to_base64(seg_overlay),
    })


@app.post("/predict/tta", summary="Akciğer grafisi analiz et (TTA — daha güvenilir)")
@limiter.limit("10/minute")
async def predict_with_tta(request: Request, file: UploadFile = File(...)):
    """
    **Test-Time Augmentation (TTA) ile tahmin.**

    Aynı görüntüyü 5 farklı dönüşümle analiz eder, ortalamasını alır.
    Standart `/predict`'e göre daha güvenilir; özellikle sınırda vakalarda
    ve apikal küçük pnömotorakslar için önerilir.

    **Ek dönen değerler:**
    - `prob_std`      : 5 tahminin standart sapması (belirsizlik ölçüsü)
    - `prob_votes`    : Her varyantın bireysel tahmini
    - `uncertainty`   : Klinik güven etiketi
    - `is_uncertain`  : Radyolog incelemesi önerilir mi?
    """
    try:
        ensemble = get_ensemble()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        gray = read_upload_as_gray(file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Görüntü okunamadı: {e}")

    try:
        tta_result = predict_tta(ensemble, gray, img_size=IMG_SIZE, seg_threshold=SEG_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTA hatası: {e}")

    prob     = tta_result["prob_mean"]
    seg_bin  = tta_result["seg_binary"]
    mask_ratio = float(seg_bin.sum()) / (255 * gray.shape[0] * gray.shape[1])
    has_ptx  = (prob >= CLS_THRESHOLD) and (mask_ratio >= 0.005)
    seg_overlay = overlay_mask_on_image(
        gray, tta_result["seg_binary"], alpha=0.4, color_bgr=(0, 200, 0)
    )

    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    try:
        gradcam_bgr, _ = generate_gradcam_result(get_gradcam_model(), gray, img_size=IMG_SIZE)
    except Exception as e:
        logger.warning(f"TTA GradCAM başarısız, orijinal görüntü kullanılıyor: {e}")
        gradcam_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if has_ptx:
        diagnosis = f"PNÖMOTORAKS TESPİT EDİLDİ — TTA olasılık: {prob:.1%}"
    else:
        diagnosis = f"Normal — TTA olasılık: {prob:.1%} (maske alanı: {mask_ratio:.1%})"

    return JSONResponse({
        "has_pneumothorax":   has_ptx,
        "probability":        round(prob, 4),
        "prob_std":           round(tta_result["prob_std"], 4),
        "prob_votes":         [round(p, 4) for p in tta_result["prob_votes"]],
        "uncertainty":        uncertainty_label(tta_result),
        "is_uncertain":       tta_result["is_uncertain"],
        "diagnosis":          diagnosis,
        "original_image":     ndarray_to_base64(original_bgr),
        "gradcam_image":      ndarray_to_base64(gradcam_bgr),
        "segmentation_image": ndarray_to_base64(seg_overlay),
    })


@app.get("/results", summary="K-fold cross validation sonuçları")
async def get_results():
    """
    `results/global_kfold_results.csv` dosyasından fold metriklerini döndürür.

    Dönen değerler:
    - `folds`   : Her fold için metrik satırları
    - `summary` : Tüm foldlar üzerinden ortalama ve std
    """
    csv_path = Path("results/global_kfold_results.csv")
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sonuç dosyası bulunamadı: results/global_kfold_results.csv",
        )

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Sayısal alanları float'a çevir
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)

    if not rows:
        return JSONResponse({"folds": [], "summary": {}})

    # Sayısal sütunlar için özet istatistik
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    summary = {}
    for key in numeric_keys:
        vals = [r[key] for r in rows if isinstance(r.get(key), float)]
        if vals:
            summary[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }

    return JSONResponse({"folds": rows, "summary": summary})
