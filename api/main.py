"""
FastAPI Arayüzü — Pnömotoraks Tespit Sistemi
Akış: Dosya Yükle → Analiz Et → Sonucu Göster

Başlatma:
    uvicorn api.main:app --reload --port 8000

Endpointler:
    POST /predict   — görüntü yükle, analiz et, sonuç döndür
    GET  /health    — servis sağlık kontrolü

TÜBİTAK 2209-A | Ahmet Demir
"""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.model.unet import PneumothoraxModel
from src.preprocessing.green_mask_extractor import load_image, overlay_mask_on_image
from src.utils.gradcam import generate_gradcam_result

# ── Uygulama ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pnömotoraks Tespit API",
    description=(
        "Akciğer grafilerinde pnömotoraksı otomatik tespit eden "
        "derin öğrenme tabanlı karar destek sistemi.\n\n"
        "TÜBİTAK 2209-A Projesi"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model yükleme ──────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "results/checkpoints/best_model.pth")
IMG_SIZE = 512
THRESHOLD = 0.5

_model: PneumothoraxModel | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model() -> PneumothoraxModel:
    """Model singleton — ilk çağrıda diskten yükler."""
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model dosyası bulunamadı: {MODEL_PATH}\n"
                "Önce modeli eğitin: python src/utils/train.py"
            )
        _model = PneumothoraxModel()
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.eval().to(_device)
    return _model


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

ACCEPTED_TYPES = {"image/png", "image/jpeg", "image/jpg", "application/dicom"}


def read_upload_as_gray(upload: UploadFile) -> np.ndarray:
    """
    UploadFile'ı uint8 grayscale numpy dizisine dönüştürür.
    PNG, JPEG ve DICOM formatlarını destekler.
    """
    content = upload.file.read()

    # DICOM: pydicom ile oku
    if upload.filename and upload.filename.lower().endswith((".dcm", ".dicom")):
        import pydicom, tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            gray, _, _ = load_image.__module__ and None  # tip kontrolü için
            import pydicom as pd_
            ds = pd_.dcmread(tmp_path)
            arr = ds.pixel_array.astype(np.float32)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr
        finally:
            os.unlink(tmp_path)

    # PNG / JPEG: Pillow ile oku
    pil_img = Image.open(io.BytesIO(content)).convert("L")  # grayscale
    return np.array(pil_img, dtype=np.uint8)


def ndarray_to_base64(img_bgr: np.ndarray) -> str:
    """BGR görüntüyü PNG base64 string'e çevirir."""
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def run_inference(gray: np.ndarray):
    """
    Modeli çalıştırır, segmentasyon maskesi ve sınıflandırma skoru döndürür.

    Returns:
        seg_mask  : float32 ndarray (IMG_SIZE x IMG_SIZE), sigmoid çıktısı
        cls_prob  : float, pnömotoraks olasılığı
    """
    model = get_model()

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(_device)

    with torch.no_grad():
        seg_pred, cls_pred = model(tensor)

    cls_prob = torch.sigmoid(cls_pred).item()
    seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
    return seg_mask, cls_prob


# ── Endpointler ───────────────────────────────────────────────────────────────

@app.get("/health", summary="Servis sağlık kontrolü")
async def health():
    """API ve model durumunu döndürür."""
    model_ready = Path(MODEL_PATH).exists()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "device": str(_device),
        "model_path": MODEL_PATH,
    }


@app.post("/predict", summary="Akciğer grafisi analiz et")
async def predict(file: UploadFile = File(...)):
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
    # Dosya uzantı kontrolü
    filename = file.filename or ""
    allowed_ext = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    if Path(filename).suffix.lower() not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü. İzin verilenler: {allowed_ext}",
        )

    # Model hazır mı?
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Görüntüyü oku
    try:
        gray = read_upload_as_gray(file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Görüntü okunamadı: {e}")

    orig_h, orig_w = gray.shape[:2]

    # Segmentasyon + sınıflandırma
    try:
        seg_mask, cls_prob = run_inference(gray)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model hatası: {e}")

    # Grad-CAM
    try:
        gradcam_bgr, _ = generate_gradcam_result(model, gray, img_size=IMG_SIZE)
    except Exception as e:
        # Grad-CAM opsiyonel — hata olursa düz görüntü döndür
        gradcam_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Segmentasyon maskesini orijinal boyuta getir
    seg_resized = cv2.resize(seg_mask, (orig_w, orig_h))
    binary_mask = (seg_resized > THRESHOLD).astype(np.uint8) * 255
    seg_overlay = overlay_mask_on_image(gray, binary_mask, alpha=0.4, color_bgr=(0, 60, 220))

    # Karar
    has_ptx = cls_prob >= THRESHOLD
    if has_ptx:
        diagnosis = f"PNÖMOTORAKS TESPİT EDİLDİ (olasılık: {cls_prob:.1%})"
    else:
        diagnosis = f"Normal — Pnömotoraks saptanmadı (olasılık: {cls_prob:.1%})"

    return JSONResponse({
        "has_pneumothorax":    has_ptx,
        "probability":         round(cls_prob, 4),
        "diagnosis":           diagnosis,
        "gradcam_image":       ndarray_to_base64(gradcam_bgr),
        "segmentation_image":  ndarray_to_base64(seg_overlay),
    })
