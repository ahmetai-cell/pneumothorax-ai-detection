"""
FastAPI — Pnömotoraks Tespit Sistemi (Production)

Başlatma:
    uvicorn api.main:app --reload --port 8000

Endpointler:
    POST /predict       — görüntü analiz (standart)
    POST /predict/tta   — TTA ile analiz
    GET  /health        — servis sağlık
    GET  /metrics       — model performans metrikleri
    GET  /results       — K-fold sonuçları

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

import asyncio
import base64
import collections
import concurrent.futures
import csv
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.models.unet import PneumothoraxModel
from src.preprocessing.green_mask_extractor import overlay_mask_on_image
from src.utils.gradcam import generate_gradcam_result
from src.utils.normalize import normalize_to_tensor
from src.utils.tta import predict_tta, uncertainty_label, CLS_THRESHOLD, SEG_THRESHOLD

logger = logging.getLogger("ptx-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── CORS ──────────────────────────────────────────────────────────────────────
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174").split(",")
    if o.strip()
]

# ── Rate limiter (Depends tabanlı — tüm FastAPI versiyonlarıyla uyumlu) ───────
_rate_store: dict[str, collections.deque] = collections.defaultdict(
    lambda: collections.deque()
)
_rate_lock = threading.Lock()

class _RateLimiter:
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls  = calls
        self.period = period

    async def __call__(self, request: Request) -> None:
        ip  = getattr(request.client, "host", "unknown")
        now = time.time()
        with _rate_lock:
            dq = _rate_store[ip]
            while dq and now - dq[0] > self.period:
                dq.popleft()
            if len(dq) >= self.calls:
                raise HTTPException(
                    status_code=429,
                    detail=f"Çok fazla istek. {self.period} saniye içinde max {self.calls}.",
                )
            dq.append(now)

_inference_limit = _RateLimiter(calls=60, period=60)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pnömotoraks Tespit API",
    description="TÜBİTAK 2209-A — Derin öğrenme tabanlı pnömotoraks tespit sistemi.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Model config ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "results/checkpoints"))
N_FOLDS  = int(os.getenv("N_FOLDS", "5"))
IMG_SIZE = 512
CLS_T    = float(os.getenv("CLS_THRESHOLD", str(CLS_THRESHOLD)))
SEG_T    = float(os.getenv("SEG_THRESHOLD", str(SEG_THRESHOLD)))

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _device = torch.device("cuda")
elif torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")

logger.info(f"Device: {_device}")

# ThreadPool avoids blocking FastAPI event loop during GPU inference.
# max_workers=1: serializes requests to prevent GPU OOM on single-GPU setups.
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# Prevents GPU memory race condition when multiple requests arrive simultaneously.
_model_lock = threading.Lock()

# ── Model singletons ──────────────────────────────────────────────────────────
_ensemble: torch.nn.Module | None = None
_gradcam_model: PneumothoraxModel | None = None


class _EnsembleModel(torch.nn.Module):
    def __init__(self, models: list[PneumothoraxModel]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor):
        seg_acc = cls_acc = None
        for m in self.models:
            out = m(x)                  # tek çağrı
            seg, cls = out[0], out[1]
            seg_acc = seg if seg_acc is None else seg_acc + seg
            cls_acc = cls if cls_acc is None else cls_acc + cls
        n = len(self.models)
        return seg_acc / n, cls_acc / n


def _load_fold(fold_i: int) -> PneumothoraxModel:
    path = CHECKPOINT_DIR / f"fold_{fold_i}_best.pth"
    if not path.exists():
        raise RuntimeError(f"Checkpoint bulunamadı: {path}")
    state = torch.load(str(path), map_location=_device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    has_aux = any("aux_head" in k for k in state)
    m = PneumothoraxModel(
        encoder_name=os.getenv("MODEL_ENCODER", "efficientnet-b2"),
        in_channels=1,
        deep_supervision=has_aux,
    )
    m.load_state_dict(state)
    return m.eval().to(_device)


def get_ensemble() -> torch.nn.Module:
    global _ensemble
    if _ensemble is None:
        models = [_load_fold(i) for i in range(1, N_FOLDS + 1)]
        _ensemble = _EnsembleModel(models).to(_device)
        _ensemble.eval()
        logger.info("Ensemble model yüklendi")
    return _ensemble


def get_gradcam_model() -> PneumothoraxModel:
    global _gradcam_model
    if _gradcam_model is None:
        _gradcam_model = _load_fold(N_FOLDS)
        logger.info("GradCAM model yüklendi")
    return _gradcam_model


# ── Blocking inference (runs in thread pool) ──────────────────────────────────

def _inference_sync(gray: np.ndarray):
    """Thread-safe GPU inference. Called via run_in_executor."""
    with _model_lock:
        ensemble = get_ensemble()
        resized  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        tensor   = normalize_to_tensor(resized, device=_device)
        with torch.inference_mode():
            seg_pred, cls_pred = ensemble(tensor)
        cls_prob = float(torch.sigmoid(cls_pred).item())
        seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
    return seg_mask, cls_prob


def _tta_sync(gray: np.ndarray):
    """Thread-safe TTA inference."""
    with _model_lock:
        ensemble = get_ensemble()
        result   = predict_tta(ensemble, gray, img_size=IMG_SIZE, seg_threshold=SEG_T)
    return result


def _gradcam_sync(gray: np.ndarray):
    """Thread-safe GradCAM."""
    with _model_lock:
        model = get_gradcam_model()
        return generate_gradcam_result(model, gray, img_size=IMG_SIZE)


async def _run_in_pool(fn, *args):
    """Awaitable wrapper — runs blocking fn in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_inference_executor, fn, *args)


# ── Upload helpers ─────────────────────────────────────────────────────────────
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_ALLOWED_EXT      = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
_DICOM_EXT        = {".dcm", ".dicom"}


def _validate_upload(upload: UploadFile) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"Desteklenmeyen uzantı '{ext}'. İzin: {sorted(_ALLOWED_EXT)}")
    return "dicom" if ext in _DICOM_EXT else "raster"


async def read_upload_as_gray(upload: UploadFile) -> np.ndarray:
    """Async file read → uint8 grayscale ndarray."""
    file_type = _validate_upload(upload)
    content   = await upload.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Dosya çok büyük. Max: {_MAX_UPLOAD_BYTES // (1024*1024)} MB")
    if len(content) == 0:
        raise HTTPException(400, "Boş dosya.")

    if file_type == "dicom":
        import pydicom, tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            try:
                ds = pydicom.dcmread(tmp_path)
            except Exception as e:
                raise HTTPException(422, f"Geçersiz DICOM: {e}")
            if not hasattr(ds, "pixel_array"):
                raise HTTPException(422, "DICOM piksel verisi bulunamadı.")
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                arr = np.clip(arr, -1500, 500)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return arr
        finally:
            os.unlink(tmp_path)

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(422, "Bozuk veya geçersiz görüntü.")
    arr = np.array(pil_img, dtype=np.uint8)
    if arr.size == 0:
        raise HTTPException(422, "Boş görüntü içeriği.")
    logger.info(f"Upload: {upload.filename} shape={arr.shape}")
    return arr


def ndarray_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — lower is better (0 = perfectly calibrated)."""
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_conf - bin_acc)
    return float(ece / len(probs))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    fold_status = {f"fold_{i}": (CHECKPOINT_DIR / f"fold_{i}_best.pth").exists()
                   for i in range(1, N_FOLDS + 1)}
    return {
        "status":         "ok",
        "ensemble_ready": all(fold_status.values()),
        "folds":          fold_status,
        "device":         str(_device),
        "cls_threshold":  CLS_T,
        "seg_threshold":  SEG_T,
    }


@app.get("/metrics")
async def get_metrics():
    """
    Aggregated model performance metrics.
    Loads from output/cv_results.json (new training) or
    results/ptx_local_kfold.csv (legacy).
    Prometheus-compatible structure included.
    """
    # Try new training results first
    result_paths = [
        Path("output/cv_results.json"),
        Path(os.getenv("OUTPUT_DIR", "output")) / "cv_results.json",
    ]
    csv_paths = [
        Path("results/ptx_local_kfold.csv"),
        Path("results/global_kfold_results.csv"),
    ]

    metrics_data: dict = {}

    for p in result_paths:
        if p.exists():
            with open(p) as f:
                cv = json.load(f)
            # Aggregate across folds
            all_vals: dict[str, list] = {}
            for fold_metrics in cv.values():
                for k, v in fold_metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        all_vals.setdefault(k, []).append(float(v))
            metrics_data = {
                k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                for k, v in all_vals.items()
            }
            metrics_data["source"] = str(p)
            break

    if not metrics_data:
        for p in csv_paths:
            if p.exists():
                rows = []
                with open(p, newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        rows.append({k: _safe_float(v) for k, v in row.items()})
                num_keys = [k for k, v in rows[0].items() if isinstance(v, float)] if rows else []
                metrics_data = {
                    k: {
                        "mean": round(float(np.mean([r[k] for r in rows if isinstance(r.get(k), float)])), 4),
                        "std":  round(float(np.std( [r[k] for r in rows if isinstance(r.get(k), float)])), 4),
                    }
                    for k in num_keys
                }
                metrics_data["source"] = str(p)
                break

    # Prometheus-compatible gauge list
    prometheus = []
    for name, vals in metrics_data.items():
        if isinstance(vals, dict) and "mean" in vals:
            prometheus.append({
                "name":  f"ptx_{name}_mean",
                "value": vals["mean"],
                "type":  "gauge",
            })

    return JSONResponse({
        "aggregated":    metrics_data,
        "thresholds":    {"cls": CLS_T, "seg": SEG_T},
        "device":        str(_device),
        "prometheus":    prometheus,
    })


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    _rate: None = Depends(_inference_limit),
):
    try:
        get_ensemble()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    try:
        gray = await read_upload_as_gray(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Görüntü okunamadı: {e}")

    orig_h, orig_w = gray.shape[:2]

    # Non-blocking inference
    try:
        seg_mask, cls_prob = await _run_in_pool(_inference_sync, gray)
    except Exception as e:
        raise HTTPException(500, f"Model hatası: {e}")

    # GradCAM (non-blocking)
    try:
        gradcam_bgr, _ = await _run_in_pool(_gradcam_sync, gray)
    except Exception as e:
        logger.warning(f"GradCAM başarısız: {e}")
        gradcam_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    seg_resized  = cv2.resize(seg_mask, (orig_w, orig_h))
    binary_mask  = (seg_resized > SEG_T).astype(np.uint8) * 255
    seg_overlay  = overlay_mask_on_image(gray, binary_mask, alpha=0.4, color_bgr=(0, 200, 0))
    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask_ratio = float(binary_mask.sum()) / max(255 * orig_h * orig_w, 1)
    has_ptx    = (cls_prob >= CLS_T) and (mask_ratio >= 0.005)

    logger.info(f"/predict | {file.filename} | prob={cls_prob:.4f} mask={mask_ratio:.4f} ptx={has_ptx}")

    diagnosis = (
        f"PNÖMOTORAKS TESPİT EDİLDİ (olasılık: {cls_prob:.1%})"
        if has_ptx else
        f"Normal — Pnömotoraks saptanmadı (olasılık: {cls_prob:.1%})"
    )

    return JSONResponse({
        "has_pneumothorax":   has_ptx,
        "probability":        round(cls_prob, 4),
        "diagnosis":          diagnosis,
        "original_image":     ndarray_to_base64(original_bgr),
        "gradcam_image":      ndarray_to_base64(gradcam_bgr),
        "segmentation_image": ndarray_to_base64(seg_overlay),
    })


@app.post("/predict/tta")
async def predict_with_tta(
    request: Request,
    file: UploadFile = File(...),
    _rate: None = Depends(_inference_limit),
):
    try:
        get_ensemble()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    try:
        gray = await read_upload_as_gray(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Görüntü okunamadı: {e}")

    try:
        tta_result = await _run_in_pool(_tta_sync, gray)
    except Exception as e:
        raise HTTPException(500, f"TTA hatası: {e}")

    prob       = tta_result["prob_mean"]
    seg_bin    = tta_result["seg_binary"]
    mask_ratio = float(seg_bin.sum()) / max(255 * gray.shape[0] * gray.shape[1], 1)
    has_ptx    = (prob >= CLS_T) and (mask_ratio >= 0.005)

    seg_overlay  = overlay_mask_on_image(gray, seg_bin, alpha=0.4, color_bgr=(0, 200, 0))
    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    try:
        gradcam_bgr, _ = await _run_in_pool(_gradcam_sync, gray)
    except Exception as e:
        logger.warning(f"TTA GradCAM başarısız: {e}")
        gradcam_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    diagnosis = (
        f"PNÖMOTORAKS TESPİT EDİLDİ — TTA: {prob:.1%}"
        if has_ptx else
        f"Normal — TTA: {prob:.1%} (maske: {mask_ratio:.1%})"
    )

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


@app.get("/results")
async def get_results():
    csv_paths = [
        Path("output/cv_results.json"),
        Path("results/global_kfold_results.csv"),
        Path("results/ptx_local_kfold.csv"),
    ]

    for p in csv_paths:
        if not p.exists():
            continue
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            rows = [{"fold": k, **v} for k, v in data.items()]
        else:
            rows = []
            with open(p, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    rows.append({k: _safe_float(v) for k, v in row.items()})

        if not rows:
            continue
        num_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
        summary  = {
            k: {"mean": round(float(np.mean([r[k] for r in rows if isinstance(r.get(k), float)])), 4),
                "std":  round(float(np.std( [r[k] for r in rows if isinstance(r.get(k), float)])), 4)}
            for k in num_keys
        }
        return JSONResponse({"folds": rows, "summary": summary, "source": str(p)})

    raise HTTPException(404, "Sonuç dosyası bulunamadı. Eğitim tamamlandıktan sonra tekrar deneyin.")
