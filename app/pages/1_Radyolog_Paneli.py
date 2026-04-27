"""
Klinik Karar Destek Sistemi v2.0 — Radyolog Paneli

Pnömotoraks Otomatik Tespit ve Analiz Dashboard'u

Bölümler:
  ① KPI Kartlar   : Olasılık · Güven Skoru · Tahmini Alan
  ② Görsel Analiz : Orijinal · Segmentasyon Overlay · Grad-CAM
  ③ Analitikler   : Risk Gauge · TTA Dağılımı · Piksel Histogramı · Alan Kıyası
  ④ DICOM Meta    : Hasta / Teknik parametreler
  ⑤ Rapor         : Otomatik bulgular metni + PDF / HTML / TXT indirme

Başlatma:
    streamlit run app/streamlit_app.py

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
Dokuz Eylül Üniversitesi Tıp Fakültesi
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje kökünü Python yoluna ekle (src modülünün bulunması için)
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import datetime
import io
import os
import tempfile

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from src.model.unet import PneumothoraxModel
from src.utils.gradcam import generate_gradcam_result, generate_integrated_gradients
from src.utils.tta import predict_tta, uncertainty_label

# ── Opsiyonel bağımlılıklar ───────────────────────────────────────────────────

try:
    from fpdf import FPDF as _FPDF_CLS
    _FPDF_OK = True
except ImportError:
    _FPDF_OK = False

# ── Sabitler ──────────────────────────────────────────────────────────────────

ENCODER_OPTIONS: dict[str, str] = {
    "EfficientNet-B0  ⚡ Hızlı":   "efficientnet-b0",
    "EfficientNet-B2  ⚖️ Dengeli": "efficientnet-b2",
    "EfficientNet-B4  🎯 Hassas":  "efficientnet-b4",
}

CHECKPOINT_DIR    = Path("results/checkpoints")
IMG_SIZE          = 512
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Varsayılan piksel boyutu — PA grafisi 512 px ≈ 42 cm → 0.082 cm/px
DEFAULT_PS_CM     = 0.082
TTA_LABELS        = ["Orijinal", "H-Flip", "Parlak+", "Parlak−", "Kontrast+"]

# Referans klinik vakalar (alan karşılaştırması için)
REF_CASES = {
    "Minimal (<2 cm²)":    1.0,
    "Hafif (2–10 cm²)":    5.8,
    "Orta (10–25 cm²)":   17.2,
    "Geniş (>25 cm²)":    31.5,
}

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pnömotoraks Klinik Analiz",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── KPI kart ─────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
    margin-bottom: 8px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}
.kpi-icon  { font-size: 2rem; margin-bottom: 6px; }
.kpi-value { font-size: 2.1rem; font-weight: 800; margin: 4px 0; }
.kpi-label { font-size: 0.78rem; color: rgba(255,255,255,0.50);
             text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-sub   { font-size: 0.75rem; color: rgba(255,255,255,0.32); margin-top: 5px; }

/* ── Teşhis banner'ı ──────────────────────────────────────────────── */
.dx-banner {
    border-radius: 10px;
    padding: 14px 24px;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 12px 0 16px 0;
    letter-spacing: 0.5px;
}

/* ── Panel başlığı ────────────────────────────────────────────────── */
.panel-title {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #4a9eff;
    border-radius: 0 6px 6px 0;
    padding: 6px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: rgba(255,255,255,0.80);
}

/* ── Sidebar branding ──────────────────────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 10px 0 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Model cache ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(encoder_name: str) -> PneumothoraxModel | None:
    """Encoder adına göre checkpoint'i yükler."""
    ckpt = CHECKPOINT_DIR / f"best_{encoder_name}.pth"
    if not ckpt.exists():
        ckpt = CHECKPOINT_DIR / "best_model.pth"    # genel fallback
    if not ckpt.exists():
        return None
    model = PneumothoraxModel(encoder_name=encoder_name)
    model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    return model.eval().to(DEVICE)


# ── Dosya okuyucu ─────────────────────────────────────────────────────────────

def file_to_gray_and_meta(
    uploaded_file,
) -> tuple[np.ndarray, tuple[int, int], dict, float]:
    """
    Yüklenen DICOM / PNG / JPEG dosyasını okur.

    Returns
    -------
    gray        : uint8 grayscale ndarray (orijinal boyut)
    orig_shape  : (H, W)
    dicom_meta  : dict — DICOM ise hasta bilgileri, değilse {}
    ps_cm       : float — cm/piksel (DICOM'dan veya varsayılan)
    """
    name  = uploaded_file.name.lower()
    data  = uploaded_file.read()
    meta  = {}
    ps_cm = DEFAULT_PS_CM

    if name.endswith((".dcm", ".dicom")):
        import pydicom

        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            ds  = pydicom.dcmread(tmp_path)
            arr = ds.pixel_array.astype(np.float32)
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            gray = arr[0] if arr.ndim == 3 else arr

            # PixelSpacing (mm → cm)
            raw_ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
            if raw_ps is not None:
                ps_cm = float(raw_ps[0]) / 10.0

            def _get(tag: str, default: str = "Bilinmiyor") -> str:
                v = getattr(ds, tag, None)
                return str(v).strip() if v not in (None, "") else default

            meta = {
                "Hasta Adı":           _get("PatientName"),
                "Hasta Yaşı":          _get("PatientAge"),
                "Cinsiyet":            _get("PatientSex"),
                "Modalite":            _get("Modality"),
                "Çekim Tarihi":        _get("StudyDate"),
                "Çekim Saati":         _get("StudyTime"),
                "Kurum":               _get("InstitutionName"),
                "Cihaz Üreticisi":     _get("Manufacturer"),
                "kVp":                 _get("KVP"),
                "Piksel Aralığı (mm)": str(raw_ps) if raw_ps else "N/A",
                "Görüntü Boyutu (px)": f"{gray.shape[1]} × {gray.shape[0]}",
                "Bit Derinliği":       str(getattr(ds, "BitsStored", "N/A")),
            }
        finally:
            os.unlink(tmp_path)
    else:
        pil  = Image.open(io.BytesIO(data)).convert("L")
        gray = np.array(pil, dtype=np.uint8)

    return gray, (gray.shape[0], gray.shape[1]), meta, ps_cm


# ── Çıkarım ───────────────────────────────────────────────────────────────────

def run_inference(
    model: PneumothoraxModel,
    gray: np.ndarray,
    use_tta: bool,
    threshold: float,
) -> dict:
    """TTA veya standart çıkarım. Tüm sonuçları dict olarak döner."""
    if use_tta:
        r   = predict_tta(model, gray, img_size=IMG_SIZE, seg_threshold=threshold)
        prob    = float(r["prob_mean"])
        seg_bin = r["seg_binary"]       # 512×512, 0/255
        std     = float(r["prob_std"])
        votes   = list(r["prob_votes"])
        uncert  = uncertainty_label(r)
        confidence = max(0.0, min(1.0, 1.0 - std * 5))
    else:
        resized   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        tensor    = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            seg_pred, cls_pred = model(tensor)
        prob       = float(torch.sigmoid(cls_pred).item())
        seg_np     = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
        seg_bin    = ((seg_np > threshold) * 255).astype(np.uint8)
        std        = None
        votes      = None
        uncert     = None
        # Uzaklık tabanlı güven: prob=0 ya da 1 → tam güven
        confidence = 2.0 * abs(prob - 0.5)

    return {
        "prob":       prob,
        "seg_bin":    seg_bin,     # 512×512, uint8 (0/255)
        "std":        std,
        "votes":      votes,
        "uncert":     uncert,
        "confidence": confidence,
        "has_ptx":    prob >= threshold,
    }


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def compute_area_cm2(
    seg_bin_512: np.ndarray,
    orig_shape: tuple[int, int],
    ps_cm: float,
) -> float:
    """Segmentasyon maskesindeki pnömotoraks alanını cm² hesaplar."""
    H, W    = orig_shape
    seg_orig = cv2.resize(seg_bin_512, (W, H), interpolation=cv2.INTER_NEAREST)
    n_pos    = int((seg_orig > 0).sum())
    scale_h  = H / IMG_SIZE
    scale_w  = W / IMG_SIZE
    px_area  = (ps_cm * scale_h) * (ps_cm * scale_w)   # cm²/piksel
    return n_pos * px_area


def make_overlay(
    gray: np.ndarray,
    seg_bin_512: np.ndarray,
    orig_shape: tuple[int, int],
    color_bgr: tuple = (0, 210, 90),
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Segmentasyon maskesini orijinal görüntüye bindirip kontur çizer.
    RGB ndarray döner.
    """
    H, W     = orig_shape
    seg_orig = cv2.resize(seg_bin_512, (W, H), interpolation=cv2.INTER_NEAREST)
    bgr      = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Yarı şeffaf renk katmanı
    colored           = bgr.copy()
    colored[seg_orig > 0] = color_bgr
    blended = cv2.addWeighted(colored, alpha, bgr, 1.0 - alpha, 0)

    # Kontur çizgisi
    contours, _ = cv2.findContours(seg_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 170), 2)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def determine_lung_side(seg_bin_512: np.ndarray) -> str:
    """PA grafide etkilenen tarafı belirler (centroid bazlı)."""
    if seg_bin_512.sum() == 0:
        return "Belirsiz"
    m = cv2.moments(seg_bin_512)
    if m["m00"] == 0:
        return "Belirsiz"
    cx = int(m["m10"] / m["m00"])
    # PA pozisyonu: imajın sol yarısı → sağ akciğer
    return "Sağ Akciğer" if cx < IMG_SIZE // 2 else "Sol Akciğer"


def classify_severity(area_cm2: float) -> tuple[str, str, str]:
    """(şiddet etiketi, emoji, klinik öneri) döner."""
    if area_cm2 < 2.0:
        return (
            "Minimal",
            "🟡",
            "Yakın klinik takip önerilir. 24 saat içinde kontrol grafisi planlanmalı.",
        )
    elif area_cm2 < 10.0:
        return (
            "Hafif",
            "🟠",
            "Göğüs cerrahisi konsültasyonu önerilir. Semptomatik tedavi değerlendirilmeli.",
        )
    elif area_cm2 < 25.0:
        return (
            "Orta Derece",
            "🔴",
            "Acil göğüs cerrahisi konsültasyonu gereklidir. Tüp torakostomi düşünülmeli.",
        )
    else:
        return (
            "Geniş (Tension Riski)",
            "🚨",
            "ACİL müdahale! İğne dekompresyonu veya tüp torakostomi derhal uygulanmalı.",
        )


# ── Normal PDF (scipy olmadan) ────────────────────────────────────────────────

def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Gauss yoğunluk fonksiyonu — scipy bağımlılığı olmadan."""
    std = max(std, 1e-6)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


# ── Plotly grafikleri ─────────────────────────────────────────────────────────

_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#ccc"},
    margin=dict(l=50, r=20, t=50, b=40),
    height=265,
)


def fig_gauge(prob: float, threshold: float) -> go.Figure:
    """Pnömotoraks riski göstergesi."""
    clr = "#e74c3c" if prob >= threshold else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": clr}},
        title={"text": "Pnömotoraks Riski", "font": {"size": 13, "color": "#aab"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#555",
                "tickfont": {"color": "#888"},
            },
            "bar": {"color": clr, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   30],  "color": "rgba(46,204,113,0.12)"},
                {"range": [30,  60],  "color": "rgba(243,156,18,0.12)"},
                {"range": [60, 100],  "color": "rgba(231,76,60,0.12)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(**_PLOT_LAYOUT, height=240, margin=dict(l=30, r=30, t=40, b=10))
    return fig


def fig_tta_distribution(
    votes: list[float],
    mean: float,
    std: float,
    threshold: float,
) -> go.Figure:
    """TTA tahmin dağılımı — çan eğrisi + bireysel oylar."""
    vis_std = max(std, 0.025)     # görsel stabilite için minimum genişlik
    x       = np.linspace(0.0, 1.0, 400)
    y       = _normal_pdf(x, mean, vis_std)

    fig = go.Figure()

    # Çan eğrisi
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        fill="tozeroy",
        line={"color": "#4a9eff", "width": 2},
        fillcolor="rgba(74,158,255,0.12)",
        name="Dağılım",
        showlegend=False,
    ))

    # Bireysel TTA tahminleri
    marker_colors = ["#f39c12", "#e74c3c", "#2ecc71", "#9b59b6", "#1abc9c"]
    for i, (v, lbl) in enumerate(zip(votes, TTA_LABELS)):
        y_pt = float(_normal_pdf(np.array([v]), mean, vis_std)[0])
        fig.add_trace(go.Scatter(
            x=[v], y=[y_pt],
            mode="markers+text",
            marker={"size": 11, "color": marker_colors[i], "symbol": "diamond",
                    "line": {"color": "white", "width": 1}},
            text=[lbl], textposition="top center",
            textfont={"size": 9, "color": "#ddd"},
            name=lbl, showlegend=False,
        ))

    # Karar eşiği
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Eşik {threshold:.2f}",
        annotation_font={"color": "#aaa", "size": 10},
    )
    # Ortalama
    fig.add_vline(
        x=mean, line_dash="solid", line_color="#4a9eff",
        annotation_text=f"Ort {mean:.3f}",
        annotation_font={"color": "#4a9eff", "size": 10},
        annotation_position="top left",
    )

    fig.update_layout(
        **_PLOT_LAYOUT,
        title={"text": f"TTA Tahmin Dağılımı  (std={std:.4f})", "font": {"size": 12}},
        xaxis={"title": "Pnömotoraks Olasılığı", "range": [0, 1],
               "gridcolor": "#2a2a2a", "color": "#aaa"},
        yaxis={"title": "Yoğunluk", "gridcolor": "#2a2a2a", "color": "#aaa"},
    )
    return fig


def fig_histogram(
    gray: np.ndarray,
    seg_bin_512: np.ndarray,
    orig_shape: tuple[int, int],
) -> go.Figure:
    """Sağlıklı vs pnömotoraks bölgesi piksel yoğunluk histogramı."""
    H, W     = orig_shape
    seg_orig = cv2.resize(seg_bin_512, (W, H), interpolation=cv2.INTER_NEAREST)
    mask     = seg_orig > 0

    ptx_px  = gray[mask].flatten().astype(float)
    norm_px = gray[~mask].flatten().astype(float)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ptx_px, nbinsx=64,
        name="Pnömotoraks Bölgesi",
        marker_color="rgba(231,76,60,0.78)",
        opacity=0.85, histnorm="percent",
    ))
    fig.add_trace(go.Histogram(
        x=norm_px, nbinsx=64,
        name="Normal Doku",
        marker_color="rgba(46,204,113,0.55)",
        opacity=0.75, histnorm="percent",
    ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        barmode="overlay",
        title={"text": "Piksel Yoğunluk Dağılımı (Bölgesel)", "font": {"size": 12}},
        xaxis={"title": "Piksel Değeri (0–255)", "gridcolor": "#2a2a2a", "color": "#aaa"},
        yaxis={"title": "Oran (%)", "gridcolor": "#2a2a2a", "color": "#aaa"},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
    )
    return fig


def fig_area_comparison(area_cm2: float) -> go.Figure:
    """Bu vakanın alanını referans klinik vakalarla kıyaslar."""
    labels = list(REF_CASES.keys()) + ["Bu Vaka  ▶"]
    values = list(REF_CASES.values()) + [round(area_cm2, 2)]
    colors = ["#4e5d78", "#f39c12", "#e74c3c", "#922b21", "#4a9eff"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v} cm²" for v in values],
        textposition="outside",
        textfont={"color": "#ddd", "size": 11},
    ))
    fig.add_hline(
        y=area_cm2,
        line_dash="dot", line_color="#4a9eff",
        annotation_text=f"Bu vaka: {area_cm2:.1f} cm²",
        annotation_font={"color": "#4a9eff", "size": 10},
    )
    fig.update_layout(
        **_PLOT_LAYOUT,
        title={"text": "Pnömotoraks Alanı — Klinik Referans Karşılaştırması",
               "font": {"size": 12}},
        xaxis={"gridcolor": "#2a2a2a", "color": "#aaa"},
        yaxis={"title": "Alan (cm²)", "gridcolor": "#2a2a2a", "color": "#aaa",
               "range": [0, max(max(values) * 1.25, 5)]},
    )
    return fig


# ── KPI kart HTML ─────────────────────────────────────────────────────────────

def kpi_html(icon: str, value: str, label: str, sub: str, color: str) -> str:
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-icon'>{icon}</div>"
        f"<div class='kpi-value' style='color:{color}'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>"
    )


# ── Rapor üretimi ─────────────────────────────────────────────────────────────

def _tr_to_ascii(text: str) -> str:
    """fpdf2 için Türkçe karakterleri ASCII'ye çevirir."""
    table = str.maketrans("şğıöüçŞĞİÖÜÇ", "sgiouscSGIOUC")
    return text.translate(table)


def generate_report_text(
    prob: float,
    area_cm2: float,
    confidence: float,
    has_ptx: bool,
    lung_side: str,
    severity: str,
    recommendation: str,
    encoder_name: str,
    use_tta: bool,
    std: float | None,
    dicom_meta: dict,
) -> str:
    """Radyoloji raporu formatında otomatik metin."""
    now  = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    mode = "TTA (5 varyant)" if use_tta else "Standart Tek Geçiş"

    lines = [
        "═" * 60,
        "  PNÖMOTORAKS YAPAY ZEKA ANALİZ RAPORU",
        "  Dokuz Eylül Üniversitesi — TÜBİTAK 2209-A",
        "═" * 60,
        "",
        f"  Analiz Tarihi  : {now}",
        f"  Model          : U-Net++ + {encoder_name}",
        f"  Çıkarım Modu   : {mode}",
    ]

    if dicom_meta:
        lines += [
            "",
            "  HASTA BİLGİLERİ",
            "  " + "─" * 44,
            f"  Hasta Adı  : {dicom_meta.get('Hasta Adı', 'N/A')}",
            f"  Yaş        : {dicom_meta.get('Hasta Yaşı', 'N/A')}",
            f"  Cinsiyet   : {dicom_meta.get('Cinsiyet', 'N/A')}",
            f"  Çekim Tar. : {dicom_meta.get('Çekim Tarihi', 'N/A')}",
            f"  Kurum      : {dicom_meta.get('Kurum', 'N/A')}",
        ]

    lines += [
        "",
        "  BULGULAR",
        "  " + "─" * 44,
    ]

    if has_ptx:
        lines += [
            f"  {lung_side} apikal bölgede pnömotoraks ile uyumlu görünüm",
            "  izlenmiştir. Plevral hat seçilmekte, plevral alanda hava",
            "  dansitesi dikkati çekmektedir. Akciğer parankiminde",
            "  kollaps bulguları mevcut olup klinik korelasyon önerilir.",
        ]
    else:
        lines += [
            "  Her iki akciğer alanı açık görünmektedir.",
            "  Belirgin pnömotoraks bulgusu saptanmamıştır.",
            "  Plevral hat normal sınırlarda izlenmektedir.",
        ]

    lines += [
        "",
        "  ÖLÇÜMLER",
        "  " + "─" * 44,
        f"  Pnömotoraks Olasılığı : %{prob * 100:.1f}",
        f"  Güven Skoru           : %{confidence * 100:.0f}",
        f"  Tahmini Alan          : {area_cm2:.2f} cm²",
        f"  Şiddet Sınıfı         : {severity}",
    ]

    if std is not None:
        lines.append(f"  TTA Std (belirsizlik) : {std:.4f}")

    if has_ptx:
        lines += [
            "",
            "  KLİNİK ÖNERİ",
            "  " + "─" * 44,
            f"  {recommendation}",
        ]

    lines += [
        "",
        "═" * 60,
        "  ⚠  UYARI",
        "  Bu rapor yapay zeka destekli otomatik analiz sistemi",
        "  tarafından üretilmiştir. Kesin tanı ve tedavi kararı",
        "  için uzman hekim değerlendirmesi zorunludur.",
        "═" * 60,
        "",
        "  TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu",
        "  Dokuz Eylül Üniversitesi Tıp Fakültesi Hastanesi",
    ]

    return "\n".join(lines)


def generate_pdf_bytes(report_text: str) -> bytes | None:
    """fpdf2 ile PDF üretir. Kurulu değilse None döner."""
    if not _FPDF_OK:
        return None
    try:
        pdf = _FPDF_CLS()
        pdf.set_margins(20, 20, 20)
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Başlık
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(10, 50, 120)
        pdf.cell(0, 10, "PNOMOTORAKS YAPAY ZEKA ANALIZ RAPORU", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "Dokuz Eylul Universitesi | TUBITAK 2209-A", ln=True, align="C")
        pdf.ln(6)

        # Rapor gövdesi
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(40, 40, 40)
        safe = _tr_to_ascii(report_text)
        for line in safe.split("\n"):
            pdf.multi_cell(0, 5, line)

        return bytes(pdf.output())
    except Exception:
        return None


def generate_html_report(report_text: str) -> str:
    """Tarayıcıda PDF olarak yazdırılabilen HTML şablonu."""
    safe = (report_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Pnömotoraks Analiz Raporu</title><style>"
        "body{font-family:'Courier New',monospace;font-size:13px;"
        "background:#fff;color:#222;padding:40px;max-width:720px;margin:auto}"
        "pre{white-space:pre-wrap;line-height:1.7}"
        "h1{color:#0a3278;border-bottom:2px solid #0a3278;padding-bottom:8px}"
        "</style></head><body>"
        "<h1>&#x1FAC1; Pnömotoraks Analiz Raporu</h1>"
        f"<pre>{safe}</pre>"
        "</body></html>"
    )


# ── Ana arayüz ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sayfa başlığı ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;padding:8px 0 2px 0'>"
        "<h1 style='font-size:2rem;margin-bottom:4px'>🫁 Pnömotoraks Klinik Analiz Sistemi</h1>"
        "<p style='color:rgba(160,190,255,0.55);font-size:0.85rem;letter-spacing:1.5px'>"
        "TÜBİTAK 2209-A &nbsp;·&nbsp; U-Net++ + EfficientNet"
        " &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp; TTA"
        " &nbsp;·&nbsp; Dokuz Eylül Üniversitesi"
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Kenar çubuğu ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'>"
            "<div style='font-size:2.5rem'>🫁</div>"
            "<div style='font-weight:700;font-size:1rem'>Pnömotoraks AI</div>"
            "<div style='font-size:0.72rem;color:#888;margin-top:3px'>"
            "TÜBİTAK 2209-A &nbsp;|&nbsp; DEÜ</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**⚙️ Analiz Ayarları**")

        encoder_label = st.selectbox(
            "🧠 Model Seçimi",
            list(ENCODER_OPTIONS.keys()),
            help="Encoder mimarisi. B0 hızlı, B4 hassas.",
        )
        encoder_name = ENCODER_OPTIONS[encoder_label]

        use_tta = st.toggle(
            "🔁 Test-Time Augmentation (TTA)",
            value=True,
            help="5 farklı görüş açısından analiz — daha güvenilir sonuç.",
        )
        threshold = st.slider(
            "📊 Karar Eşiği",
            0.10, 0.90, 0.50, 0.05,
            help="Bu değerin üzerindeki olasılıklar Pnömotoraks kabul edilir.",
        )
        show_ig = st.checkbox(
            "🔬 Integrated Gradients",
            help="Piksel katkı haritası (daha yavaş — Captum gerektirir).",
        )

        st.divider()
        st.caption("**Format:** PNG · JPEG · DICOM")
        st.caption(f"**Cihaz:** `{DEVICE}`")
        if not _FPDF_OK:
            st.caption("💡 PDF için: `pip install fpdf2`")

    # ── Dosya yükleme ──────────────────────────────────────────────────────────
    st.subheader("① Akciğer Grafisi Yükle")
    uploaded = st.file_uploader(
        "PNG, JPEG veya DICOM seçin",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info(
            "🫁  Akciğer grafisi yüklemek için yukarıdaki alanı kullanın.\n\n"
            "**Desteklenen formatlar:** PNG · JPEG · DICOM (.dcm)"
        )
        return

    # Dosyayı oku
    with st.spinner("Dosya okunuyor…"):
        gray, orig_shape, dicom_meta, ps_cm = file_to_gray_and_meta(uploaded)

    # Model yükle
    model = load_model(encoder_name)
    if model is None:
        st.warning(
            f"⚠️  Model bulunamadı.\n\n"
            f"Beklenen: `{CHECKPOINT_DIR}/best_model.pth`\n\n"
            "Eğitin: `python -m src.utils.train`"
        )
        st.image(gray, caption="Yüklenen görüntü (model yok)", use_container_width=True, clamp=True)
        return

    # ── Çıkarım ───────────────────────────────────────────────────────────────
    with st.spinner("Model analiz yapıyor…"):
        res       = run_inference(model, gray, use_tta, threshold)
        area_cm2  = compute_area_cm2(res["seg_bin"], orig_shape, ps_cm)
        lung_side = determine_lung_side(res["seg_bin"])
        sev_lbl, sev_icon, recommendation = classify_severity(area_cm2)

    has_ptx    = res["has_ptx"]
    prob       = res["prob"]
    confidence = res["confidence"]

    # ── Teşhis banner ─────────────────────────────────────────────────────────
    if has_ptx:
        banner_bg  = "rgba(192,57,43,0.18)"
        banner_brd = "#c0392b"
        banner_clr = "#e74c3c"
        banner_txt = (
            f"🔴 PNÖMOTORAKS TESPİT EDİLDİ"
            f"  ·  {lung_side}"
            f"  ·  {sev_icon} {sev_lbl}"
        )
    else:
        banner_bg  = "rgba(26,107,60,0.18)"
        banner_brd = "#1a6b3c"
        banner_clr = "#2ecc71"
        banner_txt = "🟢  NORMAL — Belirgin Pnömotoraks Bulgusu Saptanmadı"

    st.markdown(
        f"<div class='dx-banner' "
        f"style='background:{banner_bg};border:2px solid {banner_brd};color:{banner_clr}'>"
        f"{banner_txt}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI Kartlar ───────────────────────────────────────────────────────────
    kc1, kc2, kc3 = st.columns(3)

    prob_color = "#e74c3c" if prob >= threshold else "#2ecc71"
    conf_color = (
        "#e74c3c" if confidence < 0.70 else
        "#f39c12" if confidence < 0.85 else
        "#2ecc71"
    )

    with kc1:
        st.markdown(kpi_html(
            icon="🎯",
            value=f"{prob * 100:.1f}%",
            label="Pnömotoraks Olasılığı",
            sub=f"Eşik: {threshold:.0%}",
            color=prob_color,
        ), unsafe_allow_html=True)

    with kc2:
        conf_sub = res["uncert"] if res["uncert"] else "TTA kapalı"
        st.markdown(kpi_html(
            icon="🔒",
            value=f"{confidence * 100:.0f}%",
            label="Güven Skoru",
            sub=conf_sub,
            color=conf_color,
        ), unsafe_allow_html=True)

    with kc3:
        st.markdown(kpi_html(
            icon="📐",
            value=f"{area_cm2:.1f} cm²",
            label="Tahmini Pnömotoraks Alanı",
            sub=f"{sev_icon} {sev_lbl}",
            color="#4a9eff",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Görsel Panel (3 sütun) ────────────────────────────────────────────────
    st.subheader("② Görsel Analiz")
    vc1, vc2, vc3 = st.columns(3)

    seg_overlay_rgb = make_overlay(gray, res["seg_bin"], orig_shape)

    with vc1:
        st.markdown("<div class='panel-title'>📷 Orijinal X-Ray</div>", unsafe_allow_html=True)
        st.image(gray, use_container_width=True, clamp=True)

    with vc2:
        st.markdown(
            "<div class='panel-title'>🟢 Segmentasyon Maskesi</div>", unsafe_allow_html=True
        )
        st.image(seg_overlay_rgb, use_container_width=True)

    with vc3:
        st.markdown(
            "<div class='panel-title'>🌡️ Grad-CAM Isı Haritası</div>", unsafe_allow_html=True
        )
        with st.spinner("Grad-CAM hesaplanıyor…"):
            try:
                gradcam_bgr, _ = generate_gradcam_result(model, gray, img_size=IMG_SIZE)
                st.image(cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as exc:
                st.warning(f"Grad-CAM üretilemedi: {exc}")

    # Integrated Gradients (isteğe bağlı)
    if show_ig:
        with st.expander("🔬 Integrated Gradients — piksel düzeyinde katkı haritası"):
            with st.spinner("Integrated Gradients hesaplanıyor (50 adım)…"):
                try:
                    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                    tensor  = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    ig_map  = generate_integrated_gradients(model, tensor, n_steps=50)
                    ig_col  = cv2.applyColorMap(
                        (ig_map * 255).astype(np.uint8), cv2.COLORMAP_HOT
                    )
                    st.image(cv2.cvtColor(ig_col, cv2.COLOR_BGR2RGB), use_container_width=True)
                except ImportError:
                    st.warning("`pip install captum` gerekli.")
                except Exception as exc:
                    st.error(f"IG hatası: {exc}")

    # ── Analitik Grafikler ────────────────────────────────────────────────────
    st.subheader("③ Analitik Grafikler")
    tab1, tab2, tab3 = st.tabs([
        "📈 Risk Göstergesi & TTA Dağılımı",
        "🔬 Piksel Yoğunluk Histogramı",
        "📊 Alan Karşılaştırması",
    ])

    with tab1:
        gc1, gc2 = st.columns([1, 1.5])
        with gc1:
            st.plotly_chart(fig_gauge(prob, threshold), use_container_width=True)
        with gc2:
            if res["votes"] is not None:
                st.plotly_chart(
                    fig_tta_distribution(
                        res["votes"], prob, res["std"] or 0.0, threshold
                    ),
                    use_container_width=True,
                )
                # TTA detay tablosu
                tta_table = {
                    lbl: f"{v:.4f}  {'🔴' if v >= threshold else '🟢'}"
                    for lbl, v in zip(TTA_LABELS, res["votes"])
                }
                st.dataframe(tta_table, use_container_width=True)
                st.caption(
                    "Eğer çan eğrisi **dar** ise model tutarlı; "
                    "**geniş** ise farklı görüş açıları arasında anlaşmazlık var."
                )
            else:
                st.info("TTA kapalı — dağılım grafiği için sidebar'dan TTA'yı etkinleştirin.")

    with tab2:
        if res["seg_bin"].sum() > 0:
            st.plotly_chart(
                fig_histogram(gray, res["seg_bin"], orig_shape),
                use_container_width=True,
            )
            st.caption(
                "**🔴 Kırmızı:** Segmentasyon maskesi içi (pnömotoraks bölgesi) — "
                "hava yoğunluğu nedeniyle genellikle **düşük** piksel değerleri.  \n"
                "**🟢 Yeşil:** Maske dışı (normal akciğer dokusu)."
            )
        else:
            st.info("Segmentasyon maskesi boş — histogram üretilemedi.")

    with tab3:
        st.plotly_chart(fig_area_comparison(area_cm2), use_container_width=True)
        st.caption(
            "📌 Referans değerler klinik literatüre dayanmaktadır (yaklaşık).  \n"
            f"📌 Alan hesabı piksel boyutuna göre yapıldı "
            f"({ps_cm:.4f} cm/px — {'DICOM PixelSpacing' if dicom_meta else 'Varsayılan PA'})."
        )

    # ── DICOM Metadata ────────────────────────────────────────────────────────
    if dicom_meta:
        st.subheader("④ DICOM Meta Bilgileri")
        meta_items = list(dicom_meta.items())
        half       = (len(meta_items) + 1) // 2
        mc1, mc2   = st.columns(2)
        with mc1:
            st.table({k: v for k, v in meta_items[:half]})
        with mc2:
            st.table({k: v for k, v in meta_items[half:]})

    # ── Klinik Rapor ──────────────────────────────────────────────────────────
    st.subheader("⑤ Klinik Analiz Raporu")

    report_text = generate_report_text(
        prob=prob,
        area_cm2=area_cm2,
        confidence=confidence,
        has_ptx=has_ptx,
        lung_side=lung_side,
        severity=sev_lbl,
        recommendation=recommendation,
        encoder_name=encoder_name,
        use_tta=use_tta,
        std=res["std"],
        dicom_meta=dicom_meta,
    )

    with st.expander("📋 Rapor Metnini Görüntüle", expanded=True):
        st.code(report_text, language=None)

    # İndirme butonları
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dl1, dl2, dl3, _ = st.columns([1, 1, 1, 1])

    with dl1:
        pdf_bytes = generate_pdf_bytes(report_text)
        if pdf_bytes:
            st.download_button(
                label="📄 PDF İndir",
                data=pdf_bytes,
                file_name=f"pnomotoraks_{ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.download_button(
                label="🌐 HTML İndir",
                data=generate_html_report(report_text).encode("utf-8"),
                file_name=f"pnomotoraks_{ts}.html",
                mime="text/html",
                use_container_width=True,
                help="Tarayıcıda açıp Ctrl+P → PDF olarak kaydedebilirsiniz.",
            )

    with dl2:
        st.download_button(
            label="📝 TXT İndir",
            data=report_text.encode("utf-8"),
            file_name=f"pnomotoraks_{ts}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl3:
        # Segmentasyon maskesini PNG olarak indir
        mask_pil   = Image.fromarray(res["seg_bin"])
        mask_buf   = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        st.download_button(
            label="🖼️ Maske PNG",
            data=mask_buf.getvalue(),
            file_name=f"maske_{ts}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Teknik Detaylar ───────────────────────────────────────────────────────
    with st.expander("🔧 Teknik Detaylar"):
        H_o, W_o   = orig_shape
        seg_o      = cv2.resize(
            res["seg_bin"], (W_o, H_o), interpolation=cv2.INTER_NEAREST
        )
        mask_px    = int((seg_o > 0).sum())
        total_px   = H_o * W_o

        td1, td2 = st.columns(2)
        with td1:
            st.write(f"- **Görüntü boyutu:** {W_o} × {H_o} px")
            st.write(f"- **Piksel boyutu:** {ps_cm:.5f} cm/px")
            st.write(f"- **Encoder:** `{encoder_name}`")
            st.write(f"- **Çıkarım:** {'TTA (5 varyant)' if use_tta else 'Standart'}")
        with td2:
            st.write(f"- **Pnömotoraks pikseli:** {mask_px:,} ({mask_px / total_px:.3%})")
            st.write(f"- **Ham olasılık:** `{prob:.6f}`")
            if res["std"] is not None:
                st.write(f"- **TTA std:** `{res['std']:.4f}`")
            st.write(f"- **Cihaz:** `{DEVICE}`")


if __name__ == "__main__":
    main()
