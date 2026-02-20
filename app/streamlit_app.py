"""
Streamlit Demo — Pnömotoraks Tespit Sistemi
Akış: Görüntü Yükle → Analiz Et → Sonucu Göster

Başlatma:
    streamlit run app/streamlit_app.py

TÜBİTAK 2209-A | Ahmet Demir
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model.unet import PneumothoraxModel
from src.utils.gradcam import generate_gradcam_result, generate_integrated_gradients
from src.preprocessing.green_mask_extractor import overlay_mask_on_image

# ── Sayfa ayarları ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pnömotoraks Tespit Sistemi",
    page_icon="🫁",
    layout="wide",
)

# ── Sabitler ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "results/checkpoints/best_model.pth"
IMG_SIZE    = 512
THRESHOLD   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model cache ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> PneumothoraxModel | None:
    if not Path(MODEL_PATH).exists():
        return None
    model = PneumothoraxModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def file_to_gray(uploaded_file) -> np.ndarray:
    """UploadedFile → uint8 grayscale ndarray."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".dcm", ".dicom")):
        import pydicom, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            ds  = pydicom.dcmread(tmp_path)
            arr = ds.pixel_array.astype(np.float32)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 3:
                arr = arr[0]
            return arr
        finally:
            os.unlink(tmp_path)
    else:
        pil = Image.open(io.BytesIO(data)).convert("L")
        return np.array(pil, dtype=np.uint8)


def run_inference(model, gray: np.ndarray):
    """Segmentasyon + sınıflandırma."""
    resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    tensor     = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_pred, cls_pred = model(tensor)

    prob     = torch.sigmoid(cls_pred).item()
    seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
    return seg_mask, prob, tensor


# ── Arayüz ────────────────────────────────────────────────────────────────────

def main():
    # Başlık
    st.markdown(
        "<h1 style='text-align:center'>🫁 Pnömotoraks Tespit Sistemi</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:gray'>TÜBİTAK 2209-A | "
        "U-Net++ + EfficientNet-B0 | Grad-CAM (Captum)</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Model durumu
    model = load_model()
    if model is None:
        st.warning(
            "⚠️ Model dosyası bulunamadı: `results/checkpoints/best_model.pth`\n\n"
            "Önce modeli eğitin: `python src/utils/train.py`"
        )
        return

    st.success(f"✓ Model yüklendi — Cihaz: `{DEVICE}`")

    # ── Dosya yükleme ─────────────────────────────────────────────────────────
    st.subheader("1. Akciğer Grafisi Yükle")
    uploaded = st.file_uploader(
        "PNG, JPEG veya DICOM formatında akciğer grafisi yükleyin",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
    )

    if uploaded is None:
        st.info("Lütfen bir akciğer grafisi yükleyin.")
        return

    # ── Analiz ───────────────────────────────────────────────────────────────
    st.subheader("2. Analiz")

    with st.spinner("Model tahmin yapıyor…"):
        gray = file_to_gray(uploaded)
        seg_mask, prob, tensor = run_inference(model, gray)

    has_ptx = prob >= THRESHOLD
    label   = "🔴 PNÖMOTORAKS TESPİT EDİLDİ" if has_ptx else "🟢 Normal"
    color   = "red" if has_ptx else "green"

    st.markdown(
        f"<h2 style='text-align:center;color:{color}'>{label}</h2>",
        unsafe_allow_html=True,
    )
    st.metric("Pnömotoraks Olasılığı", f"{prob:.1%}")
    st.progress(prob)

    # ── Sonuç panelleri ───────────────────────────────────────────────────────
    st.subheader("3. Sonuçlar")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Orijinal X-Ray**")
        st.image(gray, use_container_width=True, clamp=True)

    with col2:
        st.markdown("**Segmentasyon Maskesi**")
        seg_resized = cv2.resize(seg_mask, (gray.shape[1], gray.shape[0]))
        binary_mask = (seg_resized > THRESHOLD).astype(np.uint8) * 255
        seg_overlay = overlay_mask_on_image(gray, binary_mask, alpha=0.4)
        seg_rgb     = cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)
        st.image(seg_rgb, use_container_width=True)

    with col3:
        st.markdown("**Grad-CAM Isı Haritası**")
        with st.spinner("Grad-CAM hesaplanıyor…"):
            gradcam_bgr, _ = generate_gradcam_result(model, gray, img_size=IMG_SIZE)
        gradcam_rgb = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)
        st.image(gradcam_rgb, use_container_width=True)

    # ── Integrated Gradients (opsiyonel) ─────────────────────────────────────
    with st.expander("🔬 Integrated Gradients (piksel düzeyinde katkı)"):
        with st.spinner("Integrated Gradients hesaplanıyor…"):
            try:
                ig_map = generate_integrated_gradients(model, tensor, n_steps=50)
                ig_u8  = (ig_map * 255).astype(np.uint8)
                ig_colored = cv2.applyColorMap(ig_u8, cv2.COLORMAP_HOT)
                st.image(cv2.cvtColor(ig_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
            except ImportError:
                st.warning("Captum kurulu değil: `pip install captum`")

    # ── Metrikler ─────────────────────────────────────────────────────────────
    with st.expander("📊 Teknik Detaylar"):
        mask_area = int(binary_mask.sum() // 255)
        total_px  = gray.shape[0] * gray.shape[1]
        st.write(f"- **Pnömotoraks alanı (piksel):** {mask_area:,}")
        st.write(f"- **Görüntü boyutu:** {gray.shape[1]} × {gray.shape[0]}")
        st.write(f"- **Alan oranı:** {mask_area / total_px:.2%}")
        st.write(f"- **Ham olasılık:** `{prob:.6f}`")
        st.write(f"- **Eşik değeri:** `{THRESHOLD}`")
        st.write(f"- **Cihaz:** `{DEVICE}`")


if __name__ == "__main__":
    main()
