"""
Streamlit Demo — Pnömotoraks Tespit Sistemi
Akış: Görüntü Yükle → Analiz Et → Sonucu Göster

Başlatma:
    streamlit run app/streamlit_app.py

TÜBİTAK 2209-A | Ahmet Demir
"""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model.unet import PneumothoraxModel
from src.utils.gradcam import (
    generate_gradcam_result,
    generate_integrated_gradients,
)
from src.utils.tta import predict_tta, uncertainty_label
from src.preprocessing.green_mask_extractor import overlay_mask_on_image

# ── Sayfa ayarları ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pnömotoraks Tespit Sistemi",
    page_icon="🫁",
    layout="wide",
)

MODEL_PATH = "results/checkpoints/best_model.pth"
IMG_SIZE   = 512
THRESHOLD  = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            return arr[0] if arr.ndim == 3 else arr
        finally:
            os.unlink(tmp_path)
    pil = Image.open(io.BytesIO(data)).convert("L")
    return np.array(pil, dtype=np.uint8)


# ── Ana arayüz ────────────────────────────────────────────────────────────────

def main():
    # Başlık
    st.markdown(
        "<h1 style='text-align:center'>🫁 Pnömotoraks Tespit Sistemi</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:gray'>"
        "TÜBİTAK 2209-A &nbsp;|&nbsp; U-Net++ + EfficientNet-B0 &nbsp;|&nbsp; "
        "Grad-CAM (Captum) &nbsp;|&nbsp; TTA</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Model kontrolü
    model = load_model()
    if model is None:
        st.warning(
            "⚠️ Model bulunamadı: `results/checkpoints/best_model.pth`\n\n"
            "Önce eğitin: `python -m src.utils.train`"
        )
        return
    st.success(f"✓ Model yüklendi — `{DEVICE}`")

    # ── Sol panel: yükleme + ayarlar ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                            help="5 farklı açıdan analiz ederek daha güvenilir tahmin üretir.")
        threshold = st.slider("Karar eşiği", 0.1, 0.9, THRESHOLD, 0.05,
                              help="Bu eşiğin üzeri Pnömotoraks olarak sınıflandırılır.")
        show_ig   = st.checkbox("Integrated Gradients göster",
                                help="Piksel düzeyinde katkı haritası (Captum gerektirir).")
        st.divider()
        st.caption("**Desteklenen formatlar:** PNG · JPEG · DICOM")

    # ── Dosya yükleme ─────────────────────────────────────────────────────────
    st.subheader("① Akciğer Grafisi Yükle")
    uploaded = st.file_uploader(
        label="PNG, JPEG veya DICOM seçin",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Lütfen bir akciğer grafisi yükleyin.")
        return

    gray = file_to_gray(uploaded)

    # ── Analiz ───────────────────────────────────────────────────────────────
    st.subheader("② Analiz")

    with st.spinner("Model tahmin yapıyor…"):
        if use_tta:
            result   = predict_tta(model, gray, img_size=IMG_SIZE, seg_threshold=threshold)
            prob     = result["prob_mean"]
            seg_bin  = result["seg_binary"]
            uncert   = uncertainty_label(result)
            votes    = result["prob_votes"]
            std      = result["prob_std"]
        else:
            resized    = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            tensor     = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                seg_pred, cls_pred = model(tensor)
            prob    = torch.sigmoid(cls_pred).item()
            seg_np  = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
            seg_res = cv2.resize(seg_np, (gray.shape[1], gray.shape[0]))
            seg_bin = (seg_res > threshold).astype(np.uint8) * 255
            uncert  = None
            votes   = None
            std     = None

    has_ptx = prob >= threshold
    label   = "🔴 PNÖMOTORAKS TESPİT EDİLDİ" if has_ptx else "🟢 Normal"
    color   = "#c0392b" if has_ptx else "#27ae60"

    st.markdown(
        f"<h2 style='text-align:center;color:{color}'>{label}</h2>",
        unsafe_allow_html=True,
    )

    # Metrik satırı
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Olasılık", f"{prob:.1%}")
    m2.metric("Eşik", f"{threshold:.0%}")
    if std is not None:
        m3.metric("Std (belirsizlik)", f"{std:.3f}")
    if uncert:
        m4.markdown(f"**Güven:** {uncert}")

    st.progress(min(prob, 1.0))

    # TTA oy dağılımı
    if votes is not None:
        with st.expander("🗳️ TTA oy dağılımı (5 varyant)"):
            labels = ["Orijinal", "H-Flip", "Parlak +", "Parlak −", "Kontrast +"]
            for lbl, v in zip(labels, votes):
                bar_color = "red" if v >= threshold else "green"
                st.markdown(
                    f"`{lbl:<12}` &nbsp; **{v:.3f}** "
                    f"{'🔴' if v >= threshold else '🟢'}",
                    unsafe_allow_html=True,
                )

    # ── Görsel paneller ───────────────────────────────────────────────────────
    st.subheader("③ Sonuçlar")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Orijinal X-Ray**")
        st.image(gray, use_container_width=True, clamp=True)

    with col2:
        st.markdown("**Segmentasyon Maskesi**")
        seg_overlay = overlay_mask_on_image(gray, seg_bin, alpha=0.4, color_bgr=(0, 60, 220))
        st.image(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.markdown("**Grad-CAM Isı Haritası**")
        with st.spinner("Grad-CAM…"):
            try:
                gradcam_bgr, _ = generate_gradcam_result(model, gray, img_size=IMG_SIZE)
                st.image(cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM üretilemedi: {e}")

    # ── Integrated Gradients ──────────────────────────────────────────────────
    if show_ig:
        with st.expander("🔬 Integrated Gradients — piksel düzeyinde katkı"):
            with st.spinner("Integrated Gradients hesaplanıyor (50 adım)…"):
                try:
                    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                    tensor  = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    ig_map  = generate_integrated_gradients(model, tensor, n_steps=50)
                    ig_u8   = (ig_map * 255).astype(np.uint8)
                    ig_col  = cv2.applyColorMap(ig_u8, cv2.COLORMAP_HOT)
                    st.image(cv2.cvtColor(ig_col, cv2.COLOR_BGR2RGB), use_container_width=True)
                except ImportError:
                    st.warning("`pip install captum` gerekli.")
                except Exception as e:
                    st.error(f"Hata: {e}")

    # ── Teknik detaylar ───────────────────────────────────────────────────────
    with st.expander("📊 Teknik Detaylar"):
        mask_area = int(seg_bin.sum() // 255)
        total_px  = gray.shape[0] * gray.shape[1]
        st.write(f"- **Mod:** {'TTA (5 varyant)' if use_tta else 'Standart'}")
        st.write(f"- **Görüntü boyutu:** {gray.shape[1]} × {gray.shape[0]}")
        st.write(f"- **Pnömotoraks alanı:** {mask_area:,} piksel ({mask_area / total_px:.2%})")
        st.write(f"- **Ham olasılık:** `{prob:.6f}`")
        if std is not None:
            st.write(f"- **Belirsizlik (std):** `{std:.4f}`")
        st.write(f"- **Cihaz:** `{DEVICE}`")


if __name__ == "__main__":
    main()
