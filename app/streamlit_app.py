"""
Pnömotoraks AI — Ana Giriş
===========================
Streamlit multi-page app. Sayfalar:
  🫁 Radyolog Paneli  → pages/1_Radyolog_Paneli.py
  📊 Eğitim İzleme   → pages/2_Egitim_Izleme.py

Başlatma:
    streamlit run app/streamlit_app.py

TÜBİTAK 2209-A | Ahmet Demir, Erkan Koçulu
"""

import streamlit as st

st.set_page_config(
    page_title="Pnömotoraks AI",
    page_icon="🫁",
    layout="wide",
)

st.markdown("## 🫁 Pnömotoraks AI — TÜBİTAK 2209-A")
st.markdown("**Dokuz Eylül Üniversitesi Tıp Fakültesi**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Radyolog_Paneli.py", label="🫁 Radyolog Paneli",
                 icon="🫁", use_container_width=True)
    st.caption("DICOM / PNG yükle → Otomatik pnömotoraks tespiti, segmentasyon, PDF rapor")

with col2:
    st.page_link("pages/2_Egitim_Izleme.py", label="📊 Eğitim İzleme",
                 icon="📊", use_container_width=True)
    st.caption("W&B'den canlı metrikler: Loss, Dice, HD95, Fold karşılaştırma, Radar")
