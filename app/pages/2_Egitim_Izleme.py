"""
Eğitim İzleme  —  W&B Canlı Dashboard
======================================

W&B API'den fold metriklerini çeker, Plotly ile görselleştirir.
Eğitim devam ediyorsa 30 sn'de bir otomatik yeniler.

TÜBİTAK 2209-A | Ahmet Demir
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Sabitler ──────────────────────────────────────────────────────────────────

WANDB_ENTITY  = "ahmet-ai-t-bi-tak"
WANDB_PROJECT = "Pneumothorax-Detection"
RESULTS_CSV   = _ROOT / "results" / "global_kfold_results.csv"
META_JSON     = _ROOT / "checkpoints" / "global_base_model_meta.json"

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Eğitim İzleme",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg,#1a1f2e,#252b3a);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:18px 14px;
    text-align:center; margin-bottom:8px;
}
.metric-val  { font-size:2rem; font-weight:800; }
.metric-lbl  { font-size:0.75rem; color:rgba(255,255,255,0.5);
               text-transform:uppercase; letter-spacing:1px; }
.metric-sub  { font-size:0.72rem; color:rgba(255,255,255,0.3); margin-top:4px; }
.status-pill {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:0.82rem; font-weight:600;
}
.running  { background:rgba(34,197,94,0.15); color:#22c55e; border:1px solid #22c55e44; }
.finished { background:rgba(99,102,241,0.15); color:#818cf8; border:1px solid #818cf844; }
.crashed  { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid #f8717144; }
</style>
""", unsafe_allow_html=True)

# ── Başlık ────────────────────────────────────────────────────────────────────

st.markdown("## 📊 Eğitim İzleme")
st.markdown("**TÜBİTAK 2209-A** — U-Net++ Global Pre-training & Fine-tuning")
st.divider()

# ── W&B bağlantısı ────────────────────────────────────────────────────────────

@st.cache_resource(ttl=30)
def get_wandb_api():
    try:
        import wandb
        return wandb.Api(timeout=15)
    except Exception:
        return None

@st.cache_data(ttl=30)
def fetch_runs() -> list[dict]:
    """
    W&B'den tüm run'ları çeker.
    Her run için: name, state, group, config, summary, history (son 200 epoch)
    """
    api = get_wandb_api()
    if api is None:
        return []
    try:
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            order="-created_at",
        )
        result = []
        for run in runs:
            try:
                hist = run.history(
                    keys=[
                        "epoch",
                        "Loss/BCE_Dice_Loss_train",
                        "Loss/BCE_Dice_Loss_val",
                        "Segmentation/Dice_Score",
                        "Segmentation/IoU",
                        "Segmentation/Hausdorff_HD95_px",
                        "Classification/AUC_ROC",
                        "Classification/Recall_Sensitivity",
                        "Classification/Specificity_Ozgukluk",
                        "LR",
                    ],
                    samples=500,
                    pandas=True,
                )
                result.append({
                    "name":    run.name,
                    "id":      run.id,
                    "state":   run.state,
                    "group":   run.config.get("wandb_group", run.group or "—"),
                    "config":  run.config,
                    "summary": dict(run.summary),
                    "history": hist,
                    "url":     run.url,
                    "created": run.created_at,
                })
            except Exception:
                continue
        return result
    except Exception as e:
        st.warning(f"W&B bağlantı hatası: {e}")
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    auto_refresh = st.toggle("Otomatik yenile (30 sn)", value=True)
    show_lr      = st.toggle("LR grafiği", value=False)
    show_hd95    = st.toggle("HD95 grafiği", value=True)
    n_runs_show  = st.slider("Gösterilecek run sayısı", 1, 20, 10)

    st.divider()
    if st.button("🔄 Şimdi Yenile"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"[🔗 W&B Dashboard](https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT})",
        unsafe_allow_html=False,
    )

# ── Veri çek ─────────────────────────────────────────────────────────────────

with st.spinner("W&B'den veriler çekiliyor…"):
    runs = fetch_runs()

if not runs:
    st.info(
        "W&B'den veri alınamadı. Olası nedenler:\n"
        "- `pip install wandb` ve `wandb login` gerekiyor\n"
        "- Eğitim henüz başlatılmadı\n"
        "- İnternet bağlantısı yok\n\n"
        "**Alternatif:** Aşağıda yerel sonuç dosyası okunuyor."
    )

# ── Aktif run özet kartları ───────────────────────────────────────────────────

if runs:
    active = [r for r in runs if r["state"] == "running"]
    if active:
        st.markdown("### 🟢 Aktif Eğitim")
        for run in active[:3]:
            summ = run["summary"]
            c1, c2, c3, c4, c5 = st.columns(5)
            dice = summ.get("Segmentation/Dice_Score", 0)
            iou  = summ.get("Segmentation/IoU", 0)
            loss = summ.get("Loss/BCE_Dice_Loss_val", 0)
            hd95 = summ.get("Segmentation/Hausdorff_HD95_px", None)
            auc  = summ.get("Classification/AUC_ROC", 0)
            epoch = summ.get("epoch", "—")

            def mcard(val, lbl, sub="", color="#4a9eff"):
                return f"""<div class="metric-card">
                    <div class="metric-val" style="color:{color}">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                    <div class="metric-sub">{sub}</div></div>"""

            c1.markdown(mcard(f"{dice:.3f}", "Dice Score", f"Epoch {epoch}", "#22c55e"), unsafe_allow_html=True)
            c2.markdown(mcard(f"{iou:.3f}",  "IoU",        run["name"],     "#818cf8"), unsafe_allow_html=True)
            c3.markdown(mcard(f"{loss:.4f}", "Val Loss",   run["group"],    "#f59e0b"), unsafe_allow_html=True)
            c4.markdown(mcard(f"{auc:.3f}",  "AUC-ROC",    "",              "#06b6d4"), unsafe_allow_html=True)
            hd_str = f"{hd95:.1f} px" if hd95 else "—"
            c5.markdown(mcard(hd_str, "HD95",  "", "#f87171"), unsafe_allow_html=True)

        st.divider()

# ── Run seçici ────────────────────────────────────────────────────────────────

if runs:
    # Grup listesi
    groups = sorted(set(r["group"] for r in runs), reverse=True)
    sel_group = st.selectbox(
        "Eğitim grubu seç (kfold run grubu)",
        options=["— Tümü —"] + groups,
    )

    filtered = runs if sel_group == "— Tümü —" else [
        r for r in runs if r["group"] == sel_group
    ]
    filtered = filtered[:n_runs_show]

    # ── Run listesi tablosu ───────────────────────────────────────────────────

    st.markdown("### 📋 Run Listesi")
    rows = []
    for r in filtered:
        s = r["summary"]
        state_html = (
            '<span class="status-pill running">▶ Devam</span>'   if r["state"] == "running"  else
            '<span class="status-pill finished">✓ Bitti</span>'  if r["state"] == "finished" else
            '<span class="status-pill crashed">✗ Hata</span>'
        )
        rows.append({
            "Run":         r["name"],
            "Grup":        r["group"],
            "Durum":       r["state"],
            "Dice ↑":      round(s.get("Segmentation/Dice_Score", 0), 4),
            "IoU ↑":       round(s.get("Segmentation/IoU", 0), 4),
            "Val Loss ↓":  round(s.get("Loss/BCE_Dice_Loss_val", 0), 4),
            "AUC ↑":       round(s.get("Classification/AUC_ROC", 0), 4),
            "Sens ↑":      round(s.get("Classification/Recall_Sensitivity", 0), 4),
        })

    df_runs = pd.DataFrame(rows)
    st.dataframe(
        df_runs.style.background_gradient(subset=["Dice ↑", "IoU ↑", "AUC ↑"], cmap="Greens")
                     .background_gradient(subset=["Val Loss ↓"], cmap="Reds_r"),
        use_container_width=True,
        height=min(40 + len(df_runs) * 35, 400),
    )

    # ── Eğitim eğrileri (çoklu fold) ─────────────────────────────────────────

    st.markdown("### 📈 Eğitim Eğrileri")

    # Kaç subplot
    n_plots = 2 + (1 if show_hd95 else 0) + (1 if show_lr else 0)
    subplot_titles = ["Train / Val Loss", "Dice Score"]
    if show_hd95:
        subplot_titles.append("HD95 (Hausdorff, px)")
    if show_lr:
        subplot_titles.append("Learning Rate")

    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    COLOR_PALETTE = px.colors.qualitative.Plotly

    for ri, run in enumerate(filtered):
        hist = run["history"]
        if hist is None or hist.empty:
            continue
        color = COLOR_PALETTE[ri % len(COLOR_PALETTE)]
        label = run["name"]

        epochs = hist.get("epoch", hist.index)

        # Loss
        if "Loss/BCE_Dice_Loss_train" in hist.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist["Loss/BCE_Dice_Loss_train"],
                name=f"{label} Train", line=dict(color=color, dash="dot", width=1.5),
                legendgroup=label, showlegend=True,
            ), row=1, col=1)
        if "Loss/BCE_Dice_Loss_val" in hist.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist["Loss/BCE_Dice_Loss_val"],
                name=f"{label} Val", line=dict(color=color, width=2),
                legendgroup=label, showlegend=False,
            ), row=1, col=1)

        # Dice
        if "Segmentation/Dice_Score" in hist.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist["Segmentation/Dice_Score"],
                name=label, line=dict(color=color, width=2),
                legendgroup=label, showlegend=False,
            ), row=1, col=2)

        col_offset = 3
        # HD95
        if show_hd95 and "Segmentation/Hausdorff_HD95_px" in hist.columns:
            hd = hist["Segmentation/Hausdorff_HD95_px"].replace([np.inf, -np.inf], np.nan)
            fig.add_trace(go.Scatter(
                x=epochs, y=hd,
                name=label, line=dict(color=color, width=2),
                legendgroup=label, showlegend=False,
            ), row=1, col=col_offset)
            col_offset += 1

        # LR
        if show_lr and "LR" in hist.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist["LR"],
                name=label, line=dict(color=color, width=1.5, dash="dot"),
                legendgroup=label, showlegend=False,
            ), row=1, col=col_offset)

    fig.update_layout(
        height=380,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="rgba(255,255,255,0.8)", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0.3)",
        ),
        margin=dict(l=40, r=20, t=40, b=80),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)

    # ── Fold karşılaştırma radar grafiği ─────────────────────────────────────

    finished_runs = [r for r in filtered if r["state"] == "finished"]
    if finished_runs:
        st.markdown("### 🕸 Fold Karşılaştırma (Radar)")

        categories = ["Dice", "IoU", "AUC", "Sensitivity", "Specificity"]
        radar_fig = go.Figure()

        for ri, run in enumerate(finished_runs[:5]):
            s = run["summary"]
            vals = [
                s.get("Segmentation/Dice_Score", 0),
                s.get("Segmentation/IoU", 0),
                s.get("Classification/AUC_ROC", 0),
                s.get("Classification/Recall_Sensitivity", 0),
                s.get("Classification/Specificity_Ozgukluk", 0),
            ]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=run["name"],
                line=dict(color=COLOR_PALETTE[ri % len(COLOR_PALETTE)], width=2),
                fill="toself",
                fillcolor=COLOR_PALETTE[ri % len(COLOR_PALETTE)].replace("rgb", "rgba").replace(")", ",0.08)"),
            ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor="#0e1117",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=9),
                ),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            ),
            paper_bgcolor="#0e1117",
            font=dict(color="rgba(255,255,255,0.8)"),
            height=420,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=60, r=60, t=30, b=60),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

# ── Yerel CSV (W&B yoksa fallback) ───────────────────────────────────────────

st.markdown("### 💾 Yerel Sonuç Dosyası")

if RESULTS_CSV.exists():
    df_local = pd.read_csv(RESULTS_CSV)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(
            df_local.style.highlight_max(
                subset=[c for c in df_local.columns if "dice" in c.lower() or "auc" in c.lower()],
                color="rgba(34,197,94,0.3)"
            ),
            use_container_width=True,
        )
    with col2:
        st.metric("Ort. Dice", f"{df_local['best_dice'].mean():.4f}",
                  f"±{df_local['best_dice'].std():.4f}")
        st.metric("Ort. AUC",  f"{df_local['best_auc'].mean():.4f}",
                  f"±{df_local['best_auc'].std():.4f}")
        if "best_sensitivity" in df_local.columns:
            st.metric("Ort. Sensitivity", f"{df_local['best_sensitivity'].mean():.4f}")
else:
    st.info(f"Henüz sonuç yok: `{RESULTS_CSV.relative_to(_ROOT)}`\nEğitim bitince otomatik görünür.")

# ── Base model meta ───────────────────────────────────────────────────────────

if META_JSON.exists():
    st.markdown("### 🏆 Base Model")
    meta = json.loads(META_JSON.read_text())
    cols = st.columns(len(meta))
    icons = {"encoder": "🧠", "img_size": "📐", "sources": "💾",
             "best_fold": "🥇", "best_dice": "🎯", "best_auc": "📊", "epochs": "🔄"}
    for i, (k, v) in enumerate(meta.items()):
        with cols[i]:
            st.metric(f"{icons.get(k,'')} {k}", v)

# ── W&B iframe (tam dashboard) ────────────────────────────────────────────────

st.markdown("### 🌐 W&B Tam Dashboard")
with st.expander("W&B Dashboard'u Göster (iframe)", expanded=False):
    wandb_url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}"
    st.markdown(
        f'<iframe src="{wandb_url}" width="100%" height="700px" '
        f'style="border:none;border-radius:8px;"></iframe>',
        unsafe_allow_html=True,
    )
    st.caption(f"[Yeni sekmede aç ↗]({wandb_url})")

# ── Oto yenile ────────────────────────────────────────────────────────────────

if auto_refresh and runs and any(r["state"] == "running" for r in runs):
    st.toast("Eğitim devam ediyor — 30 sn sonra yenilenecek", icon="🔄")
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
elif auto_refresh:
    st.caption("⏸ Aktif eğitim yok — oto yenileme pasif")
