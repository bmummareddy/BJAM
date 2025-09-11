# streamlit_app.py — BJAM Binder-Jet AM Parameter Recommender (bright UI)
# Uses shared.py (Deep_Fill v9 dataset) for loading, priors, models, and recs.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # headless backend (required on Streamlit Cloud)
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import streamlit as st

from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
)

# -----------------------------------------------------------------------------
# Page config + subtle bright styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BJAM Predictions",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #FFFDF7 0%, #FFF8EC 40%, #FFF4E2 100%); }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .stMetric { background: rgba(255,255,255,.6); border-radius: 12px; padding: 10px; }
    div[data-testid="stStatusWidget"] { opacity: .85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load data (single source of truth) + train models
# -----------------------------------------------------------------------------
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# -----------------------------------------------------------------------------
# Sidebar: data + guardrails + target
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data source: {Path(src).name} · rows={len(df_base):,}")
        st.download_button(
            "Download source dataset (CSV)",
            data=df_base.to_csv(index=False).encode("utf-8"),
            file_name=Path(src).name,
            mime="text/csv",
            help="Exports the exact dataset driving optimization & visuals.",
        )
    else:
        st.warning("No dataset found. App will use physics priors only (no few-shot).")

    st.divider()
    guardrails_on = st.toggle(
        "Guardrails",
        value=True,
        help=(
            "ON: narrows ranges to empirically stable windows and gently clips predictions.\n"
            "OFF: wider exploration; still 0–100% physically bounded."
        ),
    )
    target_green = st.slider(
        "Target green %TD", 80, 98, 90, 1,
        help="Recommendations prefer q10 ≥ target for conservative picks."
    )
    st.caption(
        "Guardrails = constraints to keep settings inside stable BJAM windows:\n"
        "• binder 60–110% · speed ~1.2–3.5 mm/s · layer ≈ 3–5×D50.\n"
        "Turn OFF to explore more aggressively."
    )

# -----------------------------------------------------------------------------
# Header + quick stats
# -----------------------------------------------------------------------------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · bright, friendly UI · toggle guardrails to explore")

m1, m2, m3 = st.columns(3)
m1.metric("Rows in dataset", f"{len(df_base):,}")
m2.metric("Materials", f"{df_base['material'].nunique() if 'material' in df_base else 0:,}")
m3.metric("Quantile models", "Trained" if models else "Physics-only")

with st.expander("Preview source data", expanded=False):
    if len(df_base):
        st.dataframe(df_base.head(25), use_container_width=True)
    else:
        st.info("No rows to preview.")

st.divider()

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Inputs", help="Pick material & D50; layer defaults to ≈4×D50.")

    # Material list from data (fallback to a simple default)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []
    material = st.selectbox("Material", options=(materials or ["316L"]), index=0)

    # D50 default: median for chosen material if available; else 30 µm
    d50_default = 30.0
    if "material" in df_base and "d50_um" in df_base and material in df_base["material"].astype(str).unique():
        try:
            d50_default = float(df_base.loc[df_base["material"].astype(str) == material, "d50_um"].dropna().median())
        except Exception:
            d50_default = 30.0

    d50_um = st.number_input("D50 (µm)", 1.0, 150.0, float(d50_default), 1.0)

    # Priors + guardrail band for layer
    pri = physics_priors(d50_um, binder_type_guess=None)
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    t_lo, t_hi = gr["layer_thickness_um"]
    layer_um = st.slider(
        "Layer thickness (µm)",
        min_value=float(round(t_lo)), max_value=float(round(t_hi)),
        value=float(round(pri["layer_thickness_um"])), step=1.0,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Prior binder %", f"{pri['binder_saturation_pct']:.0f}%")
    c2.metric("Prior speed", f"{pri['roller_speed_mm_s']:.1f} mm/s")
    c3.metric("Layer/D50", f"{layer_um/d50_um:.2f}×")

with right:
    st.subheader("Recommend parameters", help="Top-k sets aimed at your target green %TD.")
    top_k = st.slider("Number of recommendations", 3, 8, 5, 1)
    if st.button("Recommend", type="primary", use_container_width=True):
        recs = copilot(
            material=material,
            d50_um=float(d50_um),
            df_source=df_base,
            models=models,
            guardrails_on=guardrails_on,
            target_green=float(target_green),
            top_k=int(top_k),
        )
        st.dataframe(recs, use_container_width=True)
        st.caption("Ranking prefers **q10 ≥ target**, then **q50**; mild penalty for extreme binder/speed.")
    else:
        st.info("Click **Recommend** to generate top-k parameter sets.")

st.divider()

# -----------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------
tabs = st.tabs(["Heatmap (speed × saturation)", "Saturation sensitivity", "Qualitative packing", "Formulae"])

# Heatmap: speed × saturation (fixed layer & D50)
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD", help="Layer fixed to your slider; explore speed × saturation.")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    b_lo, b_hi = gr["binder_saturation_pct"]
    s_lo, s_hi = gr["roller_speed_mm_s"]

    s_vals = np.linspace(float(b_lo), float(b_hi), 55)   # saturation axis
    v_vals = np.linspace(float(s_lo), float(s_hi), 45)   # speed axis

    grid = pd.DataFrame(
        [(b, v, layer_um, d50_um, material) for b in s_vals for v in v_vals],
        columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um", "material"],
    )
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    scored = predict_quantiles(models, grid)
    Xs = sorted(scored["binder_saturation_pct"].unique())
    Ys = sorted(scored["roller_speed_mm_s"].unique())
    z = scored.sort_values(["binder_saturation_pct", "roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=Xs, y=Ys, z=z, colorbar=dict(title="%TD"), colorscale="Turbo"))
    fig.add_trace(go.Contour(
        x=Xs, y=Ys, z=z,
        contours=dict(start=90, end=90, size=1, coloring="none"),
        line=dict(width=3), showscale=False, name="90% TD",
    ))
    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
        title=f"Green %TD (q50) · Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · Source={Path(src).name if src else '—'}",
    )
    st.plotly_chart(fig, use_container_width=True)

# Saturation sensitivity: q10/q50/q90 vs binder %
with tabs[1]:
    st.subheader("Saturation sensitivity", help="Uncertainty band helps spot stable operating windows.")
    sats = np.linspace(float(b_lo), float(b_hi), 61)
    curve_df = pd.DataFrame({
        "binder_saturation_pct": sats,
        "roller_speed_mm_s": 1.6,
        "layer_thickness_um": float(layer_um),
        "d50_um": float(d50_um),
        "material": material,
        "material_class": "metal",
        "binder_type_rec": "solvent_based",
    })
    curve_scored = predict_quantiles(models, curve_df)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax2.plot(curve_scored["binder_saturation_pct"], curve_scored["td_q50"], label="q50")
    ax2.fill_between(curve_scored["binder_saturation_pct"], curve_scored["td_q10"], curve_scored["td_q90"], alpha=0.2, label="q10–q90")
    ax2.axhline(target_green, linestyle="--", linewidth=1, label=f"Target {target_green}%")
    ax2.set_xlabel("Binder saturation (%)")
    ax2.set_ylabel("Predicted green %TD")
    ax2.set_title(f"Sensitivity @ speed=1.6 mm/s, layer={layer_um:.0f} µm, D50={d50_um:.0f} µm")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# Qualitative packing slice (illustrative)
with tabs[2]:
    st.subheader("Qualitative packing", help="Illustrative 2D slice to build intuition near target packing.")
    r = 0.02
    pts, attempts = [], 0
    while len(pts) < 250 and attempts < 10000:
        x, y = np.random.rand(2)
        if all((x - px) ** 2 + (y - py) ** 2 >= (2 * r) ** 2 for px, py in pts):
            pts.append((x, y))
        attempts += 1

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax3.set_aspect("equal", "box")
    for (x, y) in pts:
        ax3.add_patch(plt.Circle((x, y), r, alpha=0.75))
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("Qualitative packing slice (~90% effective packing)")
    st.pyplot(fig3, clear_figure=True)

# Formulae (symbols)
with tabs[3]:
    st.subheader("Formulae (symbols)")
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    st.caption("These relations guide priors and plots. The few-shot model refines predictions from your dataset.")

# Diagnostics
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})
