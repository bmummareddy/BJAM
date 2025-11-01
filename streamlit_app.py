# -*- coding: utf-8 -*-
# BJAM — Digital Twin for Binder Jet AM (Fast, Guardrailed, STL-aware)
# Author: ChatGPT for Bhargavi
# Notes:
# - Reads BJAM_cleaned.csv or BJAM_All_Deep_Fill_v9.csv if present.
# - Trains quantile models (q10, q50, q90) on-the-fly with caching.
# - Physics guardrails enforce >=90% theoretical density targets where possible.
# - Produces exactly 3 water-based and 2 solvent-based options when feasible.
# - Digital Twin: accepts STL, estimates layers/build time/binder consumption,
#   produces a basic per-layer "risk" heatmap proxy.
# - Designed to be minimal-deps.

from __future__ import annotations
import os, io, math, importlib.util
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Geometry (optional)
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
if HAVE_TRIMESH:
    import trimesh  # type: ignore

# Local helper module
from shared import (
    load_dataset,
    fit_quantile_models,
    recommend_with_guardrails,
    pack_fraction_furnas,
    washburn_binder_suggestion,
    valid_binder_types,
    sanitize_material_name
)

st.set_page_config(page_title="BJAM Digital Twin", layout="wide", page_icon="🧪")

# Theming / small CSS
st.markdown(
    """
    <style>
      .small { font-size: 0.85rem; color: #6b7280; }
      .ok    { color: #059669; font-weight: 600; }
      .bad   { color: #dc2626; font-weight: 600; }
      .chip  { display:inline-block; padding:2px 8px; border-radius:12px;
               background:#111827; color:#e5e7eb; margin-right:6px; font-size:0.8rem; }
      .muted { color:#9ca3af; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("Inputs")
    material = st.text_input("Powder / material", value="316L", help="Material name (free text).")
    d50_um   = st.number_input("Particle size D50 (µm)", value=25.0, min_value=1.0, max_value=200.0, step=0.5)
    layer_um = st.number_input("Layer thickness (µm)", value=60.0, min_value=20.0, max_value=200.0, step=5.0)
    want_density = st.slider("Target green % theoretical density", 80, 99, 90, help="Guardrail target.")
    st.caption("Binder mix: app will return exactly 3 water-based and 2 solvent-based (when feasible).")

    st.divider()
    st.subheader("Digital Twin (STL)")
    st.caption("Optional: upload an STL to estimate layers, binder use and build time.")
    stl_file = st.file_uploader("STL file", type=["stl"], accept_multiple_files=False)

    recoater_width_mm = st.number_input("Recoater pass width (mm)", value=100.0, min_value=10.0, max_value=500.0, step=5.0)
    build_speed_mm_s  = st.number_input("Nominal roller traverse speed (mm/s)", value=60.0, min_value=5.0, max_value=400.0, step=1.0)
    saturation_pct_ui = st.slider("Assumed binder saturation for twin (%)", 40, 120, 80)

    st.divider()
    st.caption("Tip: models are cached; changing inputs is fast. Use 'Data Health' to inspect coverage.")

# Data + Models (cached)
@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_dataset()

@st.cache_resource(show_spinner=False)
def _fit_models(df: pd.DataFrame):
    return fit_quantile_models(df)

df = _load_data()
models = _fit_models(df)

# Headline
c1, c2 = st.columns([1.2, 1])
with c1:
    st.title("BJAM Digital Twin — Parameter Optimizer")
    st.write("Data-driven recommendations with physics guardrails and STL-aware build estimates.")
with c2:
    st.metric("Training rows", len(df))
    st.metric("Materials in dataset", df["material"].nunique() if "material" in df.columns else 0)

# Recommendations
with st.expander("Top-5 guardrailed recommendations", expanded=True):
    clean_mat = sanitize_material_name(material)
    rec_df = recommend_with_guardrails(
        models=models,
        df=df,
        material=clean_mat,
        d50_um=d50_um,
        layer_um=layer_um,
        target_td=want_density,
        require_mix=True,   # 3 water + 2 solvent
        n_total=5
    )

    if rec_df.empty:
        st.warning("No safe recommendations found near this region. Try adjusting D50 or layer thickness.")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        st.caption("Exactly 3 water-based and 2 solvent-based rows are returned when feasible; otherwise, closest-safe set.")

# Sensitivity / Heatmap
st.subheader("Parameter sweep — heatmap of predicted green %TD")
hcol1, hcol2 = st.columns([1, 1])
with hcol1:
    x_vals = np.linspace(max(0.5*layer_um, 20), min(2.0*layer_um, 200), 30)
    y_vals = np.linspace(20, 120, 30)  # binder saturation
with hcol2:
    grid = []
    for L in x_vals:
        for S in y_vals:
            grid.append({
                "material": sanitize_material_name(material),
                "d50_um": d50_um,
                "layer_thickness_um": L,
                "binder_saturation_pct": S
            })
    grid = pd.DataFrame(grid)
    _, pred50, _ = models  # q10, q50, q90
    p = pred50.predict(grid[["material","d50_um","layer_thickness_um","binder_saturation_pct"]])
    Z = p.reshape(len(x_vals), len(y_vals))

    fig = go.Figure(data=go.Heatmap(
        z=Z, x=y_vals, y=x_vals, colorbar=dict(title="%TD (q50)")
    ))
    fig.add_hline(y=layer_um, line=dict(color="#111827", dash="dash"))
    fig.add_vline(x=saturation_pct_ui, line=dict(color="#111827", dash="dash"))
    fig.update_layout(xaxis_title="Binder saturation (%)",
                      yaxis_title="Layer thickness (µm)",
                      height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Digital Twin (STL)
st.subheader("Digital Twin — STL-based estimates")
if stl_file is not None and HAVE_TRIMESH:
    raw = stl_file.read()
    mesh = trimesh.load(io.BytesIO(raw), file_type='stl', force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "dump"):
        mesh = trimesh.util.concatenate(mesh.dump())
    if isinstance(mesh, trimesh.Trimesh):
        bbox = mesh.bounds
        dims = (bbox[1] - bbox[0])  # (x,y,z)
        vol_mm3 = mesh.volume * 1000.0  # robust-ish
        layer_h_mm = layer_um / 1000.0
        n_layers = int(math.ceil(max(dims[2], 1e-6) / layer_h_mm))
        xy_area_mm2 = mesh.convex_hull.area if hasattr(mesh, "convex_hull") else (dims[0]*dims[1])
        binder_ml = xy_area_mm2 * n_layers * (saturation_pct_ui/100.0) * 1e-3
        passes = max(1, math.ceil(dims[0] / max(1.0, recoater_width_mm)))
        build_time_min = (passes * n_layers * (dims[1] / max(1.0, build_speed_mm_s))) / 60.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Layers", f"{n_layers:,}")
        c2.metric("XY area per layer (mm²)", f"{xy_area_mm2:,.0f}")
        c3.metric("Binder estimate (mL)", f"{binder_ml:,.1f}")
        c4.metric("Build time (min)", f"{build_time_min:,.1f}")

        normals = mesh.face_normals
        steep = np.abs(normals[:,2])  # 1.0 is vertical; low means steep overhangs
        cz = mesh.triangles_center[:,2] - mesh.bounds[0,2]
        li = np.clip((cz / max(1e-9, dims[2])) * n_layers, 0, n_layers-1).astype(int)
        risk = np.zeros(n_layers)
        counts = np.zeros(n_layers)
        for s, idx in zip(steep, li):
            r = 1.0 - s
            risk[idx] += r
            counts[idx] += 1
        risk = np.divide(risk, np.maximum(1, counts))

        figz = go.Figure()
        figz.add_trace(go.Scatter(y=risk, x=list(range(n_layers)), mode="lines"))
        figz.update_layout(xaxis_title="Layer index",
                           yaxis_title="Relative risk (overhang proxy)",
                           height=320, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(figz, use_container_width=True)
    else:
        st.error("Could not parse STL into a single mesh. Try a simpler file.")
elif stl_file is not None and not HAVE_TRIMESH:
    st.error("trimesh is not installed. Please add it to requirements to use the digital twin STL features.")

# Data Health
with st.expander("Data Health & Coverage", expanded=False):
    if df.empty:
        st.info("No dataset loaded.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            by_mat = df.groupby("material").size().sort_values(ascending=False).head(20)
            figm = go.Figure(go.Bar(x=by_mat.index.astype(str), y=by_mat.values))
            figm.update_layout(xaxis_title="Material (top20)", yaxis_title="Rows", height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figm, use_container_width=True)
        with g2:
            if "d50_um" in df.columns and "green_pct_td" in df.columns:
                figd = go.Figure(go.Scatter(x=df["d50_um"], y=df["green_pct_td"], mode="markers"))
                figd.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
                figd.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD", height=360, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figd, use_container_width=True)

st.caption("© BJAM AI — physics-informed recommendations with uncertainty (q10/q50/q90).")
