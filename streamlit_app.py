# -*- coding: utf-8 -*-
# BJAM — Digital Twin for Binder Jet AM (single-file version)
# Self-contained: includes data load, models, physics guardrails, and STL twin.
# Copy this file as streamlitapp.py and deploy with the provided requirements.txt.

from __future__ import annotations
import os, io, math, importlib.util
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional geometry backend (STL analysis)
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
if HAVE_TRIMESH:
    import trimesh  # type: ignore

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
DATA_CANDIDATES = [
    Path("BJAM_cleaned.csv"),
    Path("BJAM_All_Deep_Fill_v9.csv"),
    Path("/mnt/data/BJAM_cleaned.csv"),
    Path("/mnt/data/BJAM_All_Deep_Fill_v9.csv"),
]

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            # normalize column names
            rename_map = {}
            lower = [c.lower().strip() for c in df.columns]
            def find_like(names):
                for n in names:
                    if n in lower:
                        i = lower.index(n)
                        return df.columns[i]
                return None

            # required fields (best-effort)
            mcol = find_like(["material"])
            if mcol: rename_map[mcol] = "material"
            d50c = find_like(["d50_um","d50","particle_size_um","particle_size"])
            if d50c: rename_map[d50c] = "d50_um"
            lthc = find_like(["layer_thickness_um","layer_um","layer","layer_thickness"])
            if lthc: rename_map[lthc] = "layer_thickness_um"
            satc = find_like(["binder_saturation_pct","saturation_pct","binder_saturation","saturation"])
            if satc: rename_map[satc] = "binder_saturation_pct"
            gpc  = find_like(["green_pct_td","green_%td","green_density_pct","final_density_pct","green_pct","pct_td"])
            if gpc: rename_map[gpc] = "green_pct_td"
            btyp = find_like(["binder_type","binder"])
            if btyp: rename_map[btyp] = "binder_type"

            df = df.rename(columns=rename_map)
            for k in ["material","d50_um","layer_thickness_um","binder_saturation_pct"]:
                if k not in df.columns:
                    df[k] = np.nan
            return df.dropna(subset=["material"]).copy()
    # Empty fallback with expected columns
    return pd.DataFrame(columns=["material","d50_um","layer_thickness_um","binder_saturation_pct","green_pct_td","binder_type"])

# --------------------------------------------------------------------------------------
# Modeling (quantile GBMs)
# --------------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["material","d50_um","layer_thickness_um","binder_saturation_pct"]

def _make_quantile(q: float):
    cat = ["material"]
    num = ["d50_um","layer_thickness_um","binder_saturation_pct"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )
    model = GradientBoostingRegressor(
        loss="quantile", alpha=q,
        n_estimators=250, max_depth=3, learning_rate=0.06, random_state=42
    )
    return Pipeline([("prep", pre), ("gb", model)])

@st.cache_resource(show_spinner=False)
def fit_quantile_models(df: pd.DataFrame):
    if df.empty or not set(FEATURES).issubset(df.columns):
        class _Dummy:
            def predict(self, X): return np.full((len(X),), 85.0)
        return _Dummy(), _Dummy(), _Dummy()
    y = df["green_pct_td"] if "green_pct_td" in df.columns else df.iloc[:,0]*0 + 85.0
    X = df[FEATURES]
    q10 = _make_quantile(0.10).fit(X, y)
    q50 = _make_quantile(0.50).fit(X, y)
    q90 = _make_quantile(0.90).fit(X, y)
    return q10, q50, q90

# --------------------------------------------------------------------------------------
# Physics guardrails + recommendation logic
# --------------------------------------------------------------------------------------
def sanitize_material_name(s: str) -> str:
    return str(s).strip()

def pack_fraction_furnas(d50_um: float, layer_um: float) -> float:
    # Heuristic proxy for achievable packing as a function of D50 vs. layer thickness
    ratio = (d50_um / max(1e-6, layer_um))
    pf = 0.74 * (1.0 / (1.0 + 0.6 * (1/ratio)))  # clamp below
    return float(np.clip(pf, 0.45, 0.74))

def washburn_binder_suggestion(d50_um: float, target_td: float) -> float:
    # Coarse Washburn-inspired heuristic to split water vs solvent likelihood
    base = 70.0 + (target_td - 90.0) * 0.8
    adj = 15.0 * np.tanh((40.0 - d50_um)/30.0)
    return float(np.clip(base + adj, 35.0, 110.0))

def _roller_speed_rule(d50_um: float) -> float:
    # v_r ∝ 1 / (d50^alpha), alpha≈0.6 (normalized to ~60 mm/s at d50=25 µm)
    alpha = 0.6
    base = 60.0 * (25.0 ** alpha)
    v = base / (max(5.0, d50_um) ** alpha)
    return float(np.clip(v, 15.0, 200.0))

def recommend_with_guardrails(models, material: str, d50_um: float, layer_um: float,
                              target_td: float, n_total: int=5) -> pd.DataFrame:
    q10, q50, q90 = models
    Ls = np.linspace(max(20.0, 0.6*layer_um), min(200.0, 1.6*layer_um), 9)
    Ss = np.linspace(40.0, 110.0, 15)
    rows = []
    for L in Ls:
        for S in Ss:
            x = pd.DataFrame([{
                "material": material,
                "d50_um": d50_um,
                "layer_thickness_um": L,
                "binder_saturation_pct": S
            }])
            y10 = float(q10.predict(x[FEATURES])[0])
            y50 = float(q50.predict(x[FEATURES])[0])
            y90 = float(q90.predict(x[FEATURES])[0])
            pf = pack_fraction_furnas(d50_um, L) * 100.0
            if pf < 90.0:  # guardrail: theoretical packing at least 90%
                continue
            score = (max(0.0, y10 - target_td)) + 0.25*(y50-target_td)
            rows.append((L, S, y10, y50, y90, pf, score))

    if not rows:
        return pd.DataFrame()

    cand = pd.DataFrame(rows, columns=["layer_um","saturation_pct","pred_q10","pred_q50","pred_q90","theoretical_%TD","score"])
    cand = cand.sort_values("score", ascending=False)

    # Enforce mix: ideally 3 water-based + 2 solvent-based
    water_needed, solvent_needed = 3, 2
    out = []
    split_thresh = washburn_binder_suggestion(d50_um, target_td)
    for _, r in cand.iterrows():
        btype = "water_based" if r["saturation_pct"] <= split_thresh else "solvent_based"
        if btype == "water_based" and water_needed > 0:
            out.append((btype, r))
            water_needed -= 1
        elif btype == "solvent_based" and solvent_needed > 0:
            out.append((btype, r))
            solvent_needed -= 1
        if len(out) >= n_total: break

    if len(out) < n_total:
        for _, r in cand.iterrows():
            if any(np.isclose(r["layer_um"], rr[1]["layer_um"]) and np.isclose(r["saturation_pct"], rr[1]["saturation_pct"]) for rr in out):
                continue
            btype = "water_based" if r["saturation_pct"] <= split_thresh else "solvent_based"
            out.append((btype, r))
            if len(out) >= n_total: break

    if not out:
        return pd.DataFrame()

    recs = []
    for btype, r in out:
        recs.append({
            "binder_type": btype,
            "saturation_pct": float(np.round(r["saturation_pct"], 1)),
            "roller_speed_mm_s": _roller_speed_rule(d50_um),
            "layer_um": float(np.round(r["layer_um"], 1)),
            "pred_q10": float(np.round(r["pred_q10"], 1)),
            "pred_q50": float(np.round(r["pred_q50"], 1)),
            "pred_q90": float(np.round(r["pred_q90"], 1)),
            "theoretical_%TD": float(np.round(r["theoretical_%TD"], 1)),
            "material": material,
            "d50_um": float(d50_um)
        })
    outdf = pd.DataFrame(recs)
    outdf.insert(0, "id", [f"Opt-{i+1}" for i in range(len(outdf))])
    return outdf

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="BJAM Digital Twin", layout="wide", page_icon="🧪")
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
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Inputs")
    material = st.text_input("Powder / material", value="316L", help="Material name (free text).")
    d50_um   = st.number_input("Particle size D50 (µm)", value=25.0, min_value=1.0, max_value=200.0, step=0.5)
    layer_um = st.number_input("Layer thickness (µm)", value=60.0, min_value=20.0, max_value=200.0, step=5.0)
    want_density = st.slider("Target green % theoretical density", 80, 99, 90, help="Guardrail target.")
    st.caption("App returns exactly 3 water-based and 2 solvent-based options when feasible.")

    st.divider()
    st.subheader("Digital Twin (STL)")
    st.caption("Optional: upload an STL to estimate layers, binder use, and build time.")
    stl_file = st.file_uploader("STL file", type=["stl"], accept_multiple_files=False)

    recoater_width_mm = st.number_input("Recoater pass width (mm)", value=100.0, min_value=10.0, max_value=500.0, step=5.0)
    build_speed_mm_s  = st.number_input("Nominal roller traverse speed (mm/s)", value=60.0, min_value=5.0, max_value=400.0, step=1.0)
    saturation_pct_ui = st.slider("Assumed binder saturation for twin (%)", 40, 120, 80)

    st.divider()
    st.caption("Models are cached; use 'Data Health' to inspect coverage.")

# Load data and fit models
df = load_dataset()
models = fit_quantile_models(df)

# Header
c1, c2 = st.columns([1.2, 1])
with c1:
    st.title("BJAM Digital Twin — Parameter Optimizer")
    st.write("Physics-informed recommendations with uncertainty and STL-aware build estimates.")
with c2:
    st.metric("Training rows", len(df))
    st.metric("Materials in dataset", df["material"].nunique() if "material" in df.columns else 0)

# Recommendations
with st.expander("Top-5 guardrailed recommendations", expanded=True):
    clean_mat = sanitize_material_name(material)
    rec_df = recommend_with_guardrails(
        models=models,
        material=clean_mat,
        d50_um=d50_um,
        layer_um=layer_um,
        target_td=want_density,
        n_total=5
    )

    if rec_df.empty:
        st.warning("No safe recommendations found near this region. Try adjusting D50 or layer thickness.")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        st.caption("If exact 3 water + 2 solvent cannot be met, nearest safe set is returned.")

# Heatmap sweep
st.subheader("Parameter sweep — heatmap of predicted green %TD")
x_vals = np.linspace(max(0.5*layer_um, 20), min(2.0*layer_um, 200), 30)   # layer thickness
y_vals = np.linspace(20, 120, 30)                                        # binder saturation

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

fig = go.Figure(data=go.Heatmap(z=Z, x=y_vals, y=x_vals, colorbar=dict(title="%TD (q50)")))
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
        dims = (bbox[1] - bbox[0])  # (x,y,z) in STL units
        layer_h_mm = layer_um / 1000.0
        n_layers = int(math.ceil(max(dims[2], 1e-6) / layer_h_mm))
        # quick proxy for per-layer cross section
        xy_area_mm2 = mesh.convex_hull.area if hasattr(mesh, "convex_hull") else (dims[0]*dims[1])
        binder_ml = xy_area_mm2 * n_layers * (saturation_pct_ui/100.0) * 1e-3
        passes = max(1, math.ceil(dims[0] / max(1.0, recoater_width_mm)))
        build_time_min = (passes * n_layers * (dims[1] / max(1.0, build_speed_mm_s))) / 60.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Layers", f"{n_layers:,}")
        c2.metric("XY area per layer (mm²)", f"{xy_area_mm2:,.0f}")
        c3.metric("Binder estimate (mL)", f"{binder_ml:,.1f}")
        c4.metric("Build time (min)", f"{build_time_min:,.1f}")

        # Overhang-risk proxy per layer
        normals = mesh.face_normals
        steep = np.abs(normals[:,2])          # near 1 = vertical; near 0 = overhang
        cz = mesh.triangles_center[:,2] - mesh.bounds[0,2]
        li = np.clip((cz / max(1e-9, dims[2])) * n_layers, 0, n_layers-1).astype(int)
        risk = np.zeros(n_layers); counts = np.zeros(n_layers)
        for s, idx in zip(steep, li):
            risk[idx] += (1.0 - s)
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
            figm.update_layout(xaxis_title="Material (top20)", yaxis_title="Rows",
                               height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figm, use_container_width=True)
        with g2:
            if "d50_um" in df.columns and "green_pct_td" in df.columns:
                figd = go.Figure(go.Scatter(x=df["d50_um"], y=df["green_pct_td"], mode="markers"))
                figd.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
                figd.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD",
                                   height=360, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figd, use_container_width=True)

st.caption("© BJAM AI — physics-informed recommendations with uncertainty (q10/q50/q90).")
