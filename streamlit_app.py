# -*- coding: utf-8 -*-
# BJAM Digital Twin — FAST build: STL meshing + cached voxel packing + lazy ML
from __future__ import annotations
import io, math, hashlib, importlib.util
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional geometry backend
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
if HAVE_TRIMESH:
    import trimesh  # type: ignore

st.set_page_config(page_title="BJAM Digital Twin", layout="wide", page_icon="🧪")

# ============================= Sidebar ========================================
with st.sidebar:
    st.header("Inputs")
    material = st.text_input("Powder / material", value="316L")
    d50_um   = st.number_input("Particle size D50 (µm)", value=25.0, min_value=1.0, max_value=200.0, step=0.5)
    layer_um = st.number_input("Layer thickness (µm)", value=60.0, min_value=20.0, max_value=200.0, step=5.0)
    want_density = st.slider("Target green % theoretical density", 80, 99, 90)

    st.divider()
    st.subheader("Digital Twin (STL)")
    st.caption("Upload STL to preview mesh and packing by layer.")
    stl_file = st.file_uploader("STL file", type=["stl"], accept_multiple_files=False)

    fast_mode = st.toggle("Fast mode (recommended)", value=True,
                          help="Uses adaptive voxel pitch and samples ≤160 slices. Much faster.")
    recoater_width_mm = st.number_input("Recoater pass width (mm)", value=100.0, min_value=10.0, max_value=500.0, step=5.0)
    build_speed_mm_s  = st.number_input("Roller traverse speed (mm/s)", value=60.0, min_value=5.0, max_value=400.0, step=1.0)
    saturation_pct_ui = st.slider("Assumed binder saturation for twin (%)", 40, 120, 80)

    st.divider()
    go_btn = st.button("Run optimizer", type="primary", use_container_width=True)

# ============================= Data loading ===================================
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
            lower = [c.lower().strip() for c in df.columns]
            def like(names):
                for n in names:
                    if n in lower:
                        return df.columns[lower.index(n)]
                return None
            ren = {}
            mcol = like(["material"]); ren[mcol] = "material" if mcol else None
            d50c = like(["d50_um","d50","particle_size_um","particle_size"]); ren[d50c] = "d50_um" if d50c else None
            lthc = like(["layer_thickness_um","layer_um","layer","layer_thickness"]); ren[lthc] = "layer_thickness_um" if lthc else None
            satc = like(["binder_saturation_pct","saturation_pct","binder_saturation","saturation"]); ren[satc] = "binder_saturation_pct" if satc else None
            gpc  = like(["green_pct_td","green_%td","green_density_pct","final_density_pct","green_pct","pct_td"]); ren[gpc] = "green_pct_td" if gpc else None
            btyp = like(["binder_type","binder"]); ren[btyp] = "binder_type" if btyp else None
            ren = {k:v for k,v in ren.items() if k}
            df = df.rename(columns=ren)
            for k in ["material","d50_um","layer_thickness_um","binder_saturation_pct"]:
                if k not in df.columns:
                    df[k] = np.nan
            return df.dropna(subset=["material"]).copy()
    return pd.DataFrame(columns=["material","d50_um","layer_thickness_um","binder_saturation_pct","green_pct_td","binder_type"])

df = load_dataset()

# ============================= Lazy ML (imports inside) =======================
FEATURES = ["material","d50_um","layer_thickness_um","binder_saturation_pct"]

def _lazy_import_sklearn():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor
    return OneHotEncoder, ColumnTransformer, Pipeline, GradientBoostingRegressor

def _make_quantile(q: float):
    OneHotEncoder, ColumnTransformer, Pipeline, GradientBoostingRegressor = _lazy_import_sklearn()
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["material"]),
            ("num", "passthrough", ["d50_um","layer_thickness_um","binder_saturation_pct"]),
        ]
    )
    # Fewer trees for speed; good enough for steering UI
    model = GradientBoostingRegressor(
        loss="quantile", alpha=q,
        n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42
    )
    return Pipeline([("prep", pre), ("gb", model)])

@st.cache_resource(show_spinner=False)
def fit_quantile_models(df: pd.DataFrame):
    if df.empty or not set(FEATURES).issubset(df.columns):
        class _Dummy:
            def predict(self, X): return np.full((len(X),), 85.0)
        return _Dummy(), _Dummy(), _Dummy()
    y = df["green_pct_td"] if "green_pct_td" in df.columns else np.full(len(df), 85.0)
    X = df[FEATURES]
    q10 = _make_quantile(0.10).fit(X, y)
    q50 = _make_quantile(0.50).fit(X, y)
    q90 = _make_quantile(0.90).fit(X, y)
    return q10, q50, q90

# ============================= Physics helpers ================================
def sanitize_material_name(s: str) -> str:
    return str(s).strip()

def pack_fraction_furnas(d50_um: float, layer_um: float) -> float:
    ratio = (d50_um / max(1e-6, layer_um))
    pf = 0.74 * (1.0 / (1.0 + 0.6 * (1/ratio)))
    return float(np.clip(pf, 0.45, 0.74))

def washburn_binder_suggestion(d50_um: float, target_td: float) -> float:
    base = 70.0 + (target_td - 90.0) * 0.8
    adj  = 15.0 * np.tanh((40.0 - d50_um)/30.0)
    return float(np.clip(base + adj, 35.0, 110.0))

def _roller_speed_rule(d50_um: float) -> float:
    alpha = 0.6
    base = 60.0 * (25.0 ** alpha)
    v = base / (max(5.0, d50_um) ** alpha)
    return float(np.clip(v, 15.0, 200.0))

# ============================= Recommender ====================================
def recommend_with_guardrails(models, material: str, d50_um: float, layer_um: float,
                              target_td: float, n_total: int=5) -> pd.DataFrame:
    q10, q50, q90 = models
    Ls = np.linspace(max(20.0, 0.6*layer_um), min(200.0, 1.6*layer_um), 9)
    Ss = np.linspace(40.0, 110.0, 15)
    rows = []
    for L in Ls:
        for S in Ss:
            x = pd.DataFrame([{
                "material": material, "d50_um": d50_um,
                "layer_thickness_um": L, "binder_saturation_pct": S
            }])
            y10 = float(q10.predict(x[FEATURES])[0])
            y50 = float(q50.predict(x[FEATURES])[0])
            y90 = float(q90.predict(x[FEATURES])[0])
            pf = pack_fraction_furnas(d50_um, L) * 100.0
            if pf < 90.0:
                continue
            score = (max(0.0, y10 - target_td)) + 0.25*(y50 - target_td)
            rows.append((L, S, y10, y50, y90, pf, score))
    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows, columns=["layer_um","saturation_pct","pred_q10","pred_q50","pred_q90","theoretical_%TD","score"])
    cand = cand.sort_values("score", ascending=False)

    water_needed, solvent_needed = 3, 2
    out = []
    split = washburn_binder_suggestion(d50_um, target_td)
    for _, r in cand.iterrows():
        btype = "water_based" if r["saturation_pct"] <= split else "solvent_based"
        if btype=="water_based" and water_needed>0:
            out.append((btype, r)); water_needed -= 1
        elif btype=="solvent_based" and solvent_needed>0:
            out.append((btype, r)); solvent_needed -= 1
        if len(out) >= n_total: break
    if len(out) < n_total:
        for _, r in cand.iterrows():
            btype = "water_based" if r["saturation_pct"] <= split else "solvent_based"
            out.append((btype, r))
            if len(out) >= n_total: break

    recs = []
    for btype, r in out[:n_total]:
        recs.append({
            "id": f"Opt-{len(recs)+1}",
            "material": material, "d50_um": float(d50_um),
            "binder_type": btype,
            "layer_um": float(np.round(r["layer_um"], 1)),
            "saturation_pct": float(np.round(r["saturation_pct"], 1)),
            "roller_speed_mm_s": _roller_speed_rule(d50_um),
            "pred_q10": float(np.round(r["pred_q10"], 1)),
            "pred_q50": float(np.round(r["pred_q50"], 1)),
            "pred_q90": float(np.round(r["pred_q90"], 1)),
            "theoretical_%TD": float(np.round(r["theoretical_%TD"], 1)),
        })
    return pd.DataFrame(recs)

# ============================= Heatmap cache ==================================
@st.cache_data(show_spinner=False)
def cached_heatmap_pred(material: str, d50_um: float, layer_um: float, x_lo: float, x_hi: float,
                        y_lo: float, y_hi: float, nx: int, ny: int, q50_model):
    x_vals = np.linspace(x_lo, x_hi, nx)
    y_vals = np.linspace(y_lo, y_hi, ny)
    grid = pd.DataFrame([
        {"material": material, "d50_um": d50_um, "layer_thickness_um": L, "binder_saturation_pct": S}
        for L in x_vals for S in y_vals
    ])
    Z = q50_model.predict(grid[["material","d50_um","layer_thickness_um","binder_saturation_pct"]]).reshape(nx, ny)
    return x_vals, y_vals, Z

# ============================= STL voxel cache =================================
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=False)
def voxelize_cached(stl_bytes: bytes, layer_um: float, fast_mode: bool, dims_hint: Tuple[float,float,float]):
    if not HAVE_TRIMESH:
        raise RuntimeError("trimesh not available")
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type='stl', force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "dump"):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("STL did not parse into a single mesh")

    bbox = mesh.bounds
    dims = (bbox[1] - bbox[0])  # (x,y,z) in STL units (assumed mm)
    z_height_mm = float(dims[2])

    layer_h_mm = max(1e-6, layer_um / 1000.0)
    # Adaptive pitch: coarser of (layer pitch, diag/256) in FAST mode
    diag = float(np.linalg.norm(dims))
    max_voxels = 256 if fast_mode else 384
    adaptive_pitch = max(layer_h_mm, diag / max_voxels)

    pitch = adaptive_pitch if fast_mode else layer_h_mm
    vox = mesh.voxelized(pitch)
    dense = vox.matrix  # (nx,ny,nz) bool

    fill_per_layer = dense.sum(axis=(0,1))
    max_in_plane = dense.shape[0] * dense.shape[1]
    packing_pct = (fill_per_layer / max(1, max_in_plane)) * 100.0

    # Map voxel-slices to physical layer count
    n_layers = int(max(1, math.ceil(z_height_mm / layer_h_mm)))
    if fast_mode:
        # Sample to at most 160 layers evenly for speed
        target = min(160, n_layers)
        idx = np.linspace(0, max(1, len(packing_pct)-1), target).round().astype(int)
        packing_pct = packing_pct[np.clip(idx, 0, len(packing_pct)-1)]
        n_out = target
    else:
        # Resample voxel slices to number of physical layers (nearest)
        if len(packing_pct) != n_layers:
            idx = np.linspace(0, max(1, len(packing_pct)-1), n_layers).round().astype(int)
            packing_pct = packing_pct[np.clip(idx, 0, len(packing_pct)-1)]
        n_out = n_layers

    # Simplified stats for UI
    xy_span_x = float(dims[0]); xy_span_y = float(dims[1])
    return {
        "mesh_bounds": dims.tolist(),
        "n_layers": int(n_layers),
        "n_out": int(n_out),
        "packing_pct": packing_pct.astype(float),
        "xy_span_x": xy_span_x,
        "xy_span_y": xy_span_y,
        "pitch_used_mm": float(pitch),
    }

# ============================= Header =========================================
c1, c2 = st.columns([1.2, 1])
with c1:
    st.title("BJAM Digital Twin — Optimizer & Layer Packing (Fast)")
    st.write("Cached STL voxelization + lazy ML. Toggle Fast mode for instant packing plots.")
with c2:
    st.metric("Training rows", len(df))
    st.metric("Materials", df["material"].nunique() if "material" in df.columns else 0)

# ============================= Run models on demand ===========================
if go_btn:
    with st.spinner("Training quantile models…"):
        q10, q50, q90 = fit_quantile_models(df)

    with st.expander("Top-5 guardrailed recommendations", expanded=True):
        rec_df = recommend_with_guardrails(
            models=(q10, q50, q90),
            material=sanitize_material_name(material),
            d50_um=d50_um, layer_um=layer_um, target_td=want_density, n_total=5
        )
        if rec_df.empty:
            st.warning("No safe recommendations near this region. Try adjusting D50 or layer thickness.")
        else:
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # Heatmap
    st.subheader("Parameter sweep — predicted green %TD (q50)")
    nx, ny = (20, 20) if fast_mode else (30, 30)
    x_lo, x_hi = max(0.5*layer_um, 20), min(2.0*layer_um, 200)
    y_lo, y_hi = 20, 120
    x_vals, y_vals, Z = cached_heatmap_pred(
        sanitize_material_name(material), d50_um, layer_um, x_lo, x_hi, y_lo, y_hi, nx, ny, q50
    )
    fig = go.Figure(data=go.Heatmap(z=Z, x=y_vals, y=x_vals, colorbar=dict(title="%TD (q50)")))
    fig.add_hline(y=layer_um, line=dict(color="#111827", dash="dash"))
    fig.add_vline(x=saturation_pct_ui, line=dict(color="#111827", dash="dash"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Layer thickness (µm)",
                      height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ============================= STL preview & packing ==========================
st.subheader("Digital Twin — STL preview & layer packing")
if stl_file is not None:
    if not HAVE_TRIMESH:
        st.error("trimesh is not installed. Check requirements.txt.")
    else:
        raw = stl_file.read()
        file_hash = _hash_bytes(raw)
        # Quick mesh just to show preview & dims (no cache needed for preview)
        try:
            mesh = trimesh.load(io.BytesIO(raw), file_type='stl', force='mesh')
            if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "dump"):
                mesh = trimesh.util.concatenate(mesh.dump())
            if isinstance(mesh, trimesh.Trimesh):
                V, F = mesh.vertices, mesh.faces
                if len(F) > 120_000 and fast_mode:
                    mesh_s = mesh.simplify_quadratic_decimation(int(len(F) * 0.25))
                    V, F = mesh_s.vertices, mesh_s.faces
                fig3d = go.Figure(data=[go.Mesh3d(
                    x=V[:,0], y=V[:,1], z=V[:,2],
                    i=F[:,0], j=F[:,1], k=F[:,2],
                    opacity=0.7, flatshading=True
                )])
                fig3d.update_layout(scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
                                    height=420, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig3d, use_container_width=True)

                dims = (mesh.bounds[1] - mesh.bounds[0])
                # Cached voxelization (fast on repeat)
                with st.spinner("Voxelizing for packing (cached)…"):
                    voxres = voxelize_cached(raw, layer_um, fast_mode, (float(dims[0]), float(dims[1]), float(dims[2])))
                packing_pct = voxres["packing_pct"]
                n_layers = voxres["n_layers"]
                n_out = voxres["n_out"]

                passes = max(1, math.ceil(voxres["xy_span_x"] / max(1.0, recoater_width_mm)))
                build_time_min = (passes * n_out * (voxres["xy_span_y"] / max(1.0, build_speed_mm_s))) / 60.0
                avg_fill = float(np.mean(packing_pct)/100.0) if len(packing_pct) else 0.0
                area_mm2 = voxres["xy_span_x"] * voxres["xy_span_y"]
                binder_ml = area_mm2 * n_out * avg_fill * (saturation_pct_ui/100.0) * 1e-3

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Physical layers", f"{n_layers:,}")
                c2.metric("Computed slices", f"{n_out:,}")
                c3.metric("Binder estimate (mL)", f"{binder_ml:,.1f}")
                c4.metric("Build time (min)", f"{build_time_min:,.1f}")
                st.caption(f"Voxel pitch used: {voxres['pitch_used_mm']:.4f} mm  •  Fast mode: {fast_mode}")

                figp = go.Figure()
                figp.add_trace(go.Scatter(x=list(range(len(packing_pct))), y=packing_pct, mode="lines"))
                figp.add_hline(y=90, line=dict(dash="dash"), annotation_text="90% target", annotation_position="top left")
                figp.update_layout(xaxis_title="Layer index (Z ↑)", yaxis_title="Packing proxy per layer (%)",
                                   height=320, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figp, use_container_width=True)
            else:
                st.error("Could not parse STL into a single mesh. Try a simpler/manifold file.")
        except Exception as e:
            st.error(f"STL preview/voxelization failed: {e}")

# ============================= Data Health ====================================
with st.expander("Data Health & Coverage", expanded=False):
    if df.empty:
        st.info("No dataset loaded.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            by_mat = df.groupby("material").size().sort_values(ascending=False).head(20)
            figm = go.Figure(go.Bar(x=by_mat.index.astype(str), y=by_mat.values))
            figm.update_layout(xaxis_title="Material (top 20)", yaxis_title="Rows",
                               height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(figm, use_container_width=True)
        with g2:
            if "d50_um" in df.columns and "green_pct_td" in df.columns:
                figd = go.Figure(go.Scatter(x=df["d50_um"], y=df["green_pct_td"], mode="markers"))
                figd.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
                figd.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD",
                                   height=360, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(figd, use_container_width=True)

st.caption("© BJAM AI — cached STL voxel packing + lazy quantile ML. Toggle Fast mode to accelerate heavy steps.")
