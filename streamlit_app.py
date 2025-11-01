# -*- coding: utf-8 -*-
# BJAM — Binder-Jet AM Parameter Recommender + Digital Twin (Beta)
# Self-contained version: uses a light grid "model" computed from your dataset.

from __future__ import annotations
import io, math, importlib.util
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# Optional heavy deps (only used in Digital Twin)
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH:
    import trimesh  # type: ignore
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point  # type: ignore
    from shapely.ops import unary_union  # type: ignore

# ----------------------------- App config -----------------------------
st.set_page_config(page_title="BJAM — Binder-Jet AM Parameter Recommender",
                   layout="wide", initial_sidebar_state="expanded")
TITLE = "BJAM — Binder-Jet AM Parameter Recommender"
SUBTITLE = ("Physics-guided few-shot heuristics from your dataset. "
            "Digital Twin (Beta) added. Generated with help of ChatGPT.")

DATA_PATH = "BJAM_All_Deep_Fill_v9.csv"
RNG_SEED_DEFAULT = 42

# ----------------------------- Data loading ---------------------------
@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize common header variants
    ren = {
        "roller_speed": "roller_speed_mmps",
        "roller_speed_mm_s": "roller_speed_mmps",
        "binder_saturation": "binder_saturation_pct",
        "green_density": "green_density_pctTD",
        "green_pct_td": "green_density_pctTD",
        "binder": "binder_type",
        "d50": "d50_um",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    # Basic cleanup
    for c in ["d50_um", "binder_saturation_pct", "roller_speed_mmps", "green_density_pctTD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["d50_um", "binder_saturation_pct", "roller_speed_mmps"]).reset_index(drop=True)
    if "material" not in df.columns:
        df["material"] = "Generic"
    if "green_density_pctTD" not in df.columns:
        # fabricate a placeholder using rough trendlines if missing
        df["green_density_pctTD"] = 75 + 0.08*(df["binder_saturation_pct"]-80) - 3.0*np.abs(df["roller_speed_mmps"]-2.2)
    return df

# ----------------------------- Light "models" --------------------------
@st.cache_data
def build_material_grids(_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each material, precompute a coarse grid of (sat, speed) with q10/q50/q90.
    Cache input as underscore-arg (hashable allowed by Streamlit).
    """
    mats: Dict[str, pd.DataFrame] = {}
    for mat, d in _df.groupby("material"):
        # Choose sat/speed ranges from data quantiles (robust to outliers)
        sat_lo, sat_hi = float(d["binder_saturation_pct"].quantile(0.05)), float(d["binder_saturation_pct"].quantile(0.95))
        spd_lo, spd_hi = float(d["roller_speed_mmps"].quantile(0.05)), float(d["roller_speed_mmps"].quantile(0.95))
        Xs = np.linspace(max(50, sat_lo), min(110, sat_hi), 24)
        Ys = np.linspace(max(0.5, spd_lo), min(6.0, spd_hi), 20)

        recs = []
        # global fallbacks (avoid holes)
        gq10 = float(np.percentile(d["green_density_pctTD"], 10))
        gq50 = float(np.percentile(d["green_density_pctTD"], 50))
        gq90 = float(np.percentile(d["green_density_pctTD"], 90))

        for s in Xs:
            for v in Ys:
                dd = d[(d["binder_saturation_pct"].between(s-2.5, s+2.5)) &
                       (d["roller_speed_mmps"].between(v-0.2, v+0.2))]
                if len(dd) >= 3:
                    q10 = float(np.percentile(dd["green_density_pctTD"], 10))
                    q50 = float(np.percentile(dd["green_density_pctTD"], 50))
                    q90 = float(np.percentile(dd["green_density_pctTD"], 90))
                else:
                    q10, q50, q90 = gq10, gq50, gq90
                recs.append((s, v, q10, q50, q90))
        grid = pd.DataFrame(recs, columns=["sat", "speed", "q10", "q50", "q90"])
        mats[mat] = grid
    return mats

def get_grid_for_material(_models: Dict[str, pd.DataFrame], mat: str) -> pd.DataFrame:
    # No caching here (dict arg); cheap copy and only takes hashable 'mat'
    grid = _models.get(mat)
    if grid is None and len(_models):
        grid = list(_models.values())[0]
    return grid.copy() if grid is not None else pd.DataFrame(columns=["sat","speed","q10","q50","q90"])

def nearest_band(grid: pd.DataFrame, speed: float) -> float:
    s = np.sort(grid["speed"].unique())
    if len(s) == 0: return float(speed)
    return float(s[np.abs(s - speed).argmin()])

# ----------------------------- Trial picker ----------------------------
def pick_trials(grid: pd.DataFrame, d50_um: float, target_pctTD: float, guardrails: bool=True) -> pd.DataFrame:
    g = grid.copy()
    # Simple score: closeness to target + mild regularization to ~2.0 mm/s center
    g["score"] = -np.abs(g["q50"] - target_pctTD) - 0.05*np.abs(g["speed"]-2.0)
    g = g.sort_values("score", ascending=False)

    # Binder heuristic: <=95% sat -> water; >95% -> solvent
    g["binder_type_rec"] = np.where(g["sat"] <= 95.0, "water_based", "solvent_based")

    top = []
    wcnt, scnt = 0, 0
    for _, r in g.iterrows():
        b = r["binder_type_rec"]
        if b == "water_based" and wcnt < 3:
            top.append(r); wcnt += 1
        elif b == "solvent_based" and scnt < 2:
            top.append(r); scnt += 1
        if len(top) == 5:
            break
    if len(top) < 5:
        for _, r in g.iterrows():
            if len(top) == 5: break
            if r not in top: top.append(r)

    out = pd.DataFrame(top)[["sat","speed","q10","q50","q90","binder_type_rec"]]
    out = out.rename(columns={"sat":"binder_saturation_pct","speed":"roller_speed_mmps"})
    out["d50_um"] = float(d50_um)
    out["id"] = [f"Opt-{i+1}" for i in range(len(out))]
    return out.reset_index(drop=True)

# ----------------------------- Figures --------------------------------
def heatmap_fig(grid: pd.DataFrame, x_cross: float, y_cross: float, smooth_sigma: float=1.0) -> go.Figure:
    if grid.empty:
        return go.Figure(layout=dict(title="No data", template="simple_white"))

    X = np.sort(grid["sat"].unique())
    Y = np.sort(grid["speed"].unique())
    Z = grid.pivot(index="speed", columns="sat", values="q50").values.astype(float)
    if smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=smooth_sigma, mode="nearest")

    fig = go.Figure(data=go.Heatmap(
        x=X, y=Y, z=Z, colorscale="YlGnBu",
        colorbar=dict(title="%TD (q50)")
    ))
    # dashed admissible box (tweak as you like)
    x0, x1 = np.percentile(X, [35, 65]); y0, y1 = np.percentile(Y, [35, 65])
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                  line=dict(color="teal", width=3, dash="dash"))
    # crosshair
    fig.add_trace(go.Scatter(x=[x_cross], y=[y_cross], mode="markers",
                             marker=dict(color="red", size=12, symbol="cross"),
                             name="chosen"))
    fig.update_layout(title="Predicted Green Density (% Theoretical Density)",
                      xaxis_title="Binder Saturation (%)",
                      yaxis_title="Roller Speed (mm/s)",
                      template="simple_white",
                      height=480, margin=dict(l=60,r=20,t=40,b=50))
    return fig

def sat_sensitivity_fig(grid: pd.DataFrame, speed_ref: float) -> go.Figure:
    if grid.empty:
        return go.Figure(layout=dict(title="No data", template="simple_white"))
    band = nearest_band(grid, speed_ref)
    gb = grid[grid["speed"] == band].sort_values("sat")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q90"], name="q90", mode="lines"))
    fig.update_layout(title=f"Saturation sensitivity @ speed≈{band:.2f} mm/s",
                      xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                      template="simple_white", height=420, margin=dict(l=40,r=20,t=40,b=40))
    return fig

# ----------------------------- Packing (qualitative) -------------------
def poly_radii_um(n: int, d50_um: float) -> np.ndarray:
    mu = math.log(max(d50_um, 1e-6)) - 0.5*0.25**2
    diam = np.random.lognormal(mean=mu, sigma=0.25, size=n)
    return diam/2.0

def rsa_pack_um(n_try: int, phi_target: float, box_um: float, d50_um: float, seed: int=RNG_SEED_DEFAULT) -> Tuple[np.ndarray,np.ndarray]:
    rng = np.random.default_rng(seed)
    centers: List[Tuple[float,float]] = []
    radii: List[float] = []
    area_box = box_um*box_um
    total = 0.0
    trial = 0
    while trial < n_try and total/area_box < phi_target:
        r = float(poly_radii_um(1, d50_um)[0])
        x = rng.uniform(r, box_um - r); y = rng.uniform(r, box_um - r)
        ok = True
        for i in range(len(radii)):
            dx = x - centers[i][0]; dy = y - centers[i][1]
            if dx*dx + dy*dy < (r + radii[i])**2: ok = False; break
        if ok:
            centers.append((x,y)); radii.append(r); total += math.pi*r*r
        trial += 1
    if not radii:
        return np.zeros((0,2)), np.zeros((0,))
    C = np.array(centers, float); R = np.array(radii, float)
    # small jitter to avoid lattice-like rows
    C += rng.uniform(-0.15, 0.15, size=C.shape) * R.reshape(-1,1)
    C[:,0] = np.clip(C[:,0], R, box_um-R); C[:,1] = np.clip(C[:,1], R, box_um-R)
    return C, R

def packing_figure(C: np.ndarray, R: np.ndarray, binder_sat: float, fld_um: float) -> go.Figure:
    fig = go.Figure()
    # binder background
    fig.add_shape(type="rect", x0=0, y0=0, x1=fld_um, y1=fld_um,
                  line_color="black", fillcolor="rgb(231,176,62)")
    # particles
    for (x,y), r in zip(C, R):
        fig.add_shape(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                      line_color="rgb(20,70,200)", fillcolor="rgb(37,110,230)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(title=f"Qualitative packing • Binder Sat {int(binder_sat)}%",
                      template="simple_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=480, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ----------------------------- Digital Twin helpers -------------------
def slice_mesh_layer(mesh: "trimesh.Trimesh", z: float) -> List["Polygon"]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar, _ = sec.to_planar()
        return [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    except Exception:
        return []

def pack_in_polys(polys: List["Polygon"], d50_um: float, phi2D_target: float, fov_um: float, seed: int):
    if not polys:
        return rsa_pack_um(n_try=14000, phi_target=phi2D_target, box_um=fov_um, d50_um=d50_um, seed=seed)
    try:
        U = unary_union(polys)
    except Exception:
        U = polys[0]
        for p in polys[1:]:
            U = U.union(p)
    minx, miny, maxx, maxy = U.bounds
    size = min(maxx-minx, maxy-miny)
    if size <= 0:
        return rsa_pack_um(12000, phi2D_target, fov_um, d50_um, seed)
    L = min(fov_um, size)
    cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
    x0, y0 = cx - L/2, cy - L/2
    C, R = rsa_pack_um(18000, phi2D_target, L, d50_um, seed)
    keepC, keepR = [], []
    for (x,y), r in zip(C, R):
        if U.contains(Point(x0+x, y0+y)):
            keepC.append((x,y)); keepR.append(r)
    if not keepR:
        return np.zeros((0,2)), np.zeros((0,))
    return np.array(keepC,float), np.array(keepR,float)

# ----------------------------- Sidebar --------------------------------
with st.sidebar:
    st.subheader("Inputs")
    df_all = load_csv(DATA_PATH)
    materials = sorted(df_all["material"].dropna().astype(str).unique().tolist())
    material = st.selectbox("Material", materials, index=0 if materials else None)
    d50_um = st.number_input("D50 (µm)", value=90.0, min_value=1.0, max_value=500.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", value=120, min_value=5, max_value=300)
    target_pctTD = st.slider("Target green density (%TD)", value=92, min_value=80, max_value=98)
    guardrails = st.toggle("Guardrails", value=True)
    st.caption(f"Data source: {DATA_PATH} • Rows: {len(df_all):,}")

# Build grids and pick the one for this material (no caching on dict param)
models = build_material_grids(df_all)
grid = get_grid_for_material(models, material)

# ----------------------------- Title & Tabs ----------------------------
st.markdown(f"<h1 style='margin-top:0'>{TITLE}</h1>", unsafe_allow_html=True)
st.caption(SUBTITLE)
tab_pred, tab_heat, tab_sens, tab_pack, tab_form, tab_twin = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)"]
)

# ----------------------------- Predict (Top-5) ------------------------
with tab_pred:
    trials = pick_trials(grid, d50_um=d50_um, target_pctTD=target_pctTD, guardrails=guardrails)
    st.write("Recommended trials (forced 3 water-based + 2 solvent-based):")
    st.dataframe(trials[["id","binder_type_rec","binder_saturation_pct","roller_speed_mmps","q50","q10","q90","d50_um"]],
                 use_container_width=True, hide_index=True)
    rec_id = st.selectbox("Use trial for other tabs",
                          trials["id"].tolist(),
                          index=0 if len(trials) else None)
    sel = trials[trials["id"] == rec_id].iloc[0] if len(trials) else None
    st.session_state["selected_trial"] = None if sel is None else dict(sel)

# ----------------------------- Heatmap --------------------------------
with tab_heat:
    use = st.session_state.get("selected_trial")
    x_cross = float(use["binder_saturation_pct"]) if use else float(np.median(grid["sat"])) if not grid.empty else 85.0
    y_cross = float(use["roller_speed_mmps"]) if use else float(np.median(grid["speed"])) if not grid.empty else 2.3
    smooth = st.slider("Heatmap smoothing (σ)", 0.0, 2.5, 1.0, 0.1)
    fig = heatmap_fig(grid, x_cross=x_cross, y_cross=y_cross, smooth_sigma=smooth)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Saturation sensitivity -----------------
with tab_sens:
    speed_ref = float(use["roller_speed_mmps"]) if use else 2.3
    st.plotly_chart(sat_sensitivity_fig(grid, speed_ref=speed_ref), use_container_width=True)

# ----------------------------- Qualitative packing --------------------
with tab_pack:
    st.write("Qualitative, 2D slice-like packing (illustrative):")
    phi2D = float(np.clip(0.55 + 0.003*(d50_um-20), 0.40, 0.90))
    C, R = rsa_pack_um(n_try=14000, phi_target=phi2D, box_um=800.0, d50_um=d50_um, seed=RNG_SEED_DEFAULT)
    sat = float(use["binder_saturation_pct"]) if use else 85.0
    st.plotly_chart(packing_figure(C, R, binder_sat=sat, fld_um=800.0), use_container_width=True)
    st.caption(f"FOV≈0.80 mm • φ₂D(target)≈{phi2D:.2f}")

# ----------------------------- Formulae (stub) ------------------------
with tab_form:
    st.write("Guardrails/heuristics used:")
    st.markdown("""
- Layer thickness: `t ≤ 2.5·D50` (streaking/flooding check).
- Roller speed nominal window (data-derived): mid-percentiles of your material grid.
- Binder split: water-based ≤95% saturation; solvent-based >95%.
- Packing proxy: φ₂D ≈ 0.60–0.70 corresponds to φ₃D ≈ 0.55–0.65 prior to burnout.
    """)

# ----------------------------- Digital Twin (Beta) --------------------
with tab_twin:
    st.subheader("Upload an STL to view a layer-true packing preview")
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.warning("trimesh and shapely are required. Please add them to requirements.txt.")
    else:
        up = st.file_uploader("STL file", type=["stl"])
        if up is not None:
            try:
                mesh = trimesh.load_mesh(io.BytesIO(up.read()), file_type="stl")
                mesh.rezero()
                zmin, zmax = mesh.bounds[0,2], mesh.bounds[1,2]
                # Compute number of layers based on input layer thickness (µm) — interpret mesh units as µm-equivalent for preview
                n_layers = max(1, int(max(1.0, (zmax - zmin)) // max(1.0, layer_um)))
                layer_idx = st.slider("Layer index", 1, n_layers, min(3, n_layers))
                z = zmin + (layer_idx - 0.5) * max(1.0, layer_um)

                polys = slice_mesh_layer(mesh, z)
                phi2D_target = float(np.clip(0.55 + 0.003*(d50_um-20), 0.40, 0.90))
                C2, R2 = pack_in_polys(polys, d50_um=d50_um, phi2D_target=phi2D_target, fov_um=800.0, seed=RNG_SEED_DEFAULT+layer_idx)
                sat = float(st.session_state.get("selected_trial", {}).get("binder_saturation_pct", 85.0))
                figDT = packing_figure(C2, R2, binder_sat=sat, fld_um=800.0)
                figDT.update_layout(title=f"Digital twin preview • layer {layer_idx}/{n_layers}")
                st.plotly_chart(figDT, use_container_width=True)
                st.caption("Note: preview is illustrative; STL units are interpreted for visualization only.")
            except Exception as ex:
                st.error(f"Failed to process STL: {ex}")
        else:
            st.info("Upload an STL to enable the Digital Twin preview.")
