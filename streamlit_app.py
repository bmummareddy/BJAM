# -*- coding: utf-8 -*-
# BJAM — Binder-Jet AM Parameter Recommender (Stable Baseline) + Optional Digital Twin
# Works with BJAM_All_Deep_Fill_v9.csv or BJAM_cleaned.csv (or an uploaded CSV)

from __future__ import annotations
import io, os, re, math, importlib.util
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

APP_TITLE = "BJAM — Binder-Jet AM Parameter Recommender (Stable Baseline)"
DATA_FILES = ["BJAM_All_Deep_Fill_v9.csv", "BJAM_cleaned.csv"]
RNG_SEED = 42

# ------------------------------------------------------------------------------
# Utilities: normalize headers and find columns (robust; no assumptions)
# ------------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w%]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _find_col(df: pd.DataFrame, exact: List[str], contains: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    nm = {c: _norm(c) for c in df.columns}
    inv = {}
    for k, v in nm.items():
        inv.setdefault(v, k)
    for e in exact:
        v = inv.get(_norm(e))
        if v: return v
    for c, n in nm.items():
        if any(tok in n for tok in contains):
            return c
    return None

def _coerce_num(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series([np.nan]*len(df))
    return pd.to_numeric(df[col], errors="coerce")

def normalize_df(df_raw: pd.DataFrame, notes: List[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = df_raw.copy()

    # Hard maps for common variants across your files
    rename_hard = {
        # speed
        "roller_speed": "roller_speed_mmps",
        "roller_speed_mm_s": "roller_speed_mmps",
        "roller_traverse_speed_mm_s": "roller_speed_mmps",
        "recoater_speed_mm_s": "roller_speed_mmps",
        "spread_speed_mm_s": "roller_speed_mmps",
        # saturation
        "binder_saturation": "binder_saturation_pct",
        "saturation_percent": "binder_saturation_pct",
        "sat_percent": "binder_saturation_pct",
        "sat_%": "binder_saturation_pct",
        # green density
        "green_density": "green_density_pctTD",
        "green_pct_td": "green_density_pctTD",
        "green_%_td": "green_density_pctTD",
        "green_theoretical_density_pct": "green_density_pctTD",
        # particle size
        "d50": "d50_um",
        "median_particle_size": "d50_um",
        "particle_size_um": "d50_um",
        "particle_size_microns": "d50_um",
        "median_diameter_um": "d50_um",
        # binder & material
        "binder": "binder_type",
        "binder_chemistry": "binder_type",
        "binder_family": "binder_type",
        "material_name": "material",
        "powder": "material",
        "alloy": "material",
    }
    for k, v in rename_hard.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Fuzzy detect
    c_d50 = _find_col(df,
        exact=["d50_um","d50","median_particle_size","particle_size_um","particle_size_microns","median_diameter_um"],
        contains=["d50","particle_size","median_particle","median_diameter"])
    c_sat = _find_col(df,
        exact=["binder_saturation_pct","saturation_percent","sat_percent","sat_%","binder_saturation"],
        contains=["saturation","sat"])
    c_spd = _find_col(df,
        exact=["roller_speed_mmps","roller_speed_mm_s","roller_traverse_speed_mm_s","recoater_speed_mm_s","spread_speed_mm_s","roller_speed"],
        contains=["roller","recoater","spread","traverse","speed"])
    c_td  = _find_col(df,
        exact=["green_density_pctTD","green_pct_td","green_%_td","green_theoretical_density_pct","green_density"],
        contains=["green","theoretical","%td","pcttd"])
    c_bind= _find_col(df,
        exact=["binder_type","binder","binder_chemistry","binder_family"],
        contains=["binder"])
    c_mat = _find_col(df,
        exact=["material","material_name","powder","alloy"],
        contains=["material","powder","alloy"])

    norm = pd.DataFrame(index=df.index)
    norm["material"] = (df[c_mat] if c_mat else "Generic").astype(str)
    norm["binder_type"] = (df[c_bind] if c_bind else "water_based").astype(str)
    norm["d50_um"] = _coerce_num(df, c_d50)
    norm["binder_saturation_pct"] = _coerce_num(df, c_sat)
    norm["roller_speed_mmps"] = _coerce_num(df, c_spd)
    norm["green_density_pctTD"] = _coerce_num(df, c_td)

    # Fill green %TD if missing entirely so visuals still render
    if norm["green_density_pctTD"].isna().all():
        base = 80 + 0.07*(norm["binder_saturation_pct"].fillna(85)-85) - 2.5*np.abs(norm["roller_speed_mmps"].fillna(2.2)-2.2)
        norm["green_density_pctTD"] = base.clip(lower=60, upper=98)
        notes.append("green_density_pctTD missing — using a smooth placeholder trend.")

    missing = []
    if norm["d50_um"].isna().all():               missing.append("D50")
    if norm["binder_saturation_pct"].isna().all(): missing.append("binder saturation")
    if norm["roller_speed_mmps"].isna().all():     missing.append("roller speed")
    if missing:
        notes.append("Columns fully NaN or missing: " + ", ".join(missing))

    detected = dict(d50=c_d50, sat=c_sat, speed=c_spd, td=c_td, binder=c_bind, material=c_mat)
    return norm.reset_index(drop=True), detected

# ------------------------------------------------------------------------------
# Data load (no caching; explicit errors)
# ------------------------------------------------------------------------------
def read_first_available(files: List[str], notes: List[str]) -> Tuple[pd.DataFrame, str]:
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                return df, f
            except Exception as e:
                notes.append(f"Failed reading {f}: {e}")
    return pd.DataFrame(), ""

def load_data(default_files: List[str]) -> Tuple[pd.DataFrame, str, Dict[str, Optional[str]], List[str]]:
    notes: List[str] = []
    up = st.file_uploader("Upload CSV (optional)", type=["csv"], key="csv_up")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            norm, det = normalize_df(raw, notes)
            return norm, "uploaded.csv", det, notes
        except Exception as e:
            notes.append(f"Failed to read uploaded CSV: {e}. Falling back to local files.")
    raw, path = read_first_available(default_files, notes)
    if raw.empty:
        notes.append("No CSV found locally. Please upload a dataset.")
        empty = pd.DataFrame(columns=["material","binder_type","d50_um","binder_saturation_pct","roller_speed_mmps","green_density_pctTD"])
        return empty, "", {}, notes
    norm, det = normalize_df(raw, notes)
    return norm, path, det, notes

# ------------------------------------------------------------------------------
# Grids (“light models”) — simple, transparent math
# ------------------------------------------------------------------------------
def material_grid(df: pd.DataFrame, material: str) -> pd.DataFrame:
    d = df[df["material"].astype(str)==str(material)].copy()
    if d.empty:
        return pd.DataFrame(columns=["sat","speed","q10","q50","q90"])

    sat_series = d["binder_saturation_pct"].dropna()
    spd_series = d["roller_speed_mmps"].dropna()
    sat_lo, sat_hi = (sat_series.quantile(0.05), sat_series.quantile(0.95)) if len(sat_series) else (70.0, 110.0)
    spd_lo, spd_hi = (spd_series.quantile(0.05), spd_series.quantile(0.95)) if len(spd_series) else (1.2, 3.0)
    Xs = np.linspace(max(50, float(sat_lo)), min(110, float(sat_hi)), 24)
    Ys = np.linspace(max(0.4, float(spd_lo)), min(6.0, float(spd_hi)), 20)

    gq10 = float(np.percentile(d["green_density_pctTD"], 10))
    gq50 = float(np.percentile(d["green_density_pctTD"], 50))
    gq90 = float(np.percentile(d["green_density_pctTD"], 90))

    recs = []
    for s in Xs:
        for v in Ys:
            m = pd.Series(True, index=d.index)
            if sat_series.any(): m &= d["binder_saturation_pct"].between(s-2.5, s+2.5)
            if spd_series.any(): m &= d["roller_speed_mmps"].between(v-0.20, v+0.20)
            dd = d[m]
            if len(dd) >= 3:
                q10 = float(np.percentile(dd["green_density_pctTD"], 10))
                q50 = float(np.percentile(dd["green_density_pctTD"], 50))
                q90 = float(np.percentile(dd["green_density_pctTD"], 90))
            else:
                q10, q50, q90 = gq10, gq50, gq90
            recs.append((s, v, q10, q50, q90))
    return pd.DataFrame(recs, columns=["sat","speed","q10","q50","q90"])

def nearest_band(grid: pd.DataFrame, speed: float) -> float:
    u = np.sort(grid["speed"].unique())
    if len(u)==0: return float(speed)
    return float(u[np.abs(u - speed).argmin()])

# ------------------------------------------------------------------------------
# Recommender (Top-5 with 3 water + 2 solvent)
# ------------------------------------------------------------------------------
def pick_trials(grid: pd.DataFrame, d50_um: float, target_pctTD: float) -> pd.DataFrame:
    g = grid.copy()
    if g.empty:
        Xs = np.linspace(70,110,8); Ys = np.linspace(1.2,3.0,6)
        g = pd.DataFrame([(s,v,85-abs(v-2.1)*3+0.06*(s-90),88-abs(v-2.1)*2+0.08*(s-90),92-abs(v-2.1)*1+0.10*(s-90))
                          for s in Xs for v in Ys], columns=["sat","speed","q10","q50","q90"])
    g["score"] = -np.abs(g["q50"] - target_pctTD) - 0.05*np.abs(g["speed"]-2.0)
    g = g.sort_values("score", ascending=False)
    g["binder_type_rec"] = np.where(g["sat"] <= 95.0, "water_based", "solvent_based")

    top = []; wcnt=scnt=0
    for _, r in g.iterrows():
        if r["binder_type_rec"]=="water_based" and wcnt<3:
            top.append(r); wcnt+=1
        elif r["binder_type_rec"]=="solvent_based" and scnt<2:
            top.append(r); scnt+=1
        if len(top)==5: break
    if len(top)<5:
        for _, r in g.iterrows():
            if len(top)==5: break
            if r not in top: top.append(r)

    out = pd.DataFrame(top)[["sat","speed","q10","q50","q90","binder_type_rec"]]
    out = out.rename(columns={"sat":"binder_saturation_pct","speed":"roller_speed_mmps"})
    out["d50_um"] = float(d50_um)
    out["id"] = [f"Opt-{i+1}" for i in range(len(out))]
    return out.reset_index(drop=True)

# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
def heatmap_fig(grid: pd.DataFrame, x_cross: float, y_cross: float, smooth_sigma: float=1.0) -> go.Figure:
    if grid.empty:
        return go.Figure(layout=dict(title="No data", template="simple_white"))
    X = np.sort(grid["sat"].unique()); Y = np.sort(grid["speed"].unique())
    Z = grid.pivot(index="speed", columns="sat", values="q50").values.astype(float)
    if smooth_sigma>0: Z = gaussian_filter(Z, sigma=smooth_sigma, mode="nearest")
    fig = go.Figure(data=go.Heatmap(x=X, y=Y, z=Z, colorscale="YlGnBu", colorbar=dict(title="%TD (q50)")))
    x0,x1 = np.percentile(X, [35,65]); y0,y1 = np.percentile(Y, [35,65])
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="teal", width=3, dash="dash"))
    fig.add_trace(go.Scatter(x=[x_cross], y=[y_cross], mode="markers",
                             marker=dict(color="red", size=12, symbol="cross"), name="chosen"))
    fig.update_layout(title="Predicted Green Density (% Theoretical Density)",
                      xaxis_title="Binder Saturation (%)", yaxis_title="Roller Speed (mm/s)",
                      template="simple_white", height=480, margin=dict(l=60,r=20,t=40,b=50))
    return fig

def sat_sensitivity_fig(grid: pd.DataFrame, speed_ref: float) -> go.Figure:
    if grid.empty:
        return go.Figure(layout=dict(title="No data", template="simple_white"))
    band = nearest_band(grid, speed_ref)
    gb = grid[grid["speed"]==band].sort_values("sat")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=gb["sat"], y=gb["q90"], name="q90", mode="lines"))
    fig.update_layout(title=f"Saturation sensitivity @ {band:.2f} mm/s",
                      xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                      template="simple_white", height=420, margin=dict(l=40,r=20,t=40,b=40))
    return fig

# ------------------------------------------------------------------------------
# Qualitative packing (simple RSA + jitter)
# ------------------------------------------------------------------------------
def poly_radii_um(n: int, d50_um: float) -> np.ndarray:
    mu = math.log(max(d50_um, 1e-6)) - 0.5*0.25**2
    diam = np.random.lognormal(mean=mu, sigma=0.25, size=n)
    return diam/2.0

def rsa_pack_um(n_try: int, phi_target: float, box_um: float, d50_um: float, seed: int=RNG_SEED) -> Tuple[np.ndarray,np.ndarray]:
    rng = np.random.default_rng(seed)
    centers: List[Tuple[float,float]] = []
    radii: List[float]] = []
    area_box = box_um*box_um
    total = 0.0
    trial = 0
    while trial < n_try and total/area_box < phi_target:
        r = float(poly_radii_um(1, d50_um)[0])
        x = rng.uniform(r, box_um - r); y = rng.uniform(r, box_um - r)
        ok = True
        for i in range(len(radii)):
            dx = x - centers[i][0]; dy = y - centers[i][1]
            if dx*dx + dy*dy < (r + radii[i])**2: ok=False; break
        if ok:
            centers.append((x,y)); radii.append(r); total += math.pi*r*r
        trial += 1
    if not radii:
        return np.zeros((0,2)), np.zeros((0,))
    C = np.array(centers,float); R = np.array(radii,float)
    C += rng.uniform(-0.15,0.15,size=C.shape) * R.reshape(-1,1)
    C[:,0] = np.clip(C[:,0], R, box_um-R); C[:,1] = np.clip(C[:,1], R, box_um-R)
    return C,R

def packing_figure(C: np.ndarray, R: np.ndarray, binder_sat: float, fld_um: float=800.0) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=fld_um, y1=fld_um,
                  line_color="black", fillcolor="rgb(231,176,62)")
    for (x,y), r in zip(C, R):
        fig.add_shape(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                      line_color="rgb(20,70,200)", fillcolor="rgb(37,110,230)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(title=f"Qualitative packing • Binder Sat {int(binder_sat)}%",
                      template="simple_white", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=480, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ------------------------------------------------------------------------------
# Digital Twin (optional; guarded)
# ------------------------------------------------------------------------------
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point  # type: ignore
    from shapely.ops import unary_union  # type: ignore

def slice_mesh_layer(mesh: "trimesh.Trimesh", z: float) -> List["Polygon"]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        return [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    except Exception:
        return []

def pack_in_polys(polys: List["Polygon"], d50_um: float, phi2D_target: float, fov_um: float, seed: int):
    if not polys:
        return rsa_pack_um(14000, phi2D_target, fov_um, d50_um, seed)
    try:
        U = unary_union(polys)
    except Exception:
        U = polys[0]
        for p in polys[1:]:
            U = U.union(p)
    minx,miny,maxx,maxy = U.bounds
    size = min(maxx-minx, maxy-miny)
    if size <= 0:
        return rsa_pack_um(12000, phi2D_target, fov_um, d50_um, seed)
    L = min(fov_um, size)
    cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
    x0,y0 = cx - L/2, cy - L/2
    C,R = rsa_pack_um(18000, phi2D_target, L, d50_um, seed)
    keepC, keepR = [], []
    for (x,y), r in zip(C,R):
        if U.contains(Point(x0+x, y0+y)):
            keepC.append((x,y)); keepR.append(r)
    if not keepR: return np.zeros((0,2)), np.zeros((0,))
    return np.array(keepC,float), np.array(keepR,float)

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title="BJAM", layout="wide", initial_sidebar_state="expanded")
st.title(APP_TITLE)

# Sidebar: data load + inputs + debug panel
with st.sidebar:
    st.subheader("Data")
    df_all, data_path, detected, notes = load_data(DATA_FILES)
    st.caption("Loaded: " + (os.path.basename(data_path) if data_path else "uploaded.csv" if len(df_all) else "—"))
    if notes:
        st.warning(" • ".join(notes))

    st.subheader("Inputs")
    materials = sorted(pd.Series(df_all["material"], dtype=str).dropna().unique().tolist())
    material = st.selectbox("Material", materials, index=0 if materials else None)
    d50_um = st.number_input("D50 (µm)", value=90.0, min_value=1.0, max_value=500.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", value=120, min_value=5, max_value=300)
    target_pctTD = st.slider("Target green density (%TD)", value=92, min_value=80, max_value=98)
    enable_dt = st.checkbox("Enable Digital Twin (requires shapely & trimesh)", value=False)

    with st.expander("Debug • columns detected / data peek", expanded=False):
        st.write(pd.DataFrame({
            "role": ["D50","Saturation","Speed","Green %TD","Binder","Material"],
            "column": [detected.get("d50"), detected.get("sat"), detected.get("speed"),
                       detected.get("td"), detected.get("binder"), detected.get("material")]
        }))
        st.write("Rows:", len(df_all))
        if len(df_all):
            st.write(df_all.head(8))

# Build grid for this material
grid = material_grid(df_all, material) if len(df_all) else pd.DataFrame(columns=["sat","speed","q10","q50","q90"])

tab_pred, tab_heat, tab_sens, tab_pack, tab_dt = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Digital Twin (optional)"]
)

# Predict
with tab_pred:
    if grid.empty:
        st.error("No data available for this material. Check the Debug panel and detected columns.")
    else:
        trials = pick_trials(grid, d50_um=d50_um, target_pctTD=target_pctTD)
        st.dataframe(trials[["id","binder_type_rec","binder_saturation_pct","roller_speed_mmps","q50","q10","q90","d50_um"]],
                     use_container_width=True, hide_index=True)
        rec_id = st.selectbox("Use trial for other tabs", trials["id"].tolist(), index=0)
        sel = trials[trials["id"]==rec_id].iloc[0]
        st.session_state["selected_trial"] = dict(sel)

# Heatmap
with tab_heat:
    if grid.empty:
        st.info("Heatmap disabled: no grid for this material.")
    else:
        use = st.session_state.get("selected_trial")
        x_cross = float(use["binder_saturation_pct"]) if use else float(np.median(grid["sat"]))
        y_cross = float(use["roller_speed_mmps"]) if use else float(np.median(grid["speed"]))
        smooth = st.slider("Smoothing σ", 0.0, 2.5, 1.0, 0.1)
        st.plotly_chart(heatmap_fig(grid, x_cross, y_cross, smooth), use_container_width=True)

# Sensitivity
with tab_sens:
    if grid.empty:
        st.info("Sensitivity disabled: no grid for this material.")
    else:
        use = st.session_state.get("selected_trial")
        speed_ref = float(use["roller_speed_mmps"]) if use else float(np.median(grid["speed"]))
        st.plotly_chart(sat_sensitivity_fig(grid, speed_ref), use_container_width=True)

# Packing
with tab_pack:
    use = st.session_state.get("selected_trial")
    sat = float(use["binder_saturation_pct"]) if use else 85.0
    phi2D = float(np.clip(0.55 + 0.003*(d50_um-20), 0.40, 0.90))
    C,R = rsa_pack_um(14000, phi2D, 800.0, d50_um, seed=RNG_SEED)
    st.plotly_chart(packing_figure(C,R, binder_sat=sat, fld_um=800.0), use_container_width=True)
    st.caption(f"FOV≈0.80 mm • φ₂D(target)≈{phi2D:.2f}")

# Digital Twin (optional, guarded)
with tab_dt:
    if not enable_dt:
        st.info("Turn on the 'Enable Digital Twin' checkbox in the sidebar (and ensure shapely & trimesh are in requirements.txt).")
    else:
        HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
        HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
        if not (HAVE_TRIMESH and HAVE_SHAPELY):
            st.error("Missing packages: please add shapely==2.0.4 and trimesh==4.4.9 to requirements.txt and redeploy.")
        else:
            from shapely.geometry import Polygon, Point  # type: ignore
            from shapely.ops import unary_union  # type: ignore
            import trimesh  # lazy import

            def slice_mesh_layer(mesh: "trimesh.Trimesh", z: float) -> List["Polygon"]:
                try:
                    sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
                    if sec is None: return []
                    planar,_ = sec.to_planar()
                    return [Polygon(p) for p in getattr(planar, "polygons_full", [])]
                except Exception:
                    return []

            def pack_in_polys(polys: List["Polygon"], d50_um: float, phi2D_target: float, fov_um: float, seed: int):
                if not polys:
                    return rsa_pack_um(14000, phi2D_target, fov_um, d50_um, seed)
                try:
                    U = unary_union(polys)
                except Exception:
                    U = polys[0]
                    for p in polys[1:]:
                        U = U.union(p)
                minx,miny,maxx,maxy = U.bounds
                size = min(maxx-minx, maxy-miny)
                if size <= 0:
                    return rsa_pack_um(12000, phi2D_target, fov_um, d50_um, seed)
                L = min(fov_um, size)
                cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
                x0,y0 = cx - L/2, cy - L/2
                C,R = rsa_pack_um(18000, phi2D_target, L, d50_um, seed)
                keepC, keepR = [], []
                for (x,y), r in zip(C,R):
                    if U.contains(Point(x0+x, y0+y)):
                        keepC.append((x,y)); keepR.append(r)
                if not keepR: return np.zeros((0,2)), np.zeros((0,))
                return np.array(keepC,float), np.array(keepR,float)

            up = st.file_uploader("Upload STL", type=["stl"], key="dt_stl")
            if up is None:
                st.info("Upload an STL to preview packed layers.")
            else:
                try:
                    mesh = trimesh.load_mesh(io.BytesIO(up.read()), file_type="stl")
                    mesh.rezero()
                    zmin, zmax = mesh.bounds[0,2], mesh.bounds[1,2]
                    n_layers = max(1, int(max(1.0, (zmax - zmin)) // max(1.0, layer_um)))
                    layer_idx = st.slider("Layer index", 1, n_layers, min(3, n_layers))
                    z = zmin + (layer_idx - 0.5) * max(1.0, layer_um)

                    polys = slice_mesh_layer(mesh, z)
                    phi2D_target = float(np.clip(0.55 + 0.003*(d50_um-20), 0.40, 0.90))
                    C2,R2 = pack_in_polys(polys, d50_um=d50_um, phi2D_target=phi2D_target, fov_um=800.0, seed=RNG_SEED+layer_idx)
                    sat2 = float(st.session_state.get("selected_trial", {}).get("binder_saturation_pct", 85.0))
                    st.plotly_chart(packing_figure(C2,R2, binder_sat=sat2, fld_um=800.0), use_container_width=True)
                except Exception as ex:
                    st.exception(ex)
