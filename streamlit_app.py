import io
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# Only import heavy modules when DT tab is used
TRIMESH_OK = True
try:
    import trimesh
    from shapely.geometry import Polygon, Point
except Exception:
    TRIMESH_OK = False


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="BJAM — Binder-Jet AM Parameter Recommender",
                   layout="wide",
                   initial_sidebar_state="expanded")

TITLE = "BJAM — Binder-Jet AM Parameter Recommender"
SUBTITLE = ("Physics-guided few-shot models from your dataset (shared.py). "
            "Digital Twin added. Generated with help of ChatGPT.")

DATA_PATH = "BJAM_All_Deep_Fill_v9.csv"  # keep your path/name
RNG_SEED_DEFAULT = 42

# -----------------------------
# Utilities
# -----------------------------
def _nice_title(t: str) -> None:
    st.markdown(f"<h1 style='margin-top:0'>{t}</h1>", unsafe_allow_html=True)
    st.caption(SUBTITLE)


def _scale_bar(fig: go.Figure, px_per_um: float, fov_um: float, x0: float=0.65, y0: float=0.06):
    """Draw a 500 µm scale bar in normalized axes."""
    length_um = 500
    length_px = length_um * px_per_um
    fig.add_shape(type="rect",
                  x0=0.65, x1=0.65 + 0.25, y0=y0-0.015, y1=y0-0.012,
                  xref="paper", yref="paper",
                  line_color="black", fillcolor="black")


# -----------------------------
# Load + light model stubs
# -----------------------------
@st.cache_data
def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expected columns (examples):
    # material, d50_um, binder_saturation_pct, roller_speed_mmps, green_density_pctTD, binder_type
    # If your headers differ, remap here.
    rename_map = {
        'roller_speed': 'roller_speed_mmps',
        'binder_saturation': 'binder_saturation_pct',
        'green_density': 'green_density_pctTD',
        'binder': 'binder_type'
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    # Drop obvious bad rows
    for c in ['d50_um', 'binder_saturation_pct', 'roller_speed_mmps']:
        if c in df.columns:
            df = df[pd.to_numeric(df[c], errors='coerce').notna()]
    return df.reset_index(drop=True)


@st.cache_data
def _fit_quantile_models(_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Very light placeholder “models”: for each material, bin by (sat, speed),
    compute q10, q50, q90 of green %TD. Return grids we can interpolate/smooth.
    Only hashable args are cached (note the leading underscore).
    """
    mats = {}
    if 'material' not in _df.columns:
        _df = _df.assign(material='Generic')
    for mat, d in _df.groupby('material'):
        # Build coarse grid
        s_bins = np.arange(60, 111, 5)       # 60–110 %
        v_bins = np.round(np.linspace(1.2, 3.5, 10), 2)  # 1.2–3.5 mm/s
        recs = []
        for s in s_bins:
            for v in v_bins:
                dd = d.loc[(d['binder_saturation_pct']).between(s-2.5, s+2.5) &
                           (d['roller_speed_mmps']).between(v-0.15, v+0.15)]
                if len(dd) >= 2:
                    q10 = float(np.percentile(dd['green_density_pctTD'], 10))
                    q50 = float(np.percentile(dd['green_density_pctTD'], 50))
                    q90 = float(np.percentile(dd['green_density_pctTD'], 90))
                else:
                    # fallback: global medians to avoid holes
                    q10 = float(np.percentile(d['green_density_pctTD'], 10))
                    q50 = float(np.percentile(d['green_density_pctTD'], 50))
                    q90 = float(np.percentile(d['green_density_pctTD'], 90))
                recs.append((s, v, q10, q50, q90))
        grid = pd.DataFrame(recs, columns=['sat', 'speed', 'q10', 'q50', 'q90'])
        mats[mat] = {'grid': grid}
    return mats


@st.cache_data
def _cached_quantile_grid(_models: Dict[str, Dict], mat: str) -> pd.DataFrame:
    # Only return the grid for the selected material
    md = _models.get(mat)
    if md is None:
        # fall back to the very first material if needed
        md = list(_models.values())[0]
    return md['grid'].copy()


def _nearest_band(grid: pd.DataFrame, speed: float) -> float:
    s = grid['speed'].unique()
    return float(s[np.abs(s - speed).argmin()])


def _admissible_mask(grid: pd.DataFrame, q50_floor: float = 90.0) -> pd.DataFrame:
    g = grid.copy()
    g['ok'] = (g['q50'] >= q50_floor).astype(int)
    return g


# -----------------------------
# Trial picker (3 water, 2 solvent)
# -----------------------------
def _pick_trials(grid: pd.DataFrame,
                 d50_um: float,
                 target_pctTD: float,
                 guardrails: bool = True) -> pd.DataFrame:
    # Score by closeness to target and some mild speed regularization
    g = grid.copy()
    g['score'] = -np.abs(g['q50'] - target_pctTD) - 0.05*np.abs(g['speed']-2.0)
    g = g.sort_values('score', ascending=False)

    # Enforce binder mix
    def binder_from_sat(s):
        return 'water_based' if s <= 95 else 'solvent_based'
    g['binder_type_rec'] = g['sat'].map(binder_from_sat)

    top = []
    water_count = 0
    solv_count  = 0
    for _, r in g.iterrows():
        b = r['binder_type_rec']
        if b == 'water_based' and water_count < 3:
            top.append(r)
            water_count += 1
        elif b == 'solvent_based' and solv_count < 2:
            top.append(r)
            solv_count += 1
        if len(top) == 5:
            break
    if len(top) < 5:
        # backfill
        for _, r in g.iterrows():
            if r not in top:
                top.append(r)
            if len(top) == 5:
                break

    out = pd.DataFrame(top)[['sat', 'speed', 'q10', 'q50', 'q90', 'binder_type_rec']]
    out = out.rename(columns={'sat': 'binder_saturation_pct',
                              'speed': 'roller_speed_mmps'})
    out['d50_um'] = d50_um
    out['binder_sat_str'] = out['binder_saturation_pct'].map(lambda z: f"{int(z)}%")
    out['roller_str']     = out['roller_speed_mmps'].map(lambda v: f"{v:.2f} mm/s")
    return out.reset_index(drop=True)


# -----------------------------
# Heatmap figure
# -----------------------------
def _heatmap(grid: pd.DataFrame, x_cross: float, y_cross: float,
             smooth_sigma: float = 1.0) -> go.Figure:
    # Pivot q50 into 2D and (optional) smooth
    X = np.sort(grid['sat'].unique())
    Y = np.sort(grid['speed'].unique())
    Z = grid.pivot(index='speed', columns='sat', values='q50').values.astype(float)
    if smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=smooth_sigma, mode='nearest')

    fig = go.Figure(data=go.Heatmap(
        x=X, y=Y, z=Z,
        colorscale="YlGnBu",
        colorbar=dict(title="%TD (q50)")
    ))
    # admissible dashed box (example 75–95% sat, 1.9–2.6 mm/s)
    fig.add_shape(type="rect",
                  x0=75, x1=95, y0=2.0, y1=2.6,
                  line=dict(color="teal", width=3, dash="dash"))
    # crosshair
    fig.add_trace(go.Scatter(
        x=[x_cross], y=[y_cross],
        mode="markers",
        marker=dict(color="red", size=12, symbol="cross"),
        name="chosen"
    ))
    fig.update_layout(
        title="Predicted Green Density (% Theoretical Density)",
        xaxis_title="Binder Saturation (%)",
        yaxis_title="Roller Speed (mm/s)",
        template="simple_white",
        margin=dict(l=60, r=20, t=40, b=50),
        height=480
    )
    return fig


# -----------------------------
# Packing (qualitative) – RSA + jitter
# -----------------------------
def _poly_disperse_radii(n: int, d50_um: float) -> np.ndarray:
    # lognormal around d50, modest spread
    mu = math.log(d50_um) - 0.5*0.25**2
    diam = np.random.lognormal(mean=mu, sigma=0.25, size=n)
    r = (diam/2.0)
    return r


def _rsa_pack(n_try: int, target_phi2D: float, box_um: float,
              d50_um: float, seed: int = RNG_SEED_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    centers = []
    radii   = []
    area_box = box_um*box_um
    total_area = 0.0

    trial = 0
    while trial < n_try and total_area/area_box < target_phi2D:
        r = float(_poly_disperse_radii(1, d50_um)[0])
        x = np.random.uniform(r, box_um-r)
        y = np.random.uniform(r, box_um-r)
        ok = True
        for (cx, cy, cr) in zip(centers, centers[1::2], radii):  # cheap skip, replaced below
            pass
        # proper check
        ok = True
        for i in range(len(radii)):
            dx = x - centers[i][0]
            dy = y - centers[i][1]
            if dx*dx + dy*dy < (r + radii[i])**2:
                ok = False
                break
        if ok:
            centers.append((x, y))
            radii.append(r)
            total_area += math.pi * r*r
        trial += 1

    if len(radii) == 0:
        return np.zeros((0,2)), np.zeros((0,))
    # small jitter to avoid perfect rows
    C = np.array(centers, dtype=float)
    R = np.array(radii, dtype=float)
    C += np.random.uniform(-0.15, 0.15, size=C.shape) * R.reshape(-1,1)
    # clamp to box
    C[:,0] = np.clip(C[:,0], R, box_um-R)
    C[:,1] = np.clip(C[:,1], R, box_um-R)
    return C, R


def _packing_fig(centers: np.ndarray, radii: np.ndarray,
                 binder_sat: float, fld_um: float) -> go.Figure:
    # make image by polygons
    fig = go.Figure()
    # Binder background
    fig.add_shape(type="rect", x0=0, y0=0, x1=fld_um, y1=fld_um,
                  line_color="black",
                  fillcolor="rgb(231,176,62)")  # warm yellow
    # Particles
    for (x,y), r in zip(centers, radii):
        fig.add_shape(type="circle",
                      x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                      line_color="rgb(20,70,200)",
                      fillcolor="rgb(37,110,230)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"Qualitative packing • Binder Sat {int(binder_sat)}%",
        template="simple_white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=480, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


# -----------------------------
# Digital Twin helpers
# -----------------------------
def _slice_mesh(mesh: "trimesh.Trimesh", z0: float, z1: float):
    """Return a 2D polygon outline from a Z-slab [z0, z1]."""
    try:
        slab = mesh.section_multiplane(plane_origin=[0,0,z0],
                                       plane_normal=[0,0,1.0],
                                       heights=[0, z1-z0])
        if not slab:
            return []
        polys = []
        for s in slab:
            paths = s.to_planar()[0].polygons_full
            polys.extend(paths)
        return polys
    except Exception:
        return []


def _pack_in_polygon(polys: List["shapely.geometry.Polygon"],
                     d50_um: float,
                     phi2D_target: float,
                     fov_um: float,
                     seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pack inside union of polygons; crop/zoom to FOV; return centers/radii + px/µm."""
    if len(polys) == 0:
        # fall back to square box FOV
        C, R = _rsa_pack(n_try=15_000, target_phi2D=phi2D_target, box_um=fov_um, d50_um=d50_um, seed=seed)
        return C, R, 1.0
    # Merge polygons
    try:
        from shapely.ops import unary_union
        U = unary_union(polys)
    except Exception:
        U = polys[0]
        for p in polys[1:]:
            U = U.union(p)

    # Fit a square FOV inside bounds
    minx, miny, maxx, maxy = U.bounds
    size = min(maxx-minx, maxy-miny)
    if size <= 0:
        C, R = _rsa_pack(n_try=12_000, target_phi2D=phi2D_target, box_um=fov_um, d50_um=d50_um, seed=seed)
        return C, R, 1.0
    # Choose FOV size
    L = min(fov_um, size)
    # center window
    cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
    x0 = cx - L/2; y0 = cy - L/2
    # Generate RSA in square then mask to polygon
    C, R = _rsa_pack(n_try=18_000, target_phi2D=phi2D_target, box_um=L, d50_um=d50_um, seed=seed)
    inside = []
    keepR  = []
    for (x,y), r in zip(C, R):
        pt = Point(x0+x, y0+y)
        if U.contains(pt):
            inside.append((x, y))
            keepR.append(r)
    if len(inside)==0:
        return np.zeros((0,2)), np.zeros((0,)), 1.0
    C2 = np.array(inside, dtype=float)
    R2 = np.array(keepR, dtype=float)
    return C2, R2, 1.0


# -----------------------------
# Sidebar (inputs)
# -----------------------------
with st.sidebar:
    st.subheader("Inputs")
    df_all = _load_data(DATA_PATH)
    materials = sorted(df_all['material'].dropna().unique().tolist())
    material = st.selectbox("Material", materials, index=materials.index(materials[0]) if materials else 0)

    d50_um = st.number_input("D50 (µm)", value=90.0, min_value=1.0, max_value=500.0, step=1.0)
    layer_thk_um = st.slider("Layer thickness (µm)", value=120, min_value=5, max_value=300)
    target_pctTD = st.slider("Target green density (% of theoretical)", value=92, min_value=80, max_value=98)
    guardrails = st.toggle("Guardrails", value=True)
    st.caption(f"Data / Models")
    st.caption(f"Source: {DATA_PATH}")
    st.caption(f"Rows: {len(df_all)}")

# Models and grid for the chosen material
models = _fit_quantile_models(df_all)
grid   = _cached_quantile_grid(models, material)

# Main title + tabs
_nice_title(TITLE)
tabs = st.tabs(["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)"])

# -----------------------------
# Tab 1: Predict (Top-5)
# -----------------------------
with tabs[0]:
    trials = _pick_trials(grid, d50_um=d50_um, target_pctTD=target_pctTD, guardrails=guardrails)

    st.write("Recommended trials (forced 3 water-based, 2 solvent-based when guardrails are on):")
    st.dataframe(trials[['binder_type_rec','binder_saturation_pct','roller_speed_mmps','q50','q10','q90','d50_um']])

    rec_id = st.selectbox("Pick one trial",
                          [f"#{i+1} • {row['binder_type_rec']} • Sat {int(row['binder_saturation_pct'])}% • {row['roller_speed_mmps']:.2f} mm/s"
                           for i, row in trials.iterrows()],
                          index=0)
    sel = trials.iloc[int(rec_id.split('•')[0][1:])-1]
    st.session_state['selected_trial'] = dict(sel)


# -----------------------------
# Tab 2: Heatmap
# -----------------------------
with tabs[1]:
    # Use selected or default
    use = st.session_state.get('selected_trial', None)
    x_cross = float(use['binder_saturation_pct']) if use else 85.0
    y_cross = float(use['roller_speed_mmps']) if use else 2.3
    smooth = st.slider("Heatmap smoothing (σ)", 0.0, 2.5, 1.0, 0.1)

    fig = _heatmap(grid, x_cross=x_cross, y_cross=y_cross, smooth_sigma=smooth)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Tab 3: Saturation sensitivity
# -----------------------------
with tabs[2]:
    speed_ref = float(use['roller_speed_mmps']) if use else 2.3
    band = _nearest_band(grid, speed_ref)
    gband = grid[grid['speed'] == band].sort_values('sat')
    fig2 = go.Figure()
    for q, col in [('q10','rgba(50,90,200,1)'), ('q50','rgba(60,120,200,1)'), ('q90','rgba(220,70,60,1)')]:
        fig2.add_trace(go.Scatter(x=gband['sat'], y=gband[q],
                                  name=q, mode='lines'))
    fig2.update_layout(title="Saturation sensitivity at representative speed",
                       xaxis_title="Binder saturation (%)",
                       yaxis_title="Predicted green %TD",
                       template="simple_white",
                       height=420)
    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Tab 4: Qualitative packing
# -----------------------------
with tabs[3]:
    colL, colR = st.columns([1,1])
    with colL:
        st.write("Particles only")
        phi2D = min(0.90, 0.55 + 0.003*(d50_um-20))  # light scaling to reflect d50 changes
        C, R = _rsa_pack(n_try=14_000, target_phi2D=phi2D, box_um=800.0, d50_um=d50_um, seed=RNG_SEED_DEFAULT)
        figP = _packing_fig(C, R, binder_sat=use['binder_saturation_pct'] if use else 85, fld_um=800.0)
        st.plotly_chart(figP, use_container_width=True)

    with colR:
        st.write(f"{use['binder_type_rec'] if use else 'water_based'} · Sat {int(use['binder_saturation_pct']) if use else 85}%")
        figB = _packing_fig(C, R, binder_sat=use['binder_saturation_pct'] if use else 85, fld_um=800.0)
        st.plotly_chart(figB, use_container_width=True)

    st.caption(f"FOV≈0.80 mm · φ₂D(target)≈{phi2D:.2f}")


# -----------------------------
# Tab 5: Formulae (stub)
# -----------------------------
with tabs[4]:
    st.write("Key relationships used (guardrails / heuristics):")
    st.markdown("""
- Layer thickness guardrail:  `t ≤ 2.5·D50` to avoid streaking/flooding.
- Roller window:             `1.8–2.6 mm/s` nominal for D50≈50–120 µm.
- Saturation bands:          water-based ≤95%; solvent-based >95%.
- Packing target:            φ₂D≈0.8 corresponds to ~0.60–0.65 φ₃D random packing before solvent removal.
    """)


# -----------------------------
# Tab 6: Digital Twin (Beta)
# -----------------------------
with tabs[5]:
    if not TRIMESH_OK:
        st.warning("trimesh/shapely not available in this environment. Please add them to requirements.txt")
    st.write("Upload STL to preview layer-true particle packing:")
    st.caption("Tip: use modest meshes; thin sections render faster. Use the slider to scrub layers.")
    up = st.file_uploader("STL file", type=["stl"])
    if up is not None and TRIMESH_OK:
        try:
            mesh = trimesh.load_mesh(io.BytesIO(up.read()), file_type='stl')
            mesh.rezero()
            zmin, zmax = mesh.bounds[0,2], mesh.bounds[1,2]
            n_layers = max(1, int((zmax-zmin)//layer_thk_um))
            lyr = st.slider("Layer index", 0, n_layers-1, min(3, n_layers-1))
            z0 = zmin + lyr*layer_thk_um
            z1 = z0 + layer_thk_um

            polys = _slice_mesh(mesh, z0, z1)
            phi2D_target = min(0.90, 0.55 + 0.003*(d50_um-20))
            C2, R2, s = _pack_in_polygon(polys, d50_um=d50_um,
                                         phi2D_target=phi2D_target,
                                         fov_um=800.0, seed=RNG_SEED_DEFAULT+lyr)
            figDT = _packing_fig(C2, R2, binder_sat=use['binder_saturation_pct'] if use else 85, fld_um=800.0)
            figDT.update_layout(title=f"Digital twin preview • layer {lyr+1}/{n_layers}")
            st.plotly_chart(figDT, use_container_width=True)

            # Trial picker for DT: choose any of the 5 recommended
            st.divider()
            st.write("Compare all 5 recommended recipes in this STL layer:")
            choice = st.selectbox("Recipe for overlay",
                                  [f"#{i+1} • {row['binder_type_rec']} • Sat {int(row['binder_saturation_pct'])}% • {row['roller_speed_mmps']:.2f} mm/s"
                                   for i, row in trials.iterrows()],
                                  index=0)
            # (For now the visual is the same packing — the recipe choice is metadata & scale bar.)
            st.caption("Note: packing microstructure is illustrative; the recipe chiefly affects φ₂D target and binder color/theme.")

        except Exception as ex:
            st.error(f"Failed to process STL: {ex}")
    elif up is None:
        st.info("Upload an STL to enable the Digital Twin demo.")
