# -*- coding: utf-8 -*-
# Binder Jet Parameter Studio — Predictions + Digital Twin
# Generated with the help of ChatGPT (GPT-5 Thinking).

from __future__ import annotations
import io, os, math, importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import streamlit as st

# ---------------------------- Page Config ------------------------------------
st.set_page_config(page_title="Binder Jet Parameter Studio", page_icon="🧪", layout="wide")

# ---------------------------- Soft Dep Checks --------------------------------
def _need(pkg: str, pip_name: Optional[str] = None) -> None:
    """Fail gracefully with a friendly error if a module is missing."""
    if importlib.util.find_spec(pkg) is None:
        st.error(f"Missing Python package: '{pip_name or pkg}'. "
                 f"Add it to requirements.txt and redeploy.")
        st.stop()

# Core deps (required)
for _pkg, _pip in [("numpy","numpy"), ("pandas","pandas"), ("matplotlib","matplotlib"),
                   ("PIL","Pillow"), ("plotly","plotly"), ("trimesh","trimesh"),
                   ("shapely","shapely")]:
    _need(_pkg, _pip)
# SciPy is optional; app will degrade gracefully if absent.

# ---------------------------- Imports (post-check) ---------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw
import trimesh
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------------------------- Styles -----------------------------------------
st.markdown(
    """
    <style>
    .small-note {font-size: 0.85rem; color: #6b7280;}
    .caption {font-size: 0.92rem; color: #374151;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------- Palettes ---------------------------------------
BINDER_COLORS = {
    "PVOH":   "#F2B233",
    "PEG":    "#F2D06F",
    "Furan":  "#F5C07A",
    "Acrylic":"#FFD166",
    "Other":  "#F4B942",
}
PARTICLE_COLOR = "#2F6CF6"
PARTICLE_EDGE  = "#1f2937"
BORDER_COLOR   = "#111111"
VOID_COLOR     = "#FFFFFF"

def binder_color(name: str) -> str:
    for k in BINDER_COLORS:
        if k.lower() in (name or "").lower():
            return BINDER_COLORS[k]
    return BINDER_COLORS["Other"]

# ---------------------------- Header -----------------------------------------
st.title("Binder Jet Parameter Studio")
st.write(
    "Input a powder and PSD, get five viable process recipes targeting theoretical packing ≥ 0.90, "
    "then inspect layers with a physics-aware digital twin (true-scale particles, binder, and voids)."
)
st.markdown('<div class="small-note">This app and figures were generated with the help of ChatGPT (GPT-5 Thinking).</div>', unsafe_allow_html=True)

# ---------------------------- Data Loading -----------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(paths: list[str]) -> Optional[pd.DataFrame]:
    for p in paths:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

DATA_CANDIDATES = [
    os.environ.get("BJAM_DATA", ""),             # env var path
    "data/BJAM_All_Deep_Fill_v9.csv",           # repo data folder
    "BJAM_All_Deep_Fill_v9.csv",                # repo root
    "/mnt/data/BJAM_All_Deep_Fill_v9.csv",      # platform path (if mounted)
]
df_data = load_dataset(DATA_CANDIDATES)

# ---------------------------- Recommender ------------------------------------
@dataclass
class Recipe:
    id: str
    binder_type: str
    saturation_pct: float
    layer_um: float
    roller_mm_s: float
    d50_um: float
    d10_um: Optional[float]
    d90_um: Optional[float]
    phi_green: float   # target theoretical packing fraction

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def make_top5_recipes(material: str, d50_um: float) -> pd.DataFrame:
    """Builds a 5-row DataFrame of candidate recipes targeting φ_green ≥ 0.90."""
    if df_data is not None:
        df = df_data.copy()

        # Best-effort column mapping
        def find_col(*names):
            for nm in names:
                for c in df.columns:
                    if c.lower() == nm.lower():
                        return c
            return None
        c_mat   = find_col("material","powder","alloy","ceramic")
        c_d50   = find_col("d50_um","d50","D50")
        c_sat   = find_col("saturation_pct","binder_saturation_pct","binder_saturation")
        c_layer = find_col("layer_um","layer_thickness_um","layer_height_um")
        c_rs    = find_col("roller_mm_s","roller_speed_mm_s","spreader_speed")
        c_b     = find_col("binder_type","binder","binder_chemistry")
        c_phi   = find_col("phi_green","tpd","theoretical_packing_density")

        if c_mat: df["_match"] = df[c_mat].astype(str).str.contains(material, case=False, na=False)
        else:     df["_match"] = True
        if c_d50 and pd.api.types.is_numeric_dtype(df[c_d50]): df["_d50_err"] = (df[c_d50]-d50_um).abs()
        else:                                                   df["_d50_err"] = 0.0
        if c_phi and pd.api.types.is_numeric_dtype(df[c_phi]): df["_phi_ok"] = df[c_phi] >= 0.90
        else:                                                   df["_phi_ok"] = True

        df = df[df["_phi_ok"]]
        if df.empty:
            df = df_data.copy(); df["_d50_err"]=0.0; df["_match"]=True
        df = df.sort_values(by=["_match","_d50_err"], ascending=[False,True]).head(30)

        recs: list[Recipe] = []
        for _, row in df.head(5).iterrows():
            recs.append(Recipe(
                id=f"Opt-{len(recs)+1}",
                binder_type=str(row[c_b]) if c_b else "PVOH",
                saturation_pct=_safe_float(row[c_sat], 80.0)   if c_sat   else 80.0,
                layer_um=_safe_float(row[c_layer], 0.9*d50_um) if c_layer else max(10.0, 0.9*d50_um),
                roller_mm_s=_safe_float(row[c_rs], 60.0)       if c_rs    else 60.0,
                d50_um=_safe_float(row[c_d50], d50_um)         if c_d50   else d50_um,
                d10_um=None,
                d90_um=None,
                phi_green=_safe_float(row[c_phi], 0.90)        if c_phi   else 0.90,
            ))
        if recs:
            return pd.DataFrame([r.__dict__ for r in recs])

    # Fallback synthetic set (if no CSV)
    base = max(10.0, min(200.0, 0.9*d50_um))
    recs = [
        Recipe("Opt-1","PVOH",   70, base-10, 80, d50_um, 0.7*d50_um, 1.4*d50_um, 0.90),
        Recipe("Opt-2","PEG",    80, base+0,  60, d50_um, 0.7*d50_um, 1.4*d50_um, 0.90),
        Recipe("Opt-3","Acrylic",65, base-5,  70, d50_um, None,       None,       0.90),
        Recipe("Opt-4","Furan",  75, base+10, 55, d50_um, 0.8*d50_um, 1.7*d50_um, 0.90),
        Recipe("Opt-5","PVOH",   85, base+20, 50, d50_um, 0.6*d50_um, 1.5*d50_um, 0.90),
    ]
    return pd.DataFrame([r.__dict__ for r in recs])

# ---------------------------- PSD Sampling -----------------------------------
def lognormal_from_quantiles(d50_um: float, d10_um: Optional[float], d90_um: Optional[float]) -> Tuple[float, float]:
    med = max(1e-9, float(d50_um))
    if d10_um and d90_um and d90_um > d10_um > 0:
        m = np.log(med)
        s = (np.log(d90_um) - np.log(d10_um)) / (2*1.2815515655446004)  # 90th pct z-score
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    return m, s

def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m, s = lognormal_from_quantiles(d50_um, d10_um, d90_um)
    return np.exp(rng.normal(m, s, size=n))  # diameters [µm]

# ---------------------------- Packing & Slicing -------------------------------
def stl_to_plotly_mesh(mesh: trimesh.Trimesh) -> go.Mesh3d:
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                     color="lightgray", opacity=0.55, flatshading=True, name="Part")

def slice_mesh_polygons(mesh: trimesh.Trimesh, z: float) -> List[Polygon]:
    section = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
    if section is None: return []
    planar, _ = section.to_planar()
    polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    return [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]

def apply_fov_crop(polys: List[Polygon], fov_size: float) -> List[Polygon]:
    if not polys: return []
    domain = unary_union(polys)
    cx, cy = domain.centroid.x, domain.centroid.y
    half = fov_size/2.0
    window = box(cx-half, cy-half, cx+half, cy+half)
    cropped = domain.intersection(window)
    if cropped.is_empty: return []
    if isinstance(cropped, Polygon): return [cropped]
    return [g for g in cropped.geoms if isinstance(g, Polygon) and g.is_valid and g.area>1e-8]

def pack_particles_in_domain(polys: List[Polygon],
                             diam_units: np.ndarray,
                             target_area_frac: float,
                             max_particles: int,
                             max_trials: int,
                             seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Greedy Poisson-disk-like packing into union(polys); units match STL space."""
    if not polys:
        return np.empty((0,2)), np.empty((0,)), 0.0
    domain = unary_union(polys)
    minx, miny, maxx, maxy = domain.bounds
    area_domain = domain.area

    diam = np.sort(np.asarray(diam_units))[::-1]  # big→small
    placed_xy, placed_r = [], []
    area_circles = 0.0
    target_area = float(np.clip(target_area_frac, 0.05, 0.90)) * area_domain

    rng = np.random.default_rng(seed)
    cell = max((diam.max()/2.0), (maxx-minx+maxy-miny)/400.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def ok(x, y, r):
        gx, gy = int(x // cell), int(y // cell)
        for ix in range(gx-1, gx+2):
            for iy in range(gy-1, gy+2):
                for j in grid.get((ix,iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return domain.contains(Point(x,y))

    trials = 0
    for d in diam:
        r = d/2.0
        for _ in range(180):
            trials += 1
            if trials > max_trials or area_circles >= target_area or len(placed_xy) >= max_particles:
                break
            x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
            if ok(x,y,r):
                idx = len(placed_xy)
                placed_xy.append((x,y)); placed_r.append(r)
                gx, gy = int(x // cell), int(y // cell)
                grid.setdefault((gx,gy), []).append(idx)
                area_circles += math.pi * r*r
        if trials > max_trials or area_circles >= target_area or len(placed_xy) >= max_particles:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    return centers, radii, (area_circles/area_domain)

# ---------------------------- Raster & Voids ----------------------------------
def raster_particle_mask(centers: np.ndarray, radii: np.ndarray,
                         width_units: float, height_units: float, px: int) -> np.ndarray:
    """Binary mask of particles on a px×px canvas (True where particle)."""
    img = Image.new("L", (px,px), color=0)
    drw = ImageDraw.Draw(img)
    sx = px / width_units; sy = px / height_units
    for (x,y), r in zip(centers, radii):
        x0 = int((x - r)*sx); y0 = int((height_units - (y + r))*sy)
        x1 = int((x + r)*sx); y1 = int((height_units - (y - r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img) > 0)

def void_mask_from_saturation(pore_mask: np.ndarray, saturation: float, rng=None) -> np.ndarray:
    """
    Organic-looking voids: white area ≈ (1 - saturation) × pore_area.
    Uses distance transform + noise if SciPy present; falls back to area-matched dots otherwise.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pore_area = int(pore_mask.sum())
    if pore_area <= 0:
        return np.zeros_like(pore_mask, dtype=bool)
    target_void = int(round((1.0 - saturation) * pore_area))
    if target_void <= 0:
        return np.zeros_like(pore_mask, dtype=bool)

    if HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.3)
        field = dist + 0.18 * noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat)-target_void)[len(flat)-target_void]
        void_mask = np.zeros_like(pore_mask, dtype=bool)
        void_mask[pore_mask] = field[pore_mask] >= kth
        void_mask = ndi.binary_opening(void_mask, iterations=1)
        void_mask = ndi.binary_closing(void_mask, iterations=1)
        return void_mask

    # Fallback: area-matched dots
    h, w = pore_mask.shape
    void = np.zeros_like(pore_mask, dtype=bool)
    area = 0; tries = 0
    while area < target_void and tries < 120000:
        tries += 1
        r = int(np.clip(np.random.normal(3.0, 1.2), 1.0, 6.0))
        x = np.random.randint(r, w-r); y = np.random.randint(r, h-r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            void[add] = True
            area = int(void.sum())
    return void

def render_layer(ax,
                 centers: np.ndarray, radii: np.ndarray,
                 void_mask: Optional[np.ndarray],
                 binder_hex: str, fov_size: float,
                 title: Optional[str] = None, scale_um: int = 100) -> None:
    # Binder field
    ax.add_patch(Rectangle((0,0), fov_size, fov_size, facecolor=binder_hex, edgecolor=BORDER_COLOR, linewidth=1.4))
    # Voids
    if void_mask is not None:
        ys, xs = np.where(void_mask)
        if len(xs):
            xm = xs * (fov_size/void_mask.shape[1])
            ym = (void_mask.shape[0] - ys) * (fov_size/void_mask.shape[0])
            ax.scatter(xm, ym, s=0.35, c=VOID_COLOR, alpha=0.95, linewidths=0)
    # Particles
    for (x,y), r in zip(centers, radii):
        ax.add_patch(Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25))
    # Decor
    ax.set_aspect('equal', 'box'); ax.set_xlim(0, fov_size); ax.set_ylim(0, fov_size)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    # Scale bar
    scale_len = (scale_um/1000.0); pad = 0.03*fov_size; x0, y0 = fov_size - pad - scale_len, pad
    ax.plot([x0, x0+scale_len], [y0, y0], color=BORDER_COLOR, lw=2.0)
    ax.text(x0 + scale_len/2, y0 + 0.02*fov_size, f"{int(scale_um)} µm", ha="center", va="bottom", fontsize=8, color=BORDER_COLOR)

# ---------------------------- Sidebar Inputs ---------------------------------
with st.sidebar:
    st.header("Inputs")
    material = st.text_input("Material / Powder", value="Silicon Carbide (SiC)")
    d50_um   = st.number_input("PSD: D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    d10_um   = st.number_input("PSD: D10 (µm, optional)", min_value=0.0, value=20.0, step=1.0)
    d90_um   = st.number_input("PSD: D90 (µm, optional)", min_value=0.0, value=45.0, step=1.0)
    st.markdown("---")
    st.markdown("STL & View Settings")
    stl_units = st.selectbox("STL units", ["mm","m"], index=0)
    um_to_unit = 1e-3 if stl_units == "mm" else 1e-6
    fov_mm = st.slider("Field of view (mm)", 0.10, 2.0, 0.50, 0.05)
    layer_thickness_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_tpd = st.slider("Target green packing φ_TPD", 0.85, 0.95, 0.90, 0.01)
    target_phi2D = float(np.clip(target_tpd * 0.9, 0.40, 0.88))
    n_particles_cap = st.slider("Visual cap: particles/layer", 100, 2500, 1200, 50)
    vis_size_scale = st.slider("Visual particle scale (×)", 0.5, 5.0, 1.0, 0.1,
                               help="1.0 = true scale; >1 exaggerates size for demos.")
    with st.expander("Diagnostics", expanded=False):
        import importlib.metadata as md
        def v(name):
            try: return md.version(name)
            except: return "not installed"
        st.write("numpy:", v("numpy"))
        st.write("pandas:", v("pandas"))
        st.write("matplotlib:", v("matplotlib"))
        st.write("Pillow:", v("Pillow"))
        st.write("plotly:", v("plotly"))
        st.write("trimesh:", v("trimesh"))
        st.write("shapely:", v("shapely"))
        st.write("scipy (optional):", v("scipy"))

# ---------------------------- Tabs -------------------------------------------
tab_pred, tab_single, tab_compare = st.tabs(["Predict (Top-5)", "Digital Twin (Single)", "Digital Twin (Compare)"])

# ============================ PREDICT TAB =====================================
with tab_pred:
    st.subheader("Top-5 Candidate Recipes (target φ_green ≥ 0.90)")
    top5_df = make_top5_recipes(material, d50_um)
    st.dataframe(top5_df, use_container_width=True, hide_index=True)
    st.session_state["top5_recipes_df"] = top5_df

# ============================ SINGLE VIEWER ===================================
with tab_single:
    st.subheader("Digital Twin — Single Recipe Viewer")

    if "top5_recipes_df" not in st.session_state or st.session_state["top5_recipes_df"] is None:
        st.info("No recipes yet — go to the Predict tab first.")
    else:
        rec_df = st.session_state["top5_recipes_df"].copy()
        opt = st.selectbox("Choose a recipe", list(rec_df["id"]))
        rec = rec_df[rec_df["id"] == opt].iloc[0]
        sat_override = st.slider("Binder saturation for this view (%)", 50, 95, int(rec["saturation_pct"]), 1)

        # STL upload (with built-in test box)
        colA, colB = st.columns([2,1])
        with colA:
            stl_single = st.file_uploader("Upload STL", type=["stl"], key="stl_single")
        with colB:
            use_cube = st.checkbox("Use built-in 10 mm cube", value=False, key="cube_single")

        mesh = None
        if use_cube:
            mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
        elif stl_single is not None:
            try:
                mesh = trimesh.load(io.BytesIO(stl_single.read()), file_type="stl", force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump(concatenate=True)
            except Exception as e:
                st.error(f"Could not read STL: {e}")
                st.stop()
        else:
            st.info("Upload an STL or use the built-in cube.")
            st.stop()

        st.plotly_chart(go.Figure(data=[stl_to_plotly_mesh(mesh)]).update_layout(
            scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=380
        ), use_container_width=True)

        # Layer slider
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_thickness_um * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.markdown(f"Layers detected: {n_layers}  |  Z extent: {maxz - minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, 1, key="layer_single")

        # Slice & crop
        z = minz + (layer_idx - 0.5) * thickness
        polys_full = slice_mesh_polygons(mesh, z)
        polys = apply_fov_crop(polys_full, fov_mm)
        if not polys:
            st.warning("No cross-section at this layer (or FOV too small). Try another layer or increase FOV.")
            st.stop()

        # PSD → diameters (STL units)
        diam_um = sample_psd_um(7000,
                                _safe_float(rec["d50_um"], d50_um),
                                _safe_float(rec.get("d10_um", None), None),
                                _safe_float(rec.get("d90_um", None), None),
                                seed=10_000 + layer_idx)
        diam_units = diam_um * (1e-3 if stl_units=="mm" else 1e-6) * vis_size_scale

        # Pack
        centers, radii, phi2D = pack_particles_in_domain(polys, diam_units, target_phi2D,
                                                         max_particles=n_particles_cap, max_trials=220_000,
                                                         seed=10_000 + layer_idx)

        # Raster pores & voids
        px = 840
        pore_mask = ~raster_particle_mask(centers, radii, fov_mm, fov_mm, px)
        void_mask = void_mask_from_saturation(pore_mask, saturation=sat_override/100.0,
                                              rng=np.random.default_rng(123+layer_idx))

        # Render + download
        fig, ax = plt.subplots(figsize=(5.8,5.8), dpi=190)
        title = f'{rec["id"]}: {rec["binder_type"]} · Sat {int(sat_override)}% · Layer {int(rec["layer_um"])} µm'
        render_layer(ax, centers, radii, void_mask, binder_color(rec["binder_type"]), fov_mm, title=title)
        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight"); buf.seek(0)
        st.download_button("Download current layer (PNG, 300 dpi)", data=buf,
                           file_name=f"{rec['id']}_layer_{layer_idx:04d}.png", mime="image/png")

# ============================ COMPARE VIEWER ==================================
with tab_compare:
    st.subheader("Digital Twin — Compare Top-5 Recipes")

    rec_df = st.session_state.get("top5_recipes_df", None)
    if rec_df is None or rec_df.empty:
        st.info("No recipes yet — go to the Predict tab to generate options.")
        st.stop()

    ids_all = list(rec_df["id"])
    sel_ids = st.multiselect("Choose recipes to compare", ids_all, default=ids_all)
    subset = rec_df[rec_df["id"].isin(sel_ids)].reset_index(drop=True)
    if subset.empty:
        st.warning("Pick at least one recipe.")
        st.stop()

    # STL upload (with cube option)
    colA, colB = st.columns([2,1])
    with colA:
        stl_cmp = st.file_uploader("Upload STL", type=["stl"], key="stl_compare")
    with colB:
        use_cube_cmp = st.checkbox("Use built-in 10 mm cube", value=False, key="cube_compare")

    mesh = None
    if use_cube_cmp:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_cmp is not None:
        try:
            mesh = trimesh.load(io.BytesIO(stl_cmp.read()), file_type="stl", force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump(concatenate=True)
        except Exception as e:
            st.error(f"Could not read STL: {e}")
            st.stop()
    else:
        st.info("Upload an STL or use the built-in cube.")
        st.stop()

    st.plotly_chart(go.Figure(data=[stl_to_plotly_mesh(mesh)]).update_layout(
        scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360
    ), use_container_width=True)

    # Shared layer slider (use median layer among selections for navigation)
    minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    median_layer_um = float(np.median(subset["layer_um"]))
    thickness = median_layer_um * (1e-3 if stl_units=="mm" else 1e-6)
    n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
    st.markdown(f"Layers detected: {n_layers}  |  Z extent: {maxz - minz:.3f} {stl_units}")
    layer_idx = st.slider("Layer index", 1, n_layers, 1, key="layer_compare")

    # Slice & crop
    z = minz + (layer_idx - 0.5) * thickness
    polys_full = slice_mesh_polygons(mesh, z)
    polys = apply_fov_crop(polys_full, fov_mm)
    if not polys:
        st.warning("No cross-section at this layer (or FOV too small). Try another layer or increase FOV.")
        st.stop()

    # Lock microstructure for apples-to-apples
    lock_micro = st.toggle("Lock microstructure across recipes", value=True)
    seed = 20_000 + layer_idx if lock_micro else np.random.randint(0, 1_000_000)

    # Use PSD from first selected (move inside loop if you want per-recipe PSDs)
    d50 = _safe_float(subset["d50_um"].iloc[0], d50_um)
    d10 = _safe_float(subset.get("d10_um", pd.Series([None])).iloc[0], None)
    d90 = _safe_float(subset.get("d90_um", pd.Series([None])).iloc[0], None)
    diam_um = sample_psd_um(7500, d50, d10, d90, seed=seed)
    diam_units = diam_um * (1e-3 if stl_units=="mm" else 1e-6) * vis_size_scale

    centers, radii, phi2D = pack_particles_in_domain(polys, diam_units, target_phi2D,
                                                     max_particles=n_particles_cap, max_trials=240_000, seed=seed)

    # Raster once; compute voids per recipe
    px = 780
    pore_mask = ~raster_particle_mask(centers, radii, fov_mm, fov_mm, px)

    n_sel = len(subset)
    if n_sel <= 3:
        cols = st.columns(n_sel)
        for col, (_, row) in zip(cols, subset.iterrows()):
            with col:
                fig, ax = plt.subplots(figsize=(5.2,5.2), dpi=188)
                vm = void_mask_from_saturation(pore_mask, row["saturation_pct"]/100.0,
                                               rng=np.random.default_rng(123 + int(row["saturation_pct"]) + layer_idx))
                title = f'{row["id"]}: {row["binder_type"]} · Sat {int(row["saturation_pct"])}% · Layer {int(row["layer_um"])} µm'
                render_layer(ax, centers, radii, vm, binder_color(row["binder_type"]), fov_mm, title=title)
                st.pyplot(fig, use_container_width=True)
    else:
        tabs = st.tabs(list(subset["id"]))
        for tab, (_, row) in zip(tabs, subset.iterrows()):
            with tab:
                fig, ax = plt.subplots(figsize=(5.3,5.3), dpi=188)
                vm = void_mask_from_saturation(pore_mask, row["saturation_pct"]/100.0,
                                               rng=np.random.default_rng(123 + int(row["saturation_pct"]) + layer_idx))
                title = f'{row["id"]}: {row["binder_type"]} · Sat {int(row["saturation_pct"])}% · Layer {int(row["layer_um"])} µm'
                render_layer(ax, centers, radii, vm, binder_color(row["binder_type"]), fov_mm, title=title)
                st.pyplot(fig, use_container_width=True)

    st.caption(f"Shared view — FOV={fov_mm:.2f} mm · φ₂D(target)≈{target_phi2D:.2f} · φ₂D(achieved)≈{phi2D:.2f} · Porosity₂D≈{(1-phi2D):.2f}")
