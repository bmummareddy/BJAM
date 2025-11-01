# -*- coding: utf-8 -*-
# Binder Jet Parameter Studio — Predictions + Digital Twin
# Complete single-file app. Figures generated with the help of ChatGPT (GPT-5 Thinking).

from __future__ import annotations
import io, os, math, importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# ------------------------------- Page config ---------------------------------
st.set_page_config(page_title="Binder Jet Parameter Studio", page_icon="🧪", layout="wide")

# ------------------------------- Soft dep checks ------------------------------
def _need(pkg: str, pip_name: Optional[str] = None) -> None:
    if importlib.util.find_spec(pkg) is None:
        st.error(f"Missing Python package: '{pip_name or pkg}'. Add it to requirements.txt and redeploy.")
        st.stop()

for _pkg, _pip in [("numpy","numpy"),("pandas","pandas"),("plotly","plotly"),
                   ("matplotlib","matplotlib"),("PIL","Pillow")]:
    _need(_pkg, _pip)

# Optional geometry deps (only for Digital Twin)
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
HAVE_SCIPY   = importlib.util.find_spec("scipy") is not None
if HAVE_TRIMESH:
    import trimesh  # type: ignore
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
if HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore

# ------------------------------- Theme bits ----------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#FFFDF8 0%,#FFF6E9 50%,#FFF1DD 100%); }
[data-testid="stSidebar"]{ background:#fffdfa; border-right:1px solid #f3e8d9; }
.stTabs [data-baseweb="tab-list"]{ gap:12px; }
.stTabs [data-baseweb="tab"]{ background:#fff; border:1px solid #f3e8d9; border-radius:10px; padding:6px 10px; }
.block-container{ padding-top:1.1rem; padding-bottom:1.1rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------- Colors --------------------------------------
BINDER_COLORS = {"PVOH":"#F2B233","PEG":"#F2D06F","Furan":"#F5C07A","Acrylic":"#FFD166","Other":"#F4B942"}
PARTICLE_COLOR="#2F6CF6"; PARTICLE_EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"

def binder_color(name:str)->str:
    k = (name or "").lower()
    for key, val in BINDER_COLORS.items():
        if key.lower() in k: return val
    return BINDER_COLORS["Other"]

# --------------------------------- Data --------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame,str]:
    cands = [
        os.environ.get("BJAM_DATA",""),
        "data/BJAM_All_Deep_Fill_v9.csv",
        "BJAM_All_Deep_Fill_v9.csv",
        "/mnt/data/BJAM_All_Deep_Fill_v9.csv",
    ]
    for p in cands:
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                pass
    return pd.DataFrame(), ""

df_base, src = load_dataset()

# --------------------------------- Guardrails --------------------------------
def guardrail_ranges(d50_um: float, on: bool=True) -> dict:
    # Simple physics-inspired envelopes; tweak as needed
    if not on:
        return dict(
            binder_saturation_pct=(55.0, 99.0),
            roller_speed_mm_s=(1.2, 3.5),
        )
    # layer thickness guidance often ~3–5×D50; speed narrower for ceramics
    sat_lo, sat_hi = 65.0, 97.0
    v_lo = 1.2 + 0.0*np.sqrt(max(d50_um,1))  # keep flat unless you prefer a d50 coupling
    v_hi = 3.2 - 0.0*np.sqrt(max(d50_um,1))
    return dict(
        binder_saturation_pct=(sat_lo, sat_hi),
        roller_speed_mm_s=(max(1.1, v_lo), min(3.5, v_hi)),
    )

# --------------------------------- Heuristic model ---------------------------
# If CSV present, we bias around historical successes; otherwise use a smooth, unimodal surrogate.
def score_green_density(sat: np.ndarray, speed: np.ndarray, layer_um: float, d50_um: float) -> np.ndarray:
    # peak around (sat≈82, speed≈1.7) with broad rolloff; penalize extreme layer
    sat0, spd0 = 82.0, 1.7
    a_s, a_v = 0.0022, 2.2      # widths
    base = 92.5 - a_s*(sat-sat0)**2 - a_v*(speed-spd0)**2
    layer_ratio = layer_um / max(d50_um, 1.0)
    layer_pen = 3.0*abs(layer_ratio - 4.0)   # sweet spot ~ 4×D50
    return np.clip(base - layer_pen, 60, 99)

# --------------------------------- Recommender -------------------------------
@dataclass
class Recipe:
    id: str
    binder_type: str
    saturation_pct: float
    layer_um: float
    roller_mm_s: float
    d50_um: float
    phi_green: float

def make_top5_recipes(material: str, d50_um: float, layer_um: float, guard_on: bool=True) -> pd.DataFrame:
    gr = guardrail_ranges(d50_um, guard_on)
    sat_lo, sat_hi = gr["binder_saturation_pct"]; v_lo, v_hi = gr["roller_speed_mm_s"]
    # dense grid search + pick top 5
    s_axis = np.linspace(sat_lo, sat_hi, 45)
    v_axis = np.linspace(v_lo, v_hi, 45)
    S, V = np.meshgrid(s_axis, v_axis)
    Z = score_green_density(S, V, layer_um, d50_um)
    flat = pd.DataFrame({"binder_saturation_pct":S.ravel(),"roller_speed_mm_s":V.ravel(),"score":Z.ravel()})
    best = flat.sort_values("score", ascending=False).head(5).reset_index(drop=True)

    # diversify binder types cyclically
    binders = ["PVOH","PEG","Acrylic","Furan","PVOH"]
    recs = []
    for i, row in best.iterrows():
        recs.append(Recipe(id=f"Opt-{i+1}",
                           binder_type=binders[i],
                           saturation_pct=float(row["binder_saturation_pct"]),
                           layer_um=float(layer_um),
                           roller_mm_s=float(row["roller_speed_mm_s"]),
                           d50_um=float(d50_um),
                           phi_green=float(row["score"]/100.0)))
    return pd.DataFrame([r.__dict__ for r in recs])

# --------------------------------- PSD helpers -------------------------------
def lognormal_from_quantiles(d50_um: float, d10_um: Optional[float], d90_um: Optional[float]) -> Tuple[float,float]:
    med = max(1e-9, float(d50_um))
    if d10_um and d90_um and d90_um > d10_um > 0:
        m = np.log(med)
        s = (np.log(d90_um) - np.log(d10_um)) / (2*1.2815515655446004)
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    return m, s

def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m, s = lognormal_from_quantiles(d50_um, d10_um, d90_um)
    return np.exp(rng.normal(m, s, size=n))  # diameters [µm]

# --------------------------------- Digital twin helpers ----------------------
def load_mesh_from_uploader(fileobj):
    try:
        mesh = trimesh.load(io.BytesIO(fileobj.read()), file_type="stl", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

def slice_mesh_polygons(mesh, z) -> List[Polygon]:
    try:
        section = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if section is None: return []
        planar, _ = section.to_planar()
        polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
    except Exception:
        return []

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

def pack_particles_in_domain(polys: List[Polygon], diam_units: np.ndarray,
                             target_area_frac: float, max_particles: int, max_trials: int, seed: int
                             ) -> Tuple[np.ndarray, np.ndarray, float]:
    if not polys:
        return np.empty((0,2)), np.empty((0,)), 0.0
    domain = unary_union(polys)
    minx, miny, maxx, maxy = domain.bounds
    area_domain = domain.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circles = 0.0
    target_area = float(np.clip(target_area_frac, 0.05, 0.90)) * area_domain

    rng = np.random.default_rng(seed)
    cell = max((diam.max()/2.0), (maxx-minx+maxy-miny)/400.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def ok(x,y,r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1,gx+2):
            for iy in range(gy-1,gy+2):
                for j in grid.get((ix,iy), []):
                    dx, dy = x-placed_xy[j][0], y-placed_xy[j][1]
                    if dx*dx + dy*dy < (r+placed_r[j])**2: return False
        return domain.contains(Point(x,y))

    trials=0
    for d in diam:
        r=d/2.0
        for _ in range(180):
            trials += 1
            if trials>max_trials or area_circles>=target_area or len(placed_xy)>=max_particles: break
            x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
            if ok(x,y,r):
                idx = len(placed_xy); placed_xy.append((x,y)); placed_r.append(r)
                gx, gy = int(x//cell), int(y//cell)
                grid.setdefault((gx,gy), []).append(idx)
                area_circles += math.pi*r*r
        if trials>max_trials or area_circles>=target_area or len(placed_xy)>=max_particles: break
    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    return centers, radii, (area_circles/area_domain)

def raster_particle_mask(centers: np.ndarray, radii: np.ndarray, width_units: float, height_units: float, px: int) -> np.ndarray:
    img = Image.new("L", (px,px), color=0)
    drw = ImageDraw.Draw(img)
    sx = px / width_units; sy = px / height_units
    for (x,y), r in zip(centers, radii):
        x0 = int((x - r)*sx); y0 = int((height_units - (y + r))*sy)
        x1 = int((x + r)*sx); y1 = int((height_units - (y - r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img) > 0)

def void_mask_from_saturation(pore_mask: np.ndarray, saturation: float, rng=None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng(0)
    pore_area = int(pore_mask.sum())
    if pore_area <= 0: return np.zeros_like(pore_mask, dtype=bool)
    target_void = int(round((1.0 - saturation) * pore_area))
    if target_void <= 0: return np.zeros_like(pore_mask, dtype=bool)

    if HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.3)
        field = dist + 0.18*noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat)-target_void)[len(flat)-target_void]
        void_mask = np.zeros_like(pore_mask, dtype=bool)
        void_mask[pore_mask] = field[pore_mask] >= kth
        void_mask = ndi.binary_opening(void_mask, iterations=1)
        void_mask = ndi.binary_closing(void_mask, iterations=1)
        return void_mask

    # fallback: area-matched dots
    h,w = pore_mask.shape
    void = np.zeros_like(pore_mask, dtype=bool); area=0; tries=0
    while area < target_void and tries < 120000:
        tries += 1
        r = int(np.clip(np.random.normal(3.0,1.2), 1.0, 6.0))
        x = np.random.randint(r, w-r); y = np.random.randint(r, h-r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            void[add] = True
            area = int(void.sum())
    return void

# --------------------------------- Header ------------------------------------
st.title("Binder Jet Parameter Studio")
st.write("Input a powder and PSD, get five viable process recipes targeting theoretical packing ≥ 0.90, then inspect layers with a physics-aware digital twin (true-scale particles, binder, voids).")
st.caption("This app and figures were generated with the help of ChatGPT (GPT-5 Thinking).")

# --------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base.columns else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", value="Silicon Carbide (SiC)")
    d50_um = st.number_input("PSD: D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    d10_um = st.number_input("PSD: D10 (µm, optional)", min_value=0.0, value=20.0, step=1.0)
    d90_um = st.number_input("PSD: D90 (µm, optional)", min_value=0.0, value=45.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_green = st.slider("Target green density (%TD)", 80, 98, 90, 1)
    guard_on = st.toggle("Guardrails", value=True, help="Keep parameters within physics-sensible ranges.")
    st.divider()
    with st.expander("Diagnostics", expanded=False):
        st.write("Data source:", os.path.basename(src) if src else "—")
        st.write("Trimesh:", "yes" if HAVE_TRIMESH else "no")
        st.write("Shapely:", "yes" if HAVE_SHAPELY else "no")
        st.write("SciPy (voids):", "yes" if HAVE_SCIPY else "no")

# --------------------------------- Tabs --------------------------------------
tab_pred, tab_heat, tab_sens, tab_pack, tab_formula, tab_twin = st.tabs([
    "Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)"
])

# ============================ Predict (Top-5) =================================
with tab_pred:
    top5_df = make_top5_recipes(material, d50_um, layer_um, guard_on)
    st.dataframe(top5_df, use_container_width=True, hide_index=True)
    st.session_state["top5_recipes_df"] = top5_df

# ============================ Heatmap (with overlays) =========================
with tab_heat:
    gr = guardrail_ranges(d50_um, guard_on)
    b_lo, b_hi = gr["binder_saturation_pct"]; v_lo, v_hi = gr["roller_speed_mm_s"]
    X = np.linspace(b_lo, b_hi, 55)
    Y = np.linspace(v_lo, v_hi, 45)
    S, V = np.meshgrid(X, Y)
    Z = score_green_density(S, V, layer_um, d50_um)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=X, y=Y, colorscale="Viridis", zsmooth=False,
                             colorbar=dict(title="%TD (q50)", len=0.8)))
    # dashed red contour at target
    thr = float(target_green)
    fig.add_trace(go.Contour(z=Z, x=X, y=Y, contours=dict(start=thr,end=thr,size=1, coloring="lines", showlabels=False),
                             line=dict(color="crimson", dash="dash", width=3), showscale=False, hoverinfo="skip"))
    # teal ROI rectangle + crosshair
    x0, x1 = max(min(X)+2, 75), min(max(X)-2, 90)
    y0, y1 = max(min(Y)+0.1, 1.45), min(max(Y)-0.1, 1.75)
    fig.update_layout(shapes=[dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                   line=dict(color="#48B4B8", width=3, dash="dash"),
                                   fillcolor="rgba(0,0,0,0)")])
    cx, cy = 0.5*(x0+x1), 0.5*(y0+y1)
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                             marker=dict(symbol="circle-open-dot", size=14, line=dict(width=2, color="#1f2937")),
                             hoverinfo="skip", name=""))
    fig.add_shape(type="line", x0=cx-1.6, x1=cx+1.6, y0=cy, y1=cy, line=dict(color="#1f2937", width=2))
    fig.add_shape(type="line", x0=cx, x1=cx, y0=cy-0.08, y1=cy+0.08, line=dict(color="#1f2937", width=2))

    fig.update_layout(title="Predicted Green Density (% Theoretical Density)",
                      xaxis_title="Binder Saturation (%)",
                      yaxis_title="Roller Speed (mm/s)",
                      margin=dict(l=20,r=20,t=40,b=35), height=480)
    st.plotly_chart(fig, use_container_width=True)

# ============================ Sensitivity =====================================
with tab_sens:
    gr = guardrail_ranges(d50_um, guard_on)
    v_mid = 0.5*(gr["roller_speed_mm_s"][0]+gr["roller_speed_mm_s"][1])
    sat_axis = np.linspace(gr["binder_saturation_pct"][0], gr["binder_saturation_pct"][1], 60)
    z = score_green_density(sat_axis, np.full_like(sat_axis, v_mid), layer_um, d50_um)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sat_axis, y=z, mode="lines", name="q50 surrogate"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                      margin=dict(l=10,r=10,t=10,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

# ============================ Qualitative packing =============================
with tab_pack:
    np.random.seed(42)
    pts = np.random.rand(220,2)
    r = (target_green/100.0)*0.030 + 0.004
    fig3, ax3 = plt.subplots(figsize=(6.6,4.6), dpi=160)
    ax3.set_aspect("equal","box")
    for (x,y) in pts: ax3.add_patch(plt.Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25, alpha=0.85))
    ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(f"Qualitative packing (~{target_green:.0f}% effective)")
    st.pyplot(fig3, clear_figure=True)

# ============================ Formulae ========================================
with tab_formula:
    st.subheader("Formulae (symbols)")
    st.latex(r"\text{Furnas multi-size packing:}\quad \phi_{\max} \approx 1-\prod_i (1-\phi_i)")
    st.latex(r"\text{Washburn penetration:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta}\, r \, t}")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\text{Packing fraction:}\quad \phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")

# ============================ Digital Twin (Beta) =============================
with tab_twin:
    st.subheader("Digital Twin — Single & Compare trials")
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("This tab needs extra packages: trimesh and shapely. Add them to requirements.txt and redeploy.")
        st.stop()

    # Top-5 table (or demo)
    own = st.session_state.get("top5_recipes_df")
    if own is None or getattr(own, "empty", True):
        own = pd.DataFrame([
            {"id":"Opt-1","binder_type":"PVOH","saturation_pct":70,"layer_um":max(10,0.9*d50_um),"d50_um":d50_um},
            {"id":"Opt-2","binder_type":"PEG","saturation_pct":80,"layer_um":max(10,0.9*d50_um),"d50_um":d50_um},
            {"id":"Opt-3","binder_type":"Acrylic","saturation_pct":65,"layer_um":max(10,0.9*d50_um),"d50_um":d50_um},
            {"id":"Opt-4","binder_type":"Furan","saturation_pct":75,"layer_um":max(10,0.9*d50_um),"d50_um":d50_um},
            {"id":"Opt-5","binder_type":"PVOH","saturation_pct":85,"layer_um":max(10,0.9*d50_um),"d50_um":d50_um},
        ])

    colA, colB = st.columns([1.2,1])
    with colA:
        pick_one = st.selectbox("Pick a trial", list(own["id"]), index=0)
        pick_many = st.multiselect("Compare trials (optional)", list(own["id"]), default=list(own["id"])[:3])
    with colB:
        d50_loc = float(own[own["id"]==pick_one]["d50_um"].iloc[0])
        d50_loc = st.number_input("D50 (µm) for twin", min_value=1.0, value=d50_loc, step=1.0)
        d10_loc = st.number_input("D10 (µm, optional)", min_value=0.0, value=0.0, step=1.0)
        d90_loc = st.number_input("D90 (µm, optional)", min_value=0.0, value=0.0, step=1.0)

    st.divider()

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2:
        use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
    um_to_unit = 1e-3 if stl_units=="mm" else 1e-6

    fov_mm = st.slider("Field of view (mm)", 0.10, 2.00, 0.50, 0.05)
    phi_TPD_target = st.slider("Target green packing φ_TPD", 0.85, 0.95, 0.90, 0.01)
    phi2D_target = float(np.clip(0.9*phi_TPD_target, 0.40, 0.88))
    cap = st.slider("Visual cap (particles)", 100, 2500, 1200, 50)

    # Load mesh
    mesh = None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = load_mesh_from_uploader(stl_file)

    # Show 3D preview
    if mesh is not None:
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # Layer slider (robust)
    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        layer_for_slider = float(own[own["id"]==pick_one]["layer_um"].iloc[0])
        thickness = layer_for_slider * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.markdown(f"Layers: {n_layers} | Z span: {maxz - minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, 1)
    else:
        n_layers, layer_idx = 1, 1
        st.info("No STL slice available — using a centered square FOV as the packing domain.")

    # Slice (or fallback)
    polys: List[Polygon] = []
    if mesh is not None:
        z = float(mesh.bounds[0][2]) + (layer_idx - 0.5)*thickness
        polys = slice_mesh_polygons(mesh, z)
        polys = apply_fov_crop(polys, fov_mm)
    if (not polys) and HAVE_SHAPELY:
        half = fov_mm/2.0
        polys = [box(-half, -half, half, half)]

    # Build microstructure once
    diam_um = sample_psd_um(7500, d50_loc, d10_loc if d10_loc>0 else None, d90_loc if d90_loc>0 else None, seed=10_000+layer_idx)
    diam_units = diam_um * um_to_unit
    centers, radii, phi2D = pack_particles_in_domain(polys, diam_units, phi2D_target,
                                                     max_particles=cap, max_trials=240_000, seed=20_000+layer_idx)

    # Single trial viewer
    st.subheader("Single trial viewer")
    sel = own[own["id"]==pick_one].iloc[0]
    binder = str(sel.get("binder_type","PVOH"))
    sat_pct = float(sel.get("saturation_pct", 80.0))
    binder_hex = binder_color(binder)

    px = 840
    pore_mask = ~raster_particle_mask(centers, radii, fov_mm, fov_mm, px)
    vmask = void_mask_from_saturation(pore_mask, saturation=sat_pct/100.0,
                                      rng=np.random.default_rng(123+layer_idx))

    cpa, cpb = st.columns(2)
    with cpa:
        figA, axA = plt.subplots(figsize=(5.2,5.2), dpi=190)
        axA.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor="white", edgecolor=BORDER, linewidth=1.4))
        for (x,y), r in zip(centers, radii):
            axA.add_patch(Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25))
        axA.set_aspect('equal','box'); axA.set_xlim(0,fov_mm); axA.set_ylim(0,fov_mm); axA.set_xticks([]); axA.set_yticks([])
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with cpb:
        figB, axB = plt.subplots(figsize=(5.2,5.2), dpi=190)
        axB.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=binder_hex, edgecolor=BORDER, linewidth=1.4))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (fov_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (fov_mm/vmask.shape[0])
            axB.scatter(xm, ym, s=0.35, c=VOID, alpha=0.95, linewidths=0)
        for (x,y), r in zip(centers, radii):
            axB.add_patch(Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25))
        axB.set_aspect('equal','box'); axB.set_xlim(0,fov_mm); axB.set_ylim(0,fov_mm); axB.set_xticks([]); axB.set_yticks([])
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={fov_mm:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{phi2D:.2f} · Porosity₂D≈{(1-phi2D):.2f}")

    # Compare trials (use same microstructure; vary voids/colors only)
    st.subheader("Compare selected trials")
    if len(pick_many)==0:
        st.info("Pick at least one trial in ‘Compare trials’.")
    else:
        cols = st.columns(min(3, len(pick_many))) if len(pick_many) <= 3 else None
        tabs = None if cols else st.tabs(pick_many)
        for i, rid in enumerate(pick_many):
            row = own[own["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0))
            hexc = binder_color(str(row.get("binder_type","PVOH")))
            vm = void_mask_from_saturation(pore_mask, saturation=sat/100.0,
                                           rng=np.random.default_rng(987+int(sat)+layer_idx))
            title = f'{row["id"]}: {row["binder_type"]} · Sat {int(sat)}% · Layer {int(row.get("layer_um", layer_um))} µm'
            if cols:
                with cols[i]:
                    figC, axC = plt.subplots(figsize=(5.0,5.0), dpi=185)
                    axC.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=hexc, edgecolor=BORDER, linewidth=1.4))
                    ys,xs = np.where(vm)
                    if len(xs):
                        xm = xs * (fov_mm/vm.shape[1]); ym = (vm.shape[0]-ys) * (fov_mm/vm.shape[0])
                        axC.scatter(xm, ym, s=0.35, c=VOID, alpha=0.95, linewidths=0)
                    for (x,y), r in zip(centers, radii):
                        axC.add_patch(Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25))
                    axC.set_aspect('equal','box'); axC.set_xlim(0,fov_mm); axC.set_ylim(0,fov_mm); axC.set_xticks([]); axC.set_yticks([])
                    axC.set_title(title, fontsize=10)
                    st.pyplot(figC, use_container_width=True)
            else:
                with tabs[i]:
                    figC, axC = plt.subplots(figsize=(5.2,5.2), dpi=185)
                    axC.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=hexc, edgecolor=BORDER, linewidth=1.4))
                    ys,xs = np.where(vm)
                    if len(xs):
                        xm = xs * (fov_mm/vm.shape[1]); ym = (vm.shape[0]-ys) * (fov_mm/vm.shape[0])
                        axC.scatter(xm, ym, s=0.35, c=VOID, alpha=0.95, linewidths=0)
                    for (x,y), r in zip(centers, radii):
                        axC.add_patch(Circle((x,y), r, facecolor=PARTICLE_COLOR, edgecolor=PARTICLE_EDGE, linewidth=0.25))
                    axC.set_aspect('equal','box'); axC.set_xlim(0,fov_mm); axC.set_ylim(0,fov_mm); axC.set_xticks([]); axC.set_yticks([])
                    axC.set_title(title, fontsize=10)
                    st.pyplot(figC, use_container_width=True)
