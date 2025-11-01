# ======================= Digital Twin (Beta) — FULL TAB =======================
# Works with your existing Recommend table in st.session_state["top5_recipes_df"].
# If trimesh/shapely are missing or an STL layer can't be sliced, it degrades gracefully.

import importlib.util, io, math
import numpy as np
import matplotlib.pyplot as _plt
from matplotlib.patches import Circle as _Circle, Rectangle as _Rect
from PIL import Image as _Image, ImageDraw as _ImageDraw
import plotly.graph_objects as _go
import streamlit as st

# ---- soft checks so the whole app never crashes
def _need(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        st.error(f"Digital Twin needs '{pip_name or pkg}'. Add it to requirements.txt and redeploy.")
        return False
    return True

HAVE_TRIMESH = _need("trimesh")
HAVE_SHAPELY = _need("shapely")
try:
    from scipy import ndimage as _ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Imports after soft checks
if HAVE_TRIMESH: import trimesh
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union

# ---- color helpers (reuse your palette if you have one)
_PARTICLE = "#2F6CF6"; _EDGE = "#1f2937"; _BINDER = "#F4B942"; _BORDER="#111111"; _VOID="#FFFFFF"

def _binder_color(name:str):
    # simple mapper; tweak to your binder list
    m = {"pvoh":"#F2B233","peg":"#F2D06F","acrylic":"#FFD166","furan":"#F5C07A"}
    key = (name or "").lower()
    for k,v in m.items():
        if k in key: return v
    return _BINDER

# ---- PSD sampling
def _ln_params(d50, d10=None, d90=None):
    med = max(1e-9, float(d50))
    if d10 and d90 and d90 > d10 > 0:
        m = np.log(med)
        s = (np.log(d90) - np.log(d10)) / (2*1.2815515655446004)
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    return m, s

def _sample_psd_um(n, d50, d10=None, d90=None, seed=42):
    m, s = _ln_params(d50, d10, d90)
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(m, s, size=n))

# ---- STL slicing (with robust fallbacks)
def _load_mesh(fileobj):
    try:
        mesh = trimesh.load(io.BytesIO(fileobj.read()), file_type="stl", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

def _slice_polys(mesh, z):
    """Return list[Polygon] at height z. Fallback to empty list when slicing fails."""
    try:
        section = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if section is None:
            return []
        planar, _ = section.to_planar()          # map to XY
        polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
    except Exception:
        return []

def _fov_crop(polys, fov):
    if not polys: return []
    dom = unary_union(polys)
    cx, cy = dom.centroid.x, dom.centroid.y
    half = fov/2.0
    win = box(cx-half, cy-half, cx+half, cy+half)
    out = dom.intersection(win)
    if getattr(out, "is_empty", True): return []
    if isinstance(out, Polygon): return [out]
    return [g for g in out.geoms if isinstance(g, Polygon) and g.is_valid and g.area>1e-8]

# ---- Poisson-disk-ish packing inside polygon union
def _pack(polys, diam_units, target_phi2D=0.80, max_particles=1200, max_trials=200000, seed=0):
    if not polys:  # nothing to pack
        return np.empty((0,2)), np.empty((0,)), 0.0
    dom = unary_union(polys)
    minx, miny, maxx, maxy = dom.bounds
    area_dom = dom.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circles = 0.0
    target_area = float(np.clip(target_phi2D, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)
    cell = max((diam.max()/2.0), (maxx-minx+maxy-miny)/400.0)
    grid = {}

    def ok(x,y,r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1,gx+2):
            for iy in range(gy-1,gy+2):
                for j in grid.get((ix,iy), []):
                    dx, dy = x-placed_xy[j][0], y-placed_xy[j][1]
                    if dx*dx + dy*dy < (r+placed_r[j])**2: return False
        return dom.contains(Point(x,y))

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
    return centers, radii, (area_circles/area_dom)

# ---- Raster + voids
def _mask_particles(centers, radii, fov_size, px=780):
    img = _Image.new("L", (px,px), color=0)
    drw = _ImageDraw.Draw(img)
    sx = px / fov_size; sy = px / fov_size
    for (x,y), r in zip(centers, radii):
        x0 = int((x - r)*sx); y0 = int((fov_size - (y + r))*sy)
        x1 = int((x + r)*sx); y1 = int((fov_size - (y - r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img) > 0)

def _voids(pore_mask, saturation, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    pore_area = int(pore_mask.sum())
    if pore_area <= 0: return np.zeros_like(pore_mask, dtype=bool)
    target_void = int(round((1.0 - saturation) * pore_area))
    if target_void <= 0:  return np.zeros_like(pore_mask, dtype=bool)

    if HAVE_SCIPY:
        dist = _ndi.distance_transform_edt(pore_mask)
        noise = _ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.2)
        field = dist + 0.18*noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat)-target_void)[len(flat)-target_void]
        vm = np.zeros_like(pore_mask, dtype=bool)
        vm[pore_mask] = field[pore_mask] >= kth
        vm = _ndi.binary_opening(vm, iterations=1)
        vm = _ndi.binary_closing(vm, iterations=1)
        return vm

    # fallback: area-matched dots
    h,w = pore_mask.shape
    vm = np.zeros_like(pore_mask, dtype=bool); area=0; tries=0
    while area < target_void and tries < 120000:
        tries += 1
        r = int(np.clip(np.random.normal(3.0,1.2), 1.0, 6.0))
        x = np.random.randint(r, w-r); y = np.random.randint(r, h-r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            vm[add] = True
            area = int(vm.sum())
    return vm

# ---- Renderers
def _panel_particles(ax, centers, radii, fov):
    ax.add_patch(_Rect((0,0), fov, fov, facecolor="white", edgecolor=_BORDER, linewidth=1.4))
    for (x,y), r in zip(centers, radii):
        ax.add_patch(_Circle((x,y), r, facecolor=_PARTICLE, edgecolor=_EDGE, linewidth=0.25))
    ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov); ax.set_xticks([]); ax.set_yticks([])

def _panel_binder(ax, centers, radii, vmask, fov, binder_hex, title=""):
    ax.add_patch(_Rect((0,0), fov, fov, facecolor=binder_hex, edgecolor=_BORDER, linewidth=1.4))
    ys, xs = np.where(vmask)
    if len(xs):
        xm = xs * (fov/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (fov/vmask.shape[0])
        ax.scatter(xm, ym, s=0.35, c=_VOID, alpha=0.95, linewidths=0)
    for (x,y), r in zip(centers, radii):
        ax.add_patch(_Circle((x,y), r, facecolor=_PARTICLE, edgecolor=_EDGE, linewidth=0.25))
    ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)

# ============================= UI — Tab body ==================================
st.markdown("### Digital Twin (Beta)")

# Top-5 recipes from your Recommend step (or make a demo set)
_own = st.session_state.get("top5_recipes_df")
if _own is None or getattr(_own, "empty", True):
    import pandas as _pd
    _own = _pd.DataFrame([
        {"id":"Opt-1","binder_type":"PVOH","saturation_pct":70,"layer_um":100,"d50_um":30},
        {"id":"Opt-2","binder_type":"PEG","saturation_pct":80,"layer_um":120,"d50_um":30},
        {"id":"Opt-3","binder_type":"Acrylic","saturation_pct":65,"layer_um":110,"d50_um":30},
        {"id":"Opt-4","binder_type":"Furan","saturation_pct":75,"layer_um":130,"d50_um":30},
        {"id":"Opt-5","binder_type":"PVOH","saturation_pct":85,"layer_um":140,"d50_um":30},
    ])

left, right = st.columns([1.2, 1])
with left:
    pick_one = st.selectbox("Pick a trial recipe", list(_own["id"]), index=0)
    pick_many = st.multiselect("Compare recipes (optional)", list(_own["id"]), default=list(_own["id"])[:3])
with right:
    st.caption("Binder & PSD for packing")
    _d50 = st.number_input("D50 (µm)", min_value=1.0, value=float(_own[_own["id"]==pick_one]["d50_um"].iloc[0] if "d50_um" in _own else 30.0), step=1.0)
    _d10 = st.number_input("D10 (µm, optional)", min_value=0.0, value=0.0, step=1.0)
    _d90 = st.number_input("D90 (µm, optional)", min_value=0.0, value=0.0, step=1.0)

st.divider()

# Mesh inputs
cA, cB, cC = st.columns([2,1,1])
with cA:
    stl_file = st.file_uploader("Upload STL", type=["stl"], key="dt_stl")
with cB:
    use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
with cC:
    stl_units = st.selectbox("STL units", ["mm","m"], index=0)
_um2unit = 1e-3 if stl_units=="mm" else 1e-6

# Viewer settings
fov_mm = st.slider("Field of view (mm)", 0.10, 2.00, 0.50, 0.05)
phi_TPD_target = st.slider("Target green packing φ_TPD", 0.85, 0.95, 0.90, 0.01)
phi2D_target = float(np.clip(0.9*phi_TPD_target, 0.40, 0.88))
cap = st.slider("Visual cap (particles)", 100, 2500, 1200, 50)

# Load mesh (or cube)
mesh = None
if HAVE_TRIMESH:
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = _load_mesh(stl_file)

# 3D preview if we have a mesh
if mesh is not None:
    _mfig = _go.Figure(data=[_go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                                        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                                        color="lightgray", opacity=0.55, flatshading=True, name="Part")])
    _mfig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
    st.plotly_chart(_mfig, use_container_width=True)

# Layer slider
if mesh is not None:
    minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    # layer from selected trial if present
    _sel = _own[_own["id"]==pick_one].iloc[0]
    _layer_um = float(_sel.get("layer_um", 120))
    thickness = _layer_um * (1e-3 if stl_units=="mm" else 1e-6)
    n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
    st.markdown(f"Layers: {n_layers}  |  Z span: {maxz-minz:.3f} {stl_units}")
    layer_idx = st.slider("Layer index", 1, n_layers, 1)
else:
    # still allow visualization — create a synthetic window sized to FOV
    n_layers, layer_idx = 1, 1
    st.info("No STL slice available — using a centered square FOV as the packing domain.")

# Build cross-section polygons (or fallback window)
polys = []
if mesh is not None and HAVE_SHAPELY:
    z = float(mesh.bounds[0][2]) + (layer_idx - 0.5)*thickness
    polys = _slice_polys(mesh, z)
    polys = _fov_crop(polys, fov_mm)
if (not polys) and HAVE_SHAPELY:
    # centered square fallback so users can still see packing
    half = fov_mm/2.0
    polys = [box(-half, -half, half, half)]

# PSD → pack once (shared microstructure)
diam_um = _sample_psd_um(7500, _d50, _d10 if _d10>0 else None, _d90 if _d90>0 else None, seed=10_000+layer_idx)
diam_units = diam_um * _um2unit
centers, radii, phi2D = _pack(polys, diam_units, phi2D_target, max_particles=cap, max_trials=240000, seed=20_000+layer_idx)

# --------------------------- Single trial view -------------------------------
st.subheader("Single Trial Viewer")
sel = _own[_own["id"]==pick_one].iloc[0]
binder = str(sel.get("binder_type","PVOH"))
sat_pct = float(sel.get("saturation_pct", 80.0))
binder_hex = _binder_color(binder)

px = 840
pore_mask = ~_mask_particles(centers, radii, fov_mm, px)
vmask = _voids(pore_mask, saturation=sat_pct/100.0, rng=np.random.default_rng(123+layer_idx))

c1, c2 = st.columns(2)
with c1:
    figA, axA = _plt.subplots(figsize=(5.2,5.2), dpi=190)
    _panel_particles(axA, centers, radii, fov_mm)
    axA.set_title("Particles only", fontsize=10)
    st.pyplot(figA, use_container_width=True)
with c2:
    figB, axB = _plt.subplots(figsize=(5.2,5.2), dpi=190)
    _panel_binder(axB, centers, radii, vmask, fov_mm, binder_hex, title=f"{binder} · Sat {int(sat_pct)}%")
    st.pyplot(figB, use_container_width=True)

st.caption(f"FOV={fov_mm:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{phi2D:.2f} · Porosity₂D≈{(1-phi2D):.2f}")

# --------------------------- Compare multiple trials -------------------------
st.subheader("Compare Selected Trials")
if len(pick_many)==0:
    st.info("Pick at least one recipe in 'Compare recipes'.")
else:
    # reuse the same microstructure for apples-to-apples; only voids & binder color vary
    cols = st.columns(min(3, len(pick_many))) if len(pick_many) <= 3 else None
    tabs = None if cols else st.tabs(pick_many)
    for i, rid in enumerate(pick_many):
        r = _own[_own["id"]==rid].iloc[0]
        sat = float(r.get("saturation_pct", 80.0))
        bhex = _binder_color(str(r.get("binder_type","PVOH")))
        vm = _voids(pore_mask, saturation=sat/100.0, rng=np.random.default_rng(987+int(sat)+layer_idx))
        title = f'{r["id"]}: {r.get("binder_type","")} · Sat {int(sat)}% · Layer {int(r.get("layer_um",_layer_um))} µm'
        if cols:
            with cols[i]:
                figC, axC = _plt.subplots(figsize=(5.0,5.0), dpi=185)
                _panel_binder(axC, centers, radii, vm, fov_mm, bhex, title=title)
                st.pyplot(figC, use_container_width=True)
        else:
            with tabs[i]:
                figC, axC = _plt.subplots(figsize=(5.2,5.2), dpi=185)
                _panel_binder(axC, centers, radii, vm, fov_mm, bhex, title=title)
                st.pyplot(figC, use_container_width=True)
# ===================== end Digital Twin (Beta) ===============================
