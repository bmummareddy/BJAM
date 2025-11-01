# -*- coding: utf-8 -*-
# BJAM Predictions + Digital Twin (uses your shared.py models)
# Figures generated with the help of ChatGPT (GPT-5 Thinking).

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

# ---------- page ----------
st.set_page_config(page_title="BJAM Predictions", page_icon="🟨", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#FFFDF8 0%,#FFF6E9 50%,#FFF1DD 100%)}
[data-testid="stSidebar"]{background:#fffdfa;border-right:1px solid #f3e8d9}
.stTabs [data-baseweb="tab-list"]{gap:12px}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #f3e8d9;border-radius:10px;padding:6px 10px}
.block-container{padding-top:1.1rem;padding-bottom:1.1rem}
</style>
""", unsafe_allow_html=True)

# ---------- imports from your model code ----------
if importlib.util.find_spec("shared") is None:
    st.error("shared.py not found. Please keep shared.py next to this file.")
    st.stop()

from shared import (  # uses your exact training/prediction logic
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    guardrail_ranges,
    copilot,
)

# ---------- colors ----------
BINDER_COLORS = {"PVOH":"#F2B233","PEG":"#F2D06F","Furan":"#F5C07A","Acrylic":"#FFD166","Other":"#F4B942"}
PARTICLE="#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"

def binder_color(name:str)->str:
    key=(name or "").lower()
    for k,v in BINDER_COLORS.items():
        if k.lower() in key: return v
    return BINDER_COLORS["Other"]

# ---------- dataset + models ----------
df_base, src_path = load_dataset(".")
models, meta = train_green_density_models(df_base)

# ---------- header ----------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided few-shot models from your dataset (shared.py). Digital Twin added. Generated with help of ChatGPT.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", "Silicon Carbide (SiC)")
    d50_um  = st.number_input("D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_green = st.slider("Target green density (% of theoretical)", 80, 98, 90, 1)
    guardrails_on = st.toggle("Guardrails", value=True)

    st.divider()
    st.markdown("Data / Models")
    st.write("Source:", os.path.basename(src_path) if src_path else "—")
    st.write("Rows:", f"{len(df_base):,}")
    st.write("Models:", "trained" if models else "—")

# ---------- tabs ----------
tab_pred, tab_heat, tab_sens, tab_pack, tab_form, tab_twin = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)"]
)

# =============================================================================
#  PREDICT (Top-5) — uses your copilot(...) from shared.py
# =============================================================================
with tab_pred:
    st.subheader("Top-5 parameter sets (your quantile models)")
    top_k = st.selectbox("How many?", [3,5,7,10], index=1)
    if len(df_base)==0 or not models:
        st.warning("No dataset/models found by shared.py — running will still work, but predictions will be trivial. Please ensure BJAM_All_Deep_Fill_v9.csv is loaded.")
    recs = copilot(
        material=material,
        d50_um=float(d50_um),
        df_source=df_base,
        models=models,
        guardrails_on=guardrails_on,
        target_green=float(target_green),
        top_k=int(top_k),
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.session_state["top5_recipes_df"] = recs

# =============================================================================
#  HEATMAP — built from predict_quantiles(...) so it matches your models
# =============================================================================
with tab_heat:
    st.subheader("Predicted green %TD (q50) — speed × saturation")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = gr["binder_saturation_pct"]; spd_lo, spd_hi = gr["roller_speed_mm_s"]
    xs = np.linspace(float(sat_lo), float(sat_hi), 55)
    ys = np.linspace(float(spd_lo), float(spd_hi), 45)

    grid = pd.DataFrame(
        [(b, v, layer_um, d50_um, material) for b in xs for v in ys],
        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"]
    )
    # minimal categorical hints (shared.predict_quantiles tolerates extra cols)
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    pred = predict_quantiles(models, grid)  # <- your q10/q50/q90
    Xs = sorted(pred["binder_saturation_pct"].unique())
    Ys = sorted(pred["roller_speed_mm_s"].unique())
    Z = pred.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=Xs, y=Ys, colorscale="Viridis", zsmooth=False,
                             colorbar=dict(title="%TD (q50)", len=0.8)))
    thr = float(target_green)
    fig.add_trace(go.Contour(z=Z, x=Xs, y=Ys,
                             contours=dict(start=thr,end=thr,size=1, coloring="lines", showlabels=False),
                             line=dict(color="crimson", dash="dash", width=3), showscale=False, hoverinfo="skip"))
    # simple ROI & crosshair for readability (static window)
    x0,x1 = max(min(Xs)+2, 75), min(max(Xs)-2, 90)
    y0,y1 = max(min(Ys)+0.1, 1.45), min(max(Ys)-0.1, 1.75)
    fig.update_layout(shapes=[dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                   line=dict(color="#48B4B8", width=3, dash="dash"),
                                   fillcolor="rgba(0,0,0,0)")])
    cx,cy=0.5*(x0+x1),0.5*(y0+y1)
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                             marker=dict(symbol="circle-open-dot", size=14, line=dict(width=2, color="#1f2937")),
                             hoverinfo="skip", name=""))
    fig.add_shape(type="line", x0=cx-1.6, x1=cx+1.6, y0=cy, y1=cy, line=dict(color="#1f2937", width=2))
    fig.add_shape(type="line", x0=cx, x1=cx, y0=cy-0.08, y1=cy+0.08, line=dict(color="#1f2937", width=2))
    fig.update_layout(title="Predicted Green Density (% Theoretical Density)",
                      xaxis_title="Binder Saturation (%)", yaxis_title="Roller Speed (mm/s)",
                      margin=dict(l=20,r=20,t=40,b=35), height=480)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
#  SATURATION SENSITIVITY — from your models (q10/q50/q90)
# =============================================================================
with tab_sens:
    st.subheader("Saturation sensitivity at representative speed")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 50)

    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "layer_thickness_um": np.full_like(sat_axis, layer_um),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material]*len(sat_axis),
        "material_class":"metal","binder_type_rec":"solvent_based",
    })
    pred = predict_quantiles(models, grid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q90"], name="q90", mode="lines"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                      height=380, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
#  QUALITATIVE PACKING (unchanged playful sketch)
# =============================================================================
with tab_pack:
    np.random.seed(42)
    pts = np.random.rand(220,2); r = (target_green/100.0)*0.030 + 0.004
    fig3, ax3 = plt.subplots(figsize=(6.6,4.6), dpi=160); ax3.set_aspect("equal","box")
    for (x,y) in pts: ax3.add_patch(plt.Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25, alpha=0.85))
    ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(f"Qualitative packing (~{target_green:.0f}% effective)")
    st.pyplot(fig3, clear_figure=True)

# =============================================================================
#  FORMULAE
# =============================================================================
with tab_form:
    st.subheader("Formulae (symbols)")
    st.latex(r"\text{Furnas packing:}\quad \phi_{\max}\approx 1-\prod_i(1-\phi_i)")
    st.latex(r"\text{Washburn:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta}\, r \, t}")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\text{Packing fraction:}\quad \phi=\frac{V_{\text{solids}}}{V_{\text{total}}}")

# =============================================================================
#  DIGITAL TWIN (BETA) — recipe-true scaling & robust slicing
# =============================================================================
# Optional geometry deps (no hard-stop on import; we’ll soft-fail inside the tab)
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH: import trimesh  # type: ignore
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore

def sample_psd_um(n:int, d50_um:float, d10_um:Optional[float], d90_um:Optional[float], seed:int)->np.ndarray:
    rng = np.random.default_rng(seed)
    med = max(1e-9, d50_um)
    if d10_um and d90_um and d90_um>d10_um>0:
        m = np.log(med); s = (np.log(d90_um)-np.log(d10_um))/(2*1.2815515655446004); s=max(s,0.05)
    else:
        m, s = np.log(med), 0.25
    return np.exp(rng.normal(m,s,size=n))

def load_mesh(fileobj):
    try:
        mesh = trimesh.load(io.BytesIO(fileobj.read()), file_type="stl", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh): mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}"); return None

def slice_polys(mesh, z)->List[Polygon]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        out = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in out if p.is_valid and p.area>1e-8]
    except Exception:
        return []

def crop_fov(polys, fov):
    if not polys: return []
    dom = unary_union(polys); cx,cy = dom.centroid.x, dom.centroid.y
    half=fov/2.0; win = box(cx-half, cy-half, cx+half, cy+half)
    res = dom.intersection(win)
    if getattr(res, "is_empty", True): return []
    if isinstance(res, Polygon): return [res]
    return [g for g in res.geoms if isinstance(g, Polygon) and g.is_valid and g.area>1e-8]

def pack_in_domain(polys, diam_units, phi2D_target, max_particles, max_trials, seed):
    if not polys: return np.empty((0,2)), np.empty((0,)), 0.0
    dom = unary_union(polys); minx,miny,maxx,maxy = dom.bounds; area_dom=dom.area
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []; area_circ=0.0
    tgt = float(np.clip(phi2D_target,0.05,0.90))*area_dom
    rng = np.random.default_rng(seed)
    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/400.0); grid:Dict[Tuple[int,int],List[int]]={}
    def ok(x,y,r):
        gx,gy=int(x//cell),int(y//cell)
        for ix in range(gx-1,gx+2):
            for iy in range(gy-1,gy+2):
                for j in grid.get((ix,iy),[]):
                    dx,dy=x-placed_xy[j][0],y-placed_xy[j][1]
                    if dx*dx+dy*dy < (r+placed_r[j])**2: return False
        return dom.contains(Point(x,y))
    trials=0
    for d in diam:
        r=d/2.0
        for _ in range(180):
            trials+=1
            if trials>max_trials or area_circ>=tgt or len(placed_xy)>=max_particles: break
            x=rng.uniform(minx,maxx); y=rng.uniform(miny,maxy)
            if ok(x,y,r):
                idx=len(placed_xy); placed_xy.append((x,y)); placed_r.append(r)
                gx,gy=int(x//cell),int(y//cell); grid.setdefault((gx,gy),[]).append(idx)
                area_circ += math.pi*r*r
        if trials>max_trials or area_circ>=tgt or len(placed_xy)>=max_particles: break
    centers=np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii=np.array(placed_r) if placed_r else np.empty((0,))
    return centers, radii, (area_circ/area_dom)

def raster_mask(centers, radii, fov, px=840):
    img=Image.new("L",(px,px),color=0); drw=ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def voids_from_saturation(pore_mask, saturation, rng=None):
    if rng is None: rng=np.random.default_rng(0)
    pore=int(pore_mask.sum()); 
    if pore<=0: return np.zeros_like(pore_mask,bool)
    target=int(round((1.0 - saturation)*pore))
    if target<=0: return np.zeros_like(pore_mask,bool)
    if importlib.util.find_spec("scipy"):
        from scipy import ndimage as ndi
        dist=ndi.distance_transform_edt(pore_mask); noise=ndi.gaussian_filter(rng.standard_normal(pore_mask.shape),sigma=2.2)
        field=dist+0.18*noise; flat=field[pore_mask]
        kth=np.partition(flat,len(flat)-target)[len(flat)-target]
        vm=np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm=ndi.binary_opening(vm,iterations=1); vm=ndi.binary_closing(vm,iterations=1)
        return vm
    # fallback: dotted
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    while area<target and tries<120000:
        tries+=1; r=int(np.clip(np.random.normal(3.0,1.2),1.0,6.0))
        x=np.random.randint(r,w-r); y=np.random.randint(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]
            disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

with tab_twin:
    st.subheader("Digital Twin — recipe-true layer preview & compare")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Digital Twin needs 'trimesh' and 'shapely' (see requirements.txt).")
        st.stop()

    # pull Top-5
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.warning("Generate Top-5 in the Predict tab first.")
        st.stop()

    # choose recipe(s)
    left,right = st.columns([1.2,1])
    with left:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with right:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        fov_mm = st.slider("Field of view (mm)", 0.10, 2.00, 0.50, 0.05)
        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 100, 2500, 1200, 50)

    # mesh upload / cube
    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    # load/select mesh
    mesh=None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = load_mesh(stl_file)

    # recipe context (use recipe-specific layer & D50)
    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um", d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    diam_um = sample_psd_um(7500, d50_r, None, None, seed=9991)
    diam_units = diam_um * um2unit

    # layer slider based on the recipe layer
    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.markdown(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, 1)
        z = minz + (layer_idx - 0.5) * thickness
    else:
        n_layers, layer_idx, z = 1, 1, 0.0
        st.info("No STL — using a centered square FOV for microstructure.")

    # mesh preview
    if mesh is not None and show_mesh:
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # slicing and cropping; fallback to centered box
    polys: List[Polygon] = []
    if mesh is not None:
        polys = slice_polys(mesh, z)
        polys = crop_fov(polys, fov_mm)
    if (not polys):
        half=fov_mm/2.0
        polys=[box(-half,-half,half,half)]

    # pack once per layer (true-scale)
    centers, radii, phi2D = pack_in_domain(polys, diam_units, phi2D_target,
                                           max_particles=cap, max_trials=240_000, seed=20_000+layer_idx)

    # render helpers
    def draw_particles(ax, fov):
        ax.add_patch(Rectangle((0,0), fov, fov, facecolor="white", edgecolor=BORDER, linewidth=1.4))
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov); ax.set_xticks([]); ax.set_yticks([])

    def draw_binder(ax, fov, sat_pct, binder_hex):
        px=840
        pores = ~raster_mask(centers, radii, fov, px)
        vmask = voids_from_saturation(pores, saturation=float(sat_pct)/100.0,
                                      rng=np.random.default_rng(123+layer_idx))
        ax.add_patch(Rectangle((0,0), fov, fov, facecolor=binder_hex, edgecolor=BORDER, linewidth=1.4))
        ys,xs=np.where(vmask)
        if len(xs):
            xm = xs*(fov/vmask.shape[1]); ym=(vmask.shape[0]-ys)*(fov/vmask.shape[0])
            ax.scatter(xm, ym, s=0.35, c=VOID, alpha=0.95, linewidths=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov); ax.set_xticks([]); ax.set_yticks([])

    # single-trial viewer
    st.subheader("Single trial")
    sat_pct = float(rec.get("saturation_pct", 80.0)); binder = str(rec.get("binder_type","PVOH"))
    colA, colB = st.columns(2)
    with colA:
        figA, axA = plt.subplots(figsize=(5.2,5.2), dpi=190)
        draw_particles(axA, fov_mm)
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with colB:
        figB, axB = plt.subplots(figsize=(5.2,5.2), dpi=190)
        draw_binder(axB, fov_mm, sat_pct, binder_color(binder))
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={fov_mm:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{phi2D:.2f} · Porosity₂D≈{(1-phi2D):.2f}")

    # multi-trial compare (same microstructure → different voids/colors)
    st.subheader("Compare trials")
    if not picks:
        st.info("Pick one or more trials above to compare.")
    else:
        cols = st.columns(min(3, len(picks))) if len(picks)<=3 else None
        tabs = None if cols else st.tabs(picks)
        for i, rid in enumerate(picks):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0)); hexc=binder_color(str(row.get("binder_type","PVOH")))
            px=760; pores=~raster_mask(centers, radii, fov_mm, px)
            vm = voids_from_saturation(pores, saturation=sat/100.0,
                                       rng=np.random.default_rng(987+int(sat)+layer_idx))
            figC, axC = plt.subplots(figsize=(5.1,5.1), dpi=185)
            axC.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=hexc, edgecolor=BORDER, linewidth=1.4))
            ys,xs=np.where(vm)
            if len(xs):
                xm=xs*(fov_mm/vm.shape[1]); ym=(vm.shape[0]-ys)*(fov_mm/vm.shape[0])
                axC.scatter(xm, ym, s=0.35, c=VOID, alpha=0.95, linewidths=0)
            for (x,y), r in zip(centers, radii):
                axC.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
            axC.set_aspect('equal','box'); axC.set_xlim(0,fov_mm); axC.set_ylim(0,fov_mm); axC.set_xticks([]); axC.set_yticks([])
            axC.set_title(f'{row["id"]}: {row["binder_type"]} · Sat {int(sat)}% · Layer {int(row.get("layer_um", layer_r))} µm', fontsize=10)
            if cols: 
                with cols[i]: st.pyplot(figC, use_container_width=True)
            else:
                with tabs[i]: st.pyplot(figC, use_container_width=True)
