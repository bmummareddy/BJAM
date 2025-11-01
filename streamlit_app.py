# -*- coding: utf-8 -*-
from __future__ import annotations
import io, math, importlib.util
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# Optional heavy deps for Digital Twin
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
HAVE_SCIPY   = importlib.util.find_spec("scipy")   is not None
if HAVE_TRIMESH: import trimesh  # type: ignore
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
if HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore
from scipy.ndimage import gaussian_filter

# Local utils
from shared import (
    load_dataset, train_green_density_models, predict_quantiles, guardrail_ranges,
)

st.set_page_config(page_title="BJAM — Binder-Jet AM Recommender", page_icon="🟨", layout="wide")
VERSION = "2025-11-01 clean build (Py3.13 pins, cache-safe, DT slicer, balanced Top-5)"
st.caption(f"Build: {VERSION}")
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#FFFDF8 0%,#FFF6E9 55%,#FFF1DD 100%)}
[data-testid="stSidebar"]{background:#fffdfa;border-right:1px solid #f3e8d9}
.stTabs [data-baseweb="tab-list"]{gap:12px}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #f3e8d9;border-radius:10px;padding:6px 10px}
.block-container{padding-top:1rem;padding-bottom:1.1rem}
</style>
""", unsafe_allow_html=True)

# ---- Colors
BINDER_COLORS = {"water_based":"#F2D06F","solvent_based":"#F2B233","other":"#F4B942"}
PARTICLE="#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"
def binder_color(name:str)->str:
    key=(name or "").lower()
    for k,v in BINDER_COLORS.items():
        if k in key: return v
    return BINDER_COLORS["other"]

# =========================== Packing helpers ===========================
def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    med = max(1e-9, float(d50_um))
    if d10_um and d90_um and d90_um > d10_um > 0:
        m = np.log(med)
        s = (np.log(d90_um) - np.log(d10_um)) / (2*1.2815515655446004)
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    d = np.exp(rng.normal(m, s, size=n))
    return np.clip(d, 0.30*med, 3.00*med)

def draw_scale_bar(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm
    x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad
    y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.5, color="#111111")
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm", ha="center", va="bottom", fontsize=9, color="#111111")

def raster_mask(centers, radii, fov, px=1024):
    img=Image.new("L",(px,px),color=0); drw=ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def voids_from_saturation(pore_mask, saturation, rng=None):
    if rng is None: rng=np.random.default_rng(0)
    pore=int(pore_mask.sum())
    if pore<=0: return np.zeros_like(pore_mask,bool)
    target=int(round((1.0 - saturation)*pore))
    if target<=0: return np.zeros_like(pore_mask,bool)
    if HAVE_SCIPY:
        dist=ndi.distance_transform_edt(pore_mask)
        noise=ndi.gaussian_filter(rng.standard_normal(pore_mask.shape),sigma=2.0)
        field=dist+0.18*noise
        flat=field[pore_mask]
        kth=np.partition(flat,len(flat)-target)[len(flat)-target]
        vm=np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm=ndi.binary_opening(vm,iterations=1); vm=ndi.binary_closing(vm,iterations=1)
        return vm
    # fallback dots
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    while area<target and tries<120000:
        tries+=1; r=int(np.clip(np.random.normal(3.0,1.2),1.0,6.0))
        x=np.random.randint(r,w-r); y=np.random.randint(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]
            disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def slice_polys(mesh, z)->List["Polygon"]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        out = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in out if p.is_valid and p.area>1e-8]
    except Exception:
        return []

def crop_window(polys, fov):
    if not polys: return [], (0.0, 0.0)
    dom = unary_union(polys); cx,cy = dom.centroid.x, dom.centroid.y
    half=fov/2.0; xmin, ymin = cx-half, cy-half
    win = box(xmin, ymin, xmin+fov, ymin+fov)
    res = dom.intersection(win)
    if getattr(res, "is_empty", True): return [], (xmin, ymin)
    geoms = [res] if isinstance(res, Polygon) else [g for g in res.geoms if isinstance(g, Polygon)]
    return geoms, (xmin, ymin)

def to_local(polys, origin_xy):
    if not polys: return []
    ox, oy = origin_xy
    out=[]
    for p in polys:
        x,y = p.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-ox, np.array(y)-oy]))
    return out

def pack_in_domain(polys, diam_units, phi2D_target, max_particles, max_trials, seed):
    if not HAVE_SHAPELY:
        raise RuntimeError("Shapely not installed.")
    if not polys: return np.empty((0,2)), np.empty((0,)), 0.0

    dom_all = unary_union(polys)
    minx, miny, maxx, maxy = dom_all.bounds
    area_dom = dom_all.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)

    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/400.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def no_overlap(x, y, r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1, gx+2):
            for iy in range(gy-1, gy+2):
                for j in grid.get((ix, iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return True

    trials = 0
    for d in diam:
        r = d/2.0
        fit_dom = dom_all.buffer(-r)
        if getattr(fit_dom, "is_empty", True): continue
        fminx, fminy, fmaxx, fmaxy = fit_dom.bounds

        for _ in range(600):
            trials += 1
            if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
                break
            x = rng.uniform(fminx, fmaxx)
            y = rng.uniform(fminy, fmaxy)
            if not fit_dom.contains(Point(x, y)): continue
            if not no_overlap(x, y, r): continue

            idx = len(placed_xy)
            placed_xy.append((x, y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx, gy), []).append(idx)
            area_circ += math.pi * r * r

        if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi2D   = area_circ / area_dom if area_dom > 0 else 0.0
    return centers, radii, float(phi2D)

# =========================== Data & models ===========================
df_base, src_path = load_dataset(".")
models, meta      = train_green_density_models(df_base)

# =========================== Sidebar ===========================
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base.columns else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", "Silicon Carbide (SiC)")
    d50_um  = st.number_input("D50 (µm)", min_value=1.0, value=60.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_green = st.slider("Target green density (%TD)", 80, 98, 90, 1)
    guardrails_on = st.toggle("Guardrails", value=True)

    st.divider()
    st.markdown("Data / Models")
    st.write("Source:", (src_path or "in-memory demo"))
    st.write("Rows:", f"{len(df_base):,}")
    st.write("Models:", "trained" if models else "—")

# =========================== Balanced Top-5 ===========================
def recommend_balanced_top5(material, d50_um, layer_um, target_green, guardrails_on, models, df_source):
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = [float(x) for x in gr["binder_saturation_pct"]]
    spd_lo, spd_hi = [float(x) for x in gr["roller_speed_mm_s"]]

    def candidates(binder_type):
        Xs = np.linspace(sat_lo, sat_hi, 36)
        Ys = np.linspace(spd_lo, spd_hi, 28)
        g = pd.DataFrame([(b, v, d50_um) for v in Ys for b in Xs],
                         columns=["binder_saturation_pct","roller_speed_mm_s","d50_um"])
        g["material"] = material
        pred = predict_quantiles(models, g)
        g = g.join(pred[["td_q10","td_q50","td_q90"]])
        # score: closeness to target on q50 + mild penalty if q10 << target
        g["score"] = (g["td_q50"] - float(target_green)).abs() + 0.10*np.clip(float(target_green)-g["td_q10"], 0, None)
        g["binder_type_rec"] = binder_type
        return g.sort_values("score", ascending=True)

    gw = candidates("water_based")
    gs = candidates("solvent_based")

    pick_w = gw.head(3).copy()
    pick_s = gs.head(2).copy()
    if len(pick_w) < 3:
        need = 3 - len(pick_w)
        pick_w = pd.concat([pick_w, gw.iloc[len(pick_w):len(pick_w)+need]], ignore_index=True)
    if len(pick_s) < 2:
        need = 2 - len(pick_s)
        pick_s = pd.concat([pick_s, gs.iloc[len(pick_s):len(pick_s)+need]], ignore_index=True)

    out = pd.concat([pick_w, pick_s], ignore_index=True).reset_index(drop=True)
    out["layer_um"] = float(layer_um)
    out["id"] = [f"Opt-{i+1}" for i in range(len(out))]
    cols = ["id","binder_type_rec","binder_saturation_pct","roller_speed_mm_s","layer_um",
            "td_q10","td_q50","td_q90","d50_um","material"]
    return out[cols]

# =========================== Tabs ===========================
tab_pred, tab_heat, tab_sens, tab_pack, tab_form, tab_twin, tab_health = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)", "Data health"]
)

# ---------- Predict ----------
with tab_pred:
    st.subheader("Top-5 parameter sets (3 water-based + 2 solvent-based)")
    recs = recommend_balanced_top5(material=material, d50_um=float(d50_um), layer_um=float(layer_um),
                                   target_green=float(target_green), guardrails_on=guardrails_on,
                                   models=models, df_source=df_base)
    st.session_state["top5_recipes_df"] = recs.copy()
    st.dataframe(recs.rename(columns={
        "binder_type_rec":"binder_type",
        "binder_saturation_pct":"saturation_pct",
        "roller_speed_mm_s":"roller_speed",
        "td_q50":"pred_q50","td_q10":"pred_q10","td_q90":"pred_q90"
    }), use_container_width=True, hide_index=True)

# ---------- Heatmap ----------
@st.cache_data(show_spinner=False)
def _build_heatmap_table(material: str, d50_um: float, layer_um: float, binder_type: str,
                         guardrails_on: bool, models_key: str) -> pd.DataFrame:
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = gr["binder_saturation_pct"]; spd_lo, spd_hi = gr["roller_speed_mm_s"]
    Xs = np.linspace(float(sat_lo), float(sat_hi), 85)
    Ys = np.linspace(float(spd_lo), float(spd_hi), 70)
    g = pd.DataFrame([(b, v, d50_um) for v in Ys for b in Xs],
                     columns=["binder_saturation_pct","roller_speed_mm_s","d50_um"])
    g["material"] = material
    pr = predict_quantiles(models, g)
    return pd.DataFrame({
        "sat": pr["binder_saturation_pct"].astype(float),
        "spd": pr["roller_speed_mm_s"].astype(float),
        "q10": pr["td_q10"].astype(float),
        "q50": pr["td_q50"].astype(float),
        "q90": pr["td_q90"].astype(float),
    })

with tab_heat:
    st.subheader("Predicted Green Density (% Theoretical Density)")
    _trial_df = st.session_state.get("top5_recipes_df")
    default_binder = "water_based"
    if _trial_df is not None and not _trial_df.empty:
        default_binder = str(_trial_df.iloc[0]["binder_type_rec"])
    binder_for_map = st.radio("Binder for heatmap", ["water_based","solvent_based"],
                              index=0 if "water" in default_binder else 1, horizontal=True)

    dfZ = _build_heatmap_table(material, float(d50_um), float(layer_um), binder_for_map,
                               bool(guardrails_on), "models-v1")
    X = np.sort(dfZ["sat"].unique())
    Y = np.sort(dfZ["spd"].unique())
    Z = dfZ.pivot(index="spd", columns="sat", values="q50").values.astype(float)
    sigma = st.slider("Smoothing σ", 0.0, 2.0, 1.0, 0.1)
    if sigma > 0: Z = gaussian_filter(Z, sigma=sigma, mode="nearest")

    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    z0, z1 = max(40.0, zmin), min(100.0, zmax if zmax > zmin else zmin + 1.0)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=X, y=Y, zmin=z0, zmax=z1, zsmooth="best",
                             colorscale="Viridis", colorbar=dict(title="%TD (q50)", len=0.82, ticks="outside")))
    thr = float(target_green)
    fig.add_trace(go.Contour(z=Z, x=X, y=Y, contours=dict(start=thr,end=thr,size=1, coloring="lines"),
                             line=dict(color="#C21807", dash="dash", width=3.0), showscale=False, hoverinfo="skip"))

    cx, cy = float(np.median(X)), float(np.median(Y))
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                             marker=dict(symbol="circle-open-dot", size=14, line=dict(width=2, color="#1f2937")),
                             hoverinfo="skip"))
    fig.update_layout(title=f"Heatmap for {binder_for_map.replace('_',' ')} binder",
                      xaxis_title="Binder Saturation (%)", yaxis_title="Roller Speed (mm/s)",
                      margin=dict(l=10,r=20,t=40,b=35), height=480)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Saturation sensitivity ----------
with tab_sens:
    st.subheader("Saturation sensitivity at representative speed")
    # Use mid roller speed from guardrails
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 75)
    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material]*len(sat_axis),
    })
    pred = predict_quantiles(models, grid)
    q10 = pred["td_q10"].astype(float).values
    q50 = pred["td_q50"].astype(float).values
    q90 = pred["td_q90"].astype(float).values

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sat_axis, y=q90, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=sat_axis, y=q10, fill='tonexty', name="q10–q90", mode="lines",
                              line=dict(color="rgba(56,161,105,0.0)"), fillcolor="rgba(56,161,105,0.22)"))
    fig2.add_trace(go.Scatter(x=sat_axis, y=q50, name="q50 (median)", mode="lines",
                              line=dict(width=3, color="#2563eb")))
    fig2.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                       height=420, margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Qualitative packing ----------
with tab_pack:
    st.subheader("Qualitative packing (illustrative)")
    if not HAVE_SHAPELY:
        st.error("This view requires 'shapely'. Please add it to requirements.txt.")
    else:
        frame_mm = 1.8; px = 1400
        diam_um = sample_psd_um(6500, d50_um, None, None, seed=42)
        diam_mm = diam_um / 1000.0
        phi2D_target = float(np.clip(0.90 * (target_green/100.0), 0.40, 0.88))
        dom = box(0, 0, frame_mm, frame_mm)

        centers, radii, phi2D = pack_in_domain([dom], diam_mm, phi2D_target,
                                               max_particles=2600, max_trials=600_000, seed=1001)

        pore_mask = ~raster_mask(centers, radii, frame_mm, px)
        sat_frac = float(np.clip(target_green/100.0, 0.6, 0.98))
        vmask = voids_from_saturation(pore_mask, saturation=sat_frac)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
        ax.add_patch(Rectangle((0, 0), frame_mm, frame_mm, facecolor=binder_color("water_based"), edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (frame_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (frame_mm/vmask.shape[0])
            ax.scatter(xm, ym, s=0.25, c=VOID, alpha=0.95, linewidths=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0, frame_mm); ax.set_ylim(0, frame_mm)
        ax.set_xticks([]); ax.set_yticks([])
        draw_scale_bar(ax, frame_mm, length_um=500)
        ax.set_title(f"Illustrative microstructure • D50≈{d50_um:.0f} µm • Sat≈{int(sat_frac*100)}% • Layer≈{layer_um:.0f} µm",
                     fontsize=10, pad=10)
        st.pyplot(fig, clear_figure=True)

# ---------- Formulae ----------
with tab_form:
    st.subheader("Formulae (quick notes)")
    st.markdown("""
- Layer guardrail:  t ≤ 2.5·D50 (µm) to reduce streaking and binder flooding.
- Roller window:    scales weakly with D50; nominal 1.8–2.6 mm/s for ~50–120 µm powders.
- Saturation bands: water-based often ≤95%; solvent-based often >95%.
- φ₂D target:       0.75–0.85 corresponds to ~0.58–0.65 φ₃D for random loose/close pack pre-debinder.
    """)

# ---------- Digital Twin ----------
with tab_twin:
    st.subheader("Digital Twin — STL layer preview")
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Digital Twin needs 'trimesh' and 'shapely' (see requirements.txt).")
    else:
        left,right = st.columns([1.4,1])
        with left:
            stl_file = st.file_uploader("Upload STL", type=["stl"])
        with right:
            pack_full = st.checkbox("Pack full slice (auto FOV)", value=True)
            fov_mm = st.slider("Manual FOV (mm)", 0.5, 6.0, 1.2, 0.05, disabled=pack_full)
            cap = st.slider("Visual cap (particles)", 200, 4000, 1800, 50)
            phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
            phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))

        top5 = st.session_state.get("top5_recipes_df")
        if top5 is None or getattr(top5, "empty", True):
            st.info("Generate Top-5 in the Predict tab first to link a recipe.")
            binder = "water_based"; sat_pct = 85.0
        else:
            sel = top5.iloc[0]
            binder = str(sel["binder_type_rec"])
            sat_pct = float(sel["binder_saturation_pct"])

        if stl_file is not None:
            try:
                mesh = trimesh.load(io.BytesIO(stl_file.read()), file_type="stl", force="mesh", process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump(concatenate=True)

                minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
                thickness = float(layer_um) * 1e-3  # treat STL as mm
                n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
                st.markdown(f"Layers: {n_layers} • Z span: {maxz-minz:.3f} mm")
                layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
                z = minz + (layer_idx - 0.5) * thickness

                polys_world = slice_polys(mesh, z)

                if pack_full and polys_world:
                    dom = unary_union(polys_world)
                    xmin, ymin, xmax, ymax = dom.bounds
                    fov_x = xmax - xmin; fov_y = ymax - ymin
                    fov = max(fov_x, fov_y)
                    win = box(xmin, ymin, xmin+fov, ymin+fov)
                    clip = dom.intersection(win)
                    geoms = [clip] if isinstance(clip, Polygon) else [g for g in clip.geoms if g.area > 1e-9]
                    origin = (xmin, ymin)
                    polys_local = to_local(geoms, origin)
                    render_fov = fov
                else:
                    polys_clip, origin = crop_window(polys_world, float(fov_mm))
                    polys_local = to_local(polys_clip, origin)
                    render_fov = float(fov_mm)

                # Particle diameters from selected recipe’s D50 (fall back to sidebar D50)
                d50_r = float(top5.iloc[0]["d50_um"]) if top5 is not None and not top5.empty else float(d50_um)
                diam_um = sample_psd_um(9000, d50_r, None, None, seed=9991)
                diam_mm = diam_um / 1000.0

                centers, radii, phi2D = pack_in_domain(polys_local, diam_mm, phi2D_target,
                                                       max_particles=int(cap), max_trials=480_000, seed=20_000+layer_idx)

                # Render two panels
                col1, col2 = st.columns(2)
                with col1:
                    figA, axA = plt.subplots(figsize=(5.3,5.3), dpi=188)
                    axA.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor="white", edgecolor=BORDER, linewidth=1.2))
                    for (x,y), r in zip(centers, radii):
                        axA.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
                    axA.set_aspect('equal','box'); axA.set_xlim(0,render_fov); axA.set_ylim(0,render_fov); axA.set_xticks([]); axA.set_yticks([])
                    draw_scale_bar(axA, render_fov)
                    axA.set_title("Particles only", fontsize=10)
                    st.pyplot(figA, use_container_width=True)
                with col2:
                    px=900; pores=~raster_mask(centers, radii, render_fov, px)
                    vmask = voids_from_saturation(pores, saturation=float(sat_pct)/100.0)
                    figB, axB = plt.subplots(figsize=(5.3,5.3), dpi=188)
                    axB.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor=binder_color(binder), edgecolor=BORDER, linewidth=1.2))
                    ys, xs = np.where(vmask)
                    if len(xs):
                        xm = xs * (render_fov/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (render_fov/vmask.shape[0])
                        axB.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidth=0)
                    for (x,y), r in zip(centers, radii):
                        axB.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
                    axB.set_aspect('equal','box'); axB.set_xlim(0,render_fov); axB.set_ylim(0,render_fov); axB.set_xticks([]); axB.set_yticks([])
                    draw_scale_bar(axB, render_fov)
                    axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
                    st.pyplot(figB, use_container_width=True)

                st.caption(f"FOV={render_fov:.2f} mm • φ₂D(target)≈{phi2D_target:.2f} • φ₂D(achieved)≈{min(phi2D,1.0):.2f} • Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

            except Exception as ex:
                st.error(f"Failed to process STL: {ex}")
        else:
            st.info("Upload an STL to enable the Digital Twin preview.")

# ---------- Data health ----------
with tab_health:
    st.subheader("Training coverage & ≥90%TD evidence near this D50")
    d = df_base.copy()
    if "material" in d.columns:
        d = d[d["material"].astype(str)==str(material)]
    if "d50_um" in d.columns:
        lo, hi = 0.8*float(d50_um), 1.2*float(d50_um)
        d = d[(d["d50_um"]>=lo) & (d["d50_um"]<=hi)]
    c1, c2 = st.columns([1,2])
    with c1:
        if "green_pct_td" in d.columns:
            st.metric("Rows in ±20% D50 window", len(d))
            st.metric("Seen ≥90%TD cases", int((d["green_pct_td"]>=90).sum()))
            if len(d): st.metric("Best %TD", f"{float(d['green_pct_td'].max()):.1f}")
        else:
            st.info("No green %TD column found after normalization.")
    with c2:
        if not d.empty and "d50_um" in d.columns and "green_pct_td" in d.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["d50_um"], y=d["green_pct_td"], mode="markers", name="train pts"))
            fig.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
            fig.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD",
                              height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training points in this window.")
