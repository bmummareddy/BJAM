from __future__ import annotations
import io, math, random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.draw import disk as sk_disk
import trimesh

import shared

st.set_page_config(page_title="BJAM Prediction Studio", layout="wide")

APP_URL = "https://bjampredictions.streamlit.app/"

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("BJAM Prediction Studio")
st.sidebar.caption("AI-assisted recipes for Binder Jet AM (metals & ceramics) + Digital Twin preview.")
st.sidebar.markdown(f"[Open public app]({APP_URL})")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Predict",
    "Heatmap",
    "Saturation",
    "Digital Twin",
    "About"
])

# -------------------------
# Shared inputs
# -------------------------
with st.sidebar:
    st.subheader("Input")
    material = st.text_input("Material (e.g., Alumina, 316L, SiC)", value="Alumina")
    size_um = st.number_input("Particle size D50 (µm)", min_value=2.0, max_value=120.0, value=35.0, step=1.0)
    want_5 = st.checkbox("Return five recipe options (3 aqueous, 2 solvent)", value=True)

df = shared.load_bjam()

# -------------------------
# Predict tab
# -------------------------
with tabs[0]:
    st.header("Parameter suggestions")
    preds = shared.make_predictions(material, size_um, want_5_sets=want_5)

    st.write("These five options maintain the rule: at least three aqueous systems and two solvent systems.")
    rows = []
    for i, p in enumerate(preds, 1):
        rows.append({
            "Trial": i,
            "Binder": p.binder_type,
            "Saturation (%)": p.binder_saturation_pct,
            "Roller speed (mm/s)": p.roller_speed_mm_s,
            "Post-sinter (°C)": p.post_sinter_C,
            "Est. green density (%)": p.est_green_density_pct,
            "Notes": p.rationale
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.success("Goal: theoretical packing ≥ 90% after sintering. Use the Digital Twin tab to preview layer packing qualitatively.")

# -------------------------
# Heatmap tab (qualitative scoring of parameter space)
# -------------------------
def score_point(sat, speed, temp):
    # simple monotone scoring: closer to guardrail band → higher
    # sat ideal 70–90; speed 100–250; temp material rule
    sat_score = 1.0 - min(abs(sat - 80)/25.0, 1.0)
    spd_score = 1.0 - min(abs(speed - 200)/150.0, 1.0)
    tmp_score = 1.0 - min(abs(temp - shared.post_sinter_rule(material))/400.0, 1.0)
    return max(0.0, (0.45*sat_score + 0.35*spd_score + 0.20*tmp_score))

with tabs[1]:
    st.header("Qualitative feasibility heatmap")
    sat_vals = np.linspace(60, 92, 33)
    spd_vals = np.linspace(60, 350, 40)
    Z = np.zeros((len(spd_vals), len(sat_vals)))
    Tref = shared.post_sinter_rule(material)

    for i, spd in enumerate(spd_vals):
        for j, sat in enumerate(sat_vals):
            Z[i, j] = score_point(sat, spd, Tref)

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=sat_vals,
        y=spd_vals,
        colorbar=dict(title="Score"),
        zmin=0, zmax=1
    ))
    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        margin=dict(l=60,r=20,t=30,b=60),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Darker → less feasible; brighter → more feasible for this material at its typical sinter temperature.")

# -------------------------
# Saturation tab (per binder family)
# -------------------------
with tabs[2]:
    st.header("Binder saturation guidance by binder family")
    x = np.linspace(5, 80, 100)
    water_line = [shared.binder_sat_rule(s, True) for s in x]
    solv_line = [shared.binder_sat_rule(s, False) for s in x]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=water_line, mode="lines", name="Aqueous binders"))
    fig2.add_trace(go.Scatter(x=x, y=solv_line, mode="lines", name="Solvent binders"))

    fig2.add_vline(x=size_um, line_dash="dash", annotation_text=f"D50={size_um:.1f} µm", annotation_position="top right")
    fig2.update_layout(
        xaxis_title="Particle size D50 (µm)",
        yaxis_title="Suggested saturation (%)",
        height=450,
        margin=dict(l=60,r=20,t=30,b=60)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Curves are heuristic guardrails tuned for quick success; refine locally with your machine & powder.")

# -------------------------
# Digital Twin tab
# -------------------------
def poisson_disk_pack(width_px:int, height_px:int, r_px:int, target_phi:float, max_tries:int=60000):
    """
    Very simple rejection sampler to place non-overlapping disks to approximate a target packing fraction.
    Returns list of (cy, cx, r).
    """
    rng = np.random.default_rng(42)
    circles = []
    area_target = target_phi * width_px * height_px
    area = 0.0
    tries = 0
    while area < area_target and tries < max_tries:
        tries += 1
        cx = rng.integers(r_px, width_px - r_px)
        cy = rng.integers(r_px, height_px - r_px)
        ok = True
        for (yy, xx, rr) in circles:
            if (cx - xx)**2 + (cy - yy)**2 < (r_px + rr)**2:
                ok = False
                break
        if ok:
            circles.append((cy, cx, r_px))
            area += math.pi * (r_px**2)
    return circles

def draw_layer(width_px:int, height_px:int, circles:List[Tuple[int,int,int]]) -> Image.Image:
    img = Image.new("RGB", (width_px, height_px), (255,255,255))
    drw = ImageDraw.Draw(img)
    # pastel palette that reads cleanly
    for (cy, cx, rr) in circles:
        drw.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=(60,60,60), width=1, fill=(200, 230, 255))
    return img

with tabs[3]:
    st.header("Digital Twin: STL + qualitative layer packing")
    colA, colB = st.columns([1,1])

    # --- STL viewer
    with colA:
        st.subheader("Upload STL")
        st.caption("Viewer displays overall geometry; packing slices are a qualitative 2D layer proxy (fast).")
        up = st.file_uploader("Choose an STL file", type=["stl"])
        stl_mesh = None
        if up is not None:
            try:
                data = up.read()
                stl_mesh = trimesh.load(io.BytesIO(data), file_type='stl', force='mesh')
                if not isinstance(stl_mesh, trimesh.Trimesh) and hasattr(stl_mesh, 'dump'):
                    stl_mesh = stl_mesh.dump().sum()
            except Exception as e:
                st.error(f"Failed to parse STL: {e}")

        if stl_mesh is not None and isinstance(stl_mesh, trimesh.Trimesh):
            verts = stl_mesh.vertices
            faces = stl_mesh.faces
            mesh = go.Mesh3d(
                x=verts[:,0], y=verts[:,1], z=verts[:,2],
                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                opacity=0.6
            )
            figm = go.Figure(data=[mesh])
            figm.update_layout(scene_aspectmode='data', height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("Upload an STL to preview the geometry.")

    # --- Qualitative packing
    with colB:
        st.subheader("Layer packing preview")
        st.caption("Pick one of the five trials, then scrub through layers to visualize qualitative packing (2D slice).")

        pred_options = shared.make_predictions(material, size_um, want_5_sets=True)
        labels = [f"Trial {i+1}: {p.binder_type} | Sat {p.binder_saturation_pct:.1f}% | v {p.roller_speed_mm_s:.0f} mm/s | {p.post_sinter_C}°C"
                  for i,p in enumerate(pred_options)]
        choice = st.selectbox("Choose trial", labels, index=0)
        idx = labels.index(choice)
        chosen = pred_options[idx]

        # particle radius in pixels relative to D50; map 10–60 µm to 3–10 px for visibility
        r_px = int(np.interp(size_um, [10, 60], [3, 10]).item())
        r_px = max(2, r_px)

        # green density heuristic → target 2D packing fraction (cap near RCP ~0.82 in 2D)
        phi_target = float(np.clip(chosen.est_green_density_pct/100.0, 0.45, 0.82))

        width_px, height_px = 480, 360
        num_layers = 30
        layer = st.slider("Layer index", 1, num_layers, 1, key="layer_slider")

        # Regenerate circles deterministically per layer to visualize variation
        rng = np.random.default_rng(seed=layer * 1337)
        # small jitter on r to mimic polydispersity
        r_effective = int(max(2, round(r_px * float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.3)))))
        circles = poisson_disk_pack(width_px, height_px, r_effective, phi_target, max_tries=40000)
        img = draw_layer(width_px, height_px, circles)
        st.image(img, caption=f"Layer {layer} | ~{len(circles)} particles | φ≈{phi_target:.2f}")

        st.caption("This 2D slice is qualitative and sized visually; use it to compare trials (binder family & saturation) and D50 trends.")

# -------------------------
# About tab
# -------------------------
with tabs[4]:
    st.header("About")
    st.markdown("""
This app predicts BJAM parameters and previews qualitative particle packing layers.
It is designed to return five recipes by default — three aqueous binder systems and two solvent systems — while
keeping a physics-informed flavor (packing targets, d50-dependent roller speed, and binder saturation guardrails).

If you use this in your article or presentation, please include a short note:
“Figure and recipe card generated with assistance from ChatGPT.”

Public app URL: {}
""".format(APP_URL))
