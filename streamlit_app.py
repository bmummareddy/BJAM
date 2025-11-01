import io
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

from shared import (
    load_training, generate_five_recipes, make_parameter_grid,
    saturation_curve, synth_layer_particles, stl_to_binary_slices, clamp
)

st.set_page_config(page_title="Binder-Jet Parameter Assistant", layout="wide")

DATA_PATH = "/mnt/data/BJAM_All_Deep_Fill_v9.csv"
APP_URL = "https://bjampredictions.streamlit.app/"

# ---------------------------
# Utility drawing helpers
# ---------------------------

def draw_layer_image(pts, radii, size=(800, 420)):
    W, H = size
    img = Image.new("RGB", (W, H), (250, 252, 255))
    drw = ImageDraw.Draw(img, "RGBA")

    # scale input coords to canvas
    if len(pts) == 0:
        return img
    x = pts[:,0]; y = pts[:,1]
    minx, maxx = float(np.min(x)), float(np.max(x))
    miny, maxy = float(np.min(y)), float(np.max(y))
    sx = (W-30) / max(1e-6, (maxx - minx))
    sy = (H-30) / max(1e-6, (maxy - miny))

    # subtle color palette cycling by radius rank
    order = np.argsort(radii)
    for idx in order:
        cx = 15 + (pts[idx,0] - minx) * sx
        cy = 15 + (pts[idx,1] - miny) * sy
        r  = radii[idx] * 0.95 *  (sx+sy)/2.0 / max(sx,sy)

        # gradient-ish ring (simple two-tone)
        c1 = (60, 120, 200, 200)
        c2 = (90, 160, 230, 140)
        bbox = [cx-r, cy-r, cx+r, cy+r]
        drw.ellipse(bbox, fill=c1, outline=(20,70,140,220), width=2)
        inner = [cx-0.7*r, cy-0.7*r, cx+0.7*r, cy+0.7*r]
        drw.ellipse(inner, fill=c2)

    # light blur for cohesion (tiny)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img

def fig_heatmap(S, V, Z):
    fig = go.Figure(
        data=go.Heatmap(
            x=S[0, :], y=V[:, 0], z=Z,
            coloraxis="coloraxis",
            hovertemplate="Sat %{x:.0f}%<br>Roller %{y:.0f} mm/s<br>Fit %{z:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1),
        xaxis_title="Binder Saturation (%)",
        yaxis_title="Roller Traverse (mm/s)",
        margin=dict(l=40, r=10, t=10, b=40),
        height=420
    )
    return fig

def fig_saturation(x, y, d50):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Recommended band"))
    fig.add_vline(x=float(d50), line_dash="dash", annotation_text=f"D50={d50:.0f} µm", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title="Recommended saturation (%)",
        height=380, margin=dict(l=40, r=10, t=10, b=40)
    )
    return fig

def fig_slices_gallery(bin_slice):
    # convert 2D bool/0-1 to image
    arr = (bin_slice.astype(np.uint8) * 255)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    img = img.resize((600, 600), Image.NEAREST)
    fig = px.imshow(np.asarray(img))
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=10,b=10), height=620)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return fig

# ---------------------------
# Sidebar (global)
# ---------------------------

st.sidebar.header("BJAM Assistant")
st.sidebar.markdown(
    f"App link: {APP_URL}  \n"
    "Predict parameters, visualize process windows, and explore a simple digital twin."
)

# ---------------------------
# Load data (soft-fail)
# ---------------------------

@st.cache_data(show_spinner=False)
def _load_df():
    try:
        return load_training(DATA_PATH)
    except Exception:
        # allow app to run without local CSV
        cols = ["Material","Powder","D50_um","BinderType","BinderSat_%","Roller_mm_s","Sinter_T_C","AchievedRelDensity_%"]
        return pd.DataFrame(columns=cols)

df = _load_df()

# ---------------------------
# Tabs
# ---------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Heatmap", "Saturation", "Digital Twin"])

with tab1:
    st.subheader("Parameter predictions")
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        material = st.selectbox("Powder / Material",
                                sorted(list({m for m in df["Material"].dropna().unique()}) + ["Alumina","SiC","Zirconia","316L","17-4PH"]),
                                index=0)
    with colB:
        d50 = st.number_input("Particle size D50 (µm)", 5.0, 120.0, 35.0, 1.0)
    with colC:
        target = st.slider("Target relative density", 0.80, 0.99, 0.90, 0.01)
    with colD:
        st.write("") ; st.write("")
        if st.button("Generate 5 trials"):
            st.session_state["_trials"] = generate_five_recipes(material, d50, target)

    trials = st.session_state.get("_trials", generate_five_recipes(material, d50, target))
    table = pd.DataFrame(trials)[["ID","Family","BinderType","BinderSat_%","Roller_mm_s","Sinter_T_C","Score"]]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("Pick a trial to preview qualitative packing for a single layer:")
    tid = st.selectbox("Trial", [t["ID"] for t in trials])
    cur = next(t for t in trials if t["ID"] == tid)

    # Generate a representative qualitative layer
    seed = abs(hash(tid)) % (2**32-1)
    pts, radii = synth_layer_particles(px=540, py=300, d50_um=d50, layer_thickness_um=50, seed=seed)
    img = draw_layer_image(pts, radii, size=(900, 480))
    st.image(img, caption=f"Qualitative packing slice — {tid} (≈ D50: {d50:.0f} µm)", use_container_width=True)

    with st.expander("Trial details"):
        st.json(cur)

with tab2:
    st.subheader("Process window heatmap (fitness proxy)")
    col1, col2 = st.columns([1,1])
    with col1:
        material_hm = st.selectbox("Material (heatmap)", ["Alumina","SiC","Zirconia","316L","17-4PH"], index=0, key="hm_m")
    with col2:
        d50_hm = st.number_input("Particle size D50 (µm) (heatmap)", 5.0, 120.0, 35.0, 1.0, key="hm_d")
    S, V, Z = make_parameter_grid(material_hm, d50_hm, n=50)
    fig = fig_heatmap(S, V, Z)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Binder saturation recommendation vs D50")
    col1, col2 = st.columns([1,1])
    with col1:
        material_sat = st.selectbox("Material (saturation)", ["Alumina","SiC","Zirconia","316L","17-4PH"], index=0, key="sat_m")
    with col2:
        family = st.radio("Binder family", ["Water","Solvent"], horizontal=True)
    x, y = saturation_curve(material_sat, d50_um=35.0, family=family)
    fig = fig_saturation(x, y, d50=35.0)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Curve centers on a nominal D50 and shifts in the Predictions tab via the five concrete trials.")

with tab4:
    st.subheader("Digital Twin: STL → layer slices & packing preview")
    st.markdown("Upload an STL to voxelize and browse layers. The packing preview uses your currently selected trial’s D50 to render a qualitative particle slice.")

    up = st.file_uploader("Upload STL", type=["stl"])
    colZ1, colZ2 = st.columns([1,1])
    with colZ1:
        voxel_pitch = st.slider("Voxel pitch (mm)", 0.2, 1.5, 0.4, 0.1)
    with colZ2:
        max_dim = st.slider("Max voxel dimension (voxels)", 100, 300, 220, 10)

    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("STL voxels → layer slider")
        if up is not None:
            try:
                slices = stl_to_binary_slices(up.read(), voxel_pitch=float(voxel_pitch), max_dim_vox=int(max_dim))
                if len(slices) == 0:
                    st.warning("Voxelization returned 0 slices. Try larger voxel pitch or a manifold STL.")
                else:
                    z = st.slider("Layer index", 0, len(slices)-1, min(10, len(slices)-1))
                    st.plotly_chart(fig_slices_gallery(slices[z]), use_container_width=True)
                    st.caption(f"Slice {z+1}/{len(slices)}")
            except Exception as e:
                st.error(f"STL processing error: {e}")
        else:
            st.info("Upload an STL to view slices.")

    with c2:
        st.markdown("Qualitative particle layer (matches current trial D50)")
        # Consume current trial from tab1 if set, else regenerate
        trials = st.session_state.get("_trials", None)
        if not trials:
            trials = generate_five_recipes("Alumina", 35.0, 0.90)
        # pick currently selected or fallback
        chosen_id = st.selectbox("Pick one of the five trials", [t["ID"] for t in trials], index=0, key="tw_id")
        chosen = next(t for t in trials if t["ID"] == chosen_id)
        est_d50 = st.number_input("Override D50 for packing (µm)", 5.0, 120.0, 35.0, 1.0, key="tw_d50")
        pts, radii = synth_layer_particles(px=620, py=360, d50_um=est_d50, layer_thickness_um=50, seed=abs(hash(chosen_id))%(2**32-1))
        img2 = draw_layer_image(pts, radii, size=(900, 520))
        st.image(img2, caption=f"Qualitative packing slice — {chosen_id} (D50≈{est_d50:.0f} µm)", use_container_width=True)

st.markdown("---")
st.caption("Generated with the help of ChatGPT (BJAM Assistant).")
