# streamlit_app.py — BJAM Binder-Jet AM Recommender (bright UI, multi-material + packing layer)
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
    suggest_binder_family,   # to show/override binder family
)

# --------------------------- Page setup ---------------------------------------
st.set_page_config(page_title="BJAM Predictions", page_icon="🟨", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #FFFDF7 0%, #FFF8EC 40%, #FFF4E2 100%); }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stMetric { background: rgba(255,255,255,.6); border-radius: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------- Data & models ------------------------------------
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# --------------------------- Sidebar -----------------------------------------
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data source: {Path(src).name} · rows={len(df_base):,}")
        st.download_button("Download source dataset (CSV)",
                           data=df_base.to_csv(index=False).encode("utf-8"),
                           file_name=Path(src).name, mime="text/csv")
    else:
        st.warning("No dataset found. App will use physics priors only (few-shot disabled).")

    st.divider()
    guardrails_on = st.toggle("Guardrails", value=True,
                              help="ON: stable windows (binder 60–110%, speed ~1.2–3.5 mm/s, layer ≈3–5×D50). OFF: wider exploration.")
    target_green = st.slider("Target green %TD", 80, 98, 90, 1)

    st.divider()
    st.subheader("(Optional) Upload measured runs")
    up = st.file_uploader("CSV with columns like: d50_um, layer_um, roller_mm_s, binder_pct, green_td_measured",
                          type=["csv"], accept_multiple_files=False, help="Enables residuals/outliers plot.")
    df_user = None
    if up is not None:
        try:
            df_user = pd.read_csv(up)
            st.caption(f"Uploaded rows: {len(df_user):,}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# --------------------------- Header ------------------------------------------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · custom materials supported · guardrails toggle")

m1, m2, m3 = st.columns(3)
m1.metric("Rows in dataset", f"{len(df_base):,}")
m2.metric("Materials", f"{df_base['material'].nunique() if 'material' in df_base else 0:,}")
m3.metric("Quantile models", "Trained" if models else "Physics-only")

with st.expander("Preview source data", expanded=False):
    if len(df_base): st.dataframe(df_base.head(25), use_container_width=True)
    else: st.info("No rows to preview.")

st.divider()

# --------------------------- Inputs ------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Inputs")

    # Choose dataset or custom material
    mode = st.radio("Material source", ["From dataset", "Custom"], horizontal=True)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []

    if mode == "From dataset" and materials:
        material = st.selectbox("Material (from dataset)", options=materials, index=0)
        # Try to pick a sensible D50 default from dataset
        d50_default = 30.0
        if "d50_um" in df_base.columns:
            msk = (df_base["material"].astype(str) == material)
            if msk.any(): d50_default = float(df_base.loc[msk, "d50_um"].dropna().median() or 30.0)
        material_class = df_base.loc[df_base["material"].astype(str)==material, "material_class"].dropna().astype(str).iloc[0] \
                         if {"material","material_class"}.issubset(df_base.columns) and \
                            (df_base["material"].astype(str)==material).any() else "metal"
    else:
        material = st.text_input("Material (custom)", value="Al2O3")
        material_class = st.selectbox("Material class", options=["metal","oxide","carbide","other"], index=1)
        d50_default = 30.0

    d50_um = st.number_input("D50 (µm)", min_value=1.0, max_value=150.0, value=float(d50_default), step=1.0,
                             help="Layer guidance follows ≈3–5×D50.")
    pri = physics_priors(d50_um, binder_type_guess=None)
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    t_lo, t_hi = gr["layer_thickness_um"]
    layer_um = st.slider("Layer thickness (µm)", float(round(t_lo)), float(round(t_hi)),
                         float(round(pri["layer_thickness_um"])), 1.0)

    # Binder family (auto + override)
    auto_binder = suggest_binder_family(material, material_class)
    binder_choice = st.selectbox("Binder family",
                                 options=[f"auto ({auto_binder})","solvent_based","water_based"],
                                 help="Auto uses material class: water-based for oxide/carbide; solvent-based otherwise.")
    binder_family = auto_binder if binder_choice.startswith("auto") else binder_choice

    c1,c2,c3 = st.columns(3)
    c1.metric("Prior binder %", f"{pri['binder_saturation_pct']:.0f}%")
    c2.metric("Prior speed", f"{pri['roller_speed_mm_s']:.1f} mm/s")
    c3.metric("Layer/D50", f"{layer_um/d50_um:.2f}×")

with right:
    st.subheader("Recommend parameters")
    top_k = st.slider("Number of recommendations", 3, 8, 5, 1)
    if st.button("Recommend", type="primary", use_container_width=True):
        recs = copilot(material=material, d50_um=float(d50_um), df_source=df_base, models=models,
                       guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k))
        # Reflect binder family choice in table
        recs["binder_type"] = binder_family
        st.dataframe(recs, use_container_width=True)
        st.caption("Ranking prefers **q10 ≥ target**, then **q50**; mild penalty for extremes.")
    else:
        st.info("Click **Recommend** to generate top-k parameter sets.")

st.divider()

# --------------------------- Visuals ------------------------------------------
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing layer (circles)",   # <-- NEW
    "Process window",
    "Pareto frontier",
    "Importance (local)",
    "Residuals / Outliers",
    "Formulae",
])

# Helper: build scoring grid with current context
def _scoring_grid(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=55, ny=45):
    sats = np.linspace(float(b_lo), float(b_hi), nx)
    spds = np.linspace(float(s_lo), float(s_hi), ny)
    grid = pd.DataFrame([(b,v,layer_um,d50_um,material) for b in sats for v in spds],
                        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    grid["material_class"] = material_class
    grid["binder_type_rec"] = binder_family
    return grid, sats, spds

# Heatmap
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD")
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    grid, Xs, Ys = _scoring_grid(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family)
    scored = predict_quantiles(models, grid)
    Z = scored.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=list(Xs), y=list(Ys), z=Z, colorscale="Turbo", colorbar=dict(title="%TD")))
    fig.add_trace(go.Contour(x=list(Xs), y=list(Ys), z=Z, contours=dict(start=90, end=90, size=1, coloring="none"),
                             line=dict(width=3), showscale=False, name="90% TD"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Roller speed (mm/s)",
                      height=520, margin=dict(l=10,r=10,t=40,b=10),
                      title=f"Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · Material={material} ({material_class}) · Source={Path(src).name if src else '—'}")
    st.plotly_chart(fig, use_container_width=True)

# Saturation sensitivity
with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90)")
    sats = np.linspace(float(b_lo), float(b_hi), 61)
    curve_df = pd.DataFrame({"binder_saturation_pct":sats, "roller_speed_mm_s":1.6,
                             "layer_thickness_um":float(layer_um), "d50_um":float(d50_um),
                             "material":material, "material_class":material_class, "binder_type_rec":binder_family})
    cs = predict_quantiles(models, curve_df)
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    ax2.plot(cs["binder_saturation_pct"], cs["td_q50"], label="q50")
    ax2.fill_between(cs["binder_saturation_pct"], cs["td_q10"], cs["td_q90"], alpha=0.2, label="q10–q90")
    ax2.axhline(target_green, linestyle="--", linewidth=1, label=f"Target {target_green}%")
    ax2.set_xlabel("Binder saturation (%)"); ax2.set_ylabel("Predicted green %TD")
    ax2.set_title(f"Speed=1.6 mm/s · Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# ------------- NEW: Packing layer (circles) -----------------------------------
with tabs[2]:
    st.subheader("Packing layer (circles)", help="Illustrative 2D slice of one powder layer. Circles ≈ particles (drawn around D50).")

    # Controls for the packing visualization
    cA, cB, cC, cD = st.columns(4)
    width_in_D50 = cA.slider("Width (× D50)", min_value=8, max_value=40, value=18, step=1,
                             help="Horizontal span of the layer in units of D50.")
    cv_pct = cB.slider("Polydispersity (CV %)", min_value=0, max_value=60, value=20, step=5,
                       help="Coefficient of variation of particle diameter (lognormal). 0% is monodisperse.")
    max_particles = cC.slider("Max particles", min_value=100, max_value=600, value=300, step=50)
    seed = cD.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)

    # Geometry in "D50 units": D50 ≡ 1 diameter ⇒ base radius = 0.5
    H = float(layer_um) / float(d50_um)          # layer thickness in D50 units
    W = float(width_in_D50)
    rng = np.random.default_rng(int(seed))

    # Build a particle size distribution (lognormal with chosen CV, median=1 D50)
    cv = cv_pct / 100.0
    if cv <= 0.0:
        diam = np.ones(max_particles)
    else:
        # For lognormal: CV^2 = exp(sigma^2) - 1  =>  sigma = sqrt(log(1+CV^2))
        sigma = float(np.sqrt(np.log(1.0 + cv**2)))
        mu = 0.0  # median = exp(mu) = 1.0 (D50 units)
        diam = rng.lognormal(mean=mu, sigma=sigma, size=max_particles)
        diam = np.clip(diam, 0.4, 1.8)  # avoid extreme dots

    radii = 0.5 * diam
    radii.sort()
    radii = radii[::-1]  # place larger first

    # Random Sequential Addition (RSA) to avoid overlaps
    pts = []
    rs = []
    attempts = 0
    max_attempts = 20000

    def can_place(x, y, r):
        if x - r < 0 or x + r > W or y - r < 0 or y + r > H:
            return False
        for (px, py, pr) in pts:
            dx = x - px; dy = y - py
            if dx*dx + dy*dy < (r + pr)**2:
                return False
        return True

    for r in radii:
        placed = False
        for _ in range(200):  # limited tries per particle
            x = rng.uniform(r, W - r)
            y = rng.uniform(r, H - r)
            if can_place(x, y, r):
                pts.append((x, y, r)); rs.append(r)
                placed = True
                break
        attempts += 1
        if attempts > max_attempts:
            break

    # Compute 2D areal packing fraction (illustrative)
    area_circles = float(np.pi * np.sum(np.square(rs)))
    phi_area = area_circles / (W * H) if W * H > 0 else 0.0

    # Draw
    figP, axP = plt.subplots(figsize=(10, 10 * (H / max(W, 1e-6)) * 0.35), dpi=150)  # size scales with aspect
    axP.set_aspect("equal", "box")
    # Layer rectangle
    axP.add_patch(plt.Rectangle((0, 0), W, H, fill=False, linewidth=1.2))
    # Circles
    for (x, y, r) in pts:
        axP.add_patch(plt.Circle((x, y), r, alpha=0.75))
    axP.set_xlim(0, W); axP.set_ylim(0, H)
    axP.set_xticks([]); axP.set_yticks([])
    axP.set_title(f"Layer ~ {layer_um:.0f} µm (≈ {H:.2f} × D50); width = {W:.0f} × D50\n"
                  f"Particles placed: {len(pts)}  ·  Areal packing ≈ {phi_area*100:.1f}%")
    st.pyplot(figP, clear_figure=True)
    st.caption("Note: This is an illustrative 2D slice (random sequential packing). Areal packing is not equal to 3D green density, "
               "but helps visualize how layer thickness vs D50 and polydispersity affect local packing.")

# Process window (layer vs 3–5×D50)
with tabs[3]:
    st.subheader("Process window — Layer vs D50")
    fig3 = go.Figure()
    fig3.add_hrect(y0=3*d50_um, y1=5*d50_um, fillcolor="lightgray", opacity=0.3, line_width=0,
                   annotation_text="Stable band ≈ 3–5×D50", annotation_position="top left")
    fig3.add_hline(y=layer_um, line_dash="dash", annotation_text=f"Your layer = {layer_um:.0f} µm")
    fig3.update_layout(yaxis_title="Layer thickness (µm)", xaxis_title="(reference) D50 multiplier band",
                       xaxis=dict(showticklabels=False), height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig3, use_container_width=True)

# Pareto frontier (min binder %, max %TD at fixed layer & D50)
with tabs[4]:
    st.subheader("Pareto frontier — Binder vs green %TD (fixed layer, fixed D50)")
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    grid_p, Xs_p, Ys_p = _scoring_grid(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=80, ny=1)
    sc_p = predict_quantiles(models, grid_p)
    sc_p = sc_p[["binder_saturation_pct","td_q50"]].dropna().sort_values("binder_saturation_pct")
    # Non-dominated set
    pts_pf = sc_p.values
    pareto_idx = []
    best = -1
    for i,(b,td) in enumerate(pts_pf[::-1]):
        if td > best: pareto_idx.append(len(pts_pf)-1-i); best = td
    pareto_idx = sorted(pareto_idx)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=sc_p["binder_saturation_pct"], y=sc_p["td_q50"], mode="markers", name="Candidates"))
    fig4.add_trace(go.Scatter(x=sc_p.iloc[pareto_idx]["binder_saturation_pct"], y=sc_p.iloc[pareto_idx]["td_q50"],
                              mode="lines+markers", name="Pareto frontier"))
    fig4.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD (q50)",
                       height=480, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# Importance (local, around current point) — requires trained q50 model
with tabs[5]:
    st.subheader("Local importance (permutation) — q50 model")
    if models and "q50" in models:
        # Local neighborhood around current settings
        b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
        loc = pd.DataFrame({
            "binder_saturation_pct": np.linspace(max(b_lo, 0.8*80), min(b_hi, 1.2*80), 32),
            "roller_speed_mm_s": np.linspace(max(s_lo, 1.2), min(s_hi, 3.0), 32),
        }).sample(64, replace=True, random_state=0).reset_index(drop=True)
        loc["layer_thickness_um"] = float(layer_um)
        loc["d50_um"] = float(d50_um)
        loc["material"] = material
        loc["material_class"] = material_class
        loc["binder_type_rec"] = binder_family

        model = models["q50"]
        feat_cols = ["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um"]
        imp = permutation_importance(model, loc[feat_cols], model.predict(loc), n_repeats=10, random_state=0)
        order = np.argsort(imp.importances_mean)[::-1]
        fig5, ax5 = plt.subplots(figsize=(6.8, 4.4), dpi=150)
        ax5.bar([feat_cols[i] for i in order], imp.importances_mean[order], yerr=imp.importances_std[order])
        ax5.set_ylabel("Importance (Δpred when shuffled)")
        ax5.set_title("Permutation importance (q50)")
        st.pyplot(fig5, clear_figure=True)
    else:
        st.info("Quantile models not trained — importance is unavailable (using physics proxy).")

# Residuals / Outliers (requires user CSV with measured green_td_measured)
with tabs[6]:
    st.subheader("Residuals / Outliers vs measured green %TD")
    if df_user is not None and len(df_user):
        ren = {"binder_pct":"binder_saturation_pct","layer_um":"layer_thickness_um","roller_mm_s":"roller_speed_mm_s",
               "d50":"d50_um","material_name":"material","class":"material_class"}
        dfx = df_user.rename(columns=ren).copy()
        required = ["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um"]
        for c in required:
            if c not in dfx.columns: dfx[c] = np.nan
        dfx["material"] = dfx.get("material", material)
        dfx["material_class"] = dfx.get("material_class", material_class)
        dfx["binder_type_rec"] = binder_family

        scored = predict_quantiles(models, dfx[["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um",
                                                "material","material_class","binder_type_rec"]])
        if "green_td_measured" in dfx.columns:
            y = pd.to_numeric(dfx["green_td_measured"], errors="coerce")
            yhat = scored["td_q50"]
            resid = y - yhat
            fig6, ax6 = plt.subplots(figsize=(8,4.6), dpi=150)
            ax6.scatter(yhat, y, alpha=0.7)
            ax6.axline((55,55),(98,98), linestyle="--", linewidth=1, color="black")
            ax6.set_xlabel("Predicted green %TD (q50)"); ax6.set_ylabel("Measured green %TD")
            ax6.set_title("Measured vs Predicted (q50)")
            st.pyplot(fig6, clear_figure=True)

            fig7, ax7 = plt.subplots(figsize=(8,4.2), dpi=150)
            ax7.axhline(0, color="black", linewidth=1)
            ax7.scatter(range(len(resid)), resid, alpha=0.7)
            ax7.set_ylabel("Residual (measured - predicted)")
            ax7.set_xlabel("Sample")
            ax7.set_title("Residuals")
            st.pyplot(fig7, clear_figure=True)
        else:
            st.info("Your CSV needs a 'green_td_measured' column to plot residuals.")
    else:
        st.info("Upload your measured CSV in the sidebar to enable residuals.")

# Formulae
with tabs[7]:
    st.subheader("Formulae (symbols)")
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    st.caption("Few-shot model refines these physics-guided priors using your dataset.")

# Footer diagnostics
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note":"No trained models (physics-only)."})
