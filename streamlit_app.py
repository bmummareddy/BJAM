# streamlit_app.py
# BJAM - Binder-Jet AM Parameter Recommender (bright, friendly UI)
# Uses shared.py for: loading dataset, guardrails, physics priors, quantile models, and copilot.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import streamlit as st

from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
)

# -----------------------------------------------------------------------------
# Page setup (bright, inviting)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BJAM Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling with bright, inviting colors
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #FFFDF7 0%, #FFF8EC 40%, #FFF4E2 100%);
    }
    .stTabs [data-baseweb="tab"] { 
        font-weight: 600; 
    }
    .stMetric { 
        background: rgba(255,255,255,.6); 
        border-radius: 12px; 
        padding: 10px; 
    }
    div[data-testid="stStatusWidget"] { 
        opacity: .85; 
    }
    /* Additional styling for better visibility */
    .stDownloadButton button {
        background-color: rgba(255, 255, 255, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading (single source of truth; uses BJAM_All_Deep_Fill_v9.csv via shared.py)
# -----------------------------------------------------------------------------
try:
    df_base, src = load_dataset(".")
    models, meta = train_green_density_models(df_base)
except Exception as e:
    st.error(f"Error loading dataset or training models: {e}")
    df_base = pd.DataFrame()
    src = None
    models = None
    meta = {"note": "Error during initialization"}

# -----------------------------------------------------------------------------
# Sidebar - data + guardrails
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("BJAM Controls")
    
    if src and len(df_base) > 0:
        st.success(f"Data source: {Path(src).name} · rows={len(df_base):,}")
        # Download source dataset
        try:
            csv_data = df_base.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download source dataset (CSV)",
                data=csv_data,
                file_name=Path(src).name,
                mime="text/csv",
                help="Exports the exact dataset driving optimization & visuals.",
            )
        except Exception as e:
            st.warning(f"Could not prepare download: {e}")
    else:
        st.warning("No dataset found. Running on physics priors only (few-shot disabled).")

    st.divider()

    # Guardrails toggle
    guardrails_on = st.toggle(
        "Enable Guardrails",
        value=True,
        help=(
            "Guardrails ON: narrows ranges to empirically stable windows, clips overly optimistic predictions, "
            "and lightly penalizes extreme settings. OFF: wider exploration; still 0-100% physically bounded."
        ),
    )
    
    # Target density slider
    target_green = st.slider(
        "Target green %TD",
        min_value=80, 
        max_value=98, 
        value=90, 
        step=1,
        help="Use 90% for a strong starting point. Recs prefer q10 ≥ target for conservatism.",
    )

    st.caption(
        "💡 **What are guardrails?**\n\n"
        "Constraints + conservative post-processing to keep recommendations inside BJAM-stable regions: "
        "**binder 60-110%**, **speed ~1.2-3.5 mm/s**, **layer ≈ 3-5×D50**. "
        "With them OFF, you can explore a wider space."
    )

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("BJAM - Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · bright, friendly UI · toggle guardrails to explore")

# Top metrics / preview
m1, m2, m3 = st.columns(3)
m1.metric("Rows in dataset", f"{len(df_base):,}")
m2.metric("Materials", f"{df_base['material'].nunique() if 'material' in df_base and len(df_base) > 0 else 0:,}")
m3.metric("Quantile models", "Trained ✓" if models else "Physics-only")

with st.expander("📊 Preview source data", expanded=False):
    if len(df_base) > 0:
        st.dataframe(df_base.head(25), use_container_width=True)
    else:
        st.info("No rows to preview.")

st.divider()

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1])

# Material & D50
with left:
    st.subheader("Inputs", help="Set material & D50; layer default ≈ 4×D50, adjustable below.")

    # Material options from data (fallback to text input if empty)
    materials = []
    if "material" in df_base.columns and len(df_base) > 0:
        materials = sorted(df_base["material"].dropna().astype(str).unique().tolist())
    
    if not materials:
        materials = ["316L"]
    
    material = st.selectbox(
        "Material",
        options=materials,
        index=0,
        help="Choose from dataset materials. If your material is missing, type it in.",
    )

    # D50 default: median for chosen material if present, else 30 µm
    d50_default = 30.0
    if "material" in df_base.columns and "d50_um" in df_base.columns and len(df_base) > 0:
        try:
            material_data = df_base.loc[df_base["material"].astype(str) == material, "d50_um"]
            if len(material_data) > 0:
                d50_default = float(material_data.dropna().median())
                if pd.isna(d50_default):
                    d50_default = 30.0
        except Exception:
            d50_default = 30.0

    d50_um = st.number_input(
        "D50 (µm)",
        min_value=1.0, 
        max_value=150.0, 
        value=float(d50_default), 
        step=1.0,
        help="Median particle size. Layer guidance follows ≈ 3-5 × D50.",
    )

    # Default process priors (baseline)
    pri = physics_priors(d50_um, binder_type_guess=None)
    
    # Allow user to adjust layer; default 4×D50 (bounded by guardrails)
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    t_lo, t_hi = gr["layer_thickness_um"]
    
    layer_um = st.slider(
        "Layer thickness (µm)",
        min_value=float(round(t_lo)), 
        max_value=float(round(t_hi)),
        value=float(round(pri["layer_thickness_um"])),
        step=1.0,
        help="Stable spreading often near 4×D50 (band ≈ 3-5×D50).",
    )

    # Show the baseline priors (non-editable)
    c1, c2, c3 = st.columns(3)
    c1.metric("Prior binder %", f"{pri['binder_saturation_pct']:.0f}%")
    c2.metric("Prior speed", f"{pri['roller_speed_mm_s']:.1f} mm/s")
    c3.metric("Layer/D50", f"{layer_um/d50_um:.2f}×")

# Recommend
with right:
    st.subheader("Recommend parameters", help="Produces top-k sets targeting your green %TD.")
    
    top_k = st.slider("Number of recommendations", 3, 8, 5, 1)
    recommend_btn = st.button("🔍 Recommend", use_container_width=True, type="primary")

    if recommend_btn:
        try:
            with st.spinner("Generating recommendations..."):
                recs = copilot(
                    material=material,
                    d50_um=float(d50_um),
                    df_source=df_base,
                    models=models,
                    guardrails_on=guardrails_on,
                    target_green=float(target_green),
                    top_k=int(top_k),
                )
            
            st.dataframe(recs, use_container_width=True)
            st.caption(
                "Ranking favors **q10 ≥ target** (conservative), then **q50**; light penalty for extreme binder/speed. "
                "Use the visuals below to see *why* these are suggested."
            )
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    else:
        st.info("Click **Recommend** to generate top-k parameter sets.")

st.divider()

# -----------------------------------------------------------------------------
# Visuals (Heatmap, Sensitivity, Packing)
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🗺️ Heatmap (speed × saturation)",
        "📈 Saturation sensitivity",
        "🔷 Qualitative packing",
        "📐 Formulae (symbols)",
    ]
)

# -- Heatmap (speed × saturation, fixed layer & d50) --------------------------
with tabs[0]:
    st.subheader("Heatmap - Predicted green %TD", help="Layer fixed to your slider; explore speed × saturation.")
    
    try:
        # Build grid
        gr = guardrail_ranges(d50_um, on=guardrails_on)
        b_lo, b_hi = gr["binder_saturation_pct"]
        s_lo, s_hi = gr["roller_speed_mm_s"]

        s_vals = np.linspace(float(b_lo), float(b_hi), 55)   # saturation axis
        v_vals = np.linspace(float(s_lo), float(s_hi), 45)   # speed axis

        grid = pd.DataFrame(
            [(b, v, layer_um, d50_um, material) for b in s_vals for v in v_vals],
            columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um", "material"],
        )
        
        # Fill categorical hints
        grid["material_class"] = "metal"  # best-effort
        grid["binder_type_rec"] = "solvent_based"

        scored = predict_quantiles(models, grid)
        
        # Pivot to matrices for z
        Xs = sorted(scored["binder_saturation_pct"].unique())
        Ys = sorted(scored["roller_speed_mm_s"].unique())
        z = scored.sort_values(["binder_saturation_pct", "roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=Xs, y=Ys, z=z,
                colorbar=dict(title="%TD"),
                colorscale="Turbo",
            )
        )
        
        # 90% contour overlay
        fig.add_trace(
            go.Contour(
                x=Xs, y=Ys, z=z,
                contours=dict(start=90, end=90, size=1, coloring="none"),
                line=dict(width=3, color="white"),
                showscale=False,
                name="90% TD",
            )
        )
        
        fig.update_layout(
            xaxis_title="Binder saturation (%)",
            yaxis_title="Roller speed (mm/s)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=520,
            title=f"Green %TD (q50) · Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · Source={Path(src).name if src else '—'}",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")

# -- Saturation sensitivity (q10/q50/q90 vs binder %) -------------------------
with tabs[1]:
    st.subheader("Saturation sensitivity", help="Uncertainty band helps spot stable operating windows.")
    
    try:
        gr = guardrail_ranges(d50_um, on=guardrails_on)
        b_lo, b_hi = gr["binder_saturation_pct"]
        
        sats = np.linspace(float(b_lo), float(b_hi), 61)
        curve_df = pd.DataFrame({
            "binder_saturation_pct": sats,
            "roller_speed_mm_s": 1.6,
            "layer_thickness_um": layer_um,
            "d50_um": d50_um,
            "material": material,
            "material_class": "metal",
            "binder_type_rec": "solvent_based"
        })
        
        curve_scored = predict_quantiles(models, curve_df)

        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax2.plot(curve_scored["binder_saturation_pct"], curve_scored["td_q50"], 
                label="q50", linewidth=2)
        ax2.fill_between(
            curve_scored["binder_saturation_pct"], 
            curve_scored["td_q10"], 
            curve_scored["td_q90"], 
            alpha=0.2, 
            label="q10-q90"
        )
        ax2.axhline(target_green, linestyle="--", linewidth=1.5, 
                   color="red", alpha=0.7, label=f"Target {target_green}%")
        ax2.set_xlabel("Binder saturation (%)", fontsize=11)
        ax2.set_ylabel("Predicted green %TD", fontsize=11)
        ax2.set_title(f"Sensitivity @ speed=1.6 mm/s, layer={layer_um:.0f} µm, D50={d50_um:.0f} µm")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2, clear_figure=True)
        
    except Exception as e:
        st.error(f"Error generating sensitivity plot: {e}")

# -- Qualitative packing (illustrative slice near 90% effective packing) ------
with tabs[2]:
    st.subheader("Qualitative packing", help="Illustrative 2D slice to build intuition near target packing.")
    
    try:
        # Simple non-overlapping disk packing (illustrative)
        # Adjusted to reflect target packing fraction
        target_phi = target_green / 100.0
        
        # Scale disk radius and count to approximately match target packing
        # For 2D random packing: max ~64% for monodisperse circles
        # Scale visualized packing relative to this theoretical max
        scale_factor = target_phi / 0.64  # Normalize to 2D max packing
        r = 0.018 * (0.85 / scale_factor) ** 0.5  # Adjust radius inversely
        max_points = int(200 * scale_factor)  # More points for higher packing
        
        # Random sequential addition (RSA) packing
        np.random.seed(42)  # For reproducibility
        pts = []
        attempts = 0
        max_attempts = 15000
        
        while len(pts) < max_points and attempts < max_attempts:
            x, y = np.random.rand(2)
            # Check minimum distance to existing points
            if all((x - px) ** 2 + (y - py) ** 2 >= (2 * r) ** 2 for px, py in pts):
                pts.append((x, y))
            attempts += 1

        # Create figure
        fig3, ax3 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        ax3.set_aspect("equal", "box")
        
        # Draw particles
        for (x, y) in pts:
            circ = plt.Circle((x, y), r, alpha=0.8, color='steelblue', ec='darkblue', linewidth=0.5)
            ax3.add_patch(circ)
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title(
            f"Qualitative 2D packing visualization\n"
            f"(~{target_green:.0f}% target packing · {len(pts)} particles)"
        )
        ax3.set_facecolor('#f8f8f8')
        
        st.pyplot(fig3, clear_figure=True)
        
        st.caption(
            f"**Note:** This is a simplified 2D visualization. Actual 3D packing behavior "
            f"differs due to particle size distribution, shape, and process dynamics. "
            f"Particle count: {len(pts)}, Max attempts: {attempts:,}"
        )
        
    except Exception as e:
        st.error(f"Error generating packing visualization: {e}")

# -- Formulae (symbols, LaTeX) ------------------------------------------------
with tabs[3]:
    st.subheader("Formulae (symbols)", help="Key symbolic relations used for intuition and display.")
    
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\text{Packing fraction:}\quad \phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    
    st.caption(
        "These relations guide priors and plots. The few-shot model refines predictions from your dataset."
    )

# -----------------------------------------------------------------------------
# Footer / Debug
# -----------------------------------------------------------------------------
with st.expander("🔧 Diagnostics", expanded=False):
    st.write("**Models meta:**", meta if meta else {"note": "No trained models (physics-only)."})
    st.write("**Guardrails on:**", guardrails_on)
    st.write("**Source file:**", src or "—")
    st.write("**Dataset shape:**", df_base.shape if len(df_base) > 0 else "(empty)")
    
    if not len(df_base):
        st.info("No dataset rows found. Upload or add BJAM_All_Deep_Fill_v9.csv to enable few-shot.")
