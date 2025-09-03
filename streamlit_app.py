
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt

from shared import load_dataset, copilot, propose_abc, train_green_density_models

st.set_page_config(page_title="Binder‑Jet AM: Parameter Recommender", layout="wide")

st.markdown(f"""
<div class="hero">
  <h3 style="margin:6px 0 0 0;">Binder‑Jet AM: Physics‑guided parameter recommendations</h3>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
:root {
  --accent: #2D8CFF;
  --accent-2: #00C2A8;
  --bg: #ffffff;
  --card: #f7fbff;
  --ink: #0e1117;
}
.stApp { background: var(--bg); }
h1, h2, h3 { color: var(--ink); font-weight: 800; letter-spacing: 0.2px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #e8f2ff 0%, #e9fff8 100%); }
.stButton>button, .stDownloadButton>button, .stFileUploader label div {
  border-radius: 14px; border: none; padding: 0.6rem 1rem;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: white; font-weight: 700; box-shadow: 0 6px 16px rgba(45,140,255,0.35,
  /* layered soft color swells + subtle grid */
  background-image:
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%),
    radial-gradient(1200px 600px at 5% -10%, rgba(62,110,242,0.08), transparent 70%),
    radial-gradient(1000px 500px at 110% 10%, rgba(123,211,137,0.10), transparent 70%),
    radial-gradient(800px 400px at 30% 120%, rgba(255,209,102,0.10), transparent 60%),
    repeating-linear-gradient(45deg, rgba(0,0,0,0.025) 0px, rgba(0,0,0,0.025) 2px, transparent 2px, transparent 8px);
  background-blend-mode: normal, multiply, multiply, multiply, normal;
}

.stButton>button:hover, .stDownloadButton>button:hover { filter: brightness(1.08); }
div[data-testid="stMarkdownContainer"]>p, li { color: var(--ink); font-size: 0.98rem; }
.block-container { padding-top: 1.2rem; }
div.stAlert { background: var(--card); border-left: 4px solid var(--accent); }
div[data-testid="stExpander"] { background: var(--card); border-radius: 16px; }

.katex-display { margin: 0.25em 0 0.35em 0 !important; }  /* tighten block equation spacing */
</style>
""", unsafe_allow_html=True)


# Global safety toggle
st.sidebar.subheader("Safety")
guardrails_on = st.sidebar.toggle("Enable guardrails (recommended)", value=True,
    help="When off, input ranges widen and predictions are not clipped to conservative bounds.")
st.session_state["guardrails_on"] = guardrails_on

# Detailed guardrails explainer (sidebar)
st.sidebar.markdown("""
**Guardrails: what they do**  
- Keep inputs inside empirically stable BJAM **process windows**.  
- Constrain the **solver search space** and **clip outputs** to conservative ranges.

**When ON (recommended):**
- Ranges: binder 60–110 %, layer 30–150 µm, speed 1.5–3.5 mm/s.  
- Predictions: gently clipped to **60–98 %TD** to avoid over‑confidence.  
- Objective: includes small penalties for high binder and extreme speeds.  
- Warnings shown for out‑of‑window entries; heatmaps emphasize ≥90 % TD region.

**When OFF (explore/expert mode):**
- Wider ranges: binder 0–160 %, layer 5–300 µm, speed 0–10 mm/s.  
- No conservative clipping (only physical 0–100 %TD).  
- Objective prioritizes density; fewer penalties = more aggressive suggestions.  
- Use with caution: combinations may be **unstable** or **unprintable**.
""")

# Guardrails definition/info at the toggle
st.sidebar.info(
    "Guardrails: safety constraints that keep recommendations inside empirically stable BJAM ranges.\n\n"
    "• When ON: Inputs stay within tested ranges; predictions are gently clipped to conservative limits (e.g., 60–98 %TD). "
    "This avoids unstable spreading, over-binder, or unrealistic speeds.\n"
    "• When OFF: Wider input ranges and no conservative clipping (still physically bounded 0–100 %TD). "
    "Use for exploration or expert what‑ifs."
)

# ---------------- Theme / Background (Locked: Ivory Soft) ----------------
# Locked theme per user preference: "Ivory (Soft)"
_pal = {
    "accent":"#3E6EF2", "accent2":"#7BD389",
    "bg":"#FFFDF7", "bg2":"#FFF8EC",
    "card":"#FFFBF0", "ink":"#1A1C23",
    "sideStart":"#FFF3D9", "sideEnd":"#F2FFE8"
}
# Dynamic CSS using the Ivory palette
_css = f"""
<style>
:root {{
  --accent: {_pal['accent']};
  --accent-2: {_pal['accent2']};
  --bg: {_pal['bg']};
  --card: {_pal['card']};
  --ink: {_pal['ink']};
}}
.stApp {{
  background: linear-gradient(180deg, {_pal['bg']} 0%, {_pal['bg2']} 100%);
}}
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {_pal['sideStart']} 0%, {_pal['sideEnd']} 100%);
}}
h1, h2, h3 {{ color: var(--ink) !important; }}
.stButton>button, .stDownloadButton>button, .stFileUploader label div {{
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}}
div.stAlert {{ background: var(--card) !important; border-left: 4px solid var(--accent) !important; }}
div[data-testid="stExpander"] {{ background: var(--card) !important; }}
/* Fun accent chips */
.badge {{
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: rgba(62,110,242,0.12); color: var(--ink); font-weight:700; font-size:0.8rem;
  border: 1px solid rgba(62,110,242,0.25);
}}
.hero {{
  border-radius:18px; padding:14px 18px; margin-bottom:10px;
  background: linear-gradient(90deg, {_pal['accent']}22, {_pal['accent2']}22);
  border: 1px solid rgba(0,0,0,0.05);
}}
</style>
"""
st.markdown(_css, unsafe_allow_html=True)

# Plotly & Altair bright palettes
try:
    import plotly.express as px
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = ["#3E6EF2","#7BD389","#FF6B6B","#FFD166","#9B59B6","#00C2A8","#2EC4B6","#F77F00"]
    px.defaults.color_continuous_scale = "Turbo"
except Exception:
    pass

try:
    import altair as alt
    def ivory_theme():
        return {
            "config": {
                "background": "#FFFFFF",
                "view": {"strokeWidth": 0},
                "axis": {"labelColor": "#1A1C23", "titleColor": "#1A1C23"},
                "range": {
                    "category": ["#3E6EF2","#7BD389","#FF6B6B","#FFD166","#9B59B6","#00C2A8","#2EC4B6","#F77F00"],
                    "heatmap": ["#FDF4E3","#FDE7CB","#F9D29D","#F3B27E","#E68376","#B86BB6","#6BAED6","#3182BD"]
                }
            }
        }
    alt.themes.register("ivory_soft", ivory_theme)
    alt.themes.enable("ivory_soft")
except Exception:
    pass
# -------------------------------------------------------------


st.markdown("""
<style>
.stApp { background: #ffffff; }
h1, h2, h3 { color: #101010; font-weight: 700; letter-spacing: 0.1px; }
p, li, label, .stCaption { color: #1a1a1a; }
.section {border: 1px solid #e6e6e6; border-radius: 8px; padding: 16px; margin-bottom: 18px;}
.footer {font-size: 0.92rem; color:#444; border-top: 1px solid #e6e6e6; padding-top: 10px; margin-top: 18px;}
.stDownloadButton > button, .stDownloadButton button {
  background-color: #1f4ed8 !important;
  color: #ffffff !important;
  border: 1px solid #1f4ed8 !important;
  border-radius: 8px !important;
}
.stDownloadButton > button:hover, .stDownloadButton button:hover {
  background-color: #1739a6 !important;
  border-color: #1739a6 !important;
}
thead tr th { background: #f6f6f6 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Binder‑Jet Additive Manufacturing — Parameter Recommender")

df_base, src = load_dataset(".")
src_disp = src or "None (physics‑only mode)"
models, info = train_green_density_models(df_base)
mode_note = "ML quantile model (q10/q50/q90)" if models is not None else "Physics proxy (RCP‑style, ±5%)"

st.subheader("Physics baseline using few shot learning technique")
st.write("""
This tool begins with **physics‑guided priors** and an **RCP‑style packing proxy** for green density. When enough labeled **green‑density** rows are present, a calibrated **quantile ML model** (q10/q50/q90) replaces the proxy.
""")
st.markdown(r"""
- **Binder saturation prior:** 80 % (tuned later by optimization or learning).
- **Roller traverse speed prior:** 2.5 mm/s (guardrails 1.5–3.5 mm/s).
- **Packing proxy** (by material class):  
  $$\phi_{\mathrm{eff}}=\phi_{\mathrm{class}}-\alpha\,\Big(\frac{D_{50}-40}{40}\Big)^{2}-\beta\,\mathbb{1}\!\left[D_{50}<20~\mu\mathrm{m}\right]$$
  where $\phi_{\mathrm{class}}\approx 0.62$ (metals), $0.58$ (oxides), $0.55$ (carbides).
- **Green density estimate:** $\rho_{\mathrm{green}} \approx \phi_{\mathrm{eff}}\,\rho_{\mathrm{theoretical}}$ → we report **%TD** with a conservative **±5 %** band in proxy mode.
- **Infiltration/drying intuition** (for binder adjustments): Washburn‑type scaling $L^2 \propto \gamma \cos\theta/\mu \, t$ — higher viscosity or lower surface tension pushes saturation downward or speed downward.
- **Guardrails** (hard constraints): saturation 60–110 %, layer 30–150 µm, speed 1.5–3.5 mm/s.
- **Status this session:** Physics proxy (RCP‑style, ±5%).
""")

st.subheader("Upload experiments (optional)")
st.caption("Auto‑detects columns; used for learning **during this session**.")
up = st.file_uploader("CSV with any of the following columns", type=["csv"])
st.markdown("""
**Recognized columns (case/spacing flexible):**  
• **Material**, **D50 (µm)** (recommended) • **Binder Saturation (%)**, **Layer Thickness (µm)**, **Speed (mm/s)**, **Binder Type** • Density either as a single text column (e.g., `green 62%`) or two columns `final_density_state` & `final_density_pct`.  
**Units:** mm, µm/um, nm; speeds in mm/s or mm/min.
""")

def _normalize_upload(df):
    import re
    def normalize_col(c):
        c = str(c).strip().replace("%","pct").replace("µ","u").replace("μ","u")
        c = re.sub(r"$.*?$","",c); c = re.sub(r"[^A-Za-z0-9]+","_",c)
        return re.sub(r"_+","_",c).strip("_").lower()
    def to_float(x):
        if pd.isna(x): return np.nan
        if isinstance(x,(int,float)): return float(x)
        s = str(x).strip().lower().replace("–","-").replace("—","-")
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        return float(np.mean([float(n) for n in nums])) if nums else np.nan
    def to_pct(x):
        v = to_float(x); 
        if pd.isna(v): return np.nan
        if isinstance(x,str) and "%" in x: return v
        return v*100.0 if v<=1.5 else v
    def unify_speed(val):
        if pd.isna(val): return np.nan
        s = str(val).lower(); v = to_float(val)
        if "mm/min" in s or "per min" in s: return v/60.0
        if "cm/s" in s: return v*10.0
        return v
    def unify_length_um(val):
        if pd.isna(val): return np.nan
        s = str(val).lower(); v = to_float(val)
        if "mm" in s: return v*1000.0
        if "nm" in s: return v/1000.0
        return v
    def parse_density_state(x):
        if pd.isna(x): return (np.nan, np.nan)
        s = str(x).lower()
        state = "sintered" if "sinter" in s else ("green" if "green" in s else np.nan)
        return (state, to_pct(s))
    def categorize_binder_type(x):
        if pd.isna(x): return np.nan
        s = str(x).lower()
        if any(k in s for k in ["water","h2o","aqueous"]): return "water_based"
        if any(k in s for k in ["uv","photo"]): return "uv_curable"
        if any(k in s for k in ["solvent","alcohol","ipa","acetone"]): return "solvent_based"
        if any(k in s for k in ["polymer","pva","pov","pmma","peo","pla"]): return "polymer_based"
        if any(k in s for k in ["proprietary","dm","exone","markforged"]): return "proprietary"
        return "unspecified"
    d = df.copy(); d.columns = [normalize_col(c) for c in d.columns]
    def find(cands):
        for c in d.columns:
            for p in cands:
                if re.search(p, c): return c
        return None
    mat = find([r"^material", r"powder", r"alloy", r"name"])
    d50 = find([r"\bd50\b", r"median.*size", r"particle.*size", r"^d_?50"])
    sat = find([r"binder.*sat", r"sat.*binder", r"saturation"])
    lay = find([r"layer.*thick", r"thick"])
    spd = find([r"roller.*speed", r"traverse.*speed", r"\bspeed\b"])
    bty = find([r"binder.*type", r"type.*binder", r"binder$"])
    dens = find([r"(final|sinter|green).*(dens|%)", r"dens.*(final|sinter|green)"])
    dens_val = find([r"dens.*(pct|%)$", r"^density_pct$"])
    dens_state = find([r"dens.*state", r"^state$"])
    out = pd.DataFrame()
    out["material"] = d[mat] if mat else np.nan
    out["d50_um"] = d[d50].apply(unify_length_um) if d50 else np.nan
    out["binder_saturation_pct"] = d[sat].apply(to_pct) if sat else np.nan
    out["layer_thickness_um"] = d[lay].apply(unify_length_um) if lay else np.nan
    out["roller_speed_mm_s"] = d[spd].apply(unify_speed) if spd else np.nan
    out["binder_type_rec"] = d[bty].apply(categorize_binder_type) if bty else np.nan
    if dens:
        stv, dv = zip(*d[dens].apply(parse_density_state))
        out["final_density_state"] = stv; out["final_density_pct"] = dv
    elif dens_val and dens_state:
        out["final_density_state"] = d[dens_state]
        out["final_density_pct"] = d[dens_val].apply(to_pct)
    else:
        out["final_density_state"] = np.nan; out["final_density_pct"] = np.nan
    return out

df_work = df_base.copy()
uploaded_rows = 0
if up is not None:
    try:
        df_uploaded = pd.read_csv(up)
        df_norm = _normalize_upload(df_uploaded)
        uploaded_rows = len(df_norm)
        st.success(f"Ingested {uploaded_rows} rows; the model will use them during this session.")
        df_work = pd.concat([df_work, df_norm], ignore_index=True, sort=False)
    except Exception as e:
        st.error(f"Upload failed: {e}")

st.subheader("Input")
c1, c2 = st.columns([2,1])
with c1:
    mat = st.text_input("Material / Powder name", "Inconel 625")
with c2:
    d50 = st.number_input("Median particle size D50 (µm)", min_value=1.0, max_value=150.0, value=30.0, step=1.0)

run = st.button("Compute recommendation", type="primary")

if "pred_rows" not in st.session_state: st.session_state["pred_rows"] = []

if run:
    rec = copilot(mat, d50, df_work)

    st.subheader("Recommendation")
    table = pd.DataFrame({
        "Parameter":[ "Binder type", "Binder saturation", "Layer thickness", "Roller traverse speed",
                      "Green density q10", "Green density q50", "Green density q90", "Mode"],
        "Value":[ rec["binder_type"], f"{rec['binder_saturation_pct']:.1f}", f"{rec['layer_thickness_um']:.1f}", f"{rec['roller_speed_mm_s']:.2f}",
                  f"{rec['green_density_pred_q10']:.1f}", f"{rec['green_density_pred_q50']:.1f}", f"{rec['green_density_pred_q90']:.1f}", rec["green_density_mode"] ],
        "Units":[ "-", "%", "µm", "mm/s", "%TD", "%TD", "%TD", "-" ]
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots()
    x = [0]
    mu = [rec["green_density_pred_q50"]]
    err_low = [rec["green_density_pred_q50"] - rec["green_density_pred_q10"]]
    err_high = [rec["green_density_pred_q90"] - rec["green_density_pred_q50"]]
    ax.errorbar(x, mu, yerr=[err_low, err_high], fmt='o', capsize=5)
    ax.set_xticks([])
    ax.set_ylabel("Green density (%TD)")
    ax.set_title("Predicted green density with uncertainty")
    st.pyplot(fig, clear_figure=True)

    st.subheader("DOE A/B/C (print these coupons)")
    abc = propose_abc(rec["material"], rec["d50_um"], rec["binder_type"], rec["layer_thickness_um"])
    df_abc = pd.DataFrame(abc)[["recipe_label","binder_saturation_pct","layer_thickness_um","roller_speed_mm_s","material","d50_um","binder_type"]]
    df_abc.columns = ["Label","Saturation (%)","Layer (µm)","Speed (mm/s)","Material","D50 (µm)","Binder"]
    st.dataframe(df_abc, use_container_width=True)

    base_row = {k: rec[k] for k in ["material","d50_um","binder_type","green_density_pred_q10","green_density_pred_q50","green_density_pred_q90","green_density_mode"]}
    for r in abc:
        row = base_row.copy()
        row.update({"recipe_label": r["recipe_label"],
                    "binder_saturation_pct": r["binder_saturation_pct"],
                    "layer_thickness_um": r["layer_thickness_um"],
                    "roller_speed_mm_s": r["roller_speed_mm_s"]})
        st.session_state["pred_rows"].append(row)

st.subheader("Download")
if len(st.session_state["pred_rows"]) == 0:
    st.caption("Run at least one recommendation to enable downloads.")
else:
    out_df = pd.DataFrame(st.session_state["pred_rows"])
    st.download_button("Recommendations (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"bjam_recs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    st.download_button("Last DOE set (JSON)",
                       out_df.tail(3).to_json(orient="records", indent=2).encode("utf-8"),
                       file_name="bjam_last_recipes.json", mime="application/json")



# ===============================
# Visual diagnostics (all-in-one)
# ===============================
import numpy as np, pandas as pd, math, json
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

st.subheader("Visual diagnostics")
with st.expander("Show all plots and diagnostics", expanded=True):
    st.markdown(f"<div class='badge'>Safety: {'On' if st.session_state.get('guardrails_on', True) else 'Off'}</div>", unsafe_allow_html=True)
    st.caption("These controls drive the plots below (independent of the prediction pane).")
    c1, c2, c3, c4 = st.columns(4)
    max_d50 = 200.0 if st.session_state.get("guardrails_on", True) else 300.0
    min_d50 = 1.0 if st.session_state.get("guardrails_on", True) else 0.1
    d50_um = c1.number_input("D50 (µm)", min_value=float(min_d50), max_value=float(max_d50), value=30.0, step=1.0)
    max_layer = 250.0 if st.session_state.get("guardrails_on", True) else 300.0
    min_layer = 10.0 if st.session_state.get("guardrails_on", True) else 5.0
    layer_um = c2.number_input("Layer thickness (µm)", min_value=float(min_layer), max_value=float(max_layer), value=float(max(30.0, min(150.0, 4*d50_um))), step=1.0)
    sat_min, sat_max = (40.0, 130.0) if st.session_state.get("guardrails_on", True) else (0.0, 160.0)
    binder_pct = c3.number_input("Binder saturation (%)", min_value=float(sat_min), max_value=float(sat_max), value=80.0, step=1.0)
    spd_min, spd_max = (0.5, 6.0) if st.session_state.get("guardrails_on", True) else (0.0, 10.0)
    roller_mm_s = c4.number_input("Roller speed (mm/s)", min_value=float(spd_min), max_value=float(spd_max), value=2.5, step=0.1)
    demo_mode = st.toggle("Demo mode (use physics-style heuristic if no model)", value=True)

    def predict_quantiles(d50, layer, speed, sat):
        """Heuristic physics-style predictor with optional guardrails."""
        phi_class = 0.62
        try:
            phi_class = float(st.session_state.get("phi_class", phi_class))
        except Exception:
            pass
        guard = st.session_state.get("guardrails_on", True)

        phi = phi_class
        if d50 < 20: phi -= (0.03 if guard else 0.02)
        if d50 < 10: phi -= (0.05 if guard else 0.03)
        if d50 > 80: phi -= (0.01 if guard else 0.005)

        lo, hi = 3.0*d50, 5.0*d50
        if layer < lo:
            phi -= (0.002 if guard else 0.001) * (lo - layer)
        elif layer > hi:
            phi -= (0.0015 if guard else 0.0008) * (layer - hi)

        phi -= (0.0006 if guard else 0.0004) * (sat - 80.0)**2 / 10.0

        delta = abs(speed - 2.5) - (1.0 if guard else 0.0)
        phi -= (0.015 if guard else 0.006) * max(0.0, delta)

        green = 100.0 * phi
        if guard:
            green = max(60.0, min(98.0, green))
        else:
            green = max(0.0, min(100.0, green))

        spread = 2.0 + 0.02*abs(layer - 4*d50) + 0.03*abs(sat - 80.0) + (0.5 if guard else 0.25)*max(0.0, abs(speed - 2.5) - (1.0 if guard else 0.0))
        q10 = max(0.0, green - 1.5*spread)
        q50 = green
        q90 = min(100.0, green + 1.5*spread)
        return q10, q50, q90

    def model_predict_quantiles(d50, layer, speed, sat):
        try:
            predictor = st.session_state.get("bj_predictor", None)
            if predictor is None:
                raise RuntimeError("no model")
            return predictor(d50=d50, layer_um=layer, roller_mm_s=speed, binder_pct=sat)
        except Exception:
            return predict_quantiles(d50, layer, speed, sat)

    use_model = not demo_mode

    def plot_process_window(d50, layer, min_mult=3.0, max_mult=5.0):
        x = np.linspace(max(1.0, 0.5*d50), min(200.0, 3.0*d50), 200)
        band_lo = min_mult * x
        band_hi = max_mult * x
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=band_lo, mode="lines", name=f"{min_mult}×D50"))
        fig.add_trace(go.Scatter(x=x, y=band_hi, mode="lines", name=f"{max_mult}×D50", fill='tonexty'))
        fig.add_trace(go.Scatter(x=[d50], y=[layer], mode="markers", name="Your pick", marker=dict(size=12)))
        fig.update_layout(xaxis_title="D50 (µm)", yaxis_title="Layer thickness (µm)",
                          title="Stable spreading band vs your selection",
                          height=420, margin=dict(l=10,r=10,t=60,b=10))
        return fig

    st.plotly_chart(plot_process_window(d50_um, layer_um), use_container_width=True)

    # Recommendation solver (targets >= 90% TD if possible)
    st.markdown("### Parameter recommendations")
    target = st.slider("Target green %TD", 80, 98, 90, 1)
    if st.button("Recommend parameters"):
        sat_grid = np.linspace(sat_min, sat_max, 41)
        spd_grid = np.linspace(spd_min, spd_max, 41)
        lyr_grid = np.linspace(min_layer, max_layer, 31)
        candidates = []
        for s_ in sat_grid:
            for v_ in spd_grid:
                for t_ in lyr_grid:
                    q10, q50, q90 = (model_predict_quantiles if use_model else predict_quantiles)(d50_um, t_, v_, s_)
                    score = (q50 - target) - 0.02*abs(s_-80) - 0.1*max(0, v_-2.5)
                    candidates.append((q50, q10, q90, s_, v_, t_, score))
        dfc = pd.DataFrame(candidates, columns=["td_q50","td_q10","td_q90","binder_%","speed_mm_s","layer_um","score"])
        feas = dfc[dfc["td_q50"] >= target].sort_values(["td_q50","score"], ascending=[False, False]).head(5)
        if feas.empty:
            st.warning("No combinations reach the target with the current heuristic/model settings. Showing top candidates.")
            feas = dfc.sort_values(["td_q50","score"], ascending=[False, False]).head(5)
        mat_class = "metals" if float(st.session_state.get("phi_class", 0.62)) >= 0.60 else "ceramics"
        binder_type = "Solvent-based (PMMA/thermoplastic)" if mat_class=="metals" else "Water-based (PVA/PVOH)"
        feas = feas.assign(binder_type=binder_type)
        
        # Column header tooltips (hover on headers)
        _colcfg = {
            "binder_type": st.column_config.TextColumn(
                "Binder type",
                help="Suggested binder family based on material class; override per system and powder surface chemistry."
            ),
            "binder_%": st.column_config.NumberColumn(
                "Binder saturation (%)",
                help="Binder saturation as % of pore volume in the layer. ~80% typical; too high can cause slumping, too low reduces green strength.",
                format="%.2f"
            ),
            "speed_mm_s": st.column_config.NumberColumn(
                "Roller speed (mm/s)",
                help="Traverse speed for spreading. Higher speeds risk tearing/segregation; slower speeds improve stability but increase time.",
                format="%.2f"
            ),
            "layer_um": st.column_config.NumberColumn(
                "Layer thickness (µm)",
                help="Printed layer thickness. Stable window ≈ 3–5×D50; thicker layers risk poor packing.",
                format="%.0f"
            ),
            "td_q50": st.column_config.NumberColumn(
                "Predicted %TD (q50)",
                help="Median prediction of green theoretical density.",
                format="%.2f"
            ),
            "td_q10": st.column_config.NumberColumn(
                "Conservative bound (q10)",
                help="Lower 10th‑percentile prediction; ~90% chance the true value is above this.",
                format="%.2f"
            ),
            "td_q90": st.column_config.NumberColumn(
                "Optimistic bound (q90)",
                help="Upper 90th‑percentile prediction; ~10% chance the true value exceeds this.",
                format="%.2f"
            ),
        }

        st.dataframe(
            feas[["binder_type","binder_%","speed_mm_s","layer_um","td_q50","td_q10","td_q90"]].round(2),
            use_container_width=True,
            column_config=_colcfg,
            hide_index=False
        )


    sats = np.linspace(50, 110, 61)
    rows = []
    for s_ in sats:
        q10, q50, q90 = (model_predict_quantiles if use_model else predict_quantiles)(d50_um, layer_um, roller_mm_s, s_)
        rows.append((s_, q10, q50, q90))
    df_pred = pd.DataFrame(rows, columns=["saturation_pct","td_q10","td_q50","td_q90"])

    line = alt.Chart(df_pred).mark_line().encode(
        x=alt.X('saturation_pct:Q', title='Binder saturation (%)'),
        y=alt.Y('td_q50:Q', title='Predicted green %TD')
    )
    band = alt.Chart(df_pred).mark_area(opacity=0.25).encode(
        x='saturation_pct:Q', y='td_q10:Q', y2='td_q90:Q'
    )
    rule90 = alt.Chart(pd.DataFrame({'y':[90.0]})).mark_rule(strokeDash=[4,4]).encode(y='y:Q')
    st.altair_chart((band + line + rule90).properties(title="Saturation sensitivity with uncertainty"), use_container_width=True)

    speeds = np.linspace(0.5, 5.0, 60)
    grid = [(sp, sb) for sp in speeds for sb in sats]
    td50s = [ (model_predict_quantiles if use_model else predict_quantiles)(d50_um, layer_um, sp, sb)[1] for sp, sb in grid ]
    df_grid = pd.DataFrame(grid, columns=["roller_mm_s","saturation_pct"])
    df_grid["green_td"] = td50s
    heat = px.density_heatmap(df_grid, x="roller_mm_s", y="saturation_pct", z="green_td",
                              nbinsx=40, nbinsy=40, histfunc="avg",
                              labels={"roller_mm_s":"Roller speed (mm/s)", "saturation_pct":"Binder saturation (%)", "green_td":"%TD"})
    heat.update_layout(title="Response surface: speed × saturation (colored by predicted green %TD)",
                       height=480, margin=dict(l=10,r=10,t=60,b=10))
    st.plotly_chart(heat, use_container_width=True)

    iso = df_grid[np.isclose(df_grid["green_td"], 90.0, atol=1.0)]
    if not iso.empty:
        fig_iso = px.scatter(iso, x="roller_mm_s", y="saturation_pct", title="Approx. ≥90% TD contour points")
        st.plotly_chart(fig_iso, use_container_width=True)

    q10_b, q50_b, q90_b = predict_quantiles(d50_um, layer_um, roller_mm_s, binder_pct)
    q10_m, q50_m, q90_m = (model_predict_quantiles if use_model else predict_quantiles)(d50_um, layer_um, roller_mm_s, binder_pct)
    bar = pd.DataFrame({
        "Model": ["Baseline proxy","Refined model"],
        "q10":[q10_b, q10_m], "q50":[q50_b, q50_m], "q90":[q90_b, q90_m]
    })
    fig_uplift = go.Figure()
    fig_uplift.add_trace(go.Scatter(x=bar["Model"], y=bar["q10"], mode="markers", name="q10"))
    fig_uplift.add_trace(go.Scatter(x=bar["Model"], y=bar["q50"], mode="markers+lines", name="q50"))
    fig_uplift.add_trace(go.Scatter(x=bar["Model"], y=bar["q90"], mode="markers", name="q90"))
    fig_uplift.add_hline(y=90.0, line_dash="dot")
    fig_uplift.update_layout(title="Few-shot uplift (proxy vs refined)", yaxis_title="Predicted green %TD")
    st.plotly_chart(fig_uplift, use_container_width=True)

    def packing_section(d50=30.0, target_phi=0.90, box=1.0, n=120, span=0.4, seed=42):
        """
        Robust qualitative packing slice.
        """
        rng = np.random.default_rng(seed if seed is not None else int(1000*d50 + 7))
        r_med = max(1e-3, d50/2.0)
        sigma = float(np.log(1.0 + max(1e-3, span)))
        radii = rng.lognormal(mean=float(np.log(r_med)), sigma=sigma, size=n)
        A_box = box * box
        denom = float(np.pi * np.sum(radii**2))
        if denom <= 0:
            radii = np.full(n, 0.02*box)
            denom = float(np.pi * np.sum(radii**2))
        scale = float(np.sqrt((target_phi * A_box) / denom))
        radii = radii * scale
        min_r = 0.006 * box
        radii = np.maximum(radii, min_r)
        xs, ys = rng.random(n) * box, rng.random(n) * box
        shapes = []
        for x, y, r in zip(xs, ys, radii):
            shapes.append(dict(type="circle", xref="x", yref="y", x0=float(x - r), x1=float(x + r), y0=float(y - r), y1=float(y + r), line=dict(width=1)))
        fig = go.Figure()
        fig.update_layout(shapes=shapes, title=f"Qualitative packing slice (target φ ≈ {int(target_phi*100)}%, D50≈{int(d50)}µm)", height=420, margin=dict(l=10, r=10, t=60, b=10))
        fig.update_xaxes(range=[0, box], visible=False)
        fig.update_yaxes(range=[0, box], visible=False, scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    packing_section(d50_um, target_phi=0.90)

    T = np.linspace(800, 1400, 100)
    green_now = (model_predict_quantiles if use_model else predict_quantiles)(d50_um, layer_um, roller_mm_s, binder_pct)[1]
    Tmid = 0.5*(min(T)+max(T)) + 100*(0.35 - (green_now/100.0))
    a = 0.012
    td_sint = 98.0 / (1.0 + np.exp(-a*(T - Tmid)))
    td_sint = np.maximum(td_sint, green_now)
    df_sint = pd.DataFrame({"temperature_C":T, "td_pct":td_sint})
    base = alt.Chart(df_sint).encode(x=alt.X('temperature_C:Q', title='Temperature (°C)'))
    line = base.mark_line().encode(y=alt.Y('td_pct:Q', title='%TD'))
    rule90s = alt.Chart(pd.DataFrame({'y':[90.0]})).mark_rule(strokeDash=[4,4]).encode(y='y:Q')
    st.altair_chart((line + rule90s).properties(title="Predicted sintered density vs temperature (illustrative)"),
                    use_container_width=True)

    def pareto_frontier(df, xcol='saturation_pct', ycol='green_td'):
        pts = df.sort_values([xcol, ycol], ascending=[True, False]).to_numpy()
        front = []
        best = -np.inf
        for x,y in pts:
            if y > best:
                front.append((x,y)); best = y
        return pd.DataFrame(front, columns=[xcol,ycol])

    df_grid_small = df_grid[["saturation_pct","green_td"]].dropna().copy()
    df_front = pareto_frontier(df_grid_small)
    fig_par = px.scatter(df_grid_small.sample(min(2000, len(df_grid_small))), x='saturation_pct', y='green_td',
                         title='Binder vs density (Pareto frontier overlay)',
                         labels={'saturation_pct':'Binder saturation (%)','green_td':'Predicted green %TD'})
    fig_par.add_trace(go.Scatter(x=df_front['saturation_pct'], y=df_front['green_td'],
                                 mode='lines+markers', name='Frontier'))
    st.plotly_chart(fig_par, use_container_width=True)

    def local_importance(d50, layer, speed, sat, h=(1.0, 2.0, 0.1, 2.0)):
        names = ["d50_um","layer_um","roller_mm_s","binder_pct"]
        base_q = (model_predict_quantiles if use_model else predict_quantiles)(d50, layer, speed, sat)[1]
        grads = []
        for (n, delta) in zip(names, h):
            args = dict(d50=d50, layer=layer, speed=speed, sat=sat)
            if n == "d50_um": args["d50"] += delta
            elif n == "layer_um": args["layer"] += delta
            elif n == "roller_mm_s": args["speed"] += delta
            elif n == "binder_pct": args["sat"] += delta
            q = (model_predict_quantiles if use_model else predict_quantiles)(**args)[1]
            grads.append(abs(q - base_q)/max(1e-6, delta))
        imp = pd.DataFrame({"feature":names, "importance":grads}).sort_values("importance", ascending=False)
        return imp

    imp_df = local_importance(d50_um, layer_um, roller_mm_s, binder_pct)
    st.bar_chart(imp_df.set_index("feature"))

    df_exp = st.session_state.get("df_experiments", None)
    if isinstance(df_exp, pd.DataFrame) and all(col in df_exp.columns for col in ["d50_um","layer_um","roller_mm_s","binder_pct","green_td_measured"]):
        preds = []
        for _, r in df_exp.iterrows():
            preds.append((model_predict_quantiles if use_model else predict_quantiles)(
                r["d50_um"], r["layer_um"], r["roller_mm_s"], r["binder_pct"])[1])
        df_exp = df_exp.assign(predicted_green_td = preds,
                               residual = df_exp["green_td_measured"] - np.array(preds))
        sc = px.scatter(df_exp, x="predicted_green_td", y="residual",
                        hover_data=df_exp.columns, title="Residuals vs predicted (outlier diagnostics)")
        st.plotly_chart(sc, use_container_width=True)
    else:
        st.caption("Residuals/outliers plot will appear after you upload experiments with measured green density.")
st.markdown("---")
if src:
    try:
        with open(src, "rb") as f:
            st.download_button("Download source dataset", f.read(), file_name=os.path.basename(src), mime="text/csv")
    except Exception as _e:
        st.caption(f"Source dataset: {src_disp}")
else:
    st.caption("No source dataset found for this session (physics‑only mode).")

st.markdown(f"""
<div class="footer">
<strong>© {datetime.now().year} Bhargavi Mummareddy</strong> • Contact: <a href="mailto:mummareddybhargavi@gmail.com">mummareddybhargavi@gmail.com</a><br/>
<b>Guardrails:</b> saturation 60–110 %, layer 30–150 µm, speed 1.5–3.5 mm/s.
</div>
""", unsafe_allow_html=True)

