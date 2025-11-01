# shared.py — data/model/physics helpers for BJAM Digital Twin
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

DATA_CANDIDATES = [
    Path("BJAM_cleaned.csv"),
    Path("BJAM_All_Deep_Fill_v9.csv"),
    Path("/mnt/data/BJAM_cleaned.csv"),
    Path("/mnt/data/BJAM_All_Deep_Fill_v9.csv"),
]

def load_dataset() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            # normalize column names
            rename_map = {}
            lower = [c.lower().strip() for c in df.columns]
            def find_like(names):
                for n in names:
                    if n in lower:
                        i = lower.index(n)
                        return df.columns[i]
                return None

            # required fields
            mcol = find_like(["material"])
            if mcol: rename_map[mcol] = "material"
            d50c = find_like(["d50_um","d50","particle_size_um","particle_size"])
            if d50c: rename_map[d50c] = "d50_um"
            lthc = find_like(["layer_thickness_um","layer_um","layer","layer_thickness"])
            if lthc: rename_map[lthc] = "layer_thickness_um"
            satc = find_like(["binder_saturation_pct","saturation_pct","binder_saturation","saturation"])
            if satc: rename_map[satc] = "binder_saturation_pct"
            gpc  = find_like(["green_pct_td","green_%td","green_density_pct","final_density_pct","green_pct","pct_td"])
            if gpc: rename_map[gpc] = "green_pct_td"
            btyp = find_like(["binder_type","binder"])
            if btyp: rename_map[btyp] = "binder_type"

            df = df.rename(columns=rename_map)

            for k in ["material","d50_um","layer_thickness_um","binder_saturation_pct"]:
                if k not in df.columns:
                    df[k] = np.nan
            return df.dropna(subset=["material"]).copy()
    return pd.DataFrame(columns=["material","d50_um","layer_thickness_um","binder_saturation_pct","green_pct_td","binder_type"])

# Modeling
FEATURES = ["material","d50_um","layer_thickness_um","binder_saturation_pct"]

def _make_quantile(q: float):
    cat = ["material"]
    num = ["d50_um","layer_thickness_um","binder_saturation_pct"]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                      ("num", "passthrough", num)]
    )
    model = GradientBoostingRegressor(
        loss="quantile", alpha=q,
        n_estimators=250,
        max_depth=3,
        learning_rate=0.06,
        random_state=42
    )
    return Pipeline([("prep", pre), ("gb", model)])

def fit_quantile_models(df: pd.DataFrame):
    if df.empty:
        class _Dummy:
            def predict(self, X): return np.full((len(X),), 85.0)
        return _Dummy(), _Dummy(), _Dummy()
    y = df["green_pct_td"] if "green_pct_td" in df.columns else df.iloc[:,0]*0 + 85.0
    X = df[FEATURES]
    q10 = _make_quantile(0.10).fit(X, y)
    q50 = _make_quantile(0.50).fit(X, y)
    q90 = _make_quantile(0.90).fit(X, y)
    return q10, q50, q90

# Physics guardrails
def pack_fraction_furnas(d50_um: float, layer_um: float) -> float:
    # heuristic: thinner layers vs D50 -> higher packing; clamp to [0, 1]
    ratio = (d50_um / max(1e-6, layer_um))
    pf = 0.74 * (1.0 / (1.0 + 0.6 * (1/ratio)))
    return float(np.clip(pf, 0.45, 0.74))

def washburn_binder_suggestion(d50_um: float, target_td: float) -> float:
    base = 70.0 + (target_td - 90.0) * 0.8
    adj = 15.0 * np.tanh((40.0 - d50_um)/30.0)
    return float(np.clip(base + adj, 35.0, 110.0))

def valid_binder_types() -> List[str]:
    return ["water_based", "solvent_based"]

def sanitize_material_name(s: str) -> str:
    return str(s).strip()

def _roller_speed_rule(d50_um: float) -> float:
    # v_r ∝ 1 / (d50^alpha), alpha≈0.6 for spherical powders
    # Normalize to ~60 mm/s at d50=25 µm
    alpha = 0.6
    base = 60.0 * (25.0 ** alpha)
    v = base / (max(5.0, d50_um) ** alpha)
    return float(np.clip(v, 15.0, 200.0))

def recommend_with_guardrails(models, df, material: str, d50_um: float, layer_um: float,
                              target_td: float, require_mix: bool, n_total: int=5) -> pd.DataFrame:
    q10, q50, q90 = models
    Ls = np.linspace(max(20.0, 0.6*layer_um), min(200.0, 1.6*layer_um), 9)
    Ss = np.linspace(40.0, 110.0, 15)

    rows = []
    for L in Ls:
        for S in Ss:
            x = pd.DataFrame([{
                "material": material,
                "d50_um": d50_um,
                "layer_thickness_um": L,
                "binder_saturation_pct": S
            }])
            y10 = float(q10.predict(x[["material","d50_um","layer_thickness_um","binder_saturation_pct"]])[0])
            y50 = float(q50.predict(x[["material","d50_um","layer_thickness_um","binder_saturation_pct"]])[0])
            y90 = float(q90.predict(x[["material","d50_um","layer_thickness_um","binder_saturation_pct"]])[0])
            pf = pack_fraction_furnas(d50_um, L) * 100.0
            if pf < 90.0:
                continue
            score = (max(0.0, y10 - target_td)) + 0.25*(y50-target_td)
            rows.append((L, S, y10, y50, y90, pf, score))

    if not rows:
        return pd.DataFrame()

    cand = pd.DataFrame(rows, columns=["layer_um","saturation_pct","pred_q10","pred_q50","pred_q90","theoretical_%TD","score"])
    cand = cand.sort_values("score", ascending=False)

    water_needed, solvent_needed = 3, 2
    out = []
    for _, r in cand.iterrows():
        btype = "water_based" if r["saturation_pct"] <= washburn_binder_suggestion(d50_um, target_td) else "solvent_based"
        if btype=="water_based" and water_needed>0:
            out.append((btype, r))
            water_needed -= 1
        elif btype=="solvent_based" and solvent_needed>0:
            out.append((btype, r))
            solvent_needed -= 1
        if len(out) >= n_total: break

    if len(out) < n_total:
        for _, r in cand.iterrows():
            if any(np.isclose(r["layer_um"], rr[1]["layer_um"]) and np.isclose(r["saturation_pct"], rr[1]["saturation_pct"]) for rr in out):
                continue
            btype = "water_based" if r["saturation_pct"] <= washburn_binder_suggestion(d50_um, target_td) else "solvent_based"
            out.append((btype, r))
            if len(out) >= n_total: break

    if not out:
        return pd.DataFrame()

    recs = []
    for btype, r in out:
        recs.append({
            "binder_type": btype,
            "saturation_pct": float(np.round(r["saturation_pct"], 1)),
            "roller_speed_mm_s": _roller_speed_rule(d50_um),
            "layer_um": float(np.round(r["layer_um"], 1)),
            "pred_q10": float(np.round(r["pred_q10"], 1)),
            "pred_q50": float(np.round(r["pred_q50"], 1)),
            "pred_q90": float(np.round(r["pred_q90"], 1)),
            "theoretical_%TD": float(np.round(r["theoretical_%TD"], 1)),
            "material": material,
            "d50_um": float(d50_um)
        })
    outdf = pd.DataFrame(recs)
    outdf.insert(0, "id", [f"Opt-{i+1}" for i in range(len(outdf))])
    return outdf
