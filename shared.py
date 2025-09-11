# shared.py — DATA + PRIORS + MODELS + RECOMMENDER for BJAM
# Single source of truth: BJAM_All_Deep_Fill_v9.csv (env override: BJAM_DATA)

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# 0) DATA SOURCE (choose file once; everything else uses it)
# =============================================================================

DATA_CSV = os.environ.get("BJAM_DATA", "BJAM_All_Deep_Fill_v9.csv")
DATA_CANDIDATES = [
    DATA_CSV,
    "BJAM_All_Deep_Fill_v9.csv",
    "BJAM_v10_clean.csv",
    "BJAM_v9_clean_v2.csv",
    "BJAM_v9_clean.csv",
]

# Canonical column names used throughout
CANON = {
    "material": "material",
    "material_class": "material_class",
    "d50_um": "d50_um",

    "layer_thickness_um": "layer_thickness_um",
    "layer_um": "layer_thickness_um",   # alias -> canonical

    "roller_speed_mm_s": "roller_speed_mm_s",
    "speed_mm_s": "roller_speed_mm_s",  # alias

    "binder_type_rec": "binder_type_rec",
    "binder_type": "binder_type_rec",   # alias

    "binder_saturation_pct": "binder_saturation_pct",
    "binder_pct": "binder_saturation_pct",  # alias

    "final_density_state": "final_density_state",
    "final_density_pct": "final_density_pct",
}

NUMERIC_COLS = [
    "d50_um", "layer_thickness_um", "roller_speed_mm_s",
    "binder_saturation_pct", "final_density_pct",
]
CATEGORICAL_COLS = ["material", "material_class", "binder_type_rec"]


# =============================================================================
# 1) UTILITIES
# =============================================================================

def _infer_material_class(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ["316l", "inconel", "17-4", "steel", "copper", "ti ", "ti-","al "]):
        return "metal"
    if any(k in n for k in ["al2o3", "alumina", "oxide", "zirconia", "zro2"]):
        return "oxide"
    if any(k in n for k in ["wc", "carbide", "sic", "tib2"]):
        return "carbide"
    return "other"


def _rename_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Map aliases -> canonical (case-insensitive)
    mapping = {}
    lower_to_canon = {k.lower(): v for k, v in CANON.items()}
    for c in df.columns:
        mapping[c] = lower_to_canon.get(c.strip().lower(), c)
    out = df.rename(columns=mapping)

    # Ensure all expected columns exist
    for col in set(CANON.values()):
        if col not in out.columns:
            out[col] = np.nan

    # Numeric coercion
    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Normalize text columns
    if "final_density_state" in out.columns:
        out["final_density_state"] = (
            out["final_density_state"].astype(str).str.strip().str.lower()
        )
    if "binder_type_rec" in out.columns:
        out["binder_type_rec"] = (
            out["binder_type_rec"]
            .astype(str).str.strip().str.replace(" ", "_").str.lower()
        )
    # Fill missing material_class if possible
    if "material_class" in out.columns and out["material_class"].isna().all():
        out["material_class"] = out["material"].astype(str).map(_infer_material_class)

    return out


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.minimum(hi, np.maximum(lo, x)))


def suggest_binder_family(material: str, material_class: str) -> str:
    mc = (material_class or "").lower()
    if mc in ("oxide", "carbide"):
        return "water_based"
    return "solvent_based"


# =============================================================================
# 2) LOAD DATASET ONCE
# =============================================================================

def load_dataset(root: str = ".") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Returns (df, src_path). df has canonical column names.
    The first existing file from DATA_CANDIDATES is used as the source of truth.
    """
    rootp = Path(root)
    for name in DATA_CANDIDATES:
        p = rootp / name
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            df = _rename_and_clean(df)
            # If final_density_state missing: assume rows are green (best-effort)
            if "final_density_state" in df.columns and df["final_density_state"].isna().all():
                df["final_density_state"] = "green"
            return df, str(p)

    # Nothing found — return empty df with canonical columns so app can still render
    cols = sorted(set(CANON.values()))
    return pd.DataFrame(columns=cols), None


# =============================================================================
# 3) PHYSICS PRIORS + GUARDRAILS
# =============================================================================

def physics_priors(d50_um: Optional[float], binder_type_guess: Optional[str]) -> Dict[str, float | str]:
    # Stable layer ~3–5×D50 → choose ~4×; clamp to practical bounds
    t = clamp(4.0 * (d50_um or 100.0), 30.0, 150.0)
    return {
        "layer_thickness_um": float(t),
        "binder_saturation_pct": 80.0,    # %
        "roller_speed_mm_s": 1.6,         # mm/s
        "binder_type_rec": binder_type_guess or "solvent_based",
    }


def guardrail_ranges(d50_um: float, on: bool = True):
    if on:
        return {
            "binder_saturation_pct": (60.0, 110.0),
            "roller_speed_mm_s": (1.2, 3.5),
            "layer_thickness_um": (
                clamp(3.0 * d50_um, 30.0, 150.0),
                clamp(5.0 * d50_um, 30.0, 150.0),
            ),
        }
    else:
        return {
            "binder_saturation_pct": (0.0, 160.0),
            "roller_speed_mm_s": (0.5, 6.0),
            "layer_thickness_um": (
                clamp(2.0 * d50_um, 5.0, 300.0),
                clamp(6.0 * d50_um, 5.0, 300.0),
            ),
        }


# =============================================================================
# 4) MODEL TRAINING (GREEN %TD quantiles)
# =============================================================================

def _preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Predictors only (no target)
    num = [c for c in ["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um"] if c in df.columns]
    cat = [c for c in ["material", "material_class", "binder_type_rec"] if c in df.columns]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )


def train_green_density_models(df: pd.DataFrame):
    """
    Train q10/q50/q90 GradientBoosting models for GREEN-state %TD.
    Returns (models_dict | None, meta_dict).
    """
    if df.empty:
        return None, {"note": "No data found; using physics proxy."}

    # If no state column, treat all rows as green
    if "final_density_state" not in df.columns:
        gdf = df.copy()
        gdf["final_density_state"] = "green"
    else:
        gdf = df.copy()

    # Keep only labeled green rows
    mask = (gdf["final_density_state"].astype(str).str.lower() == "green")
    gdf = gdf[mask & gdf["final_density_pct"].notna()].copy()

    # Sufficiency check
    if len(gdf) < 20 or gdf["material"].nunique() < 3:
        return None, {
            "note": "Insufficient labeled green data for training; using physics proxy.",
            "n_rows": int(len(gdf)),
            "n_materials": int(gdf["material"].nunique()) if len(gdf) else 0,
        }

    X_cols = [c for c in ["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um",
                          "material", "material_class", "binder_type_rec"] if c in gdf.columns]
    y_col = "final_density_pct"

    X = gdf[X_cols].copy()
    y = gdf[y_col].astype(float)

    pre = _preprocessor(gdf)

    def gbr(loss="squared_error", alpha=None, rs=42):
        kw = dict(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=rs)
        if loss == "quantile":
            return GradientBoostingRegressor(loss="quantile", alpha=alpha, **kw)
        return GradientBoostingRegressor(loss="squared_error", **kw)

    models = {
        "q10": Pipeline([("pre", pre), ("gbr", gbr(loss="quantile", alpha=0.10))]),
        "q50": Pipeline([("pre", pre), ("gbr", gbr(loss="squared_error"))]),  # approx median
        "q90": Pipeline([("pre", pre), ("gbr", gbr(loss="quantile", alpha=0.90))]),
    }
    for k in models:
        models[k].fit(X, y)

    meta = {
        "X_cols": X_cols,
        "y_col": y_col,
        "n_rows": int(len(gdf)),
        "n_materials": int(gdf["material"].nunique()),
        "note": "Trained green-density quantile models.",
    }
    return models, meta


def predict_quantiles(models, df_points: pd.DataFrame) -> pd.DataFrame:
    """
    Append td_q10/td_q50/td_q90 to df_points.
    If models is None, return physics-proxy predictions (conservative).
    """
    out = df_points.copy()

    if models is None:
        # Physics-inspired proxy: bell vs saturation; mild speed/layer penalties around priors
        sat = np.clip(out["binder_saturation_pct"].to_numpy(dtype=float) / 100.0, 0, 2)
        spd = out["roller_speed_mm_s"].to_numpy(dtype=float)
        d50 = out["d50_um"].to_numpy(dtype=float)
        layer = out["layer_thickness_um"].to_numpy(dtype=float)

        td_base = 86.0
        td_sat  = -220.0 * (sat - 0.80) ** 2 + 12.0
        td_spd  = -18.0  * (spd - 1.6) ** 2 + 2.0
        ratio   = layer / np.clip(4.0 * d50, 1e-6, None)
        td_lay  = -25.0  * (ratio - 1.0) ** 2 + 3.0

        q50 = np.clip(td_base + td_sat + td_spd + td_lay, 55.0, 98.0)
        band = 3.0 + 1.5 * np.abs(sat - 0.80)  # wider uncertainty at extremes
        out["td_q50"] = q50
        out["td_q10"] = np.clip(q50 - band, 55.0, 98.0)
        out["td_q90"] = np.clip(q50 + band, 55.0, 98.0)
        return out

    feats = [c for c in ["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um",
                         "d50_um", "material", "material_class", "binder_type_rec"] if c in out.columns]
    X = out[feats].copy()

    out["td_q10"] = np.clip(models["q10"].predict(X), 0.0, 100.0)
    out["td_q50"] = np.clip(models["q50"].predict(X), 0.0, 100.0)
    out["td_q90"] = np.clip(models["q90"].predict(X), 0.0, 100.0)
    return out


# =============================================================================
# 5) RECOMMENDER (top-k parameters)
# =============================================================================

def _candidate_grid(d50_um: float, guardrails_on: bool) -> pd.DataFrame:
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    b_lo, b_hi = gr["binder_saturation_pct"]
    s_lo, s_hi = gr["roller_speed_mm_s"]
    t_lo, t_hi = gr["layer_thickness_um"]

    binder_vals = np.linspace(b_lo, b_hi, 21)
    speed_vals  = np.linspace(s_lo, s_hi, 17)
    layer_vals  = np.linspace(t_lo, t_hi, 9)

    grid = pd.DataFrame(
        [(b, s, t) for b in binder_vals for s in speed_vals for t in layer_vals],
        columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um"],
    )
    grid["d50_um"] = float(d50_um)
    return grid


def copilot(
    material: str,
    d50_um: float,
    df_source: pd.DataFrame,
    models=None,
    guardrails_on: bool = True,
    target_green: float = 90.0,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return ranked top-k parameter sets targeted at green %TD."""
    # Material class + binder guess
    mc = None
    if "material" in df_source.columns and "material_class" in df_source.columns:
        row = df_source[df_source["material"].astype(str).str.lower() == str(material).lower()]
        if not row.empty:
            mc = row["material_class"].dropna().astype(str).iloc[0]
    mc = mc or _infer_material_class(material)
    binder_guess = suggest_binder_family(material, mc)

    # Build candidate grid
    cand = _candidate_grid(d50_um, guardrails_on)
    cand["material"] = material
    cand["material_class"] = mc
    cand["binder_type_rec"] = binder_guess

    # Score with quantiles
    scored = predict_quantiles(models, cand)

    # Conservative feasibility: prefer q10 ≥ target
    scored["meets_target_q10"] = (scored["td_q10"] >= float(target_green))

    # Ranking key: meets_target first, then q50, then mild regularizer (prefer ~80% binder, ~1.6 mm/s)
    reg = 0.1 * np.abs(scored["binder_saturation_pct"] - 80.0) + 0.2 * np.abs(scored["roller_speed_mm_s"] - 1.6)
    scored["_key"] = (
        scored["meets_target_q10"].astype(int) * 1_000_000
        + (scored["td_q50"] * 1_000).astype(int)
        - (reg * 10).astype(int)
    )
    top = scored.sort_values("_key", ascending=False).head(int(top_k)).copy()

    # Pretty output for UI
    top["binder_type"] = top["binder_type_rec"]
    top = top.rename(columns={
        "binder_saturation_pct": "binder_%",
        "roller_speed_mm_s": "speed_mm_s",
        "layer_thickness_um": "layer_um",
        "td_q50": "predicted_%TD_q50",
        "td_q10": "predicted_%TD_q10",
        "td_q90": "predicted_%TD_q90",
    })[[
        "binder_type", "binder_%", "speed_mm_s", "layer_um",
        "predicted_%TD_q50", "predicted_%TD_q10", "predicted_%TD_q90", "meets_target_q10"
    ]]

    for c in ["binder_%", "speed_mm_s", "layer_um", "predicted_%TD_q50", "predicted_%TD_q10", "predicted_%TD_q90"]:
        top[c] = top[c].astype(float).round(2)

    return top.reset_index(drop=True)
