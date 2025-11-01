# shared.py - DATA + MODELS + RECOMMENDER for BJAM app
# Uses BJAM_All_Deep_Fill_v9.csv as the primary dataset for optimization.
# Falls back to physics + guardrails if data is insufficient.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# 0) CONFIG - Single source of truth for the dataset
# =============================================================================

# Allow override via env var, but default to your requested file.
DATA_CSV = os.environ.get("BJAM_DATA", "BJAM_All_Deep_Fill_v9.csv")

# Fallbacks in case a different file is present in the repo root
DATA_CANDIDATES = [
    DATA_CSV,
    "BJAM_All_Deep_Fill_v9.csv",
    "BJAM_v10_clean.csv",
    "BJAM_v9_clean_v2.csv",
    "BJAM_v9_clean.csv",
]

# Canonical column names used by the app/models
CANON = {
    "material": "material",
    "material_class": "material_class",
    "d50_um": "d50_um",
    "layer_thickness_um": "layer_thickness_um",
    "layer_um": "layer_thickness_um",  # alias → canonical
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
    "binder_saturation_pct", "final_density_pct"
]

CATEGORICAL_COLS = [
    "material", "material_class", "binder_type_rec"
]


# =============================================================================
# 1) UTILITIES
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value x between lo and hi."""
    return float(np.clip(x, lo, hi))


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names and ensure expected columns exist.
    Handles case-insensitive matching and creates missing columns.
    """
    # Map any known aliases to canonical names (case-insensitive match)
    mapping = {}
    lower_to_canon = {k.lower(): v for k, v in CANON.items()}
    
    for c in df.columns:
        key = c.strip().lower()
        mapping[c] = lower_to_canon.get(key, c)
    
    out = df.rename(columns=mapping)

    # Ensure expected columns exist (create empty if missing)
    for col in set(CANON.values()):
        if col not in out.columns:
            out[col] = np.nan

    # Numeric coercion
    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Normalize density state (green/sintered)
    if "final_density_state" in out.columns:
        out["final_density_state"] = (
            out["final_density_state"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # Simple binder-type normalization
    if "binder_type_rec" in out.columns:
        out["binder_type_rec"] = (
            out["binder_type_rec"]
            .astype(str)
            .str.strip()
            .str.replace(" ", "_")
            .str.lower()
        )

    # Material class best-effort inference if missing
    if out["material_class"].isna().all():
        out["material_class"] = (
            out["material"]
            .astype(str)
            .str.lower()
            .map(_infer_material_class)
        )

    return out


def _infer_material_class(name: str) -> str:
    """
    Infer material class from material name.
    Returns: 'metal', 'oxide', 'carbide', or 'other'
    """
    n = (name or "").lower()
    
    if any(k in n for k in ["316l", "inconel", "17-4", "steel", "copper", "ti", "al "]):
        return "metal"
    
    if any(k in n for k in ["al2o3", "alumina", "oxide", "zirconia", "zro2"]):
        return "oxide"
    
    if any(k in n for k in ["wc", "carbide", "sic", "tib2"]):
        return "carbide"
    
    return "other"


def suggest_binder_family(material: str, material_class: Optional[str]) -> str:
    """
    Suggest binder family based on material class.
    Returns: 'water_based' or 'solvent_based'
    """
    mc = (material_class or "").lower()
    
    if mc in ("oxide", "carbide"):
        return "water_based"
    
    # default for metals & other
    return "solvent_based"


# =============================================================================
# 2) DATA LOADING
# =============================================================================

def load_dataset(root: str = ".") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load the BJAM dataset from the first available file in DATA_CANDIDATES.
    Uses data_preprocessing module for BJAM_All_Deep_Fill_v9.csv format.
    
    Args:
        root: Root directory to search for data files
        
    Returns:
        (df, src_path): DataFrame with canonical columns and path to source file
    """
    rootp = Path(root)
    
    # Try to import preprocessing module
    try:
        from data_preprocessing import load_and_clean_bjam_data
        use_preprocessing = True
    except ImportError:
        use_preprocessing = False
    
    for name in DATA_CANDIDATES:
        p = rootp / name
        
        if p.exists():
            try:
                # Use specialized preprocessing for BJAM files
                if use_preprocessing and "BJAM" in name:
                    df, _ = load_and_clean_bjam_data(str(p))
                    if len(df) > 0:
                        return df, str(p)
                
                # Fallback to standard CSV loading
                df = pd.read_csv(p)
                df = _rename_columns(df)
                
                if len(df) > 0:
                    return df, str(p)
            except Exception as e:
                # If a bad CSV slips in, continue to next candidate
                print(f"Warning: Could not load {p}: {e}")
                continue
    
    # None found - return empty dataframe with canonical columns
    return pd.DataFrame(columns=list(set(CANON.values()))), None


# =============================================================================
# 3) PHYSICS PRIORS & GUARDRAILS
# =============================================================================

def physics_priors(
    d50_um: Optional[float], 
    binder_type_guess: Optional[str]
) -> Dict[str, float | str]:
    """
    Generate physics-based prior values for process parameters.
    
    Args:
        d50_um: Median particle size in microns
        binder_type_guess: Suggested binder type (optional)
        
    Returns:
        Dictionary with layer_thickness_um, binder_saturation_pct, 
        roller_speed_mm_s, and binder_type_rec
    """
    # Stable layer ~ 3-5×D50 → pick 4×
    if pd.notna(d50_um) and d50_um is not None:
        t = clamp(4.0 * d50_um, 30.0, 150.0)
    else:
        t = 100.0
    
    sat = 80.0   # % of pore volume
    spd = 1.6    # mm/s baseline for smooth spreading
    binder = binder_type_guess or "solvent_based"
    
    return {
        "layer_thickness_um": float(t),
        "binder_saturation_pct": float(sat),
        "roller_speed_mm_s": float(spd),
        "binder_type_rec": binder,
    }


def guardrail_ranges(d50_um: float, on: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Define parameter ranges with or without guardrails.
    
    Args:
        d50_um: Median particle size in microns
        on: Whether to apply strict guardrails
        
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    if on:
        # Strict, empirically-stable ranges
        return {
            "binder_saturation_pct": (60.0, 110.0),
            "roller_speed_mm_s": (1.2, 3.5),
            "layer_thickness_um": (
                clamp(3.0 * d50_um, 30.0, 150.0),
                clamp(5.0 * d50_um, 30.0, 150.0)
            )
        }
    else:
        # Wider exploration ranges
        return {
            "binder_saturation_pct": (0.0, 160.0),
            "roller_speed_mm_s": (0.5, 6.0),
            "layer_thickness_um": (
                clamp(2.0 * d50_um, 5.0, 300.0),
                clamp(6.0 * d50_um, 5.0, 300.0)
            )
        }


# =============================================================================
# 4) MODEL TRAINING - GREEN %TD quantile models
# =============================================================================

def _preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create a preprocessor for numeric and categorical features.
    """
    num_cols = [c for c in NUMERIC_COLS if c != "final_density_pct"]  # predictors only
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )


def train_green_density_models(
    df: pd.DataFrame
) -> Tuple[Optional[Dict[str, Pipeline]], Dict[str, Any]]:
    """
    Train 3 GradientBoostingRegressor models to estimate q10, q50, q90
    for green-state density (final_density_pct where final_density_state == 'green').
    
    Args:
        df: Input dataframe with process parameters and densities
        
    Returns:
        (models_dict, meta_dict): Models for q10/q50/q90 and metadata.
        If data insufficient, returns (None, {'note': ...}).
    """
    needed = {
        "final_density_state", "final_density_pct",
        "binder_saturation_pct", "roller_speed_mm_s",
        "layer_thickness_um", "d50_um", "material", "binder_type_rec"
    }
    
    if not needed.issubset(df.columns):
        return None, {"note": "Dataset missing required columns for model training."}

    # Filter to green-state data with valid density values
    gdf = df[
        (df["final_density_state"] == "green") & 
        df["final_density_pct"].notna()
    ].copy()
    
    # Basic sufficiency checks
    if gdf.empty or len(gdf) < 20:
        return None, {
            "note": "Insufficient green-density labels for training.",
            "n_rows": int(len(gdf)),
            "n_materials": 0
        }
    
    if gdf["material"].nunique() < 3:
        return None, {
            "note": "Insufficient material diversity for training.",
            "n_rows": int(len(gdf)),
            "n_materials": int(gdf["material"].nunique())
        }

    # Define features and target
    X_cols = [
        "binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", 
        "d50_um", "material", "material_class", "binder_type_rec"
    ]
    X_cols = [c for c in X_cols if c in gdf.columns]
    y_col = "final_density_pct"

    X = gdf[X_cols].copy()
    y = gdf[y_col].astype(float)

    # Create preprocessor
    pre = _preprocessor(gdf)

    def create_gbr(loss: str = "squared_error", alpha: Optional[float] = None):
        """Helper to create GradientBoostingRegressor with consistent params."""
        kw = dict(
            n_estimators=300, 
            max_depth=3, 
            learning_rate=0.05, 
            random_state=42
        )
        
        if loss == "quantile" and alpha is not None:
            return GradientBoostingRegressor(loss="quantile", alpha=alpha, **kw)
        
        return GradientBoostingRegressor(loss="squared_error", **kw)

    # Train three models for different quantiles
    models = {
        "q10": Pipeline([("pre", pre), ("gbr", create_gbr(loss="quantile", alpha=0.10))]),
        "q50": Pipeline([("pre", pre), ("gbr", create_gbr(loss="squared_error"))]),
        "q90": Pipeline([("pre", pre), ("gbr", create_gbr(loss="quantile", alpha=0.90))]),
    }

    for k in models:
        models[k].fit(X, y)

    meta = {
        "X_cols": X_cols,
        "y_col": y_col,
        "n_rows": int(len(gdf)),
        "materials": sorted(gdf["material"].astype(str).unique().tolist()),
        "note": "Trained green-density quantile models."
    }
    
    return models, meta


def predict_quantiles(
    models: Optional[Dict[str, Pipeline]], 
    df_points: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict quantiles (q10, q50, q90) for given process parameters.
    
    Args:
        models: Dictionary of trained models (or None for physics-proxy)
        df_points: DataFrame with canonical feature columns
        
    Returns:
        df_points with added columns: td_q10, td_q50, td_q90 (clipped 0..100)
    """
    out = df_points.copy()

    # If no models, provide a simple, conservative physics-proxy
    if models is None:
        # Heuristic: bell shape vs saturation; mild penalty for speed away from 1.6; 
        # layer near 4×D50
        sat = np.clip(out["binder_saturation_pct"].to_numpy(dtype=float) / 100.0, 0, 2)
        spd = out["roller_speed_mm_s"].to_numpy(dtype=float)
        d50 = out["d50_um"].to_numpy(dtype=float)
        layer = out["layer_thickness_um"].to_numpy(dtype=float)

        # Base density
        td_base = 86.0
        
        # Saturation effect (parabolic with peak around 80%)
        td_sat = -220.0 * (sat - 0.80)**2 + 12.0
        
        # Speed effect (penalty away from 1.6 mm/s)
        td_spd = -18.0 * (spd - 1.6)**2 + 2.0
        
        # Layer effect (penalty away from 4×D50)
        ratio = layer / np.clip(4.0 * d50, 1e-6, None)
        td_layer = -25.0 * (ratio - 1.0)**2 + 3.0

        # Combined prediction
        td50 = np.clip(td_base + td_sat + td_spd + td_layer, 55.0, 98.0)
        
        # Uncertainty band (wider at extremes)
        band = 3.0 + 1.5 * np.abs(sat - 0.80)
        
        out["td_q50"] = td50
        out["td_q10"] = np.clip(td50 - band, 55.0, 98.0)
        out["td_q90"] = np.clip(td50 + band, 55.0, 98.0)
        
        return out

    # With trained models
    feats = [
        "binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um",
        "d50_um", "material", "material_class", "binder_type_rec"
    ]
    feats = [c for c in feats if c in out.columns]
    X = out[feats].copy()

    out["td_q10"] = np.clip(models["q10"].predict(X), 0.0, 100.0)
    out["td_q50"] = np.clip(models["q50"].predict(X), 0.0, 100.0)
    out["td_q90"] = np.clip(models["q90"].predict(X), 0.0, 100.0)
    
    return out


# =============================================================================
# 5) RECOMMENDER ("copilot")
# =============================================================================

def _candidate_grid(d50_um: float, guardrails_on: bool) -> pd.DataFrame:
    """
    Generate a grid of candidate process parameters.
    
    Args:
        d50_um: Median particle size
        guardrails_on: Whether to use strict ranges
        
    Returns:
        DataFrame with candidate parameter combinations
    """
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    b_lo, b_hi = gr["binder_saturation_pct"]
    s_lo, s_hi = gr["roller_speed_mm_s"]
    t_lo, t_hi = gr["layer_thickness_um"]

    # Modest grid for responsiveness; adjust step for density
    binder_vals = np.linspace(b_lo, b_hi, 21)
    speed_vals = np.linspace(s_lo, s_hi, 17)
    layer_vals = np.linspace(t_lo, t_hi, 9)

    grid = pd.DataFrame(
        [(b, s, t) for b in binder_vals for s in speed_vals for t in layer_vals],
        columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um"]
    )
    
    grid["d50_um"] = d50_um
    
    return grid


def copilot(
    material: str,
    d50_um: float,
    df_source: pd.DataFrame,
    models: Optional[Dict[str, Pipeline]] = None,
    guardrails_on: bool = True,
    target_green: float = 90.0,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend top-k parameter sets meeting/approaching target_green %TD.
    
    Args:
        material: Material name
        d50_um: Median particle size
        df_source: Source dataset
        models: Trained quantile models (optional)
        guardrails_on: Whether to apply strict guardrails
        target_green: Target green density percentage
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with top-k recommended parameter sets
    """
    # Choose binder family guess
    mc = None
    if "material_class" in df_source.columns and not pd.isna(material):
        try:
            material_matches = df_source[
                df_source["material"].astype(str).str.lower() == str(material).lower()
            ]
            if len(material_matches) > 0:
                mc = material_matches["material_class"].dropna().iloc[0]
        except Exception:
            mc = None
    
    binder_guess = suggest_binder_family(material, mc or "metal")

    # Build candidate grid
    cand = _candidate_grid(d50_um, guardrails_on)
    cand["material"] = material
    cand["material_class"] = mc or _infer_material_class(material)
    cand["binder_type_rec"] = binder_guess

    # Predict q10/q50/q90
    scored = predict_quantiles(models, cand)

    # Feasibility: prefer q10 >= target to be conservative (~90% chance true ≥ target)
    scored["meets_target_q10"] = (scored["td_q10"] >= target_green)

    # Ranking: primary by meets_target_q10, then q50 descending, 
    # then regularize (prefer 80% binder, 1.6 mm/s)
    reg = (
        0.1 * np.abs(scored["binder_saturation_pct"] - 80.0) + 
        0.2 * np.abs(scored["roller_speed_mm_s"] - 1.6)
    )
    
    scored["_rank_key"] = (
        scored["meets_target_q10"].astype(int) * 1_000_000 +
        (scored["td_q50"] * 1_000).astype(int) -
        (reg * 10).astype(int)
    )
    
    out = scored.sort_values("_rank_key", ascending=False).head(top_k).copy()

    # Pretty columns for the UI table
    out["binder_type"] = out["binder_type_rec"]
    out = out[[
        "binder_type", "binder_saturation_pct", "roller_speed_mm_s", 
        "layer_thickness_um", "td_q50", "td_q10", "td_q90", "meets_target_q10"
    ]].rename(columns={
        "binder_saturation_pct": "binder_%",
        "roller_speed_mm_s": "speed_mm_s",
        "layer_thickness_um": "layer_um",
        "td_q50": "predicted_%TD_q50",
        "td_q10": "predicted_%TD_q10",
        "td_q90": "predicted_%TD_q90",
    })

    # Round for display
    for c in ["binder_%", "speed_mm_s", "layer_um", 
              "predicted_%TD_q50", "predicted_%TD_q10", "predicted_%TD_q90"]:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)

    return out.reset_index(drop=True)
