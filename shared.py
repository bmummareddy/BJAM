# -*- coding: utf-8 -*-
"""
shared.py — data/model utilities for BJAM app.

Provides:
- load_dataset(data_dir=".") -> (DataFrame, path)
- train_green_density_models(df) -> (models_dict, meta)
- predict_quantiles(models, X_df) -> DataFrame with td_q10/td_q50/td_q90
- guardrail_ranges(d50_um, on=True) -> dict of parameter ranges
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Canonical feature names expected everywhere else in the app
FEATURE_SAT   = "binder_saturation_pct"   # %
FEATURE_SPEED = "roller_speed_mm_s"       # mm/s
FEATURE_D50   = "d50_um"                  # µm
TARGET_Y      = "green_pct_td"            # % theoretical density (green)
FEATURES      = [FEATURE_SAT, FEATURE_SPEED, FEATURE_D50]

CSV_NAME = "BJAM_All_Deep_Fill_v9.csv"

# --------- Column header normalization (common synonyms from literature/app drafts)
RENAME_MAP = {
    "binder_saturation": FEATURE_SAT,
    "binder_sat_pct": FEATURE_SAT,
    "saturation_pct": FEATURE_SAT,

    "roller_speed": FEATURE_SPEED,
    "roller_speed_mmps": FEATURE_SPEED,
    "spreader_speed": FEATURE_SPEED,

    "D50": FEATURE_D50,
    "D50_um": FEATURE_D50,
    "d50": FEATURE_D50,

    "green_density": TARGET_Y,
    "green_%TD": TARGET_Y,
    "green_%_TD": TARGET_Y,
    "green_density_pct": TARGET_Y,
    "green_density_pctTD": TARGET_Y,

    "binder": "binder_type",
    "material_name": "material",
}


# --------- Public API

@st.cache_data(show_spinner=False)
def load_dataset(data_dir: str = ".") -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Find and load BJAM csv; standardize columns; coerce numerics; impute per-material medians.
    Returns (df, absolute_path_or_None).
    """
    # Resolve path
    candidates = [
        os.path.join(data_dir, CSV_NAME),
        CSV_NAME,
        os.path.join(data_dir, "data", CSV_NAME),
    ]
    csv_path = None
    for p in candidates:
        if os.path.isfile(p):
            csv_path = p
            break

    if csv_path is None:
        # Create a tiny fallback DF so app still launches
        df = pd.DataFrame({
            "material": ["Generic"]*10,
            FEATURE_D50: np.linspace(20, 120, 10),
            FEATURE_SAT: np.linspace(70, 100, 10),
            FEATURE_SPEED: np.linspace(1.6, 2.8, 10),
            TARGET_Y: np.linspace(82, 95, 10),
        })
        return df, None

    df = pd.read_csv(csv_path)

    # Normalize headers
    for k, v in RENAME_MAP.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Ensure required columns exist
    if "material" not in df.columns:
        df["material"] = "Generic"

    for col in FEATURES + [TARGET_Y]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numerics
    for col in FEATURES + [TARGET_Y]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute per-material medians; fallback to global
    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        med = g[FEATURES + [TARGET_Y]].median(numeric_only=True)
        g[FEATURES + [TARGET_Y]] = g[FEATURES + [TARGET_Y]].fillna(med)
        return g

    df = df.groupby(df["material"].fillna("Generic"), dropna=False, as_index=False).apply(_impute_group)
    df[FEATURES + [TARGET_Y]] = df[FEATURES + [TARGET_Y]].fillna(df[FEATURES + [TARGET_Y]].median(numeric_only=True))
    df = df.dropna(subset=FEATURES + [TARGET_Y]).reset_index(drop=True)

    # Light physical clipping
    df[FEATURE_SAT]   = df[FEATURE_SAT].clip(40, 120)
    df[FEATURE_SPEED] = df[FEATURE_SPEED].clip(0.4, 12.0)
    df[FEATURE_D50]   = df[FEATURE_D50].clip(0.5, 500)
    df[TARGET_Y]      = df[TARGET_Y].clip(0, 100)

    return df, os.path.abspath(csv_path)


def _make_quantile(alpha: float) -> Pipeline:
    # Use sklearn's quantile-capable GBR for q!=0.5; absolute_error at 0.5
    loss = "absolute_error" if abs(alpha - 0.5) < 1e-9 else "quantile"
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("gbr", GradientBoostingRegressor(
            loss=loss, alpha=alpha,
            max_depth=3, n_estimators=350, learning_rate=0.05,
            random_state=42))
    ])


@st.cache_data(show_spinner=False)
def train_green_density_models(df: pd.DataFrame):
    """
    Fit q10/q50/q90 models per-material (one model set for all, using 'material' as a categorical dummy).
    For simplicity we train a single set on all rows; material influences through the data distribution.
    """
    X = df[FEATURES].values
    y = df[TARGET_Y].values
    # Bulletproofing: if any residual NaNs in y, replace by global median
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmedian(y), y)

    model_q10 = _make_quantile(0.10).fit(X, y)
    model_q50 = _make_quantile(0.50).fit(X, y)
    model_q90 = _make_quantile(0.90).fit(X, y)

    models = {"q10": model_q10, "q50": model_q50, "q90": model_q90}
    meta = {"features": FEATURES}
    return models, meta


def predict_quantiles(models: Dict[str, Pipeline], X_df: pd.DataFrame) -> pd.DataFrame:
    """
    X_df must contain FEATURES and may include extra columns; returns the same rows with td_q10/td_q50/td_q90.
    """
    cols = FEATURES
    X = X_df[cols].astype(float).values
    td_q10 = models["q10"].predict(X)
    td_q50 = models["q50"].predict(X)
    td_q90 = models["q90"].predict(X)
    out = X_df.copy()
    out["td_q10"] = td_q10
    out["td_q50"] = td_q50
    out["td_q90"] = td_q90
    return out


def guardrail_ranges(d50_um: float, on: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Simple, physics-guided windows that scale with D50.
    - Saturation narrower for water-based inks near mid D50; broader when off.
    - Roller speed widens gently with coarser powders.
    """
    if not on:
        return {
            "binder_saturation_pct": (60.0, 110.0),
            "roller_speed_mm_s": (1.0, 4.0)
        }

    # Saturation window
    sat_center = 86.0 + 0.04 * (d50_um - 50.0)   # small drift with size
    sat_half   = 10.0
    sat_lo, sat_hi = sat_center - sat_half, sat_center + sat_half
    sat_lo, sat_hi = max(60.0, sat_lo), min(110.0, sat_hi)

    # Speed window (mm/s)
    v_center = 2.2 + 0.003 * max(0.0, d50_um - 50.0)
    v_half   = 0.8
    v_lo, v_hi = max(0.6, v_center - v_half), min(4.0, v_center + v_half)

    return {
        "binder_saturation_pct": (float(sat_lo), float(sat_hi)),
        "roller_speed_mm_s": (float(v_lo), float(v_hi)),
    }
