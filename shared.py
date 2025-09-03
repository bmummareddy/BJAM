
import pandas as pd, numpy as np, re, os, json
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

DATA_CANDIDATES = ["BJAM_v10_clean.csv","BJAM_v9_clean.csv","BJAM_v9_clean_v2.csv"]

def load_dataset(root="."):
    for name in DATA_CANDIDATES:
        path = os.path.join(root, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path
    return pd.DataFrame(), None

def binder_prior_from_material(name:str):
    s = (name or "").lower()
    if any(k in s for k in ["alumina","zirconia","silica","oxide","ceram"]): return "water_based"
    if any(k in s for k in ["sic","carbide","graphite"]): return "polymer_based"
    if any(k in s for k in ["inconel","steel","316","625","718","nickel","copper","bronze","iron","aluminum","titanium"]): return "solvent_based"
    return "water_based"

def clamp(v, lo, hi):
    if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
    return float(max(lo, min(hi, v)))

def physics_priors(d50_um: float, binder_type_guess:str):
    t = clamp(4.0*(d50_um or np.nan), 30.0, 150.0) if pd.notna(d50_um) else 100.0
    sat = 80.0
    spd = 2.5
    return {"layer_thickness_um": t, "binder_saturation_pct": sat, "roller_speed_mm_s": spd, "binder_type": binder_type_guess}

def train_green_density_models(df):
    ok = all(c in df.columns for c in ["final_density_state","final_density_pct","d50_um","layer_thickness_um","binder_saturation_pct","roller_speed_mm_s","binder_type_rec","material"])
    if not ok: return None, {"note":"Dataset missing columns for model training."}
    gdf = df[(df["final_density_state"]=="green") & df["final_density_pct"].notna()].copy()
    if gdf.empty or gdf["material"].nunique() < 3 or len(gdf) < 10:
        return None, {"note":"Insufficient green-density labels for robust model.", "n_green": int(len(gdf)), "n_materials": int(gdf["material"].nunique())}
    def pri_row(r):
        d50 = r.get("d50_um")
        bt = r.get("binder_type_rec") if pd.notna(r.get("binder_type_rec")) else "water_based"
        t = np.nan if pd.isna(d50) else clamp(4.0*float(d50),30.0,150.0)
        return pd.Series({
            "layer_thickness_um": r["layer_thickness_um"] if pd.notna(r["layer_thickness_um"]) else t,
            "binder_saturation_pct": r["binder_saturation_pct"] if pd.notna(r["binder_saturation_pct"]) else 80.0,
            "roller_speed_mm_s": r["roller_speed_mm_s"] if pd.notna(r["roller_speed_mm_s"]) else 2.5,
            "binder_type_rec": bt
        })
    fill = gdf.apply(pri_row, axis=1)
    for c in ["layer_thickness_um","binder_saturation_pct","roller_speed_mm_s","binder_type_rec"]:
        gdf[c] = fill[c]
    gdf = gdf.dropna(subset=["d50_um","layer_thickness_um","binder_saturation_pct","roller_speed_mm_s","binder_type_rec"])
    num_cols = ["d50_um","layer_thickness_um","binder_saturation_pct","roller_speed_mm_s"]
    cat_cols = ["binder_type_rec"]
    X = gdf[num_cols + cat_cols]
    y = gdf["final_density_pct"].astype(float)
    pre = ColumnTransformer([("num","passthrough", num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    def mk(alpha): return Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42))])
    models = {"q10": mk(0.10), "q50": mk(0.50), "q90": mk(0.90)}
    for k, m in models.items(): m.fit(X, y)
    return models, {"note":"trained", "n_green": int(len(gdf)), "n_materials": int(gdf["material"].nunique())}

def packing_estimate(material_name:str, d50_um: float):
    s = (material_name or "").lower()
    if any(k in s for k in ["inconel","316","625","718","nickel","steel","copper","bronze","aluminum","titanium"]):
        base = 0.62
    elif any(k in s for k in ["alumina","zirconia","silica","oxide","ceram"]):
        base = 0.58
    elif any(k in s for k in ["sic","carbide","graphite","b4c","wc"]):
        base = 0.55
    else:
        base = 0.58
    if pd.notna(d50_um):
        delta = (d50_um - 40.0)/40.0
        penalty = 0.04 * (delta**2)
        if d50_um < 20: penalty += 0.04
        base = max(0.45, min(0.68, base - penalty))
    return base * 100.0

def copilot(material:str, d50_um: float, df):
    binder_guess = binder_prior_from_material(material)
    pri = physics_priors(d50_um, binder_guess)
    models, info = train_green_density_models(df)
    if models is not None:
        row = pd.DataFrame([{
            "d50_um": d50_um,
            "layer_thickness_um": pri["layer_thickness_um"],
            "binder_saturation_pct": pri["binder_saturation_pct"],
            "roller_speed_mm_s": pri["roller_speed_mm_s"],
            "binder_type_rec": binder_guess
        }])
        q10 = float(models["q10"].predict(row)[0])
        q50 = float(models["q50"].predict(row)[0])
        q90 = float(models["q90"].predict(row)[0])
        gd = {"q10": q10, "q50": q50, "q90": q90, "mode": "ML"}
    else:
        q50 = packing_estimate(material, d50_um)
        gd = {"q10": max(0.0, q50-5.0), "q50": q50, "q90": min(100.0, q50+5.0), "mode":"physics"}
    return {
        "material": material,
        "d50_um": d50_um,
        "binder_type": binder_guess,
        "binder_saturation_pct": pri["binder_saturation_pct"],
        "layer_thickness_um": pri["layer_thickness_um"],
        "roller_speed_mm_s": pri["roller_speed_mm_s"],
        "green_density_pred_q10": gd["q10"],
        "green_density_pred_q50": gd["q50"],
        "green_density_pred_q90": gd["q90"],
        "green_density_mode": gd["mode"],
    }

def propose_abc(material, d50_um, binder_type, center_t_um):
    def clamp(v, lo, hi): return float(max(lo, min(hi, v)))
    def pack(lbl, sat, t, spd):
        return {"recipe_label": lbl, "binder_saturation_pct": sat, "layer_thickness_um": t, "roller_speed_mm_s": spd,
                "material": material, "d50_um": d50_um, "binder_type": binder_type}
    sat0, spd0 = 80.0, 2.5
    A = pack("A", clamp(sat0-10,60,110), clamp(center_t_um-10,30,150), clamp(spd0-0.5,1.5,3.5))
    B = pack("B", sat0, center_t_um, spd0)
    C = pack("C", clamp(sat0+10,60,110), clamp(center_t_um+10,30,150), clamp(spd0+0.5,1.5,3.5))
    return [A,B,C]
