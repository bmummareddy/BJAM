import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import trimesh

# ---------------------------
# Data & simple rule models
# ---------------------------

def load_training(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning / standard columns we rely on; add soft guards
    for col in ["Material","Powder","D50_um","BinderType","BinderSat_%","Roller_mm_s","Sinter_T_C","AchievedRelDensity_%"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def water_or_solvent_defaults(material:str, d50_um:float, target_rel=0.90):
    """
    Physics-aware heuristics tuned for BJAM:
    - Saturation higher for finer d50 (capillarity), lower for coarser.
    - Roller speed inversely scales with sqrt(d50) to avoid bulldozing & streaks.
    - Sinter temperature is a prior per material family + small d50 tweak.
    Returns two families: water-based and solvent-based.
    """
    # Material priors (very conservative defaults if not in dict)
    material_priors = {
        # metal-like priors (example)
        "17-4PH": {"Ts": 1280, "span": 80},
        "316L":   {"Ts": 1290, "span": 80},
        # ceramic priors
        "Alumina": {"Ts": 1550, "span": 100},
        "SiC":     {"Ts": 2100, "span": 100},
        "Zirconia":{"Ts": 1450, "span": 100}
    }
    prior = material_priors.get(material, {"Ts": 1400, "span": 100})

    # Saturation model (in % of pore volume)
    # baseline maps d50 → sat% ~ a + b / sqrt(d50)
    a_w, b_w = 40.0, 70.0
    a_s, b_s = 35.0, 55.0
    sat_w = clamp(a_w + b_w / max(5.0, np.sqrt(d50_um)), 35, 85)
    sat_s = clamp(a_s + b_s / max(5.0, np.sqrt(d50_um)), 30, 80)

    # Roller mm/s (inverse with sqrt d50, but guard)
    base = 120.0
    vr = clamp(base / max(2.0, np.sqrt(d50_um)), 15, 180)

    # Sinter T by prior ± tweak for fine/coarse
    tweak = -25 if d50_um < 20 else (25 if d50_um > 60 else 0)
    Ts_nom = prior["Ts"] + tweak
    Ts_lo, Ts_hi = Ts_nom - prior["span"]//2, Ts_nom + prior["span"]//2

    # Build 3 water, 2 solvent recipes
    water = [
        {"BinderType":"Water-PVOH","BinderSat_%": clamp(sat_w-6, 35, 85), "Roller_mm_s": vr*0.9, "Sinter_T_C": Ts_lo},
        {"BinderType":"Water-PVOH","BinderSat_%": sat_w,                  "Roller_mm_s": vr,      "Sinter_T_C": Ts_nom},
        {"BinderType":"Water-PAA", "BinderSat_%": clamp(sat_w+5, 35, 85), "Roller_mm_s": vr*0.8,  "Sinter_T_C": Ts_hi}
    ]
    solvent = [
        {"BinderType":"Solvent-PEG","BinderSat_%": clamp(sat_s-5, 30, 80), "Roller_mm_s": vr*1.05, "Sinter_T_C": Ts_nom},
        {"BinderType":"Solvent-PVB","BinderSat_%": clamp(sat_s+3, 30, 80), "Roller_mm_s": vr,      "Sinter_T_C": Ts_hi}
    ]
    return water, solvent

def score_recipe(d50_um, binder_sat, roller, t_sint, target_rel=0.90):
    """
    Simple multi-penalty score → higher is better.
    Encourages binder_sat in [38..78], roller in [25..140], and moderate sinter.
    """
    penalties = 0.0
    penalties += 0.5 * max(0, 38 - binder_sat) + 0.3 * max(0, binder_sat - 78)
    penalties += 0.02 * max(0, 25 - roller) + 0.02 * max(0, roller - 140)
    penalties += 0.01 * abs(t_sint)  # normalize
    base = 100 - penalties
    return base

def generate_five_recipes(material:str, d50_um:float, target_rel=0.90):
    water, solvent = water_or_solvent_defaults(material, d50_um, target_rel)
    # score and sort inside each family, keep 3 water + 2 solvent
    for r in water:
        r["Score"] = score_recipe(d50_um, r["BinderSat_%"], r["Roller_mm_s"], r["Sinter_T_C"], target_rel)
        r["Family"] = "Water"
    for r in solvent:
        r["Score"] = score_recipe(d50_um, r["BinderSat_%"], r["Roller_mm_s"], r["Sinter_T_C"], target_rel)
        r["Family"] = "Solvent"
    water = sorted(water, key=lambda x: -x["Score"])[:3]
    solvent = sorted(solvent, key=lambda x: -x["Score"])[:2]
    recipes = water + solvent
    # add IDs
    for i, r in enumerate(recipes, 1):
        r["ID"] = f"Trial {i} ({r['Family']})"
    return recipes

# ---------------------------
# Heatmaps & saturation maps
# ---------------------------

def make_parameter_grid(material:str, d50_um:float, n=40):
    """
    Construct a clean grid of BinderSat vs Roller speed with a smooth 'fitness' (proxy for 90%+ densification).
    """
    sat = np.linspace(30, 85, n)
    vr  = np.linspace(20, 160, n)
    S, V = np.meshgrid(sat, vr)

    # Center optimum from heuristics
    water, solvent = water_or_solvent_defaults(material, d50_um)
    # nominal best guess
    best = ( (water[1]["BinderSat_%"] + solvent[1]["BinderSat_%"])/2.0,
             (water[1]["Roller_mm_s"]  + solvent[1]["Roller_mm_s"])/2.0 )
    s0, v0 = best
    # Gaussian-like response
    Z = np.exp(-((S - s0)**2/(2*8.0**2) + (V - v0)**2/(2*25.0**2)))
    return S, V, Z

def saturation_curve(material:str, d50_um:float, family="Water", x=None):
    if x is None:
        x = np.linspace(5, 120, 300)
    water, solvent = water_or_solvent_defaults(material, d50_um)
    ref = water[1] if family=="Water" else solvent[1]
    # Map d50 to recommended saturation band
    rec = ref["BinderSat_%"]
    y = rec + 6.0*np.exp(-(x-d50_um)**2/(2*18.0**2)) - 3.0
    return x, y

# ---------------------------
# Qualitative 2D packing
# ---------------------------

def poisson_disk_sampling(width, height, r, k=30, seed=0):
    """
    Bridson Poisson-disk sampling for minimum spacing r.
    returns Nx2 array of (x,y)
    """
    rng = np.random.default_rng(seed)
    grid_r = r/np.sqrt(2)
    cols, rows = int(width/grid_r)+1, int(height/grid_r)+1
    grid = -np.ones((rows, cols), dtype=int)

    def grid_coords(p):
        return int(p[1]//grid_r), int(p[0]//grid_r)

    def fits(p):
        gy, gx = grid_coords(p)
        y0, y1 = max(gy-2,0), min(gy+3, rows)
        x0, x1 = max(gx-2,0), min(gx+3, cols)
        for yy in range(y0,y1):
            for xx in range(x0,x1):
                i = grid[yy,xx]
                if i!=-1:
                    if np.linalg.norm(p - samples[i]) < r:
                        return False
        return True

    # init
    p0 = np.array([rng.random()*width, rng.random()*height])
    samples = [p0]
    active = [0]
    gy,gx = grid_coords(p0)
    grid[gy,gx] = 0

    while active:
        idx = rng.choice(active)
        found = False
        for _ in range(k):
            ang = rng.random()*2*np.pi
            rad = r*(1+rng.random())
            p = samples[idx] + rad*np.array([np.cos(ang), np.sin(ang)])
            if 0<=p[0]<width and 0<=p[1]<height and fits(p):
                samples.append(p)
                active.append(len(samples)-1)
                gy,gx = grid_coords(p)
                grid[gy,gx] = len(samples)-1
                found = True
                break
        if not found:
            active.remove(idx)
    return np.array(samples)

def synth_layer_particles(px=512, py=256, d50_um=35.0, layer_thickness_um=50.0, seed=0):
    """
    Create a 2D layer with realistic particle sizes (lognormal) and spacing (Poisson disk).
    Pixel scale is arbitrary; we scale radii to preserve relative size & porosity.
    """
    rng = np.random.default_rng(seed)
    # Map microns to pixels via a heuristic scale (more pixels for finer powders to see voids)
    um_per_px = clamp(0.15*d50_um, 1.5, 12.0)  # coarse → larger px conversion to keep reasonable particle count
    mean_r_px = (0.5*d50_um)/um_per_px
    # enforce minimum visible size
    mean_r_px = max(1.8, mean_r_px)

    # Set Poisson radius to avoid overlap
    r_min = max(1.2, 0.9*mean_r_px)
    pts = poisson_disk_sampling(px, py, r=r_min, seed=seed)

    # Assign radii via lognormal around mean
    s = 0.25  # spread
    radii = rng.lognormal(mean=np.log(mean_r_px), sigma=s, size=len(pts))
    radii = np.clip(radii, 0.6*mean_r_px, 1.8*mean_r_px)

    return pts, radii

# ---------------------------
# STL → voxel slices
# ---------------------------

def stl_to_binary_slices(stl_bytes:bytes, voxel_pitch:float=0.4, max_dim_vox=220):
    """
    Voxelize STL into a boolean volume and return axial slices (list of 2D numpy arrays).
    Uses trimesh voxelization; avoids VTK/pyvista for Streamlit Cloud stability.
    """
    mesh = trimesh.load(stl_bytes, file_type='stl')
    if not isinstance(mesh, trimesh.Trimesh):
        # scene → merge
        mesh = mesh.dump().sum()

    # Normalize size to keep voxel grid manageable on Streamlit
    extents = mesh.extents
    scale = max(extents) / max_dim_vox
    if scale > voxel_pitch:
        factor = (max_dim_vox * voxel_pitch) / max(extents)
        mesh.apply_scale(factor)

    # Voxelization
    vox = mesh.voxelized(pitch=voxel_pitch)
    vol = vox.matrix  # (Z,Y,X) boolean

    # Convert to slices along build (Z)
    slices = [vol[z, :, :].astype(np.uint8) for z in range(vol.shape[0])]
    return slices
