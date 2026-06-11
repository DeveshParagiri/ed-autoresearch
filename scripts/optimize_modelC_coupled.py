"""
optimize_modelC_coupled.py
Retune Model C against GFED5, hybrid drivers:
  - GPP from Lei's coupled ED dump (area-weighted sum of ntrl/scnd/past)
  - D_bar, T_air, P_ann, P_month from CRUJRA (climate observations)

Rationale: Lei's correctness ask was about GPP source consistency (so the
parameters transfer to the coupled run). ED's internal D_bar / T_air / P_*
are derived diagnostics with less direct skill against satellite GFED5 than
CRUJRA, the upstream observation. Using CRUJRA for climate + dump for GPP
reclaims ILAMB skill while keeping the GPP source matched to coupled ED.

Scoring: ILAMB-aligned Collier-2018 (centralized RMSE, crms normalizer, mass
weighting) — matches ILAMB ConfBurntArea to within ~0.01 internally vs official.
Reference: GFED5 (ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc).

Time window: 2001-2016 (192 months) — same as the existing pipeline.
Grid: optimizer works at 1-deg; the 0.5-deg dump and GFED5 ref are coarsened by
a 2x2 mean.

Output:
  models/C/params.json                    (overwrite — becomes the new Model C)
  ilamb/MODELS/ED-ModelC-final/burntArea.nc   (regenerated under new params)
  models/C/params.PRE-coupled.json        (backup of previous params)
"""
from __future__ import annotations
import gc, json, os, sys, time, shutil
from pathlib import Path

import cftime
import numpy as np
import optuna
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_modelC import fire_C, add_cf_bounds, uncoarsen, coarsen

REPO         = Path(__file__).resolve().parents[1]
MODELS_DIR   = REPO / "models" / "C"
DUMP_NC      = REPO / "global_baseline_modelC_inputs_1997-2016.nc"
GFED5_NC     = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
ILAMB_OUT_NC = REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc"
ILAMB_OUT_NC.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

YEARS         = list(range(2001, 2017))
N_MONTHS      = 192                                    # 2001-01 .. 2016-12
DUMP_START_YEAR = 1997
DUMP_SLICE_START = (YEARS[0] - DUMP_START_YEAR) * 12   # 48
DUMP_SLICE_STOP  = DUMP_SLICE_START + N_MONTHS         # 240
FIRE_MAX_RATE = float(os.environ.get("FIRE_MAX_RATE", 5.0))
DT_YEARS      = 1.0
# SEASONAL_TRANSFORM=1 uses the physically-correct per-month disturbance fraction
# 1 - exp(-rate/12) instead of the legacy even-spread (1 - exp(-rate))/12. The
# legacy form hard-caps any month at 1/12 and flattens seasonality; the corrected
# form lifts high-rate (dry-season) cells, raising the spatial std ratio sigma
# toward 1 (the 1:1-plot fix). OFF by default -> canonical behavior unchanged.
SEASONAL_TRANSFORM = os.environ.get("SEASONAL_TRANSFORM", "0") == "1"
# RATE_AMP=1 adds `fire_amp` to the search space so the annual fire rate can exceed
# 1.0 (savanna multi-burn). fire_C applies it only when the param is present, so the
# canonical search space is unchanged when off. (Workstream A.)
RATE_AMP      = os.environ.get("RATE_AMP", "0") == "1"
# FUEL_AMP=1 adds the fuel-scaled amplitude (fuel_k, fuel_half) so the fire rate
# grows with the cell's fuel capacity (period-mean GPP) - the physical form of the
# rate>1 lever, and the missing positive-fuel term in savanna (Africa residual). Takes
# precedence over RATE_AMP's constant fire_amp inside fire_C. (Workstream A physical + C.)
FUEL_AMP      = os.environ.get("FUEL_AMP", "0") == "1"
# SPATIAL_OBJ=1 makes the FIRST NSGA-II objective the spatial-pattern Taylor skill on
# burning cells (the meeting's criterion that weights the spatial pattern) instead of
# ILAMB Overall. ILAMB stays tracked as a user_attr. Off by default. (Workstream B.)
SPATIAL_OBJ   = os.environ.get("SPATIAL_OBJ", "0") == "1"
# SEAS_W blends a region seasonal-cycle score into the SPATIAL_OBJ first objective:
# primary = (1-SEAS_W)*spatial_taylor + SEAS_W*seasonal. 0 => pure spatial (the
# per-continent fits traded seasonal away; SEAS_W>0 protects the seasonal cycle).
SEAS_W        = float(os.environ.get("SEAS_W", 0.0))
N_TRIALS      = int(os.environ.get("N_TRIALS", 2500))
MAG_PENALTY   = float(os.environ.get("MAG_PENALTY", 0.0))
# Hard magnitude band: trials where mean is outside [1/MAG_BAND, MAG_BAND]x
# the GFED5 mean get a strongly penalized score. 0 disables.
MAG_BAND      = float(os.environ.get("MAG_BAND", 0.0))
SEED          = int(os.environ.get("SEED", 42))
SAMPLER       = os.environ.get("SAMPLER", "tpe")  # tpe | cmaes
TAG           = os.environ.get("TAG", "")


# ── 1. Hybrid driver loader: CRUJRA climate + coupled GPP ─────────────────
def load_coupled_drivers():
    print(f"[setup] CRUJRA climate from data/crujra/, GPP from {DUMP_NC.name} ...")
    cru = REPO / "data" / "crujra"
    d = {
        "dbar":    np.load(cru / "dbar_monthly.npy").astype(np.float32),
        "p_ann":   np.load(cru / "p_ann_monthly.npy").astype(np.float32),
        "p_month": np.load(cru / "p_month_monthly.npy").astype(np.float32),
        "t_air":   np.load(cru / "t_air_monthly.npy").astype(np.float32),
    }

    ds = xr.open_dataset(DUMP_NC)
    sl = slice(DUMP_SLICE_START, DUMP_SLICE_STOP)

    def grab(name):
        return np.nan_to_num(ds[name].isel(time=sl).values.astype(np.float32), nan=0.0)

    gpp_n = np.clip(grab("GPP_month_ntrl"), 0, None)
    gpp_s = np.clip(grab("GPP_month_scnd"), 0, None)
    gpp_p = np.clip(grab("GPP_month_past"), 0, None)
    af_n  = grab("area_frac_ntrl")
    af_s  = grab("area_frac_scnd")
    af_p  = grab("area_frac_past")
    gpp_total = (gpp_n * af_n + gpp_s * af_s + gpp_p * af_p).astype(np.float32)
    ds.close()

    d["gpp_monthly"] = coarsen(gpp_total)

    # AGB for vegetation-aware suppression
    agb_p = REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc"
    if agb_p.exists():
        ds_agb = xr.open_dataset(agb_p)
        agb = np.nan_to_num(ds_agb["AGB"].isel(time=sl).values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        ds_agb.close()
        d["agb"] = coarsen(agb)
    return d


def load_gfed5_1deg():
    print(f"[setup] loading GFED5 reference from {GFED5_NC.name} ...")
    ds = xr.open_dataset(GFED5_NC)
    # GFED5 ILAMB ref is at 0.5-deg, units percent, 2001-2020.
    # Slice to 2001-2016 (first 192 months) and convert % -> fraction.
    a = ds["burntArea"].isel(time=slice(0, N_MONTHS)).values.astype(np.float32)
    a = np.nan_to_num(a, nan=0.0) / 100.0
    ds.close()
    # Coarsen 0.5-deg -> 1-deg by 2x2 mean
    return coarsen(a)


# ── 2. Setup ────────────────────────────────────────────────────────────────
drivers = load_coupled_drivers()                              # all 1-deg, 192 months
obs     = load_gfed5_1deg()                                   # (192, 180, 360) monthly fraction

# Held-out-years validation: FIT_Y0/FIT_YF restrict the FIT to a year window (the
# data is 2001-2016). Default = full range (bit-identical). reproduce/assemble keep
# the full period, so a model fit on e.g. 2001-2012 can be scored on unseen 2013-2016.
FIT_Y0  = int(os.environ.get("FIT_Y0", 2001))
FIT_YF  = int(os.environ.get("FIT_YF", 2016))
N_YEARS = FIT_YF - FIT_Y0 + 1
FIT_YEARS = list(range(FIT_Y0, FIT_YF + 1))
if (FIT_Y0, FIT_YF) != (2001, 2016):
    _m0 = (FIT_Y0 - 2001) * 12
    _m1 = (FIT_YF - 2001 + 1) * 12
    obs = obs[_m0:_m1]
    for _k in list(drivers.keys()):
        drivers[_k] = drivers[_k][_m0:_m1]
    N_MONTHS = _m1 - _m0
    print(f"[setup] HELD-OUT fit window {FIT_Y0}-{FIT_YF}  ({N_MONTHS} months, {N_YEARS} yr)")

lat_1   = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
cos_lat = np.cos(np.deg2rad(lat_1)).astype(np.float32)
w2      = np.broadcast_to(cos_lat[:, None], (180, 360)).astype(np.float64)
w3      = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360)).astype(np.float64)

land_mask = (obs > 0).any(axis=0)
w2_burn   = (w2 * land_mask).astype(np.float64)
w3_land   = (w3 * land_mask[None, :, :]).astype(np.float64)
print(f"[setup] land cells (GFED5 burnable): {int(land_mask.sum())} / {land_mask.size} "
      f"({100*land_mask.mean():.1f}%)")
print(f"[setup] GFED5 land-mean monthly frac: {float((obs*w3_land).sum()/w3_land.sum()):.4g}")

# ILAMB-aligned (Collier-2018) precomputes
gfed_tm   = obs.mean(axis=0).astype(np.float64)
gfed_std  = obs.std(axis=0).clip(1e-12).astype(np.float64)  # crms normaliser
gfed_cyc  = obs.reshape(N_YEARS, 12, 180, 360).mean(axis=0)
gfed_peak_month = np.argmax(gfed_cyc, axis=0).astype(np.float32)
# ILAMB ConfBurntArea uses mass_weighting=True: cos(lat) * ref_timeint
mass_w      = (w2 * gfed_tm).astype(np.float64)
mass_w_burn = (mass_w * land_mask).astype(np.float64)
obs_anom    = (obs - gfed_tm[None, :, :]).astype(np.float64)


# ── 3. ED-consistent transform ──────────────────────────────────────────────
def ed_transform(rate_yr):
    rate_capped = np.minimum(rate_yr, FIRE_MAX_RATE)
    if SEASONAL_TRANSFORM:
        # per-month disturbance fraction (no 1/12 cap; concentrates dry-season fire)
        return (1.0 - np.exp(-rate_capped * DT_YEARS / 12.0)).astype(np.float32)
    annual_frac = 1.0 - np.exp(-rate_capped * DT_YEARS)
    return (annual_frac / 12.0).astype(np.float32)


# ── 4a. Per-cell unweighted physical objective (penalizes false positives) ───
# Computed once: GFED5 annual map (sum of 12 monthly fractions)
gfed_annual = obs.reshape(N_YEARS, 12, 180, 360).sum(axis=1).mean(axis=0).astype(np.float64)
# Three tiers: quiet (<0.001 annual frac) — should not burn
# medium (0.001-0.01) — light fire
# active (>0.01) — true hotspots
QUIET_THRESH = float(os.environ.get("QUIET_THRESH", 0.001))
gfed_quiet = (gfed_annual < QUIET_THRESH).astype(np.float64) * land_mask
gfed_active = (gfed_annual >= QUIET_THRESH).astype(np.float64) * land_mask

# Workstream C: optionally restrict the fit to ONE continent so it can be tuned to
# that region's fire regime (per-continent models). REGION="" => global (unchanged).
# The spatial objective, magnitude band, and false-positive score all restrict to
# REGION_MASK; the resulting params describe that continent only.
REGION = os.environ.get("REGION", "")
_REGION_BOX = {
    "Africa": (-20, 52, -36, 18), "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74), "Boreal": (40, 180, 48, 78),
    "SEAsia": (60, 150, -11, 30), "Australia": (112, 154, -44, -10),
    "Europe": (-12, 40, 36, 72),
}
if REGION:
    _lat = -89.5 + np.arange(180) * 1.0
    _lon = -179.5 + np.arange(360) * 1.0
    _LON, _LAT = np.meshgrid(_lon, _lat)
    _b = _REGION_BOX[REGION]
    REGION_MASK = ((_LON >= _b[0]) & (_LON <= _b[1]) &
                   (_LAT >= _b[2]) & (_LAT <= _b[3]) & land_mask)
    print(f"[opt] REGION={REGION}: fitting {int(REGION_MASK.sum())} land cells only")
else:
    REGION_MASK = land_mask.copy()
# Magnitude weight that follows REGION (== w3_land when global)
W3_OBJ = (w3 * REGION_MASK[None, :, :]).astype(np.float64)
# Restrict the false-positive / hotspot masks to REGION too (idempotent when global,
# since REGION_MASK == land_mask and these are already land-masked 0/1 fields).
gfed_quiet = gfed_quiet * REGION_MASK
gfed_active = gfed_active * REGION_MASK


def physical_score(pred_monthly):
    """Penalize false positives. Returns lower = better fit.
    Returns score in [0,1] where 1 = perfect physical match.
    """
    pred_annual = pred_monthly.reshape(N_YEARS, 12, 180, 360).sum(axis=1).mean(axis=0).astype(np.float64)
    # False-positive amplitude: pred fire in cells GFED rarely fires
    fp = (pred_annual * gfed_quiet).sum() / (gfed_quiet.sum() + 1e-12)
    # True hotspot match: |pred - gfed| in active cells, normalized by gfed mean
    diff = np.abs(pred_annual - gfed_annual) * gfed_active
    gfed_active_mean = (gfed_annual * gfed_active).sum() / (gfed_active.sum() + 1e-12)
    hotspot_err = diff.sum() / (gfed_active.sum() + 1e-12) / (gfed_active_mean + 1e-12)
    # Combine: lower fp + lower hotspot_err = better
    return float(np.exp(-10.0 * fp)), float(np.exp(-hotspot_err))


# ── 4. ILAMB-aligned scoring (Collier-2018 + mass weighting) ─────────────────
def score_BA(pred_monthly):
    pred_tm = pred_monthly.mean(axis=0).astype(np.float64)

    # Bias score: exp(-|bias / crms|), mass-weighted
    bias_per_cell = pred_tm - gfed_tm
    bias_score_per_cell = np.exp(-np.abs(bias_per_cell) / gfed_std)
    bias = float((bias_score_per_cell * mass_w).sum() / (mass_w.sum() + 1e-12))

    # RMSE score: centralized RMSE / crms, mass-weighted
    pred_anom = pred_monthly.astype(np.float64) - pred_tm[None, :, :]
    crmse_per_cell = np.sqrt(((pred_anom - obs_anom) ** 2).mean(axis=0))
    rmse_score_per_cell = np.exp(-crmse_per_cell / gfed_std)
    rmse_s = float((rmse_score_per_cell * mass_w).sum() / (mass_w.sum() + 1e-12))

    # Seasonal score: phase-shift of month-of-peak
    pred_cyc = pred_monthly.reshape(N_YEARS, 12, 180, 360).mean(axis=0)
    pred_peak = np.argmax(pred_cyc, axis=0).astype(np.float32)
    shift = pred_peak - gfed_peak_month
    shift = np.where(shift >  6, shift - 12, shift)
    shift = np.where(shift < -6, shift + 12, shift)
    seas_per_cell = (1.0 + np.cos(np.abs(shift) / 12.0 * 2.0 * np.pi)) * 0.5
    seas = float((seas_per_cell * mass_w_burn).sum() / (mass_w_burn.sum() + 1e-12))

    # Spatial score: Taylor diagram (ILAMB Variable.py)
    obs_flat  = gfed_tm[land_mask]
    pred_flat = pred_tm[land_mask]
    pw_flat   = w2_burn[land_mask]
    if pw_flat.sum() > 0:
        ow = (obs_flat * pw_flat).sum() / pw_flat.sum()
        pw = (pred_flat * pw_flat).sum() / pw_flat.sum()
        oa = obs_flat - ow
        pa = pred_flat - pw
        std0 = max(float(np.sqrt(((oa**2) * pw_flat).sum() / pw_flat.sum())), 1e-12)
        std  = max(float(np.sqrt(((pa**2) * pw_flat).sum() / pw_flat.sum())), 1e-12)
        denom = float(np.sqrt(((pa**2)*pw_flat).sum() * ((oa**2)*pw_flat).sum()))
        rho   = float((pa * oa * pw_flat).sum() / (denom + 1e-12))
        sigma = std / std0
        spatial = 2.0 * (1.0 + rho) / ((sigma + 1.0 / max(sigma, 1e-12)) ** 2)
    else:
        spatial = 0.0

    # FIX 1 (2026-06-08): match official ILAMB aggregation, which weights RMSE
    # double: Overall = (Bias + 2*RMSE + Seasonal + Spatial) / 5. The old
    # unweighted mean of the four components made the selection target diverge
    # from the official Overall used for promotion (verified: this weighting
    # reproduces official Overall to 4 decimals for both canonical and tropfix).
    overall = float((bias + 2.0 * rmse_s + seas + float(spatial)) / 5.0)
    return overall, dict(bias=bias, rmse=rmse_s, seas=seas, spatial=float(spatial),
                         overall=overall)


def predict(params):
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(drivers, params)
    rate = rate * land_mask[None, :, :]
    return ed_transform(rate)


# ── 5. Optuna ───────────────────────────────────────────────────────────────
LOG_PARAMS = {
    "k1":              (1e-5, 1e-1),
    "D_low":           (1e0,  1e4),
    "k2":              (1e-5, 1e-1),
    "D_high":          (1e1,  1e5),
    # fire_exp constrained to >= 1 so the multiplicative-suppression cascade
    # behaves physically — values < 1 cause global smearing of fire signal.
    "fire_exp":        (1.0,  10.0),
    "P_half":          (1e0,  1e4),
    "pre_dampen_half": (1e-2, 1e3),
    "gpp_af":          (1e-3, 1e2),
    "gpp_b":           (1e-5, 1e1),
    "gpp_d":           (1e-2, 1e3),
    "ign_k":           (1e-3, 1e1),
    "ign_c":           (1e-1, 1e3),
    # Tropical closed-canopy suppression: kill false-positive fire in high-AGB
    # tropical forest (Amazon/Congo/SE-Asia). Gated to |lat|<23.5 inside fire_C.
    # Closed-canopy tropical forest AGB ~ 10-25 kgC/m2; savanna grass < 2 kgC/m2,
    # so a crit in this band leaves savanna untouched. (The earlier GLOBAL veg
    # term k_veg/agb_crit did not help and is intentionally dropped here.)
    "trop_agb_crit":   (1e0,  3e1),
    "trop_k_veg":      (5e-1, 8e0),
}
# Workstream A: let the annual fire rate exceed 1.0 (savanna multi-burn). Searched
# only when RATE_AMP=1; absent => fire_C leaves the rate <= 1 (canonical). Range
# 1..6: the diag sweep showed the per-cell ceiling reaches GFED5's 0.104 near ~2.
if RATE_AMP:
    LOG_PARAMS["fire_amp"] = (1.0, 6.0)
# Fuel-scaled amplitude (physical rate>1 + the savanna fuel term). fuel_k is the gain
# (amplitude up to 1+fuel_k), fuel_half the GPP half-saturation of the fuel response.
if FUEL_AMP:
    LOG_PARAMS["fuel_k"] = (1e-2, 6.0)
    LOG_PARAMS["fuel_half"] = (1e-1, 1e2)

# Warm start from $WARM env var (path to a params.{tag}.json) if set,
# otherwise from params.PRE-coupled.json (the original benchmark).
WARM_ENV = os.environ.get("WARM", "")
if WARM_ENV:
    WARM_FILE = MODELS_DIR / WARM_ENV
elif (MODELS_DIR / "params.PRE-coupled.json").exists():
    WARM_FILE = MODELS_DIR / "params.PRE-coupled.json"
else:
    WARM_FILE = MODELS_DIR / "params.json"
print(f"[opt] warm-start from {WARM_FILE.name}")
WARM_START = json.load(open(WARM_FILE))["params"]
# Leave the warm-start point WITHOUT the tropical params so the smoke-test/warm
# trial reproduces the canonical baseline exactly; Optuna samples trop_* for the
# enqueued trial. (Old global k_veg/agb_crit are no longer in the search space.)
# Seed fire_amp=1.0 (canonical, rate<=1) for the warm trial so the search opens at
# the known baseline and grows the amplitude from there.
if RATE_AMP and "fire_amp" not in WARM_START:
    WARM_START = {**WARM_START, "fire_amp": 1.0}
WARM_START = {k: v for k, v in WARM_START.items() if k in LOG_PARAMS}


GFED5_LM = float((obs * W3_OBJ).sum() / (W3_OBJ.sum() + 1e-12))


USE_PHYSICAL = bool(int(os.environ.get("PHYSICAL", "0")))


def spatial_taylor(pred):
    """Workstream-B objective: pixel-level spatial-pattern Taylor skill on the
    burning cells (GFED5 annual fraction > 0.1%), computed on the period-mean map.
    Returns (taylor in [0,1], r, sigma). Rewards r->1 AND sigma->1, i.e. the
    per-cell 1:1 band climbing onto the diagonal. Same form as score_spatial.py
    but on the optimizer's internal 1deg grid for speed."""
    pm = pred.mean(axis=0)
    gm = obs.mean(axis=0)
    mask = REGION_MASK & (gfed_annual > 0.001)
    m = pm[mask].astype(np.float64); g = gm[mask].astype(np.float64)
    ms, gs = m.std(), g.std()
    r = float(((m - m.mean()) * (g - g.mean())).mean() / (ms * gs + 1e-30))
    sigma = float(ms / (gs + 1e-30))
    taylor = 4.0 * (1.0 + r) / (((sigma + 1.0 / (sigma + 1e-30)) ** 2) * 2.0)
    return taylor, r, sigma


def seasonal_score_region(pred):
    """Region-weighted seasonal-cycle score (same month-of-peak phase-shift form as
    score_BA, restricted to REGION_MASK burning cells). In [0,1]. Used to keep the
    per-continent fit from trading away seasonal timing for spatial pattern."""
    pred_cyc = pred.reshape(N_YEARS, 12, 180, 360).mean(axis=0)
    pred_peak = np.argmax(pred_cyc, axis=0).astype(np.float32)
    shift = pred_peak - gfed_peak_month
    shift = np.where(shift > 6, shift - 12, shift)
    shift = np.where(shift < -6, shift + 12, shift)
    seas_per_cell = (1.0 + np.cos(np.abs(shift) / 12.0 * 2.0 * np.pi)) * 0.5
    w = mass_w_burn * REGION_MASK
    return float((seas_per_cell * w).sum() / (w.sum() + 1e-12))


def objective(trial):
    p = {name: trial.suggest_float(name, lo, hi, log=True)
         for name, (lo, hi) in LOG_PARAMS.items()}
    pred = predict(p)
    overall, _ = score_BA(pred)
    pred_lm = float((pred * W3_OBJ).sum() / (W3_OBJ.sum() + 1e-12))
    ratio = max(pred_lm, 1e-12) / GFED5_LM

    if SAMPLER == "nsga2":
        # Multi-objective: return (-ILAMB_Overall, false_positive_rate)
        fp_score, hot_score = physical_score(pred)
        # fp_score = exp(-10*fp), so minimize -log(fp_score)/10 = fp
        # but cleaner to return 1 - fp_score (lower is better)
        trial.set_user_attr("ilamb_overall", overall)
        trial.set_user_attr("fp_score", fp_score)
        trial.set_user_attr("hot_score", hot_score)
        trial.set_user_attr("pred_lm_ratio", ratio)
        # Workstream B: first objective is the spatial-pattern Taylor skill when
        # SPATIAL_OBJ=1, else ILAMB Overall (canonical). Both tracked as user_attrs.
        if SPATIAL_OBJ:
            taylor, r_sp, sigma_sp = spatial_taylor(pred)
            trial.set_user_attr("spatial_taylor", taylor)
            trial.set_user_attr("spatial_r", r_sp)
            trial.set_user_attr("spatial_sigma", sigma_sp)
            if SEAS_W > 0:
                seas_r = seasonal_score_region(pred)
                trial.set_user_attr("spatial_seas", seas_r)
                primary = (1.0 - SEAS_W) * taylor + SEAS_W * seas_r
            else:
                primary = taylor
        else:
            primary = overall
        if MAG_BAND > 0:
            viol = max(ratio - MAG_BAND, (1.0 / MAG_BAND) - ratio, 0.0)
            if viol > 0.0:
                # Graded penalty so NSGA-II has a gradient TOWARD the band: any
                # feasible point (objectives <= ~1) dominates all infeasible ones,
                # and among infeasible the smaller magnitude violation is preferred.
                # (A flat constant here gives zero selection pressure and the search
                # degenerates to random when the warm start is outside the band.)
                return (10.0 + viol, 10.0 + viol)
        return (-primary, 1.0 - fp_score)

    if MAG_BAND > 0 and (ratio > MAG_BAND or ratio < 1.0 / MAG_BAND):
        return -(overall - 1.0)
    if USE_PHYSICAL:
        fp_score, hot_score = physical_score(pred)
        w_ilamb = float(os.environ.get("W_ILAMB", 0.50))
        w_fp    = float(os.environ.get("W_FP",    0.30))
        w_hot   = float(os.environ.get("W_HOT",   0.20))
        combined = w_ilamb * overall + w_fp * fp_score + w_hot * hot_score
        trial.set_user_attr("fp_score", fp_score)
        trial.set_user_attr("hot_score", hot_score)
        trial.set_user_attr("ilamb_overall", overall)
        return -combined
    if MAG_PENALTY > 0:
        penalty = MAG_PENALTY * np.log(ratio) ** 2
        return -(overall - penalty)
    return -overall


def write_ba_nc(pred, path):
    """Write a monthly-fraction BA prediction (1deg, land-masked) to a 0.5deg CF
    NetCDF at `path`, creating parent dirs. Shared by the primary output and the
    FIX-2 top-K candidate dump so every candidate is encoded identically and can
    be re-scored with official ILAMB without surprises."""
    pred_hd = uncoarsen(np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in FIT_YEARS for m in range(1, 13)]
    lat   = np.arange(-89.75, 90.0, 0.5)
    lon   = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time","lat","lon"), pred_hd,
                       {"units":"1", "standard_name":"burnt_area_fraction",
                        "long_name":"Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (coupled-consistent retune, GFED5)",
               "Conventions": "CF-1.7",
               "transform": (f"monthly_frac = 1 - exp(-min(rate_yr, {FIRE_MAX_RATE}) / 12)"
                             if SEASONAL_TRANSFORM else
                             f"monthly_frac = (1 - exp(-min(rate_yr, {FIRE_MAX_RATE}) * 1yr)) / 12")})
    ds = add_cf_bounds(ds)
    enc = {"burntArea":   {"zlib":True, "complevel":4, "_FillValue":1e20},
           "time":        {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"},
           "time_bounds": {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, path)
    return path


def main():
    backup = MODELS_DIR / "params.PRE-coupled.json"
    if not backup.exists():
        shutil.copy(MODELS_DIR / "params.json", backup)
        print(f"[backup] {backup}")

    # Smoke test the warm-start point first
    warm_pred = predict(WARM_START)
    warm_overall, warm_breakdown = score_BA(warm_pred)
    print(f"[warm] Overall={warm_overall:.4f}  "
          f"bias={warm_breakdown['bias']:.4f} rmse={warm_breakdown['rmse']:.4f} "
          f"seas={warm_breakdown['seas']:.4f} spatial={warm_breakdown['spatial']:.4f}")

    if SAMPLER == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=SEED, x0=WARM_START)
        print(f"[opt] sampler=CmaEsSampler seed={SEED}")
        study   = optuna.create_study(direction="minimize", sampler=sampler)
    elif SAMPLER == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(seed=SEED, population_size=64)
        print(f"[opt] sampler=NSGAIISampler seed={SEED}")
        # Two objectives: minimize -ILAMB_Overall, minimize false_positive_rate
        study   = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    else:
        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
        print(f"[opt] sampler=TPESampler seed={SEED}")
        study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(WARM_START)

    t0 = time.time()
    last_print = t0

    def cb(st, tr):
        nonlocal last_print
        if time.time() - last_print > 20:
            elapsed = (time.time() - t0) / 60.0
            if SAMPLER == "nsga2":
                front = st.best_trials
                if front:
                    best_il = max(t.user_attrs.get("ilamb_overall", 0) for t in front)
                    best_ty = max(t.user_attrs.get("spatial_taylor", 0) for t in front)
                    extra = f"best_taylor={best_ty:.4f}  " if SPATIAL_OBJ else ""
                    print(f"  trial {len(st.trials):4d}/{N_TRIALS}  pareto_size={len(front)}  "
                          f"{extra}best_ilamb={best_il:.4f}  ({elapsed:.1f} min)")
                else:
                    print(f"  trial {len(st.trials):4d}/{N_TRIALS}  no_pareto  ({elapsed:.1f} min)")
            else:
                best = -st.best_value
                print(f"  trial {len(st.trials):4d}/{N_TRIALS}  best_overall={best:.4f}  "
                      f"({elapsed:.1f} min)")
            last_print = time.time()

    print(f"[opt] running {N_TRIALS} trials against GFED5 ...")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    elapsed = (time.time() - t0) / 60.0

    if SAMPLER == "nsga2":
        # Pareto front: pick trial with highest ILAMB Overall AND fp_score >= threshold
        FP_MIN = float(os.environ.get("FP_MIN", 0.80))
        candidates = [t for t in study.best_trials
                      if t.user_attrs.get("fp_score", 0) >= FP_MIN]
        if not candidates:
            candidates = study.best_trials
        # Rank by the objective we actually optimized. Under SPATIAL_OBJ with a
        # seasonal blend (SEAS_W>0), rank by the SAME blend (taylor + seasonal) so the
        # picker does not silently revert to the pure-spatial best; else spatial Taylor;
        # else ILAMB Overall. (Re-score the top-K officially regardless.)
        def rank_score(t):
            if SPATIAL_OBJ and SEAS_W > 0:
                return ((1.0 - SEAS_W) * t.user_attrs.get("spatial_taylor", 0)
                        + SEAS_W * t.user_attrs.get("spatial_seas", 0))
            if SPATIAL_OBJ:
                return t.user_attrs.get("spatial_taylor", 0)
            return t.user_attrs.get("ilamb_overall", 0)
        candidates.sort(key=lambda t: -rank_score(t))
        chosen = candidates[0]
        print(f"\n[done] {len(study.trials)} trials in {elapsed:.1f} min")
        print(f"[pareto] {len(study.best_trials)} non-dominated trials, "
              f"{len(candidates)} with fp_score >= {FP_MIN}")
        print(f"[chosen] trial {chosen.number}: "
              f"ILAMB={chosen.user_attrs['ilamb_overall']:.4f}  "
              f"fp_score={chosen.user_attrs['fp_score']:.4f}  "
              f"ratio={chosen.user_attrs.get('pred_lm_ratio',0):.2f}")
        best_p = chosen.params
    else:
        best_overall = -study.best_value
        print(f"\n[done] {len(study.trials)} trials in {elapsed:.1f} min")
        print(f"       best Overall = {best_overall:.4f}")
        best_p = study.best_params

    def build_out(params, breakdown):
        return {
            "model": "Model C (coupled-consistent retune; drivers + GPP from Lei's ED dump; GFED5 target)",
            "drivers_source": str(DUMP_NC.name),
            "reference":      "GFED5 (ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc)",
            "mechanisms":     ["gpp_monthly", "precip", "t_air_ign"],
            "n_mechanisms":   3,
            "n_params":       len(params),
            "objective":      "ILAMB Overall (Bias + 2*RMSE + Seas + Spatial) / 5 against GFED5 monthly fraction",
            "ed_consistency": {
                "interpretation": "Model C output is an annual fire rate (yr^-1)",
                "fire_max_rate":   FIRE_MAX_RATE,
                "seasonal_transform": SEASONAL_TRANSFORM,
                "transform":       ("burnt_monthly_frac = 1 - exp(-min(rate, fire_max) / 12)"
                                    if SEASONAL_TRANSFORM else
                                    "burnt_monthly_frac = (1 - exp(-min(rate, fire_max) * 1yr)) / 12"),
            },
            "scores_internal": breakdown,
            "warm_start_internal": warm_breakdown,
            "n_trials":   len(study.trials),
            "runtime_min": round(elapsed, 2),
            "params":     params,
        }

    pred   = predict(best_p)
    overall, breakdown = score_BA(pred)
    print(f"[best] Bias={breakdown['bias']:.4f}  RMSE={breakdown['rmse']:.4f}  "
          f"Seas={breakdown['seas']:.4f}  Spatial={breakdown['spatial']:.4f}  "
          f"Overall={overall:.4f}")
    raw  = fire_C(drivers, best_p) * land_mask[None, :, :]
    raw_lm = float((raw * w3_land).sum() / w3_land.sum())
    print(f"[diag] raw rate land-mean (yr^-1): {raw_lm:.4g}  "
          f"(target ~ {12*float((obs * w3_land).sum() / w3_land.sum()):.4g})")

    out_name = f"params.{TAG}.json" if TAG else "params.json"
    (MODELS_DIR / out_name).write_text(json.dumps(build_out(best_p, breakdown), indent=2))
    print(f"[write] {MODELS_DIR / out_name}")

    nc_out = ILAMB_OUT_NC.with_name(f"burntArea.{TAG}.nc") if TAG else ILAMB_OUT_NC
    write_ba_nc(pred, nc_out)
    print(f"[write] {nc_out}  ({nc_out.stat().st_size/1e6:.1f} MB)")

    # FIX 2 (2026-06-08): dump the top-K Pareto candidates, not just the single
    # internal-best, each as its own params JSON + a scoreable burntArea.nc in its
    # own model dir. The internal Overall is on a different grid/weighting than
    # official ILAMB, so the internal-best is NOT necessarily the official winner;
    # score every candidate with official ILAMB and promote the official best.
    TOPK = int(os.environ.get("TOPK", 5))
    if SAMPLER == "nsga2" and TOPK > 0:
        topk = candidates[:TOPK]
        tag = TAG or "run"
        topk_root = REPO / "ilamb" / f"MODELS_TOPK_{tag}"
        print(f"\n[FIX2] dumping top-{len(topk)} Pareto candidates -> {topk_root}")
        manifest = []
        for rank, t in enumerate(topk, 1):
            p = t.params
            pr = predict(p)
            ov, bd = score_BA(pr)
            pj = build_out(p, bd)
            pj["pareto_rank"]   = rank
            pj["trial_number"]  = t.number
            pj["fp_score"]      = t.user_attrs.get("fp_score")
            pj["pred_lm_ratio"] = t.user_attrs.get("pred_lm_ratio")
            pjpath = MODELS_DIR / f"params.{tag}.k{rank}.json"
            pjpath.write_text(json.dumps(pj, indent=2))
            mdir = topk_root / f"ED-ModelC-{tag}-k{rank}"
            write_ba_nc(pr, mdir / "burntArea.nc")
            manifest.append({
                "rank": rank, "trial": t.number,
                "internal_overall": round(ov, 4),
                "spatial_taylor": round(float(t.user_attrs.get("spatial_taylor", 0)), 4),
                "spatial_r":      round(float(t.user_attrs.get("spatial_r", 0)), 4),
                "spatial_sigma":  round(float(t.user_attrs.get("spatial_sigma", 0)), 4),
                "spatial_seas":   round(float(t.user_attrs.get("spatial_seas", 0)), 4),
                "fp_score":      round(float(t.user_attrs.get("fp_score", 0)), 4),
                "pred_lm_ratio": round(float(t.user_attrs.get("pred_lm_ratio", 0)), 4),
                "fire_amp":      round(float(p.get("fire_amp", 1.0)), 3),
                "params_json": pjpath.relative_to(REPO).as_posix(),
                "model_dir":   mdir.relative_to(REPO).as_posix(),
            })
            print(f"  k{rank}: trial {t.number}  internal_overall={ov:.4f}  "
                  f"taylor={float(t.user_attrs.get('spatial_taylor',0)):.4f}  "
                  f"sigma={float(t.user_attrs.get('spatial_sigma',0)):.3f}  "
                  f"fp={float(t.user_attrs.get('fp_score',0)):.3f}  "
                  f"ratio={float(t.user_attrs.get('pred_lm_ratio',0)):.2f}  "
                  f"amp={float(p.get('fire_amp',1.0)):.2f}")
        (MODELS_DIR / f"topk.{tag}.json").write_text(json.dumps(manifest, indent=2))
        print(f"[FIX2] manifest -> {MODELS_DIR / f'topk.{tag}.json'}")
        print(f"[FIX2] re-score all candidates with official ILAMB (CLAUDE.md recipe; "
              f"clean ._* first), e.g. --model_root \"$PWD/ilamb/MODELS_TOPK_{tag}\"")
        print(f"[FIX2] promote the candidate with the best OFFICIAL Overall, "
              f"NOT the internal one.")


if __name__ == "__main__":
    main()
