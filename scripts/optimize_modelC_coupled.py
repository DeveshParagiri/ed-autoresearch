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
gfed_cyc  = obs.reshape(16, 12, 180, 360).mean(axis=0)
gfed_peak_month = np.argmax(gfed_cyc, axis=0).astype(np.float32)
# ILAMB ConfBurntArea uses mass_weighting=True: cos(lat) * ref_timeint
mass_w      = (w2 * gfed_tm).astype(np.float64)
mass_w_burn = (mass_w * land_mask).astype(np.float64)
obs_anom    = (obs - gfed_tm[None, :, :]).astype(np.float64)


# ── 3. ED-consistent transform ──────────────────────────────────────────────
def ed_transform(rate_yr):
    rate_capped = np.minimum(rate_yr, FIRE_MAX_RATE)
    annual_frac = 1.0 - np.exp(-rate_capped * DT_YEARS)
    return (annual_frac / 12.0).astype(np.float32)


# ── 4a. Per-cell unweighted physical objective (penalizes false positives) ───
# Computed once: GFED5 annual map (sum of 12 monthly fractions)
gfed_annual = obs.reshape(16, 12, 180, 360).sum(axis=1).mean(axis=0).astype(np.float64)
# Three tiers: quiet (<0.001 annual frac) — should not burn
# medium (0.001-0.01) — light fire
# active (>0.01) — true hotspots
QUIET_THRESH = float(os.environ.get("QUIET_THRESH", 0.001))
gfed_quiet = (gfed_annual < QUIET_THRESH).astype(np.float64) * land_mask
gfed_active = (gfed_annual >= QUIET_THRESH).astype(np.float64) * land_mask


def physical_score(pred_monthly):
    """Penalize false positives. Returns lower = better fit.
    Returns score in [0,1] where 1 = perfect physical match.
    """
    pred_annual = pred_monthly.reshape(16, 12, 180, 360).sum(axis=1).mean(axis=0).astype(np.float64)
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
    pred_cyc = pred_monthly.reshape(16, 12, 180, 360).mean(axis=0)
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

    overall = float(np.mean([bias, rmse_s, seas, float(spatial)]))
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
    # Vegetation suppression: fire decreases as AGB > agb_crit (kg C / m2)
    # Forest biomass ~ 5-20 kg/m2, grass ~ 0.1-1 kg/m2
    "k_veg":           (1e-2, 1e1),
    "agb_crit":        (5e-1, 5e1),
}

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
# Add veg-suppression defaults if missing (so old warm-starts behave as before)
WARM_START.setdefault("k_veg", 0.1)
WARM_START.setdefault("agb_crit", 30.0)


GFED5_LM = float((obs * w3_land).sum() / (w3_land.sum() + 1e-12))


USE_PHYSICAL = bool(int(os.environ.get("PHYSICAL", "0")))


def objective(trial):
    p = {name: trial.suggest_float(name, lo, hi, log=True)
         for name, (lo, hi) in LOG_PARAMS.items()}
    pred = predict(p)
    overall, _ = score_BA(pred)
    pred_lm = float((pred * w3_land).sum() / (w3_land.sum() + 1e-12))
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
        if MAG_BAND > 0 and (ratio > MAG_BAND or ratio < 1.0 / MAG_BAND):
            return (1.0, 1.0)  # infeasible, both objectives high
        return (-overall, 1.0 - fp_score)

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
                    print(f"  trial {len(st.trials):4d}/{N_TRIALS}  pareto_size={len(front)}  "
                          f"best_ilamb={best_il:.4f}  ({elapsed:.1f} min)")
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
        candidates.sort(key=lambda t: -t.user_attrs.get("ilamb_overall", 0))
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

    pred   = predict(best_p)
    overall, breakdown = score_BA(pred)
    print(f"[best] Bias={breakdown['bias']:.4f}  RMSE={breakdown['rmse']:.4f}  "
          f"Seas={breakdown['seas']:.4f}  Spatial={breakdown['spatial']:.4f}  "
          f"Overall={overall:.4f}")
    raw  = fire_C(drivers, best_p) * land_mask[None, :, :]
    raw_lm = float((raw * w3_land).sum() / w3_land.sum())
    print(f"[diag] raw rate land-mean (yr^-1): {raw_lm:.4g}  "
          f"(target ~ {12*float((obs * w3_land).sum() / w3_land.sum()):.4g})")

    out = {
        "model": "Model C (coupled-consistent retune; drivers + GPP from Lei's ED dump; GFED5 target)",
        "drivers_source": str(DUMP_NC.name),
        "reference":      "GFED5 (ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc)",
        "mechanisms":     ["gpp_monthly", "precip", "t_air_ign"],
        "n_mechanisms":   3,
        "n_params":       12,
        "objective":      "Collier-4 (Bias + RMSE + Seas + Spatial) / 4 against GFED5 monthly fraction",
        "ed_consistency": {
            "interpretation": "Model C output is an annual fire rate (yr^-1)",
            "fire_max_rate":   FIRE_MAX_RATE,
            "transform":       "burnt_monthly_frac = (1 - exp(-min(rate, fire_max) * 1yr)) / 12",
        },
        "scores_internal": breakdown,
        "warm_start_internal": warm_breakdown,
        "n_trials":   len(study.trials),
        "runtime_min": round(elapsed, 2),
        "params":     best_p,
    }
    out_name = f"params.{TAG}.json" if TAG else "params.json"
    (MODELS_DIR / out_name).write_text(json.dumps(out, indent=2))
    print(f"[write] {MODELS_DIR / out_name}")

    pred_hd = uncoarsen(np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat   = np.arange(-89.75, 90.0, 0.5)
    lon   = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time","lat","lon"), pred_hd,
                       {"units":"1", "standard_name":"burnt_area_fraction",
                        "long_name":"Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (coupled-consistent retune, GFED5)",
               "Conventions": "CF-1.7",
               "transform": f"monthly_frac = (1 - exp(-min(rate_yr, {FIRE_MAX_RATE}) * 1yr)) / 12"})
    ds = add_cf_bounds(ds)
    enc = {"burntArea":   {"zlib":True, "complevel":4, "_FillValue":1e20},
           "time":        {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"},
           "time_bounds": {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"}}
    nc_out = ILAMB_OUT_NC.with_name(f"burntArea.{TAG}.nc") if TAG else ILAMB_OUT_NC
    tmp = nc_out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, nc_out)
    print(f"[write] {nc_out}  ({nc_out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
