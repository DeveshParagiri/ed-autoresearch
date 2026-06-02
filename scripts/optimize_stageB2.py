"""
optimize_stageB2.py — Joint Optuna optimization (Model C) with fuel-availability prefactor.

Stage B2 adds PHI ∈ [0.05, 1.0] (log) as a global fuel-availability prefactor on
C_combust, giving TPE a direct mass knob to hit GFED's 2.0 Pg/yr without an
asterisk.

Parameters searched (18 total):
  Burn (12): k1, D_low, k2, D_high, fire_exp, P_half, pre_dampen_half,
             gpp_af, gpp_b, gpp_d, ign_k, ign_c
  Combust (6): PHI, D_REF, BETA_MAX_leaf, BETA_MAX_fine, BETA_MAX_coarse, POOL_FRAC_fine

Objective (minimized):
  L = -( 0.6 · Score_BA_overall  +  0.4 · Score_fFire_overall )

Score_BA (Collier-4):        mean(Bias, RMSE, Seasonal, Spatial)
Score_fFire (JASMIN-style):  (Bias + Spatial) / 2

Output:
  models/C-stageB2/params.json
  out_ffire_stageB2/ED-ModelC-stageB2-fFire/fFire.nc
  out_stageB2/ED-ModelC-stageB2/burntArea.nc
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

import cftime
import h5py
import numpy as np
import optuna
import xarray as xr
import netCDF4 as nc4

# Reuse the existing model code
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_v2 import (
    load_drivers, load_gfed_1deg, fire_C, coarsen, uncoarsen, _add_time_bounds,
)

REPO      = Path(__file__).resolve().parents[1]
ED_SIM    = Path(r"C:\Users\owusu\OneDrive\Documents\UMD STUFF\ED AUTORESEARCH\fire_autoresearch\data\raw\ed_simulation\EDv3_global_simulation_1981_2016.nc")
GFED_FF   = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED4.1S" / "fFire.nc"
OUT_MODEL = REPO / "models" / "C-stageB2"
OUT_BA    = REPO / "out_stageB2"   / "ED-ModelC-stageB2"
OUT_FF    = REPO / "out_ffire_stageB2" / "ED-ModelC-stageB2-fFire"
for d in [OUT_MODEL, OUT_BA, OUT_FF]:
    d.mkdir(parents=True, exist_ok=True)

YEARS        = list(range(2001, 2017))
N_MONTHS     = 192
START_IDX    = (2001 - 1981) * 12
SEC_PER_MON  = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float64) * 86400.0
SEC_TILE     = np.tile(SEC_PER_MON, 16).astype(np.float32)[:, None, None]   # (192,1,1)
N_TRIALS     = 1500
W_BA, W_FF   = 0.6, 0.4

# ─────────────────────────────────────────────────────────────────────────────
# One-time loads: drivers, GFED refs, ED carbon pools
# ─────────────────────────────────────────────────────────────────────────────

print("[setup] Loading drivers (CRUJRA + GPP + ED static) ...")
drivers = load_drivers()                             # 1-deg
gc.collect()

print("[setup] Loading GFED burntArea (1-deg) ...")
gfed_ba_1 = load_gfed_1deg()                         # (192,180,360) fraction
land_mask_1 = (gfed_ba_1 > 0).any(axis=0)            # (180,360) bool
lat_1d  = np.arange(-89.5, 90.0, 1.0)
cos_lat = np.cos(np.deg2rad(lat_1d)).astype(np.float32)
w3_1    = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360)).astype(np.float32)
w2_1    = np.broadcast_to(cos_lat[:, None], (180, 360)).astype(np.float32)
w2_1_burn = w2_1 * land_mask_1

# GFED BA: precompute reference time-mean, std, seasonal anomaly, spatial anomaly
gfed_tm   = gfed_ba_1.mean(axis=0).astype(np.float32)
gfed_std  = gfed_ba_1.std(axis=0).astype(np.float32) + 1e-9
gfed_cyc  = gfed_ba_1.reshape(16, 12, 180, 360).mean(axis=0)
gfed_cyc_anom = gfed_cyc - gfed_cyc.mean(axis=0, keepdims=True)
gfed_cyc_ss   = (gfed_cyc_anom ** 2).sum(axis=0)
gfed_mean = float((gfed_ba_1 * w3_1).sum() / w3_1.sum())
gfed_tm_anom  = gfed_tm - gfed_mean                   # (180,360)
gfed_anom_flat = gfed_tm_anom.flatten()
pw_flat = w2_1_burn.flatten()

print("[setup] Loading GFED fFire (0.5-deg reference) ...")
with xr.open_dataset(GFED_FF, decode_times=False) as ds:
    gfed_ff_05 = ds["fFire"].values.astype(np.float32)        # (192,360,720) kg/m²/s
    lat05 = ds["lat"].values
# Coarsen fFire reference to 1-deg for fast scoring (block mean 2x2)
gfed_ff_1 = gfed_ff_05.reshape(N_MONTHS, 180, 2, 360, 2).mean(axis=(2, 4))
gfed_ff_1 = np.nan_to_num(gfed_ff_1, nan=0.0).astype(np.float32)
ff_land_mask = (gfed_ff_1 > 0).any(axis=0)            # (180,360)
w2_ff = w2_1 * ff_land_mask
gfed_ff_tm = gfed_ff_1.mean(axis=0).astype(np.float32)
gfed_ff_mean = float((gfed_ff_1 * w3_1).sum() / w3_1.sum())
gfed_ff_tm_anom = gfed_ff_tm - gfed_ff_mean
gfed_ff_anom_flat = gfed_ff_tm_anom.flatten()
pw_ff_flat = w2_ff.flatten()

print("[setup] Loading ED carbon pools (AGB, soil_C) at 1-deg ...")
with nc4.Dataset(ED_SIM) as ds:
    agb05   = np.array(ds["AGB"][START_IDX:START_IDX+N_MONTHS, ::-1, :], dtype=np.float32)
    soilC05 = np.array(ds["soil_C"][START_IDX:START_IDX+N_MONTHS, ::-1, :], dtype=np.float32)
agb05   = np.nan_to_num(agb05,   nan=0.0, posinf=0.0, neginf=0.0)
soilC05 = np.nan_to_num(soilC05, nan=0.0, posinf=0.0, neginf=0.0)
# Coarsen to 1-deg for in-loop scoring
agb_1   = agb05.reshape(N_MONTHS,180,2,360,2).mean(axis=(2,4)).astype(np.float32)
soilC_1 = soilC05.reshape(N_MONTHS,180,2,360,2).mean(axis=(2,4)).astype(np.float32)
dbar_1  = drivers["dbar"].astype(np.float32)                     # (192,180,360)

# Keep 0.5-deg copies for final NC write (after study)
del gfed_ff_05   # not needed during loop
gc.collect()

print(f"[setup] Memory footprint OK. Running {N_TRIALS} trials with weights "
      f"(BA={W_BA}, fFire={W_FF}) ...\n")


# ─────────────────────────────────────────────────────────────────────────────
# Fast scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_BA_fast(pred_1):
    """Collier-4 ILAMB-style on 1-deg arrays. pred_1: (192,180,360)."""
    # Rescale pred to GFED land-only mean (matching reproduce_v2.py)
    w3_land = w3_1 * land_mask_1[None, :, :]
    pm = float((pred_1 * w3_land).sum() / (w3_land.sum() + 1e-12))
    om = float((gfed_ba_1 * w3_land).sum() / (w3_land.sum() + 1e-12))
    if pm > 0:
        pred_1 = pred_1 * (om / pm)

    pred_tm = pred_1.mean(axis=0)
    # Bias (all cells)
    rel  = np.abs(pred_tm - gfed_tm) / (np.abs(gfed_tm) + 1e-9)
    bias = float((np.exp(-rel) * w2_1).sum() / w2_1.sum())
    # RMSE
    rmse_f = np.sqrt(((pred_1 - gfed_ba_1) ** 2).mean(axis=0))
    rmse_s = float((np.exp(-rmse_f / gfed_std) * w2_1).sum() / w2_1.sum())
    # Seasonal (burnable only)
    pred_cyc = pred_1.reshape(16, 12, 180, 360).mean(axis=0)
    pa = pred_cyc - pred_cyc.mean(axis=0, keepdims=True)
    num = (pa * gfed_cyc_anom).sum(axis=0)
    den = np.sqrt((pa**2).sum(axis=0) * gfed_cyc_ss) + 1e-9
    corr = np.clip(num / den, -1, 1)
    seas = float(((1 + corr) / 2 * w2_1_burn).sum() / (w2_1_burn.sum() + 1e-12))
    # Spatial (Taylor, burnable)
    pred_mean = float((pred_1 * w3_1).sum() / w3_1.sum())
    pp = (pred_tm - pred_mean).flatten()
    oo = gfed_anom_flat
    rho = float((pp * oo * pw_flat).sum() /
                (np.sqrt(((pp**2)*pw_flat).sum() * ((oo**2)*pw_flat).sum()) + 1e-9))
    sr  = float(np.sqrt(((pp**2)*pw_flat).sum()/(pw_flat.sum()+1e-12)) /
                (np.sqrt(((oo**2)*pw_flat).sum()/(pw_flat.sum()+1e-12)) + 1e-9))
    spatial = float(((1 + max(rho, 0)) / 2) * min(sr, 1.0/sr))
    overall = float(np.mean([bias, rmse_s, seas, spatial]))
    return overall, (pred_1, dict(bias=bias, rmse=rmse_s, seas=seas, spatial=spatial, overall=overall))


def score_fFire_fast(pred_ff_1):
    """JASMIN-style: (Bias + Spatial)/2 on 1-deg arrays. pred_ff_1: (192,180,360)."""
    pred_tm = pred_ff_1.mean(axis=0)
    rel  = np.abs(pred_tm - gfed_ff_tm) / (np.abs(gfed_ff_tm) + 1e-12)
    bias = float((np.exp(-rel) * w2_ff).sum() / (w2_ff.sum() + 1e-12))
    pred_mean = float((pred_ff_1 * w3_1).sum() / w3_1.sum())
    pp = (pred_tm - pred_mean).flatten()
    oo = gfed_ff_anom_flat
    rho = float((pp * oo * pw_ff_flat).sum() /
                (np.sqrt(((pp**2)*pw_ff_flat).sum() * ((oo**2)*pw_ff_flat).sum()) + 1e-9))
    sr  = float(np.sqrt(((pp**2)*pw_ff_flat).sum()/(pw_ff_flat.sum()+1e-12)) /
                (np.sqrt(((oo**2)*pw_ff_flat).sum()/(pw_ff_flat.sum()+1e-12)) + 1e-9))
    spatial = float(((1 + max(rho, 0)) / 2) * min(sr, 1.0/sr))
    overall = 0.5 * bias + 0.5 * spatial
    return overall, dict(bias=bias, spatial=spatial, overall=overall)


def compute_ffire_1deg(pred_ba_1, phi, d_ref, b_leaf, b_fine, b_coarse, pf_fine):
    """Compute fFire [kg/m²/s] at 1-deg. PHI is a global fuel-availability prefactor."""
    # Pool-partitioning at 1-deg (AGB and soil_C already coarsened)
    pf_coarse = max(0.05, 0.95 - pf_fine)
    C_leaf   = 0.05     * agb_1
    C_fine   = pf_fine  * agb_1
    C_coarse = pf_coarse* agb_1
    C_lit    = 0.15     * soilC_1
    m_theta  = np.clip(dbar_1 / d_ref, 0.0, 1.0)
    C_combust = phi * (C_leaf   * (b_leaf   * m_theta) +
                       C_fine   * (b_fine   * m_theta) +
                       C_coarse * (b_coarse * m_theta) +
                       C_lit    * (b_leaf   * m_theta))  # litter uses leaf-like β
    fuel_consumed = pred_ba_1 * C_combust          # kg C / m² / month
    return (fuel_consumed / SEC_TILE).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    p = {
        # Burn params (Model C, 12)
        "k1":              trial.suggest_float("k1",              1e-5, 1e-2, log=True),
        "D_low":           trial.suggest_float("D_low",           10,   3000, log=True),
        "k2":              trial.suggest_float("k2",              1e-5, 1e-1, log=True),
        "D_high":          trial.suggest_float("D_high",          1000, 60000, log=True),
        "fire_exp":        trial.suggest_float("fire_exp",        0.5,  4.0),
        "P_half":          trial.suggest_float("P_half",          10,   5000, log=True),
        "pre_dampen_half": trial.suggest_float("pre_dampen_half", 0.1,  500,  log=True),
        "gpp_af":          trial.suggest_float("gpp_af",          0.1,  5.0),
        "gpp_b":           trial.suggest_float("gpp_b",           0.001,0.5, log=True),
        "gpp_d":           trial.suggest_float("gpp_d",           0.5,  50),
        "ign_k":           trial.suggest_float("ign_k",           0.005, 5.0, log=True),
        "ign_c":           trial.suggest_float("ign_c",           -10, 40),
    }
    # Combust params
    phi      = trial.suggest_float("PHI",             0.05, 1.0, log=True)
    d_ref    = trial.suggest_float("D_REF",           50, 2000, log=True)
    b_leaf   = trial.suggest_float("BETA_MAX_leaf",   0.50, 0.95)
    b_fine   = trial.suggest_float("BETA_MAX_fine",   0.30, 0.95)
    b_coarse = trial.suggest_float("BETA_MAX_coarse", 0.10, 0.60)
    pf_fine  = trial.suggest_float("POOL_FRAC_fine",  0.05, 0.25)

    # Model-C burntArea
    pred = fire_C(drivers, p)
    if not np.all(np.isfinite(pred)):
        return 2.0  # bad trial

    # Score BA (internally rescales pred to land mean)
    ba_overall, (pred_rescaled, _) = score_BA_fast(pred)

    # fFire from rescaled pred_BA
    pred_ff = compute_ffire_1deg(pred_rescaled, phi, d_ref, b_leaf, b_fine, b_coarse, pf_fine)
    ff_overall, _ = score_fFire_fast(pred_ff)

    loss = -(W_BA * ba_overall + W_FF * ff_overall)
    trial.set_user_attr("ba_overall", ba_overall)
    trial.set_user_attr("ff_overall", ff_overall)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Study
# ─────────────────────────────────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
t0 = time.time()

# Seed from Stage B best (if present) + PHI ≈ 2.0/3.18 ≈ 0.63 to hit GFED total
stageB_path = REPO / "models" / "C-stageB" / "params.json"
if stageB_path.exists():
    warm_params = dict(json.load(open(stageB_path))["params"])
    warm_params["PHI"] = 0.63
    print("[warm-start] seeded from Stage B best + PHI=0.63")
else:
    warm_params = json.load(open(REPO / "models" / "C" / "params.json"))["params"]
    warm_params.update({"PHI": 0.63, "D_REF": 1000.0,
                        "BETA_MAX_leaf": 0.80, "BETA_MAX_fine": 0.75,
                        "BETA_MAX_coarse": 0.30, "POOL_FRAC_fine": 0.10})

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
study.enqueue_trial(warm_params)

last_print = [time.time()]
def report(study, trial):
    if time.time() - last_print[0] > 20 or trial.number == N_TRIALS - 1:
        b = study.best_trial
        ba = b.user_attrs.get("ba_overall", 0)
        ff = b.user_attrs.get("ff_overall", 0)
        elapsed = (time.time() - t0) / 60
        print(f"  trial {trial.number+1:4d}/{N_TRIALS}  "
              f"best loss={b.value:+.4f}  BA={ba:.4f}  fF={ff:.4f}  "
              f"(elapsed {elapsed:.1f} min)")
        last_print[0] = time.time()

study.optimize(objective, n_trials=N_TRIALS, callbacks=[report])

best = study.best_trial
ba_o = best.user_attrs["ba_overall"]
ff_o = best.user_attrs["ff_overall"]
print(f"\n[done] {N_TRIALS} trials in {(time.time()-t0)/60:.1f} min")
print(f"       Best: loss={best.value:+.4f}  BA_overall={ba_o:.4f}  fFire_overall={ff_o:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Write best params + best NC outputs (burntArea 0.5-deg, fFire 0.5-deg)
# ─────────────────────────────────────────────────────────────────────────────
best_p = dict(best.params)
json_out = {
    "model":      "Model C-stageB2 (joint BA + fFire; with PHI prefactor)",
    "mechanisms": ["gpp_monthly", "precip", "t_air_ign"],
    "n_burn_params":    12,
    "n_combust_params":  6,
    "scores_internal": {"ba_overall": ba_o, "ffire_overall": ff_o,
                        "loss": float(best.value),
                        "weights": {"BA": W_BA, "fFire": W_FF}},
    "params":      best_p,
    "n_trials":    N_TRIALS,
    "runtime_min": (time.time() - t0) / 60,
}
(OUT_MODEL / "params.json").write_text(json.dumps(json_out, indent=2))
print(f"\n[write] {OUT_MODEL/'params.json'}")


# --- re-run best at 0.5-deg + write NC files -------------------------------
burn_p    = {k: v for k, v in best_p.items() if k in
             ("k1","D_low","k2","D_high","fire_exp","P_half","pre_dampen_half",
              "gpp_af","gpp_b","gpp_d","ign_k","ign_c")}
phi       = best_p["PHI"]
d_ref     = best_p["D_REF"];      b_leaf   = best_p["BETA_MAX_leaf"]
b_fine    = best_p["BETA_MAX_fine"]; b_coarse = best_p["BETA_MAX_coarse"]
pf_fine   = best_p["POOL_FRAC_fine"]; pf_coarse = max(0.05, 0.95 - pf_fine)

# 1-deg prediction
pred_1 = fire_C(drivers, burn_p)
w3_land = w3_1 * land_mask_1[None, :, :]
pm = float((pred_1 * w3_land).sum() / (w3_land.sum() + 1e-12))
om = float((gfed_ba_1 * w3_land).sum() / (w3_land.sum() + 1e-12))
pred_1_scaled = pred_1 * (om / pm) if pm > 0 else pred_1

# burntArea NC at 0.5-deg
pred_masked = pred_1_scaled.copy()
pred_masked[:, ~land_mask_1] = np.nan
pred_05 = uncoarsen(pred_masked).astype(np.float32)
times   = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
lat05   = np.arange(-89.75, 90.0, 0.5); lon05 = np.arange(-179.75, 180.0, 0.5)
ds_ba = xr.Dataset(
    {"burntArea": (("time","lat","lon"), pred_05,
                   {"units":"1", "standard_name":"burnt_area_fraction",
                    "long_name":"Burnt Area Fraction"})},
    coords={"time":("time",times), "lat":("lat",lat05), "lon":("lon",lon05)},
    attrs={"title":"ED-Model C-stageB2 burntArea (joint-optimized, PHI prefactor)"})
ds_ba = _add_time_bounds(ds_ba)
TIME_UNITS = "days since 2001-01-01 00:00:00"
enc = {"burntArea": {"zlib":True,"complevel":4,"_FillValue":1e20},
       "time":       {"units":TIME_UNITS,"calendar":"noleap","dtype":"float64"},
       "time_bounds":{"units":TIME_UNITS,"calendar":"noleap","dtype":"float64"}}
tmp = OUT_BA/"burntArea.nc.tmp"; ds_ba.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
os.replace(tmp, OUT_BA/"burntArea.nc")
print(f"[write] {OUT_BA/'burntArea.nc'}")

# fFire NC at 0.5-deg
# Recompute fFire at 0.5-deg (AGB/soilC 0.5-deg, dbar upsampled)
dbar_05 = np.repeat(np.repeat(dbar_1, 2, axis=1), 2, axis=2)   # (192,360,720)
pred_ba_05 = np.nan_to_num(pred_05, nan=0.0)
C_leaf   = 0.05      * agb05
C_fine   = pf_fine   * agb05
C_coarse = pf_coarse * agb05
C_lit    = 0.15      * soilC05
m_theta  = np.clip(dbar_05 / d_ref, 0.0, 1.0).astype(np.float32)
C_combust = phi * (C_leaf   * (b_leaf   * m_theta)
                 + C_fine   * (b_fine   * m_theta)
                 + C_coarse * (b_coarse * m_theta)
                 + C_lit    * (b_leaf   * m_theta))
SEC_TILE_05 = np.tile(SEC_PER_MON, 16).astype(np.float32)[:, None, None]
pred_ff_05 = (pred_ba_05 * C_combust / SEC_TILE_05).astype(np.float32)
pred_ff_05[:, ~np.repeat(np.repeat(land_mask_1, 2, axis=0), 2, axis=1)] = np.nan

# Build time_bounds for fFire file
def monthly_bnds():
    lo = [cftime.DatetimeNoLeap(y, m, 1) for y in YEARS for m in range(1,13)]
    hi = []
    for y in YEARS:
        for m in range(1,13):
            ny = y + (1 if m==12 else 0); nm = 1 if m==12 else m+1
            hi.append(cftime.DatetimeNoLeap(ny,nm,1))
    return np.array(list(zip(lo,hi)))
TBNDS = monthly_bnds()

ds_ff = xr.Dataset(
    {"fFire":      (("time","lat","lon"), pred_ff_05,
                    {"units":"kg m-2 s-1","long_name":"Fire Carbon Flux",
                     "standard_name":"fire_carbon_flux"}),
     "time_bounds":(("time","nv"), TBNDS)},
    coords={"time":("time",times,{"bounds":"time_bounds","standard_name":"time","axis":"T"}),
            "lat":("lat",lat05.astype(np.float64)),
            "lon":("lon",lon05.astype(np.float64))},
    attrs={"title":"ED-Model C-stageB2 fFire (joint-optimized, Stage B)",
           "method": "process-based; burntArea × PHI · Σ_k (C_k × β_k(θ)) / Δt",
           "optuna_trials": N_TRIALS,
           "weights":"BA=0.6, fFire=0.4",
           "Conventions":"CF-1.7"})
enc2 = dict(enc); enc2["fFire"] = enc2.pop("burntArea")
tmp = OUT_FF/"fFire.nc.tmp"; ds_ff.to_netcdf(tmp, encoding=enc2, format="NETCDF4_CLASSIC")
os.replace(tmp, OUT_FF/"fFire.nc")
print(f"[write] {OUT_FF/'fFire.nc'}")

# Total diagnostic
R = 6.371e6; dlon = np.deg2rad(0.5)
a1d = (R**2)*dlon*(np.sin(np.deg2rad(lat05+0.25))-np.sin(np.deg2rad(lat05-0.25)))
area05 = np.broadcast_to(a1d[:,None],(360,720)).astype(np.float64)
total = float(np.nansum(pred_ff_05*area05[None,:,:]*SEC_TILE_05[:,:,:]))/1e12/16
print(f"[diag] 2001-2016 annual global total fFire: {total:.3f} Pg/yr  (target 2.0)")
print(f"\nStage B complete.  Scores to compare to JASMIN best (CLM6.0 = 0.621):")
print(f"  ED-Model C-stageB2   fFire Overall = {ff_o:.4f}")
print(f"  ED-Model C-stageB2   BA    Overall = {ba_o:.4f}  (prev C = 0.7369)")
