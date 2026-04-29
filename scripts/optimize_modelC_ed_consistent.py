"""
optimize_modelC_ed_consistent.py
Retune Model C against an ED-coupled-consistent objective.

Differences from the prior calibration:
  - Model C output is interpreted as an annual disturbance rate (yr^-1), NOT a
    monthly burnt-fraction.
  - Cap at fire_max_disturbance_rate = 0.2 yr^-1 (matches fire.cc).
  - Apply ED's saturating annual->fractional transform:
        burnt_annual = 1 - exp(-rate_capped * 1 yr)
  - Distribute uniformly across the year for TRENDY-compatible monthly output:
        burnt_monthly_fraction = burnt_annual / 12
  - Score that against GFED monthly fraction (no post-hoc land-mean rescale).

Background: in the previous pipeline the optimiser fit Model C output directly
to GFED monthly fraction, then a post-hoc land-mean rescale absorbed both unit
mismatch (x12) and absolute-magnitude calibration. When the resulting "rate" is
plugged into ED as disturbance_rate[1], ED's exponential saturation and yearly
patch_dynamics produce values systematically smaller than what the offline
pipeline trained for. Per Lei (Apr 27 2026), this is what was breaking the
coupled propagation.

Note on the partitioning logic in disturbance.cc/patch.cc: ED's max-compare
(fire vs treefall) plus partitioning through total_disturbance_rate algebraically
collapses to area_burned = patch_area * (1 - exp(-fire_rate * 1yr)) for the
TRENDY burntArea diagnostic, so we do NOT need to model treefall here. The
max-compare matters for cohort mortality and patch-creation accounting, not
for the area_burned output.

Output:
  models/C/params.json                    (overwrite — this becomes "the" Model C)
  ilamb/MODELS/ED-ModelC-final/burntArea.nc   (regenerated under new transform)
  models/C/params.OLD-rescale.json        (backup of previous params)
"""
from __future__ import annotations
import gc, json, os, sys, time, shutil
from pathlib import Path

import cftime
import h5py
import numpy as np
import optuna
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_modelC import fire_C, load_drivers, load_gfed_1deg, add_cf_bounds, uncoarsen

REPO         = Path(__file__).resolve().parents[1]
MODELS_DIR   = REPO / "models" / "C"
ILAMB_OUT_NC = REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc"
ILAMB_OUT_NC.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

YEARS         = list(range(2001, 2017))
N_MONTHS      = 192
FIRE_MAX_RATE = float(os.environ.get("FIRE_MAX_RATE", 0.2))
# default 0.2 (ED_params.defaults.cfg:137); set FIRE_MAX_RATE=5.0 to relax for offline
# tuning when the cap binds and Spatial score collapses (Lei: "we could consider
# relaxing it in ED and offline models").
DT_YEARS      = 1.0          # ED's patch_dynamics integrates over 1 year
N_TRIALS      = int(os.environ.get("N_TRIALS", 2500))


# ── 1. Data loads ───────────────────────────────────────────────────────────
print("[setup] loading drivers + GFED ...")
drivers = load_drivers()                                      # 1-deg
obs     = load_gfed_1deg()                                    # (192, 180, 360) monthly fraction

lat_1   = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
cos_lat = np.cos(np.deg2rad(lat_1)).astype(np.float32)
w2      = np.broadcast_to(cos_lat[:, None], (180, 360)).astype(np.float32)
w3      = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360)).astype(np.float32)

land_mask = (obs > 0).any(axis=0)                             # cells that ever burned in GFED
w2_burn   = (w2 * land_mask).astype(np.float32)
w3_land   = (w3 * land_mask[None, :, :]).astype(np.float32)
print(f"[setup] land cells: {int(land_mask.sum())} / {land_mask.size} "
      f"({100*land_mask.mean():.1f}%)")

# Pre-compute GFED quantities (constant across trials)
gfed_tm        = obs.mean(axis=0)
gfed_std       = obs.std(axis=0) + 1e-9
gfed_cyc       = obs.reshape(16, 12, 180, 360).mean(axis=0)
gfed_cyc_anom  = gfed_cyc - gfed_cyc.mean(axis=0, keepdims=True)
gfed_cyc_ss    = (gfed_cyc_anom ** 2).sum(axis=0)
gfed_mean      = float((obs * w3).sum() / w3.sum())
gfed_anom_flat = (gfed_tm - gfed_mean).flatten()
pw_flat        = w2_burn.flatten()


# ── 2. ED-consistent transform ──────────────────────────────────────────────
def ed_transform(rate_yr):
    """Map an annual fire rate (yr^-1) to TRENDY-format monthly burnt fraction.

        rate_capped = min(rate_yr, fire_max_disturbance_rate)
        annual_frac = 1 - exp(-rate_capped * 1 yr)
        monthly     = annual_frac / 12

    This exactly mirrors what ED writes to TRENDY burntArea.nc, given the
    max-compare collapse explained in the module docstring.
    """
    rate_capped = np.minimum(rate_yr, FIRE_MAX_RATE)
    annual_frac = 1.0 - np.exp(-rate_capped * DT_YEARS)
    return (annual_frac / 12.0).astype(np.float32)


# ── 3. Scoring (Collier-4 — match ILAMB ConfBurntArea aggregation) ──────────
def score_BA(pred_monthly):
    """Score predicted monthly fraction against GFED. NO post-hoc rescale."""
    pred_tm = pred_monthly.mean(axis=0)
    # Bias
    rel  = np.abs(pred_tm - gfed_tm) / (np.abs(gfed_tm) + 1e-9)
    bias = float((np.exp(-rel) * w2).sum() / w2.sum())
    # RMSE
    rmse_f = np.sqrt(((pred_monthly - obs) ** 2).mean(axis=0))
    rmse_s = float((np.exp(-rmse_f / gfed_std) * w2).sum() / w2.sum())
    # Seasonal cycle (burnable cells only)
    pred_cyc = pred_monthly.reshape(16, 12, 180, 360).mean(axis=0)
    pa  = pred_cyc - pred_cyc.mean(axis=0, keepdims=True)
    num = (pa * gfed_cyc_anom).sum(axis=0)
    den = np.sqrt((pa**2).sum(axis=0) * gfed_cyc_ss) + 1e-9
    corr = np.clip(num / den, -1, 1)
    seas = float(((1 + corr) / 2 * w2_burn).sum() / (w2_burn.sum() + 1e-12))
    # Spatial (Taylor, burnable)
    pred_mean = float((pred_monthly * w3).sum() / w3.sum())
    pp = (pred_tm - pred_mean).flatten()
    rho = float((pp * gfed_anom_flat * pw_flat).sum() /
                (np.sqrt(((pp**2)*pw_flat).sum() *
                         ((gfed_anom_flat**2)*pw_flat).sum()) + 1e-9))
    sr  = float(np.sqrt(((pp**2)*pw_flat).sum()/(pw_flat.sum()+1e-12)) /
                (np.sqrt(((gfed_anom_flat**2)*pw_flat).sum()/(pw_flat.sum()+1e-12)) + 1e-9))
    sr_safe = max(sr, 1e-12)
    spatial = float(((1 + max(rho, 0)) / 2) * min(sr_safe, 1.0/sr_safe))
    overall = float(np.mean([bias, rmse_s, seas, spatial]))
    return overall, dict(bias=bias, rmse=rmse_s, seas=seas, spatial=spatial,
                         overall=overall)


# ── 4. Forward model: drivers + params -> monthly fraction ──────────────────
def predict(params):
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(drivers, params)                     # raw model output (yr^-1 by interpretation)
    rate = rate * land_mask[None, :, :]                    # mask ocean
    return ed_transform(rate)


# ── 5. Optuna objective ─────────────────────────────────────────────────────
LOG_PARAMS = {
    "k1":              (1e-5, 1e-1),
    "D_low":           (1e0,  1e4),
    "k2":              (1e-5, 1e-1),
    "D_high":          (1e1,  1e5),
    "fire_exp":        (0.5,  10.0),
    "P_half":          (1e0,  1e4),
    "pre_dampen_half": (1e-2, 1e3),
    "gpp_af":          (1e-3, 1e2),
    "gpp_b":           (1e-5, 1e1),
    "gpp_d":           (1e-2, 1e3),
    "ign_k":           (1e-3, 1e1),
    "ign_c":           (1e-1, 1e3),
}

# Warm start: take the existing Model C params (which were fit under the old
# rescale interpretation) and bump them so the raw output is roughly in the
# right yr^-1 regime. The diagnostic at the top of this session showed raw
# output land-mean was ~3.5e-5; we want ~0.04. Optuna will refine.
WARM_START = json.load(open(MODELS_DIR / "params.json"))["params"]


def objective(trial):
    p = {name: trial.suggest_float(name, lo, hi, log=True)
         for name, (lo, hi) in LOG_PARAMS.items()}
    pred = predict(p)
    overall, _ = score_BA(pred)
    return -overall


def main():
    # Backup existing params first
    backup = MODELS_DIR / "params.OLD-rescale.json"
    if not backup.exists():
        shutil.copy(MODELS_DIR / "params.json", backup)
        print(f"[backup] {backup}")

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(WARM_START)

    t0 = time.time()
    last_print = t0

    def cb(st, tr):
        nonlocal last_print
        if time.time() - last_print > 20:
            best = -st.best_value
            elapsed = (time.time() - t0) / 60.0
            print(f"  trial {len(st.trials):4d}/{N_TRIALS}  best_overall={best:.4f}  "
                  f"({elapsed:.1f} min)")
            last_print = time.time()

    print(f"[opt] running {N_TRIALS} trials ...")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    elapsed = (time.time() - t0) / 60.0
    best_overall = -study.best_value
    print(f"\n[done] {len(study.trials)} trials in {elapsed:.1f} min")
    print(f"       best Overall = {best_overall:.4f}")

    # Score the best
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

    # Write new params.json
    out = {
        "model": "Model C (ED-coupled-consistent retune; output in yr^-1, ED applies (1-exp)/12)",
        "mechanisms": ["gpp_monthly", "precip", "t_air_ign"],
        "n_mechanisms": 3,
        "n_params": 12,
        "objective": "fast Collier-4 (Bias + RMSE + Seas + Spatial) / 4 against GFED monthly fraction",
        "ed_consistency": {
            "interpretation": "Model C output is an annual fire rate (yr^-1)",
            "fire_max_rate":   FIRE_MAX_RATE,
            "transform":       "burnt_monthly_frac = (1 - exp(-min(rate, fire_max) * 1yr)) / 12",
            "max_compare_note": "ED's max-compare (fire vs treefall) algebraically collapses to (1-exp(-fire)) for area_burned diagnostic.",
        },
        "scores_internal": breakdown,
        "n_trials":   len(study.trials),
        "runtime_min": round(elapsed, 2),
        "params":     best_p,
    }
    (MODELS_DIR / "params.json").write_text(json.dumps(out, indent=2))
    print(f"[write] {MODELS_DIR / 'params.json'}")

    # Regenerate burntArea.nc using the new params + new transform
    pred_hd = uncoarsen(np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat   = np.arange(-89.75, 90.0, 0.5)
    lon   = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time","lat","lon"), pred_hd,
                       {"units":"1", "standard_name":"burnt_area_fraction",
                        "long_name":"Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (ED-coupled-consistent retune)",
               "Conventions": "CF-1.7",
               "transform": "monthly_frac = (1 - exp(-min(rate_yr, 0.2) * 1yr)) / 12"})
    ds = add_cf_bounds(ds)
    enc = {"burntArea":   {"zlib":True, "complevel":4, "_FillValue":1e20},
           "time":        {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"},
           "time_bounds": {"units":"days since 2001-01-01 00:00:00", "calendar":"noleap", "dtype":"float64"}}
    tmp = ILAMB_OUT_NC.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, ILAMB_OUT_NC)
    print(f"[write] {ILAMB_OUT_NC}  ({ILAMB_OUT_NC.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
