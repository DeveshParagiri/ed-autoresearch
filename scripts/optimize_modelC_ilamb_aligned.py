"""
optimize_modelC_ilamb_aligned.py
Second-pass retune. Same ED-consistent transform as
optimize_modelC_ed_consistent.py, but with a scorer that more closely matches
real ilamb-run's ConfBurntArea aggregation.

Differences from the prior fast scorer:
  - Bias score: exp(-|bias / obs_std|), not exp(-rel_error). Matches
    ILAMB's normaliser (per-cell time-std).
  - Spatial score: 4(1+R) / ((sigma + 1/sigma)^2 * (1+R0)), R0=1 — the exact
    Taylor-diagram formula from ILAMB Variable.py:2032.
  - Seasonal score: phase-shift form, (1 + cos(|peak_month_shift|/12 * 2pi))/2,
    matching ILAMB ilamblib.py:1300. Computed on per-cell month-of-peak.

Diagnostic from first pass: real ilamb-run scored our retune at
Bias=0.689 RMSE=0.476 Seasonal=0.643 Spatial=0.534 Overall=0.5639,
while the fast scorer reported 0.7790 / 0.7741 / 0.6491 / 0.6844 / 0.7216.
The 0.16 gap was driven by RMSE (1.6x too generous) and Spatial (1.3x). This
script targets that gap.

Cap is overridable via FIRE_MAX_RATE env var; default 5.0 (relaxed from 0.2).
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
FIRE_MAX_RATE = float(os.environ.get("FIRE_MAX_RATE", 5.0))
DT_YEARS      = 1.0
N_TRIALS      = int(os.environ.get("N_TRIALS", 2500))


# ── 1. Data loads ───────────────────────────────────────────────────────────
print(f"[setup] FIRE_MAX_RATE = {FIRE_MAX_RATE} yr^-1")
print("[setup] loading drivers + GFED ...")
drivers = load_drivers()
obs     = load_gfed_1deg()                                    # (192, 180, 360) monthly fraction

lat_1   = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
cos_lat = np.cos(np.deg2rad(lat_1)).astype(np.float32)
w2      = np.broadcast_to(cos_lat[:, None], (180, 360)).astype(np.float64)
w3      = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360)).astype(np.float64)

land_mask = (obs > 0).any(axis=0)
w2_burn   = (w2 * land_mask).astype(np.float64)
w3_land   = (w3 * land_mask[None, :, :]).astype(np.float64)
print(f"[setup] land cells: {int(land_mask.sum())} / {land_mask.size} "
      f"({100*land_mask.mean():.1f}%)")

# Pre-compute GFED quantities
gfed_tm     = obs.mean(axis=0).astype(np.float64)
gfed_std    = obs.std(axis=0).clip(1e-12).astype(np.float64)  # ILAMB normaliser ("crms")
gfed_cyc    = obs.reshape(16, 12, 180, 360).mean(axis=0)
gfed_peak_month = np.argmax(gfed_cyc, axis=0).astype(np.float32)
# ILAMB ConfBurntArea uses mass_weighting=True: spatial weight is cos(lat)*obs_mean.
# This means high-fire cells dominate the score; near-zero-fire cells contribute
# almost nothing.
mass_w = (w2 * gfed_tm).astype(np.float64)
mass_w_burn = (mass_w * land_mask).astype(np.float64)


# ── 2. ED-consistent transform ──────────────────────────────────────────────
def ed_transform(rate_yr):
    rate_capped = np.minimum(rate_yr, FIRE_MAX_RATE)
    annual_frac = 1.0 - np.exp(-rate_capped * DT_YEARS)
    return (annual_frac / 12.0).astype(np.float32)


# ── 3. ILAMB-aligned scoring ────────────────────────────────────────────────
def score_BA_ilamb(pred_monthly):
    """Bias / RMSE / Seasonal / Spatial scores aligned with ILAMB's ConfBurntArea."""
    pred_tm = pred_monthly.mean(axis=0)

    # -- Bias score: exp(-|bias / obs_std|), MASS-weighted spatial mean ----
    # ILAMB uses weight = cos(lat) * obs_time_mean (mass_weighting=True).
    bias_per_cell = pred_tm - gfed_tm
    bias_score_per_cell = np.exp(-np.abs(bias_per_cell) / gfed_std)
    bias = float((bias_score_per_cell * mass_w).sum() / (mass_w.sum() + 1e-12))

    # -- RMSE score: ILAMB uses CENTERED RMSE (anomaly vs anomaly), then
    #    normalises by obs centered RMS (= obs time-std). See ilamblib:1503.
    pred_anom = pred_monthly - pred_tm[None, :, :]
    obs_anom  = obs - gfed_tm[None, :, :]
    crmse_per_cell = np.sqrt(((pred_anom - obs_anom) ** 2).mean(axis=0))
    rmse_score_per_cell = np.exp(-crmse_per_cell / gfed_std)
    rmse_s = float((rmse_score_per_cell * mass_w).sum() / (mass_w.sum() + 1e-12))

    # -- Seasonal score: phase-shift, ILAMB style --------------------------
    # peak month per cell, then |shift| with wrap
    pred_cyc = pred_monthly.reshape(16, 12, 180, 360).mean(axis=0)
    pred_peak = np.argmax(pred_cyc, axis=0).astype(np.float32)
    shift = pred_peak - gfed_peak_month
    shift = np.where(shift >  6, shift - 12, shift)
    shift = np.where(shift < -6, shift + 12, shift)
    seas_per_cell = (1.0 + np.cos(np.abs(shift) / 12.0 * 2.0 * np.pi)) * 0.5
    # ILAMB mass-weights seasonal too, with normalizer = obs_time_mean.
    seas = float((seas_per_cell * mass_w_burn).sum() / (mass_w_burn.sum() + 1e-12))

    # -- Spatial score: ILAMB Taylor formula -------------------------------
    # std0 = obs spatial std (over masked land cells), std = pred spatial std
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

    overall = float(np.mean([bias, rmse_s, seas, spatial]))
    return overall, dict(bias=bias, rmse=rmse_s, seas=seas, spatial=float(spatial),
                         overall=overall)


# ── 4. Forward model ────────────────────────────────────────────────────────
def predict(params):
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(drivers, params)
    rate = rate * land_mask[None, :, :]
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

WARM_START = json.load(open(MODELS_DIR / "params.json"))["params"]


def objective(trial):
    p = {name: trial.suggest_float(name, lo, hi, log=True)
         for name, (lo, hi) in LOG_PARAMS.items()}
    pred = predict(p)
    overall, _ = score_BA_ilamb(pred)
    return -overall


def main():
    backup = MODELS_DIR / "params.PRE-ilamb-aligned.json"
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

    print(f"[opt] running {N_TRIALS} trials with ILAMB-aligned scorer ...")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
    elapsed = (time.time() - t0) / 60.0
    best_overall = -study.best_value
    print(f"\n[done] {len(study.trials)} trials in {elapsed:.1f} min")
    print(f"       best Overall = {best_overall:.4f}")

    best_p = study.best_params
    pred   = predict(best_p)
    overall, breakdown = score_BA_ilamb(pred)
    print(f"[best] Bias={breakdown['bias']:.4f}  RMSE={breakdown['rmse']:.4f}  "
          f"Seas={breakdown['seas']:.4f}  Spatial={breakdown['spatial']:.4f}  "
          f"Overall={overall:.4f}")
    raw = fire_C(drivers, best_p) * land_mask[None, :, :]
    raw_lm = float((raw * w3_land).sum() / w3_land.sum())
    print(f"[diag] raw rate land-mean (yr^-1): {raw_lm:.4g}")
    print(f"[diag] cap binding cells (rate > {FIRE_MAX_RATE}): "
          f"{int(((raw > FIRE_MAX_RATE) & land_mask[None,:,:]).sum())}")

    out = {
        "model":      "Model C (ED-coupled-consistent + ILAMB-aligned scorer; cap %g/yr)" % FIRE_MAX_RATE,
        "mechanisms": ["gpp_monthly", "precip", "t_air_ign"],
        "n_mechanisms": 3,
        "n_params":   12,
        "objective":  "ILAMB-aligned: Bias=exp(-|b|/std), RMSE=exp(-rmse/std), Seas=phase-shift, Spatial=Taylor",
        "ed_consistency": {
            "interpretation": "Model C output is annual fire rate (yr^-1)",
            "fire_max_rate":   FIRE_MAX_RATE,
            "transform":       "burnt_monthly_frac = (1 - exp(-min(rate, fire_max) * 1yr)) / 12",
        },
        "scores_internal_ilamb_aligned": breakdown,
        "n_trials":    len(study.trials),
        "runtime_min": round(elapsed, 2),
        "params":      best_p,
    }
    (MODELS_DIR / "params.json").write_text(json.dumps(out, indent=2))
    print(f"[write] {MODELS_DIR / 'params.json'}")

    pred_hd = uncoarsen(np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat   = np.arange(-89.75, 90.0, 0.5)
    lon   = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time","lat","lon"), pred_hd,
                       {"units":"1", "standard_name":"burnt_area_fraction",
                        "long_name":"Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (ED-consistent + ILAMB-aligned)",
               "Conventions": "CF-1.7",
               "transform": f"monthly_frac = (1 - exp(-min(rate_yr, {FIRE_MAX_RATE}) * 1yr)) / 12"})
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
