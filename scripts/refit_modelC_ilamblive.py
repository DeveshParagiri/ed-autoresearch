"""
Single-objective Model C retune with REAL ILAMB-in-the-loop scoring.

Each trial:
  1. Predict burned area at 0.5deg from candidate params
  2. Write to a single-model dir under a tmp leaderboard
  3. Run ilamb-run --skip_plots on that single model
  4. Parse Overall Score from scalar_database.csv
  5. Loss = 1 - Overall Score

Slow (~5-10 sec per trial) but bulletproof: no proxy mismatch.
Seeded from ED-ModelC-ILAMB best params.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, tempfile
from pathlib import Path
import cftime, numpy as np, optuna, xarray as xr
import pandas as pd

sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs

REPO = Path(__file__).resolve().parents[1]
ILAMB_REF = REPO / "ilamb_ref_official"
ILAMB_CFG = REPO / "ilamb" / "burntArea_official.cfg"


def write_burnt_nc(pred_monthly_05deg, lat, lon, year_start, year_end, out_path, title):
    times = [cftime.DatetimeNoLeap(y, m, 15)
             for y in range(year_start, year_end + 1) for m in range(1, 13)]
    ds = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), pred_monthly_05deg.astype(np.float32),
                       {"units": "1", "standard_name": "burnt_area_fraction",
                        "long_name": "Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": title, "Conventions": "CF-1.7"})
    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds = ds.assign(time_bounds=(("time", "nb"), tb))
    ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    ds = ds.assign(lat_bounds=(("lat", "nb"), np.stack([lat - 0.25, lat + 0.25], axis=1)))
    ds.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
    ds = ds.assign(lon_bounds=(("lon", "nb"), np.stack([lon - 0.25, lon + 0.25], axis=1)))
    ds.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east", "standard_name": "longitude", "axis": "X"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_units = f"days since {year_start}-01-01 00:00:00"
    enc = {"burntArea": {"zlib": False, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    ds.to_netcdf(out_path, encoding=enc, format="NETCDF4_CLASSIC")


def score_via_ilamb(burnt_3d, lat, lon, year_start, year_end, work_root):
    """Run ilamb on one synthetic model dir, return Overall Score."""
    model_root = work_root / "MODELS"
    model_dir = model_root / "Trial"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    write_burnt_nc(burnt_3d, lat, lon, year_start, year_end,
                   model_dir / "burntArea.nc", "Trial")
    build_dir = work_root / "out"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    env = os.environ.copy()
    env["ILAMB_ROOT"] = str(ILAMB_REF)
    cmd = ["ilamb-run", "--config", str(ILAMB_CFG),
           "--model_root", str(model_root),
           "--regions", "global",
           "--build_dir", str(build_dir),
           "--skip_plots"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        # ILAMB sometimes returns non-zero due to post-processing failures
        # (e.g., ilamb.cfg not found for JSON generation), but the scalar CSV
        # is written before that. Don't bail on rc != 0; check the CSV.
    except subprocess.SubprocessError as e:
        print(f"  subprocess err: {e}", flush=True)
        return None
    csv = build_dir / "scalar_database.csv"
    if not csv.exists():
        print(f"  csv missing at {csv}", flush=True)
        return None
    df = pd.read_csv(csv)
    print(f"  csv has {len(df)} rows", flush=True)
    df = df[(df["Region"] == "global") & (df["Source"] == "GFED4.1S") & (df["Model"] == "Trial")]
    out = {}
    for name in ("Bias Score", "RMSE Score", "Seasonal Cycle Score",
                 "Spatial Distribution Score"):
        rows = df[df["ScalarName"] == name]
        if len(rows): out[name] = float(rows.iloc[0]["Data"])
    # ILAMB computes Overall Score across multiple models in a leaderboard;
    # for a single-model run we compute it ourselves from tier-2 weights:
    #   Overall = (2*Bias + 2*RMSE + Seasonal + Spatial) / 6
    if all(k in out for k in ("Bias Score", "RMSE Score", "Seasonal Cycle Score", "Spatial Distribution Score")):
        out["Overall Score"] = (2 * out["Bias Score"] + 2 * out["RMSE Score"]
                                 + out["Seasonal Cycle Score"] + out["Spatial Distribution Score"]) / 6.0
    if not out:
        return None
    return out


def predict_05deg(d_full, params):
    """Predict at full Lei NC resolution (already 0.5 deg)."""
    return predict_monthly(d_full, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=3000)
    ap.add_argument("--timeout-h", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-live"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C ILAMB-IN-THE-LOOP refit. Real ILAMB scoring per trial.")

    print("Loading drivers from Lei's NC (full 1997-2016, but predict 2001-2016)...")
    ds_full = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
    yr_all = np.array([d.year for d in ds_full["time"].values])
    mask_2001_2016 = (yr_all >= 2001) & (yr_all <= 2016)
    d_full = load_inputs(ds_full.isel(time=mask_2001_2016))
    lat = ds_full["lat"].values
    lon = ds_full["lon"].values
    # Land mask from area_frac to keep NaNs aligned
    land = (d_full["frac"][0] + d_full["frac"][1] + d_full["frac"][2]) > 0
    land_any = land.any(axis=0)  # (lat, lon)

    work_root = Path(tempfile.mkdtemp(prefix="ilamblive_", dir="/tmp"))
    print(f"  work_root = {work_root}")

    seed_p = json.load(open(REPO / "models" / "C-ilamb" / "params.ilambweighted.json"))["params"]

    # Score the seed first so we know our starting point
    pred_seed = predict_05deg(d_full, seed_p)
    pred_seed = np.where(land_any[None, :, :], pred_seed, np.nan).astype(np.float32)
    seed_scores = score_via_ilamb(pred_seed, lat, lon, 2001, 2016, work_root)
    print(f"  seed ILAMB Overall = {seed_scores['Overall Score']:.4f}" if seed_scores else "  seed scoring failed")

    def objective(trial):
        params = dict(
            k1     = trial.suggest_float("k1",     1e-6, 1e-2, log=True),
            D_low  = trial.suggest_float("D_low",  1e2,  1e5,  log=True),
            k2     = trial.suggest_float("k2",     1e-7, 1e-3, log=True),
            D_high = trial.suggest_float("D_high", 1e4,  5e6,  log=True),
            fire_exp = trial.suggest_float("fire_exp", 0.1, 2.0),
            P_half          = trial.suggest_float("P_half",          1e0, 5e3, log=True),
            pre_dampen_half = trial.suggest_float("pre_dampen_half", 1e-1, 1e2, log=True),
            gpp_af = trial.suggest_float("gpp_af", 1e-2, 1e1, log=True),
            gpp_b  = trial.suggest_float("gpp_b",  1e-3, 1e1, log=True),
            gpp_d  = trial.suggest_float("gpp_d",  1e-1, 1e3, log=True),
            ign_k  = trial.suggest_float("ign_k",  1e-2, 1e1, log=True),
            ign_c  = trial.suggest_float("ign_c", -10.0, 35.0),
        )
        if params["D_high"] <= params["D_low"] * 1.5:
            return 1.0
        pred = predict_05deg(d_full, params)
        pred = np.where(land_any[None, :, :], pred, np.nan).astype(np.float32)
        scores = score_via_ilamb(pred, lat, lon, 2001, 2016, work_root)
        if scores is None:
            return 1.0
        for k, v in scores.items():
            trial.set_user_attr(k.replace(" ", "_"), v)
        overall = scores.get("Overall Score", 0.0)
        return float(1.0 - overall)

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                            multivariate=True, group=True,
                                            n_startup_trials=50)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(seed_p)
    try:
        seed_mag = json.load(open(REPO / "models" / "C" / "params.lei-magaware-annual.json"))["params"]
        study.enqueue_trial(seed_mag)
    except Exception:
        pass

    print(f"\nTPE  n_trials={args.n_trials}  timeout={args.timeout_h:.1f}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    print(f"  done. n_trials={len(study.trials)}  wall={dt/60:.1f} min")

    best = study.best_trial
    print(f"\nBest #{best.number}: ILAMB Overall = {1 - best.value:.4f}")
    for k in ("Bias_Score", "RMSE_Score", "Seasonal_Cycle_Score",
              "Spatial_Distribution_Score", "Overall_Score"):
        v = best.user_attrs.get(k, "?")
        print(f"  {k:30s} {v}")

    json.dump({"model": "Model C, ILAMB-in-the-loop refit on Lei NC",
               "loss": "1 - actual_ILAMB_Overall_Score",
               "train_window": [2001, 2016],
               "params": best.params},
              open(out_dir / "params.live.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs.get(k) for k in ("Bias_Score", "RMSE_Score", "Seasonal_Cycle_Score", "Spatial_Distribution_Score", "Overall_Score")},
               "best_params": best.params},
              open(out_dir / "refit_live_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.live.json")

    shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
