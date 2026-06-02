"""
Refit Model C against GFED5 burned area, using the same ILAMB-weighted
Taylor-aware loss that produced the ED-ModelC-ILAMB ship variant for GFED4.

Reference: ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc (0.5°, % per month).
Inputs: global_baseline_modelC_inputs_1997-2016.nc (Lei's NC).
Train: 2001-2010. Test: 2011-2016 (GFED5 covers 2001-2020 so we have plenty).

Loss: 1 - (2*Bias + 2*RMSE + Seasonal + Spatial) / 6 with Taylor-aware spatial.
Seeded from ED-ModelC-ILAMB best (warm start from GFED4-tuned params).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, optuna, xarray as xr
import sys
sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs
from refit_modelA_multiobj import four_scores

REPO = Path(__file__).resolve().parents[1]


def load_gfed5_fraction(years):
    """Load GFED5 reference NC, slice to years, convert % to fraction."""
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    ds = xr.open_dataset(p)
    yrs = np.array([t.year for t in ds["time"].values])
    mask = np.isin(yrs, list(years))
    arr_pct = ds["burntArea"].values[mask]  # (N, 360, 720) in %
    arr_frac = (arr_pct / 100.0).astype(np.float32)
    return arr_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--timeout-h", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-gfed5"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C ILAMB-weighted Taylor-aware refit, GFED5 target")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_tr = load_gfed5_fraction(range(2001, 2011))
    print(f"  train shape {gfed_tr.shape}, mean fraction {float(np.nanmean(gfed_tr)):.4g}")

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
    print(f"  fire-active cells (GFED5): {land_fire.sum()} / {land_fire.size}")

    seed_p = json.load(open(REPO / "models" / "C-ilamb" / "params.ilambweighted.json"))["params"]

    # Score the seed first
    pred_seed = predict_monthly(d_tr, seed_p)
    b, r, s, sp = four_scores(pred_seed, gfed_tr, w2)
    seed_overall = (2 * b + 2 * r + s + sp) / 6.0
    print(f"  seed (ED-ModelC-ILAMB, GFED4-tuned) on GFED5: "
          f"overall={seed_overall:.4f} (bias={b:.3f} rmse={r:.3f} seas={s:.3f} spat={sp:.3f})")

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
        pred = predict_monthly(d_tr, params)
        bias, rmse, seas, spat = four_scores(pred, gfed_tr, w2)
        overall = (2 * bias + 2 * rmse + seas + spat) / 6.0
        trial.set_user_attr("bias_score", bias)
        trial.set_user_attr("rmse_score", rmse)
        trial.set_user_attr("seasonal_score", seas)
        trial.set_user_attr("spatial_score", spat)
        trial.set_user_attr("overall_score", overall)
        return float(1.0 - overall)

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                            multivariate=True, group=True,
                                            n_startup_trials=300)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(seed_p)

    print(f"\nTPE  n_trials={args.n_trials}  timeout={args.timeout_h:.1f}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    print(f"  done. n_trials={len(study.trials)}  wall={dt/60:.1f} min")

    best = study.best_trial
    print(f"\nBest #{best.number}: overall={best.user_attrs['overall_score']:.4f}")
    print(f"  bias={best.user_attrs['bias_score']:.4f}  rmse={best.user_attrs['rmse_score']:.4f}")
    print(f"  seas={best.user_attrs['seasonal_score']:.4f}  spat={best.user_attrs['spatial_score']:.4f}")

    json.dump({"model": "Model C, ILAMB-weighted Taylor-aware refit on Lei NC, GFED5 target",
               "loss": "1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6",
               "reference": "GFED5",
               "train_window": [2001, 2010], "params": best.params},
              open(out_dir / "params.gfed5.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "seed_overall": seed_overall,
               "best_params": best.params},
              open(out_dir / "refit_gfed5_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.gfed5.json")


if __name__ == "__main__":
    main()
