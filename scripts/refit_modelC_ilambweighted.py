"""
Single-objective Model C retune with ILAMB-style tier-2 weighted score as loss.

Loss = 1 - (2*Bias + 2*RMSE + Seasonal + Spatial) / 6
This is the exact aggregation ILAMB uses to compute its Overall Score in
scalar_database.csv. Optimizing this directly should better match ILAMB's
ranking than multi-obj NSGA-II over proxy metrics.

12 params, Lei's NC inputs (coupled-consistent), train 2001-2010.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, optuna, xarray as xr
import sys
sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs, load_gfed
from refit_modelA_multiobj import four_scores

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--timeout-h", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-ilamb"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C single-objective ILAMB-weighted refit on Lei's NC")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_tr = load_gfed(range(2001, 2011))

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    # Seed point: Model C-Ours magaware-annual best params (already a strong starting point)
    seed_p = json.load(open(REPO / "models" / "C" / "params.lei-magaware-annual.json"))["params"]

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

    json.dump({"model": "Model C, single-obj ILAMB-weighted refit on Lei NC",
               "loss": "1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6",
               "train_window": [2001, 2010], "params": best.params},
              open(out_dir / "params.ilambweighted.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "best_params": best.params},
              open(out_dir / "refit_ilambweighted_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.ilambweighted.json")


if __name__ == "__main__":
    main()
