"""
Single-objective Model C retune with ILAMB-faithful component scoring.

The four_scores in refit_modelA_multiobj uses GLOBAL aggregation for Bias and
RMSE, which diverges from ILAMB's CELL-WISE aggregation. This script uses
cell-wise formulas exactly mirroring ILAMB's ConfBurntArea, so the optimizer
loss matches ILAMB's actual scoring.

Components (ILAMB-faithful):
  Bias score   = weighted mean of exp(-|pred_tm - obs_tm| / |obs_tm|) per cell
  RMSE score   = weighted mean of exp(-rmse_per_cell / obs_std_per_cell) per cell
  Seasonal     = weighted mean of (1 + corr_per_cell(monthly clim)) / 2
  Spatial      = Taylor: 0.5*(1+r) * exp(-|log(sigma_p/sigma_o)|)

Loss = 1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6 (ILAMB tier-2 weighting)

Seeded from iteration 3's best (ED-ModelC-ILAMB params).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, optuna, xarray as xr
import sys
sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs, load_gfed

REPO = Path(__file__).resolve().parents[1]


def faithful_four_scores(pred_monthly, obs_monthly, w2):
    """ILAMB-faithful Bias, RMSE, Seasonal, Spatial. All cell-wise then weighted."""
    n = pred_monthly.shape[0]
    pred_monthly = np.nan_to_num(pred_monthly, nan=0.0, posinf=0.0, neginf=0.0)
    obs_monthly  = np.nan_to_num(obs_monthly,  nan=0.0, posinf=0.0, neginf=0.0)

    p_tm = pred_monthly.mean(axis=0)
    o_tm = obs_monthly.mean(axis=0)
    w_sum = float(w2.sum()) + 1e-12

    # Bias: cell-wise relative bias, exp(-...), then weighted mean
    abs_o = np.abs(o_tm)
    rel_bias = np.abs(p_tm - o_tm) / (abs_o + 1e-9)
    bias_field = np.exp(-rel_bias)
    bias_score = float((bias_field * w2).sum() / w_sum)
    if not np.isfinite(bias_score): bias_score = 0.0

    # RMSE: cell-wise RMSE / cell-wise obs_std, exp(-...), then weighted mean
    rmse_per_cell = np.sqrt(((pred_monthly - obs_monthly) ** 2).mean(axis=0))
    obs_std = obs_monthly.std(axis=0) + 1e-9
    norm_rmse = rmse_per_cell / obs_std
    rmse_field = np.exp(-norm_rmse)
    rmse_score = float((rmse_field * w2).sum() / w_sum)
    if not np.isfinite(rmse_score): rmse_score = 0.0

    # Seasonal: cell-wise correlation of monthly climatology, (1+r)/2, weighted mean
    n_yr = n // 12
    pred_clim = pred_monthly.reshape(n_yr, 12, *pred_monthly.shape[1:]).mean(axis=0)
    obs_clim  = obs_monthly.reshape(n_yr, 12, *obs_monthly.shape[1:]).mean(axis=0)
    pa = pred_clim - pred_clim.mean(axis=0, keepdims=True)
    oa = obs_clim  - obs_clim.mean(axis=0, keepdims=True)
    num = (pa * oa).sum(axis=0)
    den = np.sqrt((pa**2).sum(axis=0) * (oa**2).sum(axis=0)) + 1e-30
    corr = np.clip(num / den, -1, 1)
    valid_seas = (den > 1e-20) & (w2 > 0)
    if valid_seas.sum() == 0:
        seasonal_score = 0.0
    else:
        seasonal_score = float((((1 + corr) / 2 * w2)[valid_seas]).sum() / w2[valid_seas].sum())
    if not np.isfinite(seasonal_score): seasonal_score = 0.0

    # Spatial Taylor: weighted Pearson over time-mean + std-ratio penalty
    m = (w2 > 0)
    p_v = p_tm[m]; o_v = o_tm[m]; w_v = w2[m]
    pm = (p_v * w_v).sum() / w_v.sum(); om = (o_v * w_v).sum() / w_v.sum()
    cov = (w_v * (p_v - pm) * (o_v - om)).sum() / w_v.sum()
    vp  = (w_v * (p_v - pm) ** 2).sum() / w_v.sum()
    vo  = (w_v * (o_v - om) ** 2).sum() / w_v.sum()
    r = float(cov / (np.sqrt(vp * vo) + 1e-30))
    sigma_p = float(np.sqrt(vp)); sigma_o = float(np.sqrt(vo))
    sigma_ratio = (sigma_p + 1e-12) / (sigma_o + 1e-12)
    std_penalty = float(np.exp(-abs(np.log(max(sigma_ratio, 1e-9)))))
    spatial_score = float(((1 + r) / 2) * std_penalty)
    if not np.isfinite(spatial_score): spatial_score = 0.0

    return bias_score, rmse_score, seasonal_score, spatial_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=15000)
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-faithful"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C ILAMB-FAITHFUL refit. Cell-wise Bias and RMSE.")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_tr = load_gfed(range(2001, 2011))

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")

    # Seed from ED-ModelC-ILAMB best (iter 3)
    seed_p = json.load(open(REPO / "models" / "C-ilamb" / "params.ilambweighted.json"))["params"]

    # Seed score
    pred_seed = predict_monthly(d_tr, seed_p)
    b_s, r_s, sea_s, spa_s = faithful_four_scores(pred_seed, gfed_tr, w2)
    overall_seed = (2 * b_s + 2 * r_s + sea_s + spa_s) / 6.0
    print(f"  seed faithful overall = {overall_seed:.4f}  "
          f"(bias={b_s:.3f} rmse={r_s:.3f} seas={sea_s:.3f} spat={spa_s:.3f})")

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
        bias, rmse, seas, spat = faithful_four_scores(pred, gfed_tr, w2)
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
    # Also seed with magaware-annual best
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
    print(f"\nBest #{best.number}: overall={best.user_attrs['overall_score']:.4f}")
    print(f"  bias={best.user_attrs['bias_score']:.4f}  rmse={best.user_attrs['rmse_score']:.4f}")
    print(f"  seas={best.user_attrs['seasonal_score']:.4f}  spat={best.user_attrs['spatial_score']:.4f}")

    json.dump({"model": "Model C, ILAMB-faithful single-obj refit on Lei NC",
               "loss": "1 - (2*Bias_cw + 2*RMSE_cw + Seasonal_cw + Spatial_taylor)/6",
               "train_window": [2001, 2010], "params": best.params},
              open(out_dir / "params.faithful.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "best_params": best.params},
              open(out_dir / "refit_faithful_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.faithful.json")


if __name__ == "__main__":
    main()
