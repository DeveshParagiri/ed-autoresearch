"""Tune Hurtt β + D_REF against GFED5 fFire with a global-magnitude penalty.

Adds a penalty proportional to |global_PgC_per_yr - 2.0| / 2.0 so the optimizer
cannot inflate global emissions to game the ILAMB spatial term.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np, optuna, xarray as xr
sys.path.insert(0, "scripts")
from refit_modelA_multiobj import four_scores
from tune_hurtt_betas import load_inputs, fFire_from_betas, FF_SCALE, SEC_PER_MONTH

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba-model", default="ED-ModelC-GFED5")
    ap.add_argument("--n-trials", type=int, default=4000)
    ap.add_argument("--timeout-h", type=float, default=2.0)
    ap.add_argument("--target-pgc", type=float, default=2.0)
    ap.add_argument("--penalty", type=float, default=0.1,
                    help="Score penalty per relative unit of global-mean error")
    ap.add_argument("--out", default=str(REPO / "models" / "hurtt-betas-constrained"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs (BA={args.ba_model}) ...")
    ba, agb, csoil, dbar, obs, lat = load_inputs(args.ba_model)
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = ((obs / FF_SCALE) > 0).any(axis=0) | (ba > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    R = 6.371e6; dlon = np.deg2rad(0.5)
    area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
    area_2d = np.broadcast_to(np.abs(area_lat)[:, None], (len(lat), 720)).astype(np.float64)

    def global_PgC(pred_scaled):
        pred = pred_scaled / FF_SCALE  # back to kg/m2/s
        # 10 years of training data
        return float((pred * area_2d[None, :, :]).sum() * SEC_PER_MONTH / 1e12) / 10.0

    def objective(trial):
        betas = {
            "leaf":   trial.suggest_float("beta_leaf",   0.0, 1.0),
            "fine":   trial.suggest_float("beta_fine",   0.0, 1.0),
            "coarse": trial.suggest_float("beta_coarse", 0.0, 1.0),
            "litter": trial.suggest_float("beta_litter", 0.0, 1.0),
        }
        d_ref = trial.suggest_float("D_REF", 100.0, 30000.0, log=True)
        pred = fFire_from_betas(ba, agb, csoil, dbar, betas, d_ref)
        b, r, s, sp = four_scores(pred, obs, w2)
        overall = (2*b + 2*r + s + sp) / 6.0
        pgc = global_PgC(pred)
        rel_err = abs(pgc - args.target_pgc) / args.target_pgc
        score = overall - args.penalty * rel_err
        trial.set_user_attr("overall", overall); trial.set_user_attr("global_PgC", pgc)
        trial.set_user_attr("bias", b); trial.set_user_attr("rmse", r)
        trial.set_user_attr("seas", s); trial.set_user_attr("spat", sp)
        trial.set_user_attr("score", score)
        return float(1.0 - score)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True, n_startup_trials=200)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial({"beta_leaf": 0.90, "beta_fine": 0.80, "beta_coarse": 0.35,
                         "beta_litter": 0.80, "D_REF": 1000.0})
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_h*3600,
                   show_progress_bar=False, gc_after_trial=True)
    print(f"done, {len(study.trials)} trials in {(time.time()-t0)/60:.1f} min")
    best = study.best_trial
    print(f"Best: overall={best.user_attrs['overall']:.4f} PgC={best.user_attrs['global_PgC']:.2f} score={best.user_attrs['score']:.4f}")
    print(f"  {best.params}")
    json.dump({"ba_model": args.ba_model, "target_pgc": args.target_pgc,
               "best_params": best.params,
               "user_attrs": {k: best.user_attrs[k] for k in ("overall","global_PgC","bias","rmse","seas","spat","score")}},
              open(out_dir / "betas.gfed5.json", "w"), indent=2)


if __name__ == "__main__":
    main()
