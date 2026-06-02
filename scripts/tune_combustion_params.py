"""Tune Hurtt-formula β_max + D_REF against GFED5 fFire.

Model C BA frozen at criterion (a) = ED-ModelC-GFED5.
Parameters: β_leaf, β_fine, β_coarse, β_litter (each in [0, 1]),
            D_REF (mm, in [100, 30000] log).
Loss: 1 - Overall, where Overall = (2*Bias + 2*RMSE + Seas + Spat)/6
      against GFED5 fFire with cos(lat) weighting over fire-active cells.

The optimization is fast because BA, AGB, cSoil, Dbar are precomputed.
Each trial only updates beta and recomputes fFire = BA × Σ_k (C_k × β_k(θ)) / Δt.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, optuna, xarray as xr
import sys
sys.path.insert(0, "scripts")
from refit_modelA_multiobj import four_scores

REPO = Path(__file__).resolve().parents[1]
SEC_PER_MONTH = (365.25 / 12) * 86400.0
POOL_FRAC = {"leaf": 0.05, "fine": 0.10, "coarse": 0.85, "litter": 0.15}
FF_SCALE = 1e10  # scale fFire to avoid underflow in spatial correlation


def load_inputs(model_name, years=(2001, 2010)):
    ds_ba = xr.open_dataset(REPO / "ilamb" / "MODELS_LEADERBOARD" / model_name / "burntArea.nc")
    yr_ba = np.array([t.year for t in ds_ba["time"].values])
    m_ba = (yr_ba >= years[0]) & (yr_ba <= years[1])
    ba = np.nan_to_num(ds_ba["burntArea"].values[m_ba].astype(np.float32))
    lat = ds_ba["lat"].values

    ds_fuel = xr.open_dataset(REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc")
    yr_f = np.array([t.year for t in ds_fuel["time"].values])
    m_f = (yr_f >= years[0]) & (yr_f <= years[1])
    agb = np.nan_to_num(ds_fuel["AGB"].values[m_f].astype(np.float32))

    ds_csoil = xr.open_dataset(REPO / "data" / "trendy_v14" / "EDv3_S3_cSoil.nc")
    yr_c = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in ds_csoil["time"].values])
    m_c = (yr_c >= years[0]) & (yr_c <= years[1])
    annual = ds_csoil["cSoil"].values[m_c].astype(np.float32)
    if ds_csoil["latitude"].values[0] > ds_csoil["latitude"].values[-1]:
        annual = annual[:, ::-1, :]
    csoil = np.nan_to_num(np.repeat(annual, 12, axis=0))

    dbar_1 = np.load(REPO / "data" / "crujra" / "dbar_monthly.npy")
    n_months = (years[1] - years[0] + 1) * 12
    dbar_1 = dbar_1[:n_months]
    dbar = np.repeat(np.repeat(dbar_1, 2, axis=-2), 2, axis=-1).astype(np.float32)

    ds_ref = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc")
    yr_r = np.array([t.year for t in ds_ref["time"].values])
    m_r = (yr_r >= years[0]) & (yr_r <= years[1])
    obs = ds_ref["fFire"].values[m_r].astype(np.float32) * FF_SCALE

    return ba, agb, csoil, dbar, obs, lat


def fFire_from_betas(ba, agb, csoil, dbar, betas, d_ref):
    m_theta = np.clip(dbar / d_ref, 0.0, 1.0).astype(np.float32)
    C_combust = (POOL_FRAC["leaf"]   * agb   * betas["leaf"]   * m_theta
               + POOL_FRAC["fine"]   * agb   * betas["fine"]   * m_theta
               + POOL_FRAC["coarse"] * agb   * betas["coarse"] * m_theta
               + POOL_FRAC["litter"] * csoil * betas["litter"] * m_theta).astype(np.float32)
    return ((ba * C_combust) / SEC_PER_MONTH * FF_SCALE).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba-model", default="ED-ModelC-GFED5")
    ap.add_argument("--n-trials", type=int, default=4000)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "hurtt-betas"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs (BA={args.ba_model}, train 2001-2010) ...")
    ba, agb, csoil, dbar, obs, lat = load_inputs(args.ba_model)
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = ((obs / FF_SCALE) > 0).any(axis=0) | (ba > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    seed_betas = {"leaf": 0.90, "fine": 0.80, "coarse": 0.35, "litter": 0.80}
    seed_pred = fFire_from_betas(ba, agb, csoil, dbar, seed_betas, 1000.0)
    b, r, s, sp = four_scores(seed_pred, obs, w2)
    seed_overall = (2 * b + 2 * r + s + sp) / 6.0
    print(f"  CLM5-defaults seed: overall={seed_overall:.4f} (b={b:.3f} r={r:.3f} s={s:.3f} sp={sp:.3f})")

    def objective(trial):
        betas = {
            "leaf":   trial.suggest_float("beta_leaf",   0.0, 1.0),
            "fine":   trial.suggest_float("beta_fine",   0.0, 1.0),
            "coarse": trial.suggest_float("beta_coarse", 0.0, 1.0),
            "litter": trial.suggest_float("beta_litter", 0.0, 1.0),
        }
        d_ref = trial.suggest_float("D_REF", 100.0, 30000.0, log=True)
        pred = fFire_from_betas(ba, agb, csoil, dbar, betas, d_ref)
        bias, rmse, seas, spat = four_scores(pred, obs, w2)
        overall = (2 * bias + 2 * rmse + seas + spat) / 6.0
        trial.set_user_attr("bias", bias); trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("seas", seas); trial.set_user_attr("spat", spat)
        trial.set_user_attr("overall", overall)
        return float(1.0 - overall)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True,
                                          n_startup_trials=200)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial({"beta_leaf": 0.90, "beta_fine": 0.80, "beta_coarse": 0.35,
                         "beta_litter": 0.80, "D_REF": 1000.0})

    print(f"\nTPE n_trials={args.n_trials} timeout={args.timeout_h}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_h * 3600,
                   show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    best = study.best_trial
    print(f"\ndone, {len(study.trials)} trials in {dt/60:.1f} min")
    print(f"Best #{best.number}: overall={best.user_attrs['overall']:.4f}")
    print(f"  b={best.user_attrs['bias']:.3f} r={best.user_attrs['rmse']:.3f} "
          f"s={best.user_attrs['seas']:.3f} sp={best.user_attrs['spat']:.3f}")
    print(f"  params: {best.params}")

    json.dump({"ba_model": args.ba_model, "seed_overall": float(seed_overall),
               "best_overall": best.user_attrs["overall"], "best_params": best.params,
               "scores": {k: best.user_attrs[k] for k in ("bias","rmse","seas","spat","overall")}},
              open(out_dir / "betas.gfed5.json", "w"), indent=2)


if __name__ == "__main__":
    main()
