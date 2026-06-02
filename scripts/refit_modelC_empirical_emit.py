"""
Refit Model C directly against GFED5 fire carbon emissions (fFire), not burned area.

Pipeline:
  Model C predicts monthly burned-area fraction.
  Convert to BA per second: BA_per_sec = BA_frac / SEC_PER_MONTH.
  Apply per-cell GFED5-derived EF: pred_fFire = BA_per_sec * EF.
  Score against GFED5 fFire with ILAMB-weighted Taylor-aware loss.

This tells us whether tuning Model C for emissions gives different parameters than
tuning for burned area. If the answer is "essentially the same params", then the
fFire story rides on the BA fit and we have one model. If different, we need to
think about whether GCB wants a BA-tuned or fFire-tuned variant.
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
SEC_PER_MONTH = (365.25 / 12) * 86400.0


def load_gfed5_ffire(years):
    p = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"
    ds = xr.open_dataset(p)
    yrs = np.array([t.year for t in ds["time"].values])
    mask = np.isin(yrs, list(years))
    return ds["fFire"].values[mask].astype(np.float32)  # kg/m2/s


def load_gfed5_ba_fraction(years):
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    ds = xr.open_dataset(p)
    yrs = np.array([t.year for t in ds["time"].values])
    mask = np.isin(yrs, list(years))
    return (ds["burntArea"].values[mask] / 100.0).astype(np.float32)


def compute_per_cell_EF_full():
    """Per-cell EF using the full GFED5 record, same convention as score_fFire_gfed5.py."""
    ff = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc")["fFire"].values
    ba_pct = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")["burntArea"].values
    n = min(ff.shape[0], ba_pct.shape[0])
    ff = ff[:n]; ba_pct = ba_pct[:n]
    ba_per_sec = (ba_pct / 100.0) / SEC_PER_MONTH
    num = np.nansum(ff, axis=0)
    den = np.nansum(ba_per_sec, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        EF = num / den
    EF = np.where(np.isfinite(EF) & (den > 0), EF, 0.0).astype(np.float32)
    return EF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-gfed5-ffire"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C refit, GFED5 fFire target, ILAMB-weighted Taylor-aware loss")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_ba_tr = load_gfed5_ba_fraction(range(2001, 2011))
    gfed_ff_tr_raw = load_gfed5_ffire(range(2001, 2011))

    EF = compute_per_cell_EF_full()
    print(f"  EF range [{float(np.nanmin(EF[EF>0])):.3g}, {float(np.nanmax(EF)):.3g}] kgC/m2 per unit BA-frac/sec")
    print(f"  fFire train shape {gfed_ff_tr_raw.shape}, mean {float(np.nanmean(gfed_ff_tr_raw)):.3g} kg/m2/s")

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    # Loss is over cells with fire activity in either BA or fFire (using BA as proxy).
    land_fire = (gfed_ba_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    seed_p = json.load(open(REPO / "models" / "C-gfed5" / "params.gfed5.json"))["params"]

    # Scale to gC/m2/yr-ish range so spatial correlation does not underflow
    # the 1e-30 floor in four_scores. Both pred and obs are scaled identically,
    # so all four scores are mathematically invariant.
    FF_SCALE = 1e10
    gfed_ff_tr = (gfed_ff_tr_raw * FF_SCALE).astype(np.float32)

    def ba_to_ffire(pred_ba_frac):
        # pred_ba_frac: (T, lat, lon), monthly burned-area fraction
        return (pred_ba_frac / SEC_PER_MONTH) * EF[None, :, :] * FF_SCALE

    pred_seed_ba = predict_monthly(d_tr, seed_p)
    pred_seed_ff = ba_to_ffire(pred_seed_ba)
    b, r, s, sp = four_scores(pred_seed_ff, gfed_ff_tr, w2)
    seed_overall = (2 * b + 2 * r + s + sp) / 6.0
    print(f"\n  seed (BA-tuned ED-ModelC-GFED5) scored on fFire: overall={seed_overall:.4f} "
          f"(bias={b:.3f} rmse={r:.3f} seas={s:.3f} spat={sp:.3f})")

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
        pred_ba = predict_monthly(d_tr, params)
        pred_ff = ba_to_ffire(pred_ba)
        bias, rmse, seas, spat = four_scores(pred_ff, gfed_ff_tr, w2)
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

    json.dump({"model": "Model C, GFED5 fFire target, ILAMB-weighted Taylor refit",
               "loss": "1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6 vs GFED5 fFire",
               "reference": "GFED5 fFire, per-cell EF from GFED5",
               "train_window": [2001, 2010],
               "seed_ffire_overall": float(seed_overall),
               "params": best.params},
              open(out_dir / "params.gfed5-ffire.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "seed_ffire_overall": float(seed_overall),
               "best_params": best.params},
              open(out_dir / "refit_ffire_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.gfed5-ffire.json")


if __name__ == "__main__":
    main()
