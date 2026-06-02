"""
Refit Model C against GFED5 burned area with "every fire type equal" weighting.

This is criterion (c) of the Hurtt robustness triple. The per-cell weight w2 is
re-normalized so each fire type contributes equally to the loss regardless of
how many fire-active cells it has.

Fire types are assigned by a lat + AGB rule consistent with the regimes that
GFED5 and CLM5/CLM6 separate out:

  boreal                 lat > 50 N
  tropical-deforestation tropical lat (-25..25), AGB > 8 kgC/m2, inside
                         Amazon box (15.5 S..10.5 N, 91 W..30.5 W) OR
                         SE Asia box (-10..10 N, 95..145 E)
  tropical-savanna       tropical lat (-25..25), not classified above
  temperate              everything else with fire activity

Cells with no fire activity in 2001-2010 get weight 0.
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
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    ds = xr.open_dataset(p)
    yrs = np.array([t.year for t in ds["time"].values])
    mask = np.isin(yrs, list(years))
    return (ds["burntArea"].values[mask] / 100.0).astype(np.float32)


TYPES = ["boreal", "tropical-deforestation", "tropical-savanna", "temperate"]


def build_firetype_weights(lat, lon, agb_mean, land_fire):
    """Per-cell weight with cos(lat) AND per-type normalization.
    Each fire type's total weight is 1/N_types of the global sum.
    Cells outside all types or with no fire activity get weight 0.

    Returns (w2, type_id, type_counts).
    """
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    lat_grid = np.broadcast_to(lat[:, None], land_fire.shape).astype(np.float32)
    lon_grid = np.broadcast_to(lon[None, :], land_fire.shape).astype(np.float32)
    base = (cos_lat[:, None] * land_fire).astype(np.float32)

    type_id = np.full(land_fire.shape, -1, dtype=np.int8)

    # 0 boreal: lat > 50 N
    boreal = (lat_grid > 50.0) & land_fire
    type_id[boreal] = 0

    # tropics: -25..25 lat
    tropics = (lat_grid >= -25.0) & (lat_grid <= 25.0) & land_fire & (type_id == -1)

    # Amazon box: lat -15.5..10.5, lon -91..-30.5
    amazon = ((lat_grid >= -15.5) & (lat_grid <= 10.5)
              & (lon_grid >= -91.0) & (lon_grid <= -30.5))
    # SE Asia box: lat -10..10, lon 95..145
    seasia = ((lat_grid >= -10.0) & (lat_grid <= 10.0)
              & (lon_grid >= 95.0) & (lon_grid <= 145.0))

    # 1 tropical-deforestation: tropical + high AGB + inside Amazon or SE Asia
    defo = tropics & (agb_mean > 8.0) & (amazon | seasia)
    type_id[defo] = 1

    # 2 tropical-savanna: remaining tropics
    savanna = tropics & (type_id == -1)
    type_id[savanna] = 2

    # 3 temperate: everything else with fire
    temperate = land_fire & (type_id == -1)
    type_id[temperate] = 3

    w2 = np.zeros_like(base)
    counts = {}
    for i, name in enumerate(TYPES):
        m = (type_id == i)
        counts[name] = int(m.sum())
        s = float(base[m].sum())
        if s > 0:
            w2[m] = (base[m] / s).astype(np.float32)
    return w2, type_id, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-gfed5-firetype"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C refit, GFED5 target, 'every fire type equal' weighting")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_tr = load_gfed5_fraction(range(2001, 2011))

    lat = ds["lat"].values
    lon = ds["lon"].values
    # AGB only used to classify deforestation cells. Pull it from the fuel input file
    # (cVeg*0.8) which has the same grid. Predict path is untouched.
    if "cVeg" in ds:
        agb_mean = np.nanmean(ds["cVeg"].isel(time=train_mask).values, axis=0) * 0.8
    elif "AGB" in ds:
        agb_mean = np.nanmean(ds["AGB"].isel(time=train_mask).values, axis=0)
    else:
        fuel_p = REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc"
        if fuel_p.exists():
            print(f"  AGB not in {args.input}, loading from {fuel_p.name} for classification only")
            ds_fuel = xr.open_dataset(fuel_p)
            yr_fuel = np.array([d.year for d in ds_fuel["time"].values])
            m_fuel = (yr_fuel >= 2001) & (yr_fuel <= 2010)
            agb_mean = np.nanmean(ds_fuel["AGB"].values[m_fuel], axis=0)
        else:
            print("  warning: no cVeg/AGB and no fuel input, using 5 kgC/m2 (defo class will be empty)")
            agb_mean = np.full(gfed_tr.shape[1:], 5.0, dtype=np.float32)

    land_fire = (gfed_tr > 0).any(axis=0)
    w2, type_id, counts = build_firetype_weights(lat, lon, agb_mean, land_fire)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")
    for name, n in counts.items():
        print(f"  {name:25s} fire-active cells: {n}")

    seed_p = json.load(open(REPO / "models" / "C-gfed5" / "params.gfed5.json"))["params"]
    pred_seed = predict_monthly(d_tr, seed_p)
    b, r, s, sp = four_scores(pred_seed, gfed_tr, w2)
    seed_overall = (2 * b + 2 * r + s + sp) / 6.0
    print(f"\n  seed (ED-ModelC-GFED5) re-scored with fire-type-equal weights: "
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

    json.dump({"model": "Model C, GFED5 target, fire-type-equal weighting",
               "loss": "1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6 with type-normalized weights",
               "reference": "GFED5",
               "types": TYPES,
               "type_counts": counts,
               "train_window": [2001, 2010],
               "params": best.params},
              open(out_dir / "params.gfed5-firetype.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "seed_overall_with_firetype_weighting": float(seed_overall),
               "type_counts": counts,
               "best_params": best.params},
              open(out_dir / "refit_firetype_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.gfed5-firetype.json")


if __name__ == "__main__":
    main()
