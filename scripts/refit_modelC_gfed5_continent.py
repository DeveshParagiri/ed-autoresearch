"""
Refit Model C against GFED5 burned area with "every continent equal" weighting.

Loss is the same ILAMB-weighted Taylor-aware form as refit_modelC_gfed5.py,
but the per-cell weight w2 is re-normalized so each continent contributes
equally to the loss regardless of how many fire-active cells it has.

Continents defined by lat/lon boxes (rough):
  North America   15°N..75°N,   -170°W..-50°W
  South America   -60°S..15°N,  -85°W..-30°W
  Africa          -35°S..38°N,  -20°W..52°E
  Europe          36°N..75°N,   -10°W..60°E
  Asia            -10°S..75°N,  60°E..180°E
  Australia       -50°S..-10°N, 110°E..180°E

For cells outside all boxes (mostly ocean and Antarctica), weight = 0.

Cells in overlapping continents get assigned to the first matching continent
in the order above.
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


CONTINENTS = [
    ("North America",  (15.0,  75.0), (-170.0, -50.0)),
    ("South America",  (-60.0, 15.0), (-85.0,  -30.0)),
    ("Africa",         (-35.0, 38.0), (-20.0,   52.0)),
    ("Europe",         (36.0,  75.0), (-10.0,   60.0)),
    ("Asia",           (-10.0, 75.0), (60.0,   180.0)),
    ("Australia",      (-50.0,-10.0), (110.0,  180.0)),
]


def build_continent_weights(lat, lon, land_fire):
    """Per-cell weight with cos(lat) AND per-continent normalization.
    Each continent's total weight is 1/N_continents of the global sum.
    Cells outside all boxes get weight 0."""
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    lat_grid = np.broadcast_to(lat[:, None], land_fire.shape).astype(np.float32)
    lon_grid = np.broadcast_to(lon[None, :], land_fire.shape).astype(np.float32)
    base = (cos_lat[:, None] * land_fire).astype(np.float32)
    # Assign each cell to first matching continent; -1 means unassigned
    cont_id = np.full(land_fire.shape, -1, dtype=np.int8)
    for i, (name, (lat_lo, lat_hi), (lon_lo, lon_hi)) in enumerate(CONTINENTS):
        in_box = ((lat_grid >= lat_lo) & (lat_grid <= lat_hi)
                  & (lon_grid >= lon_lo) & (lon_grid <= lon_hi)
                  & land_fire)
        mask = (cont_id == -1) & in_box
        cont_id[mask] = i
    # Normalize per continent
    w2 = np.zeros_like(base)
    for i in range(len(CONTINENTS)):
        m = (cont_id == i)
        s = float(base[m].sum())
        if s > 0:
            w2[m] = (base[m] / s).astype(np.float32)  # each continent sums to 1
    # Each continent total = 1, so they're all equal regardless of fire counts
    return w2, cont_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=10000)
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-gfed5-continent"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Model C refit, GFED5 target, 'every continent equal' weighting")

    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_inputs(ds.isel(time=train_mask))
    gfed_tr = load_gfed5_fraction(range(2001, 2011))

    lat = ds["lat"].values
    lon = ds["lon"].values
    land_fire = (gfed_tr > 0).any(axis=0)
    w2, cont_id = build_continent_weights(lat, lon, land_fire)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")
    for i, (name, _, _) in enumerate(CONTINENTS):
        n = int((cont_id == i).sum())
        print(f"  {name:15s} fire-active cells: {n}")

    seed_p = json.load(open(REPO / "models" / "C-gfed5" / "params.gfed5.json"))["params"]
    pred_seed = predict_monthly(d_tr, seed_p)
    b, r, s, sp = four_scores(pred_seed, gfed_tr, w2)
    seed_overall = (2 * b + 2 * r + s + sp) / 6.0
    print(f"\n  seed (ED-ModelC-GFED5) re-scored with continent-equal weights: "
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

    json.dump({"model": "Model C, GFED5 target, continent-equal weighting",
               "loss": "1 - (2*Bias + 2*RMSE + Seasonal + Spatial)/6 with continent-normalized weights",
               "reference": "GFED5",
               "train_window": [2001, 2010],
               "params": best.params},
              open(out_dir / "params.gfed5-continent.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_score")},
               "seed_overall_with_continent_weighting": float(seed_overall),
               "best_params": best.params},
              open(out_dir / "refit_continent_summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.gfed5-continent.json")


if __name__ == "__main__":
    main()
