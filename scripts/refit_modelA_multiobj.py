"""
Multi-objective Model A retune. NSGA-II sampler over four ILAMB-style
components: Bias, RMSE, Seasonal Cycle, Spatial Distribution. Optuna minimizes
each, so we pass `1 - score`-style objectives where lower is better.

Component proxies (mirroring ILAMB's ConfBurntArea conceptually):
  Bias score        = exp(-|pred_mean - obs_mean| / |obs_mean|)
  RMSE score        = exp(-rmse / obs_std)
  Seasonal score    = (1 + corr(pred_monthly_clim, obs_monthly_clim)) / 2
  Spatial score     = (1 + corr(pred_time_mean, obs_time_mean)) / 2

We minimize (1 - score) for each, so the Pareto front contains solutions that
trade these four off honestly. After the search, we pick the Pareto-optimal
trial that maximizes the mean of the four scores (i.e., the proxy for ILAMB's
Overall Score).

Train 2001-2010 only for the optimization; we will re-evaluate on full
2001-2016 with ILAMB after the refit is done.

Outputs in models/A-multiobj/:
  params.multiobj-best.json
  refit_multiobj_summary.json
  pareto_front.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import h5py, numpy as np, optuna, xarray as xr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
YEARS = list(range(2001, 2017))


def sig(x, k, c):  return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))
def supp(x, k, c): return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))
def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def coarsen(arr):
    return arr.reshape(*arr.shape[:-2], 180, 2, 360, 2).mean(axis=(-3, -1)).astype(np.float32)


def _gpp_anom_terms(gpp_monthly, anom_k, anom_c, fuel_anom_k):
    gpp_mean = gpp_monthly.mean(axis=0, keepdims=True)
    gpp_anom = gpp_monthly - gpp_mean
    anom_supp = supp(gpp_anom, anom_k, anom_c)
    neg_anom = np.clip(-gpp_anom, 0, None)
    anom_boost = 1.0 - np.exp(-neg_anom / (fuel_anom_k + 1e-9))
    return anom_supp, anom_boost


def fire_A(d, p):
    onset    = sig(d["dbar"],  p["k1"], p["D_low"])
    suppress = supp(d["dbar"], p["k2"], p["D_high"])
    p_floor  = d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))
    fuel_mod = hump(p["af"] * d["agb"], p["fb"], p["fd"])
    soil_t   = sig(d["t_deep"], p["ss2"], p["sc2"])
    dt_deep  = np.diff(d["t_deep"], axis=0, prepend=d["t_deep"][[0]])
    warming  = sig(dt_deep, p["rate_k"], p["rate_c"])
    h_supp   = supp(d["h_natr"], p["h_k"], p["h_crit"])
    gpp_mod  = hump(p["gpp_af"] * d["gpp_monthly"], p["gpp_b"], p["gpp_d"])
    anom_sup, anom_boost = _gpp_anom_terms(d["gpp_monthly"], p["anom_k"], p["anom_c"], p["fuel_anom_k"])
    t_surf   = sig(d["t_surf"], p["ts_k"], p["ts_c"])
    ign      = sig(d["t_air"],  p["ign_k"], p["ign_c"])
    product  = (onset * suppress * p_floor * p_damp * fuel_mod
                * soil_t * warming * h_supp
                * gpp_mod * anom_sup * anom_boost
                * t_surf * ign)
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def predict_monthly(d, p, fire_max_rate=5.0):
    rate = fire_A(d, p)
    rate_capped = np.minimum(rate, fire_max_rate)
    return ((1.0 - np.exp(-rate_capped)) / 12.0).astype(np.float32)


def load_all_drivers():
    cru = DATA / "crujra"
    d = {n: np.nan_to_num(np.load(cru / f"{n}_monthly.npy"), nan=0.0).astype(np.float32)
         for n in ("dbar", "t_deep", "p_ann", "t_air", "p_month", "t_surf")}
    tv = DATA / "trendy_v14"
    ds = xr.open_dataset(tv / "EDv3_S3_gpp.nc", decode_times=False)
    gpp_raw = ds["gpp"].isel(time=slice(3612, 3804)).values.astype(np.float32) * 86400 * 365
    lat_v = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    if lat_v[0] > 0: gpp_raw = gpp_raw[:, ::-1, :]
    d["gpp_monthly"] = coarsen(np.nan_to_num(gpp_raw, nan=0.0))
    ds.close()
    est = DATA / "ed_static"
    d["h_natr"] = np.nan_to_num(np.load(est / "h_natr_monthly.npy"), nan=0.0).astype(np.float32)
    cl = xr.open_dataset(tv / "EDv3_S3_cLeaf.nc", decode_times=False)
    cw = xr.open_dataset(tv / "EDv3_S3_cWood.nc", decode_times=False)
    cleaf = cl["cLeaf"].isel(time=slice(301, 317)).values
    cwood = cw["cWood"].isel(time=slice(301, 317)).values
    if lat_v[0] > 0:
        cleaf = cleaf[:, ::-1, :]; cwood = cwood[:, ::-1, :]
    agb = np.nan_to_num(cleaf + cwood, nan=0.0).astype(np.float32)
    d["agb"] = np.repeat(coarsen(agb), 12, axis=0)
    cl.close(); cw.close()
    return d


def split_drivers(d, mask):
    return {k: v[mask] for k, v in d.items()}


def load_gfed_1deg(years):
    out = np.zeros((len(years) * 12, 180, 360), dtype=np.float32)
    idx = 0
    for yr in years:
        with h5py.File(DATA / "gfed" / f"GFED4.1s_{yr}.hdf5", "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:][::-1, :]
                out[idx] = arr.reshape(180, 4, 360, 4).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


def four_scores(pred_monthly, obs_monthly, w2):
    """Returns Bias, RMSE, Seasonal, Spatial scores in [0, 1]. Higher is better.
    Pred and obs are (n_months, lat, lon) monthly fractions. NaN-safe."""
    n = pred_monthly.shape[0]
    # Replace NaN in pred with 0 (ocean / non-finite cells)
    pred_monthly = np.nan_to_num(pred_monthly, nan=0.0, posinf=0.0, neginf=0.0)
    obs_monthly  = np.nan_to_num(obs_monthly,  nan=0.0, posinf=0.0, neginf=0.0)
    # Bias
    p_tm = pred_monthly.mean(axis=0)
    o_tm = obs_monthly.mean(axis=0)
    w_sum = float(w2.sum()) + 1e-12
    p_mean = float((p_tm * w2).sum() / w_sum)
    o_mean = float((o_tm * w2).sum() / w_sum)
    bias_score = float(np.exp(-abs(p_mean - o_mean) / (abs(o_mean) + 1e-9)))
    if not np.isfinite(bias_score): bias_score = 0.0
    # RMSE — global, weighted
    global_rmse = float(np.sqrt((((pred_monthly - obs_monthly) ** 2) * w2[None, :, :]).sum() / (n * w_sum)))
    global_std  = float(np.sqrt(((obs_monthly - obs_monthly.mean(axis=0)) ** 2 * w2[None, :, :]).sum() / (n * w_sum)) + 1e-9)
    rmse_score = float(np.exp(-global_rmse / global_std))
    if not np.isfinite(rmse_score): rmse_score = 0.0
    # Seasonal: monthly climatology
    n_yr = n // 12
    pred_clim = pred_monthly.reshape(n_yr, 12, *pred_monthly.shape[1:]).mean(axis=0)  # (12, lat, lon)
    obs_clim  = obs_monthly.reshape(n_yr, 12, *obs_monthly.shape[1:]).mean(axis=0)
    pa = pred_clim - pred_clim.mean(axis=0, keepdims=True)
    oa = obs_clim  - obs_clim.mean(axis=0, keepdims=True)
    num = (pa * oa).sum(axis=0)
    den = np.sqrt((pa**2).sum(axis=0) * (oa**2).sum(axis=0)) + 1e-30
    corr = np.clip(num / den, -1, 1)
    valid = (den > 1e-20) & (w2 > 0)
    if valid.sum() == 0 or w2[valid].sum() < 1e-12:
        seasonal_score = 0.0
    else:
        seasonal_score = float((((1 + corr) / 2 * w2)[valid]).sum() / w2[valid].sum())
    if not np.isfinite(seasonal_score):
        seasonal_score = 0.0
    # Spatial: time-mean spatial correlation
    pred_tm = pred_monthly.mean(axis=0)
    obs_tm  = obs_monthly.mean(axis=0)
    # weighted Pearson r over space
    m = (w2 > 0)
    p_v = pred_tm[m]; o_v = obs_tm[m]; w_v = w2[m]
    pm = (p_v * w_v).sum() / w_v.sum(); om = (o_v * w_v).sum() / w_v.sum()
    cov = (w_v * (p_v - pm) * (o_v - om)).sum() / w_v.sum()
    vp  = (w_v * (p_v - pm) ** 2).sum() / w_v.sum()
    vo  = (w_v * (o_v - om) ** 2).sum() / w_v.sum()
    r = float(cov / (np.sqrt(vp * vo) + 1e-30))
    # Taylor-diagram-style: penalize std-ratio drift in addition to correlation
    sigma_p = float(np.sqrt(vp)); sigma_o = float(np.sqrt(vo))
    sigma_ratio = (sigma_p + 1e-12) / (sigma_o + 1e-12)
    # Std penalty: 1 when ratio=1, decays for ratio far from 1 in either direction
    std_penalty = float(np.exp(-abs(np.log(max(sigma_ratio, 1e-9)))))
    # Combined Taylor skill, range [0, 1]
    spatial_score = float(((1 + r) / 2) * std_penalty)
    if not np.isfinite(spatial_score): spatial_score = 0.0
    return bias_score, rmse_score, seasonal_score, spatial_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=8000)
    ap.add_argument("--timeout-h", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "A-multiobj"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model A multi-objective refit, NSGA-II, 4 objectives")

    print("Loading drivers...")
    d_all = load_all_drivers()
    yr_idx = np.repeat(np.arange(2001, 2017), 12)
    train_mask = (yr_idx >= 2001) & (yr_idx <= 2010)
    d_tr = split_drivers(d_all, train_mask)

    print("Loading GFED...")
    gfed = load_gfed_1deg(YEARS)
    gfed_tr = gfed[train_mask]

    lat = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")

    def objective(trial):
        params = dict(
            k1     = trial.suggest_float("k1",     1e-6, 1e-2, log=True),
            D_low  = trial.suggest_float("D_low",  1e0,  1e5,  log=True),
            k2     = trial.suggest_float("k2",     1e-7, 1e-3, log=True),
            D_high = trial.suggest_float("D_high", 1e2,  5e6,  log=True),
            fire_exp = trial.suggest_float("fire_exp", 0.1, 2.0),
            P_half          = trial.suggest_float("P_half",          1e0, 5e3, log=True),
            pre_dampen_half = trial.suggest_float("pre_dampen_half", 1e-1, 1e2, log=True),
            af = trial.suggest_float("af", 1e-2, 1e1, log=True),
            fb = trial.suggest_float("fb", 1e-3, 1e2, log=True),
            fd = trial.suggest_float("fd", 1e-1, 1e3, log=True),
            ss2    = trial.suggest_float("ss2",    1e-2, 1e1, log=True),
            sc2    = trial.suggest_float("sc2",    -10.0, 35.0),
            rate_k = trial.suggest_float("rate_k", 1e-2, 1e1, log=True),
            rate_c = trial.suggest_float("rate_c", -5.0, 5.0),
            h_k    = trial.suggest_float("h_k",    1e-2, 1e1, log=True),
            h_crit = trial.suggest_float("h_crit", 0.1, 30.0, log=True),
            gpp_af = trial.suggest_float("gpp_af", 1e-2, 1e1, log=True),
            gpp_b  = trial.suggest_float("gpp_b",  1e-3, 1e1, log=True),
            gpp_d  = trial.suggest_float("gpp_d",  1e-1, 1e3, log=True),
            anom_k      = trial.suggest_float("anom_k",      1e-2, 1e1, log=True),
            anom_c      = trial.suggest_float("anom_c",     -1.0, 1.0),
            fuel_anom_k = trial.suggest_float("fuel_anom_k", 1e-3, 1e1, log=True),
            ts_k = trial.suggest_float("ts_k", 1e-2, 1e1, log=True),
            ts_c = trial.suggest_float("ts_c", -10.0, 35.0),
            ign_k = trial.suggest_float("ign_k", 1e-2, 1e1, log=True),
            ign_c = trial.suggest_float("ign_c", -10.0, 35.0),
        )
        if params["D_high"] <= params["D_low"] * 1.5:
            return 1.0, 1.0, 1.0, 1.0  # bad solution
        pred = predict_monthly(d_tr, params)
        bias, rmse, seas, spat = four_scores(pred, gfed_tr, w2)
        trial.set_user_attr("bias_score", bias)
        trial.set_user_attr("rmse_score", rmse)
        trial.set_user_attr("seasonal_score", seas)
        trial.set_user_attr("spatial_score", spat)
        trial.set_user_attr("overall_proxy", (bias + rmse + seas + spat) / 4)
        return 1 - bias, 1 - rmse, 1 - seas, 1 - spat

    sampler = optuna.samplers.NSGAIISampler(seed=args.seed, population_size=64)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize"]*4, sampler=sampler)

    print(f"\nNSGA-II  n_trials={args.n_trials}  timeout={args.timeout_h:.1f}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    print(f"  done. n_trials={len(study.trials)}  wall={dt/60:.1f} min")

    # Pick the trial that maximizes the overall_proxy
    completed = [t for t in study.trials if t.values is not None and "overall_proxy" in t.user_attrs]
    completed.sort(key=lambda t: -t.user_attrs["overall_proxy"])
    best = completed[0]
    print(f"\nBest by overall proxy: trial #{best.number}")
    print(f"  overall_proxy={best.user_attrs['overall_proxy']:.4f}")
    print(f"  bias={best.user_attrs['bias_score']:.4f}  rmse={best.user_attrs['rmse_score']:.4f}")
    print(f"  seasonal={best.user_attrs['seasonal_score']:.4f}  spatial={best.user_attrs['spatial_score']:.4f}")

    # Save best params + summary + pareto front
    json.dump({"model": "Model A, multi-objective NSGA-II refit (4 components)",
               "components": ["Bias", "RMSE", "Seasonal Cycle", "Spatial Distribution"],
               "selection": "trial maximizing mean of four component scores",
               "train_window": [2001, 2010],
               "params": best.params}, open(out_dir / "params.multiobj-best.json", "w"), indent=2)
    json.dump({"n_trials": len(study.trials), "wall_seconds": dt, "best_trial": best.number,
               "scores": {k: best.user_attrs[k] for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_proxy")},
               "best_params": best.params}, open(out_dir / "refit_multiobj_summary.json", "w"), indent=2)
    pareto = []
    for t in study.best_trials:
        pareto.append({"trial": t.number, "params": t.params,
                        "scores": {k: t.user_attrs.get(k) for k in ("bias_score", "rmse_score", "seasonal_score", "spatial_score", "overall_proxy")}})
    json.dump(pareto, open(out_dir / "pareto_front.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/params.multiobj-best.json")
    print(f"wrote {out_dir}/refit_multiobj_summary.json")
    print(f"wrote {out_dir}/pareto_front.json  ({len(pareto)} Pareto-optimal trials)")


if __name__ == "__main__":
    main()
