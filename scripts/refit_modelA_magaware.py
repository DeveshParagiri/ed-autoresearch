"""
Magnitude-aware annual refit of Model A (8-mechanism, 27 parameters) on
canonical 1-degree CRUJRA + ED static + TRENDY v14 drivers.

Identical pipeline to the existing reproduce_v2.fire_A formula, with a magnitude
penalty added to the loss so the optimizer cannot crush magnitude to chase
pattern.

Loss: (1 - r) + LAMBDA * |pred_mean - obs_mean| / (obs_mean + eps), annual mode.
Train 2001-2010, test 2011-2016, cos-lat weighted, GFED-active mask.

Outputs in models/A/:
  params.magaware-annual.json
  refit_magaware-annual_summary.json
"""
from __future__ import annotations
import argparse, gc, json, time
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


def metrics(pred, obs, weight):
    m = np.isfinite(pred) & np.isfinite(obs) & (weight > 0)
    if not m.any():
        return dict(r=-1.0, mse=np.inf, bias=np.inf, pred_mean=0.0, obs_mean=0.0)
    p = pred[m].astype(np.float64); o = obs[m].astype(np.float64)
    w = weight[m].astype(np.float64); ws = w.sum()
    pm = (p * w).sum() / ws; om = (o * w).sum() / ws
    cov = (w * (p - pm) * (o - om)).sum() / ws
    vp  = (w * (p - pm) ** 2).sum() / ws
    vo  = (w * (o - om) ** 2).sum() / ws
    r = cov / (np.sqrt(vp * vo) + 1e-30)
    mse = (w * (p - o) ** 2).sum() / ws
    return dict(r=float(r), mse=float(mse), bias=float(pm - om),
                pred_mean=float(pm), obs_mean=float(om))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=6000)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=str(REPO / "models" / "A"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model A magaware-annual refit, LAMBDA={args.lam}")

    print("Loading drivers (1 deg, 192 months)...")
    d_all = load_all_drivers()

    yr_idx = np.repeat(np.arange(2001, 2017), 12)
    train_mask = (yr_idx >= 2001) & (yr_idx <= 2010)
    test_mask  = (yr_idx >= 2011) & (yr_idx <= 2016)
    d_tr = split_drivers(d_all, train_mask)
    d_te = split_drivers(d_all, test_mask)
    n_yr_tr, n_yr_te = 10, 6

    print("Loading GFED 4.1s 1 deg...")
    gfed = load_gfed_1deg(YEARS)
    gfed_tr = gfed[train_mask]
    gfed_te = gfed[test_mask]

    lat = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    target_tr = gfed_tr.reshape(n_yr_tr, 12, 180, 360).sum(axis=1) * 100.0
    target_te = gfed_te.reshape(n_yr_te, 12, 180, 360).sum(axis=1) * 100.0
    w_tr = np.broadcast_to(w2[None, :, :], target_tr.shape)
    w_te = np.broadcast_to(w2[None, :, :], target_te.shape)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")
    print(f"  obs_mean train (annual %): {(target_tr*w_tr).sum()/w_tr.sum():.4g}")

    def score(d, params, n_yr, target, w):
        pred_m = predict_monthly(d, params)
        pred = pred_m.reshape(n_yr, 12, 180, 360).sum(axis=1) * 100.0
        return metrics(pred, target, w)

    def objective(trial):
        params = dict(
            # Dryness
            k1     = trial.suggest_float("k1",     1e-6, 1e-2, log=True),
            D_low  = trial.suggest_float("D_low",  1e0,  1e5,  log=True),
            k2     = trial.suggest_float("k2",     1e-7, 1e-3, log=True),
            D_high = trial.suggest_float("D_high", 1e2,  5e6,  log=True),
            fire_exp = trial.suggest_float("fire_exp", 0.1, 2.0),
            # Precip
            P_half          = trial.suggest_float("P_half",          1e0, 5e3, log=True),
            pre_dampen_half = trial.suggest_float("pre_dampen_half", 1e-1, 1e2, log=True),
            # Fuel hump on AGB (kg/m^2)
            af = trial.suggest_float("af", 1e-2, 1e1, log=True),
            fb = trial.suggest_float("fb", 1e-3, 1e2, log=True),
            fd = trial.suggest_float("fd", 1e-1, 1e3, log=True),
            # Soil temp deep sigmoid + warming rate sigmoid
            ss2    = trial.suggest_float("ss2",    1e-2, 1e1, log=True),
            sc2    = trial.suggest_float("sc2",    -10.0, 35.0),
            rate_k = trial.suggest_float("rate_k", 1e-2, 1e1, log=True),
            rate_c = trial.suggest_float("rate_c", -5.0, 5.0),
            # Canopy height
            h_k    = trial.suggest_float("h_k",    1e-2, 1e1, log=True),
            h_crit = trial.suggest_float("h_crit", 0.1, 30.0, log=True),
            # GPP hump
            gpp_af = trial.suggest_float("gpp_af", 1e-2, 1e1, log=True),
            gpp_b  = trial.suggest_float("gpp_b",  1e-3, 1e1, log=True),
            gpp_d  = trial.suggest_float("gpp_d",  1e-1, 1e3, log=True),
            # GPP anomaly
            anom_k      = trial.suggest_float("anom_k",      1e-2, 1e1, log=True),
            anom_c      = trial.suggest_float("anom_c",     -1.0, 1.0),
            fuel_anom_k = trial.suggest_float("fuel_anom_k", 1e-3, 1e1, log=True),
            # Surface T sigmoid
            ts_k = trial.suggest_float("ts_k", 1e-2, 1e1, log=True),
            ts_c = trial.suggest_float("ts_c", -10.0, 35.0),
            # Ignition T sigmoid
            ign_k = trial.suggest_float("ign_k", 1e-2, 1e1, log=True),
            ign_c = trial.suggest_float("ign_c", -10.0, 35.0),
        )
        if params["D_high"] <= params["D_low"] * 1.5:
            return 5.0
        m = score(d_tr, params, n_yr_tr, target_tr, w_tr)
        rel_bias = abs(m["pred_mean"] - m["obs_mean"]) / (m["obs_mean"] + 1e-9)
        loss = (1.0 - m["r"]) + args.lam * rel_bias
        trial.set_user_attr("r", m["r"])
        trial.set_user_attr("rel_bias", rel_bias)
        trial.set_user_attr("pred_mean", m["pred_mean"])
        trial.set_user_attr("obs_mean", m["obs_mean"])
        return float(loss)

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                            multivariate=True, group=True,
                                            n_startup_trials=300)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nOptuna  n_trials={args.n_trials}  timeout={args.timeout_h:.1f}h  (27 params)")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    print(f"  done. n_trials={len(study.trials)}  wall={dt/60:.1f} min")

    best = study.best_trial
    print(f"\nBest #{best.number}: loss={best.value:.4f}  r={best.user_attrs['r']:.4f}  "
          f"rel_bias={best.user_attrs['rel_bias']:.4f}")

    m_te = score(d_te, best.params, n_yr_te, target_te, w_te)
    rel_bias_te = abs(m_te["pred_mean"] - m_te["obs_mean"]) / (m_te["obs_mean"] + 1e-9)
    print(f"\nHELD-OUT: r={m_te['r']:.4f}  pred_mean={m_te['pred_mean']:.4g}  "
          f"obs_mean={m_te['obs_mean']:.4g}  rel_bias={rel_bias_te:.4f}")

    out_params = out_dir / "params.magaware-annual.json"
    json.dump({
        "model": "Model A, 27-param 8-mechanism, magaware-annual refit on 1deg canonical",
        "loss": f"(1 - r) + {args.lam} * |pred_mean - obs_mean| / (obs_mean + eps)",
        "train_window": [2001, 2010],
        "test_window": [2011, 2016],
        "params": best.params,
    }, open(out_params, "w"), indent=2)
    print(f"\nwrote {out_params}")

    summary = out_dir / "refit_magaware-annual_summary.json"
    json.dump({
        "mode": "annual",
        "with_fuel": True,
        "n_mechanisms": 8,
        "n_params": 27,
        "lambda": args.lam,
        "n_trials_completed": len(study.trials),
        "wall_seconds": dt,
        "best_trial": best.number,
        "train_loss": best.value,
        "train_r": best.user_attrs["r"],
        "train_rel_bias": best.user_attrs["rel_bias"],
        "train_pred_mean": best.user_attrs["pred_mean"],
        "train_obs_mean": best.user_attrs["obs_mean"],
        "test": {**m_te, "rel_bias": rel_bias_te},
        "best_params": best.params,
    }, open(summary, "w"), indent=2)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
