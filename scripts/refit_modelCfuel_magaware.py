"""
Magnitude-aware refit of Model C-fuel.

Model C-fuel adds a multiplicative fuel factor to the existing Model C structure,
mirroring CTEM's three-factor probability (Pf = Pb · Pm · Pi). The fuel factor is
a sigmoid in AGB, suppressing fire where biomass is below a threshold and allowing
it where biomass is sufficient.

Predictor (per landuse, then area-weighted):
  product = onset(D) * suppress(D) * p_floor(Pa) * p_damp(Pm) *
            gpp_mod(GPP_lu) * ign_mod(T) * fuel_mod(AGB)
  rate    = product^fire_exp
  monthly = (1 - exp(-min(rate, fire_max)))/12

Two new params:
  fuel_low : AGB threshold (kg/m^2) below which fuel suppresses fire
  fuel_k   : sigmoid steepness on AGB

Loss: (1 - r) + LAMBDA * |pred_mean - obs_mean| / (obs_mean + eps), annual mode.

Inputs: global_baseline_modelCfuel_inputs_1997-2016.nc (Lei NC + AGB).
Train 2001-2010, test 2011-2016, cos-lat weighted, GFED-active mask.

Outputs in models/C-fuel/:
  params.lei-magaware-annual.json
  refit_lei_magaware-annual_summary.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import h5py
import numpy as np
import optuna
import xarray as xr

REPO = Path(__file__).resolve().parents[1]


def sig(x, k, c):  return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))
def supp(x, k, c): return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))
def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def load_inputs(ds):
    return dict(
        D=ds["D_bar"].values.astype(np.float32),
        T=ds["T_air"].values.astype(np.float32),
        Pa=ds["P_ann"].values.astype(np.float32),
        Pm=ds["P_month"].values.astype(np.float32),
        AGB=ds["AGB"].values.astype(np.float32),
        gpp=[ds[f"GPP_month_{lu}"].values.astype(np.float32) for lu in ("ntrl","scnd","past")],
        frac=[np.where(np.isfinite(ds[f"area_frac_{lu}"].values),
                       ds[f"area_frac_{lu}"].values, 0.0).astype(np.float32)
              for lu in ("ntrl","scnd","past")],
    )


def predict_monthly(d, p, fire_max_rate=5.0):
    onset    = sig(d["D"],  p["k1"], p["D_low"])
    suppress = supp(d["D"], p["k2"], p["D_high"])
    p_floor  = d["Pa"] / (d["Pa"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["Pm"] / (p["pre_dampen_half"] + 1e-12))
    t_ign    = sig(d["T"], p["ign_k"], p["ign_c"])
    # Fuel mechanism with explicit alpha gate:
    #   fuel_mod = (1 - alpha) + alpha * sig(AGB; fuel_k, fuel_low)
    # When alpha = 0: fuel_mod = 1.0 everywhere (truly inert, baseline recovery).
    # When alpha = 1: fuel_mod = sigmoid in AGB.
    alpha = p.get("fuel_alpha", 1.0)
    fuel_sig = sig(d["AGB"], p["fuel_k"], p["fuel_low"])
    fuel_mod = (1.0 - alpha) + alpha * fuel_sig
    common   = (onset * suppress * p_floor * p_damp * t_ign * fuel_mod).astype(np.float32)
    rate = np.zeros_like(d["D"], dtype=np.float32)
    for gpp_lu, frac_lu in zip(d["gpp"], d["frac"]):
        gm = hump(p["gpp_af"] * gpp_lu, p["gpp_b"], p["gpp_d"]).astype(np.float32)
        rate += frac_lu * gm * common
    rate = np.power(np.clip(rate, 0, None), p["fire_exp"]).astype(np.float32)
    rate_capped = np.minimum(rate, fire_max_rate)
    return ((1.0 - np.exp(-rate_capped)) / 12.0).astype(np.float32)


def load_gfed(years):
    out = np.zeros((len(years) * 12, 360, 720), dtype=np.float32)
    idx = 0
    for yr in years:
        with h5py.File(REPO / "data" / "gfed" / f"GFED4.1s_{yr}.hdf5", "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:][::-1, :]
                out[idx] = arr.reshape(360, 2, 720, 2).mean(axis=(1, 3))
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
    ap.add_argument("--input", default="global_baseline_modelCfuel_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=6000)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default=str(REPO / "models" / "C-fuel"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MODE=annual  LAMBDA={args.lam}  with FUEL factor")

    print(f"Loading inputs from {args.input}")
    ds = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    test_mask  = (yr_all >= 2011) & (yr_all <= 2016)
    d_tr = load_inputs(ds.isel(time=train_mask))
    d_te = load_inputs(ds.isel(time=test_mask))
    gfed_tr_m = load_gfed(range(2001, 2011))
    gfed_te_m = load_gfed(range(2011, 2017))

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr_m > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)

    target_tr = gfed_tr_m.reshape(10, 12, 360, 720).sum(axis=1) * 100.0
    target_te = gfed_te_m.reshape(6, 12, 360, 720).sum(axis=1) * 100.0
    w_tr = np.broadcast_to(w2[None, :, :], target_tr.shape)
    w_te = np.broadcast_to(w2[None, :, :], target_te.shape)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")
    print(f"  AGB range: {np.nanmin(d_tr['AGB']):.4g} to {np.nanmax(d_tr['AGB']):.4g} kg/m^2")

    def score(d, params, n_yr, target, w):
        pred_m = predict_monthly(d, params)
        pred = pred_m.reshape(n_yr, 12, 360, 720).sum(axis=1) * 100.0
        return metrics(pred, target, w)

    # Seed point: magaware-annual best params from Model C, plus pass-through fuel
    seed_p = json.load(open(REPO / "models" / "C" / "params.lei-magaware-annual.json"))["params"]
    seed_p = dict(seed_p)
    seed_p.update(dict(fuel_k=2.0, fuel_low=0.5))   # gentle fuel transition

    t0 = time.time()
    m0 = score(d_tr, seed_p, 10, target_tr, w_tr)
    dt_pred = time.time() - t0
    print(f"  one predict+metrics: {dt_pred:.2f}s  (seed r={m0['r']:.3f}  "
          f"pred_m={m0['pred_mean']:.4g}  obs_m={m0['obs_mean']:.4g})")

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
            ign_k = trial.suggest_float("ign_k", 1e-2, 1e1, log=True),
            ign_c = trial.suggest_float("ign_c", -10.0, 35.0),
            # Fuel factor: AGB sigmoid
            fuel_k    = trial.suggest_float("fuel_k",    1e-2, 1e1, log=True),
            fuel_low  = trial.suggest_float("fuel_low",  1e-2, 2e1, log=True),
        )
        if params["D_high"] <= params["D_low"] * 1.5:
            return 5.0
        m = score(d_tr, params, 10, target_tr, w_tr)
        rel_bias = abs(m["pred_mean"] - m["obs_mean"]) / (m["obs_mean"] + 1e-9)
        loss = (1.0 - m["r"]) + args.lam * rel_bias
        trial.set_user_attr("r", m["r"])
        trial.set_user_attr("rel_bias", rel_bias)
        trial.set_user_attr("pred_mean", m["pred_mean"])
        trial.set_user_attr("obs_mean", m["obs_mean"])
        return float(loss)

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                            multivariate=True, group=True,
                                            n_startup_trials=200)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(seed_p)

    print(f"\nOptuna  n_trials={args.n_trials}  timeout={args.timeout_h:.1f}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False, gc_after_trial=True)
    dt = time.time() - t0
    print(f"  done. n_trials={len(study.trials)}  wall={dt/60:.1f} min")

    best = study.best_trial
    print(f"\nBest #{best.number}: loss={best.value:.4f}  r={best.user_attrs['r']:.4f}  "
          f"rel_bias={best.user_attrs['rel_bias']:.4f}")
    for k, v in best.params.items():
        print(f"    {k:18s} {v:.6g}")

    m_te = score(d_te, best.params, 6, target_te, w_te)
    rel_bias_te = abs(m_te["pred_mean"] - m_te["obs_mean"]) / (m_te["obs_mean"] + 1e-9)
    print(f"\nHELD-OUT 2011..2016: r={m_te['r']:.4f}  pred_mean={m_te['pred_mean']:.4g}  "
          f"obs_mean={m_te['obs_mean']:.4g}  rel_bias={rel_bias_te:.4f}")

    out_params = out_dir / "params.lei-magaware-annual.json"
    json.dump({
        "model": "Model C-fuel, magnitude-aware annual refit on Lei NC + AGB (cVeg*0.8)",
        "loss": f"(1 - r) + {args.lam} * |pred_mean - obs_mean| / (obs_mean + eps)",
        "fuel_factor": "sigmoid(AGB; fuel_k, fuel_low)",
        "train_window": [2001, 2010],
        "test_window": [2011, 2016],
        "params": best.params,
    }, open(out_params, "w"), indent=2)
    print(f"\nwrote {out_params}")

    summary = out_dir / "refit_lei_magaware-annual_summary.json"
    json.dump({
        "mode": "annual",
        "with_fuel": True,
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
