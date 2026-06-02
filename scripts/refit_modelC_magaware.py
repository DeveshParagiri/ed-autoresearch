"""
Magnitude-aware refit of Model C against Lei's NC inputs and GFED4.1s.

Two modes selectable with --mode:
  monthly  : score on monthly cell-level fractions (the 'as-is' loss target).
  annual   : score on annual cell sums in percent units.

Loss is (1 - r) + LAMBDA * |pred_mean - obs_mean| / (obs_mean + eps),
which penalises magnitude mismatch on top of pattern correlation.

Train 2001-2010, test 2011-2016, cos-lat weighted, GFED-active mask.

Outputs in models/C/:
  params.lei-magaware-{mode}.json
  refit_lei_magaware-{mode}_summary.json
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
    common   = (onset * suppress * p_floor * p_damp * t_ign).astype(np.float32)
    rate = np.zeros_like(d["D"], dtype=np.float32)
    for gpp_lu, frac_lu in zip(d["gpp"], d["frac"]):
        gm = hump(p["gpp_af"] * gpp_lu, p["gpp_b"], p["gpp_d"]).astype(np.float32)
        rate += frac_lu * gm * common
    rate = np.power(np.clip(rate, 0, None), p["fire_exp"]).astype(np.float32)
    rate_capped = np.minimum(rate, fire_max_rate)
    return ((1.0 - np.exp(-rate_capped)) / 12.0).astype(np.float32)


def to_annual_pct(monthly_frac, n_years):
    return monthly_frac.reshape(n_years, 12, 360, 720).sum(axis=1) * 100.0


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
    ap.add_argument("--mode", choices=["monthly", "annual"], required=True)
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=6000)
    ap.add_argument("--timeout-h", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="weight on relative-bias penalty in loss")
    ap.add_argument("--out", default=str(REPO / "models" / "C"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MODE={args.mode}  LAMBDA={args.lam}")

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

    if args.mode == "monthly":
        target_tr = gfed_tr_m
        target_te = gfed_te_m
        w_tr = np.broadcast_to(w2[None, :, :], gfed_tr_m.shape)
        w_te = np.broadcast_to(w2[None, :, :], gfed_te_m.shape)
        n_yr_tr, n_yr_te = 10, 6
    else:  # annual
        target_tr = gfed_tr_m.reshape(10, 12, 360, 720).sum(axis=1) * 100.0
        target_te = gfed_te_m.reshape(6, 12, 360, 720).sum(axis=1) * 100.0
        w_tr = np.broadcast_to(w2[None, :, :], target_tr.shape)
        w_te = np.broadcast_to(w2[None, :, :], target_te.shape)
        n_yr_tr, n_yr_te = 10, 6

    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")
    print(f"  target shape: train={target_tr.shape}  test={target_te.shape}")
    print(f"  obs_mean train: {(target_tr*w_tr).sum()/w_tr.sum():.4g}")

    def score(d, params, n_yr, target, w):
        pred_m = predict_monthly(d, params)
        if args.mode == "annual":
            pred = pred_m.reshape(n_yr, 12, 360, 720).sum(axis=1) * 100.0
        else:
            pred = pred_m
        return metrics(pred, target, w)

    seed_p = dict(k1=2.87e-3, D_low=1206.0, k2=2.1e-6, D_high=10008.0,
                  fire_exp=0.575, P_half=3.18, pre_dampen_half=5.71,
                  gpp_af=0.46, gpp_b=5.2e-5, gpp_d=128.6,
                  ign_k=0.81, ign_c=20.1)

    t0 = time.time()
    m0 = score(d_tr, seed_p, n_yr_tr, target_tr, w_tr)
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

    m_te = score(d_te, best.params, n_yr_te, target_te, w_te)
    rel_bias_te = abs(m_te["pred_mean"] - m_te["obs_mean"]) / (m_te["obs_mean"] + 1e-9)
    print(f"\nHELD-OUT 2011..2016: r={m_te['r']:.4f}  pred_mean={m_te['pred_mean']:.4g}  "
          f"obs_mean={m_te['obs_mean']:.4g}  rel_bias={rel_bias_te:.4f}")

    tag = f"magaware-{args.mode}"
    out_params = out_dir / f"params.lei-{tag}.json"
    json.dump({
        "model": f"Model C, magnitude-aware refit ({args.mode}) on Lei's NC",
        "loss": f"(1 - r) + {args.lam} * |pred_mean - obs_mean| / (obs_mean + eps)",
        "train_window": [2001, 2010],
        "test_window": [2011, 2016],
        "params": best.params,
    }, open(out_params, "w"), indent=2)
    print(f"\nwrote {out_params}")

    summary = out_dir / f"refit_lei_{tag}_summary.json"
    json.dump({
        "mode": args.mode,
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
        "lei_target_r": 0.63,
    }, open(summary, "w"), indent=2)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
