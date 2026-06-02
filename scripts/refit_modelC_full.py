"""
Full 12-parameter Optuna refit of Model C against Lei's hand-off inputs.

All 12 params free (no canonical-dbar values frozen). Bounds widened to make
sense for ED's INTERNAL scales (D_bar mean ~5e4 mm, GPP per-LU ~kg/m2/yr,
P_ann ~774 mm/yr, T_air mean ~12 C).

Train: 2001-2010. Test: 2011-2016.
Loss: 1 - r (Pearson, cos-lat-weighted, on cells with any GFED activity in train).

Outputs (in models/C/):
  params.lei-full-refit.json
  refit_lei_full_summary.json
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
    """Return all driver arrays as float32 numpy arrays. Also area_frac per LU."""
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


def predict(d, p, fire_max_rate=5.0):
    """Full Model C, per-landuse, area-weighted, ED saturation transform."""
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
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=5000)
    ap.add_argument("--timeout-h", type=float, default=8.0,
                    help="hard wall-clock cap in hours")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs from {args.input}")
    ds = xr.open_dataset(REPO / args.input)
    yr = np.array([d.year for d in ds["time"].values])
    train_mask = (yr >= 2001) & (yr <= 2010)
    test_mask  = (yr >= 2011) & (yr <= 2016)
    print(f"  train: 2001..2010 ({train_mask.sum()} months)   "
          f"test: 2011..2016 ({test_mask.sum()} months)")

    print("Loading driver arrays ...")
    t0 = time.time()
    d_tr = load_inputs(ds.isel(time=train_mask))
    d_te = load_inputs(ds.isel(time=test_mask))
    print(f"  done in {time.time()-t0:.1f}s")

    print("Loading GFED 4.1s ...")
    gfed_tr = load_gfed(range(2001, 2011))
    gfed_te = load_gfed(range(2011, 2017))

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
    w3_tr = np.broadcast_to(w2[None, :, :], gfed_tr.shape)
    w3_te = np.broadcast_to(w2[None, :, :], gfed_te.shape)
    print(f"  fire-active cells: {land_fire.sum()} / {land_fire.size}")

    # --- One predict timing for budgeting ---
    seed_p = dict(k1=2.87e-3, D_low=1206.0, k2=2.1e-6, D_high=10008.0,
                   fire_exp=0.575, P_half=3.18, pre_dampen_half=5.71,
                   gpp_af=0.46, gpp_b=5.2e-5, gpp_d=128.6,
                   ign_k=0.81, ign_c=20.1)
    t0 = time.time()
    p0 = predict(d_tr, seed_p)
    m0 = metrics(p0, gfed_tr, w3_tr)
    dt_pred = time.time() - t0
    print(f"  one predict+metrics on train: {dt_pred:.2f}s   (seed r={m0['r']:.3f})")

    def objective(trial):
        # Bounds chosen for ED-internal magnitudes.
        params = dict(
            # Dryness asymptotes
            k1     = trial.suggest_float("k1",     1e-6, 1e-2, log=True),
            D_low  = trial.suggest_float("D_low",  1e2,  1e5,  log=True),
            k2     = trial.suggest_float("k2",     1e-7, 1e-3, log=True),
            D_high = trial.suggest_float("D_high", 1e4,  5e6,  log=True),
            # Final exponent
            fire_exp = trial.suggest_float("fire_exp", 0.1, 2.0),
            # Precip
            P_half          = trial.suggest_float("P_half",          1e0, 5e3, log=True),
            pre_dampen_half = trial.suggest_float("pre_dampen_half", 1e-1, 1e2, log=True),
            # GPP hump  (GPP ~ 0–6 kg/m2/yr per LU; gpp_af scales it)
            gpp_af = trial.suggest_float("gpp_af", 1e-2, 1e1, log=True),
            gpp_b  = trial.suggest_float("gpp_b",  1e-3, 1e1, log=True),
            gpp_d  = trial.suggest_float("gpp_d",  1e-1, 1e3, log=True),
            # Temperature ignition  (T_air range roughly -50..40 C)
            ign_k = trial.suggest_float("ign_k", 1e-2, 1e1, log=True),
            ign_c = trial.suggest_float("ign_c", -10.0, 35.0),
        )
        # Cheap sanity guards
        if params["D_high"] <= params["D_low"] * 1.5:
            return 2.0
        pred = predict(d_tr, params)
        m = metrics(pred, gfed_tr, w3_tr)
        trial.set_user_attr("mse", m["mse"])
        trial.set_user_attr("bias", m["bias"])
        return 1.0 - m["r"]

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                            multivariate=True, group=True,
                                            n_startup_trials=200)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Warm-start with the seed point
    study.enqueue_trial(seed_p)

    print(f"\nRunning Optuna  n_trials={args.n_trials}  "
          f"timeout={args.timeout_h:.1f}h  (~{dt_pred:.1f}s/trial)")
    t0 = time.time()
    study.optimize(objective,
                    n_trials=args.n_trials,
                    timeout=args.timeout_h * 3600,
                    show_progress_bar=False,
                    gc_after_trial=True)
    dt = time.time() - t0
    n_done = len(study.trials)
    print(f"  done. n_trials_completed={n_done}  wall={dt/60:.1f} min  "
          f"({dt/max(n_done,1):.2f}s/trial)")

    best = study.best_trial
    print(f"\nBest trial #{best.number}:  loss=1-r={best.value:.4f}  "
          f"r={1-best.value:.4f}  mse={best.user_attrs['mse']:.4g}  "
          f"bias={best.user_attrs['bias']:.4g}")
    for k, v in best.params.items():
        print(f"    {k:18s} {v:.6g}")

    # Held-out
    pred_te = predict(d_te, best.params)
    m_te = metrics(pred_te, gfed_te, w3_te)
    print(f"\nHELD-OUT 2011..2016:  r={m_te['r']:.4f}  mse={m_te['mse']:.4g}  "
          f"bias={m_te['bias']:.4g}  pred_mean={m_te['pred_mean']:.4g}  "
          f"obs_mean={m_te['obs_mean']:.4g}")

    # Save
    out_params = out_dir / "params.lei-full-refit.json"
    json.dump({
        "model": "Model C — full 12-param refit on ED internal D_bar (Lei hand-off, 1997-2016 NC)",
        "train_window": [2001, 2010],
        "test_window": [2011, 2016],
        "params": best.params,
    }, open(out_params, "w"), indent=2)
    print(f"\nwrote {out_params}")

    out_summary = out_dir / "refit_lei_full_summary.json"
    json.dump({
        "n_trials_requested": args.n_trials,
        "n_trials_completed": n_done,
        "wall_seconds": dt,
        "best_trial": best.number,
        "train_loss_1minusR": best.value,
        "train_pearson_r": 1 - best.value,
        "train_mse": best.user_attrs["mse"],
        "train_bias": best.user_attrs["bias"],
        "test": m_te,
        "best_params": best.params,
        "lei_target_r": 0.63,
    }, open(out_summary, "w"), indent=2)
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
