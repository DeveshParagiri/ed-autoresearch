"""
Retune Model C against ED's INTERNAL D_bar (Lei's hand-off file).

Optimizes only {k1, D_low, k2, D_high} — README option 2. The other 8 params
(precip floor/dampen, GPP hump, T_air ignition, fire_exp) are kept fixed at
their canonical-dbar values, so non-dryness mechanisms stay anchored.

Train on 2001-2010 (per Lei's hand-off recommendation), score on 2011-2016
held-out.

Loss: 1 - r  (Pearson, cos-lat-weighted, on cells with any GFED activity).
Reports MSE and bias as well.

Outputs (in models/C/):
  params.lei-refit.json   — full param dict (8 frozen + 4 retuned)
  refit_lei_summary.json  — metrics + best trial info
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
import optuna
import xarray as xr

REPO = Path(__file__).resolve().parents[1]


# ---------- Model C primitives ----------
def sig(x, k, c):  return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))
def supp(x, k, c): return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))
def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def precompute_static(ds, params):
    """Compute everything that does NOT depend on {k1,D_low,k2,D_high}.

    Returns dict with:
      base_lu : (3, T, lat, lon) — for each landuse, the product
                gpp_mod_lu * (precip_floor * precip_damp * t_ign) * area_frac_lu
      D_bar   : (T, lat, lon)
    """
    D = ds["D_bar"].values.astype(np.float32)
    T = ds["T_air"].values.astype(np.float32)
    Pa = ds["P_ann"].values.astype(np.float32)
    Pm = ds["P_month"].values.astype(np.float32)

    p_floor = (Pa / (Pa + params["P_half"] + 1e-12)).astype(np.float32)
    p_damp  = (1.0 / (1.0 + Pm / (params["pre_dampen_half"] + 1e-12))).astype(np.float32)
    t_ign   = sig(T, params["ign_k"], params["ign_c"]).astype(np.float32)
    common  = p_floor * p_damp * t_ign

    base_lu = []
    for lu in ("ntrl", "scnd", "past"):
        gpp = ds[f"GPP_month_{lu}"].values.astype(np.float32)
        frac = np.where(np.isfinite(ds[f"area_frac_{lu}"].values),
                        ds[f"area_frac_{lu}"].values, 0.0).astype(np.float32)
        gpp_mod = hump(params["gpp_af"] * gpp, params["gpp_b"], params["gpp_d"]).astype(np.float32)
        base_lu.append(frac * gpp_mod * common)
    base_lu = np.stack(base_lu, axis=0)   # (3, T, lat, lon)
    return dict(D_bar=D, base_lu=base_lu)


def predict_from_static(static, k1, D_low, k2, D_high, fire_exp,
                         fire_max_rate=5.0):
    D = static["D_bar"]
    onset    = sig(D,  k1, D_low)
    suppress = supp(D, k2, D_high)
    dryness  = (onset * suppress).astype(np.float32)        # (T, lat, lon)
    # rate per landuse, then sum across LU
    rate = (static["base_lu"] * dryness[None, ...]).sum(axis=0)   # (T, lat, lon)
    rate = np.power(np.clip(rate, 0, None), fire_exp).astype(np.float32)
    rate_capped = np.minimum(rate, fire_max_rate)
    monthly = (1.0 - np.exp(-rate_capped)) / 12.0
    return monthly.astype(np.float32)


# ---------- GFED loader ----------
def load_gfed_halfdeg(years):
    out = np.zeros((len(years) * 12, 360, 720), dtype=np.float32)
    idx = 0
    for yr in years:
        path = REPO / "data" / "gfed" / f"GFED4.1s_{yr}.hdf5"
        with h5py.File(path, "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:][::-1, :]
                out[idx] = arr.reshape(360, 2, 720, 2).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


# ---------- Weighted Pearson + MSE + bias ----------
def weighted_metrics(pred, obs, weight):
    m = np.isfinite(pred) & np.isfinite(obs) & (weight > 0)
    if not m.any():
        return dict(r=-1.0, mse=np.inf, bias=np.inf)
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
    ap.add_argument("--params", default=str(REPO / "models" / "C" / "params.json"))
    ap.add_argument("--train-end", type=int, default=2010,
                    help="last training year (inclusive); test = train_end+1..2016")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C"))
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs from {args.input}")
    ds = xr.open_dataset(REPO / args.input)
    yr = np.array([d.year for d in ds["time"].values])
    train_mask = (yr >= 2001) & (yr <= args.train_end)
    test_mask  = (yr >= args.train_end + 1) & (yr <= 2016)
    print(f"  train: 2001..{args.train_end}  ({train_mask.sum()} months)"
          f"   test: {args.train_end+1}..2016  ({test_mask.sum()} months)")

    base_params = json.load(open(args.params))["params"]
    print(f"  starting params D_low={base_params['D_low']:.4g}  "
          f"D_high={base_params['D_high']:.4g}  k1={base_params['k1']:.4g}  "
          f"k2={base_params['k2']:.4g}")

    print("Pre-computing static (non-dryness) terms ...")
    t0 = time.time()
    ds_train = ds.isel(time=train_mask)
    ds_test  = ds.isel(time=test_mask)
    static_tr = precompute_static(ds_train, base_params)
    static_te = precompute_static(ds_test,  base_params)
    print(f"  done in {time.time()-t0:.1f}s")

    print("Loading GFED 4.1s ...")
    gfed_tr = load_gfed_halfdeg(range(2001, args.train_end + 1))
    gfed_te = load_gfed_halfdeg(range(args.train_end + 1, 2017))

    # cos-lat weight, gated to cells with any GFED activity in train window
    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = (gfed_tr > 0).any(axis=0)
    w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
    w3_tr = np.broadcast_to(w2[None, :, :], gfed_tr.shape)
    w3_te = np.broadcast_to(w2[None, :, :], gfed_te.shape)

    fire_exp = base_params["fire_exp"]

    # Optuna objective: 1 - r on training set.
    def objective(trial):
        # Search ranges chosen given Lei's D_bar mean ~5e4, range 0..5e6.
        D_low  = trial.suggest_float("D_low",   1e2, 1e5,  log=True)
        D_high = trial.suggest_float("D_high",  1e4, 5e6,  log=True)
        if D_high <= D_low * 1.5:
            return 2.0
        k1     = trial.suggest_float("k1", 1e-6, 1e-2, log=True)
        k2     = trial.suggest_float("k2", 1e-7, 1e-3, log=True)
        pred = predict_from_static(static_tr, k1, D_low, k2, D_high, fire_exp)
        m = weighted_metrics(pred, gfed_tr, w3_tr)
        trial.set_user_attr("mse", m["mse"])
        trial.set_user_attr("bias", m["bias"])
        return 1.0 - m["r"]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print(f"Running Optuna ({args.n_trials} trials) ...")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    dt = time.time() - t0
    print(f"  done in {dt:.0f}s  ({dt/args.n_trials:.2f}s/trial)")

    best = study.best_trial
    print(f"\nBest trial #{best.number}:  loss=1-r={best.value:.4f}  "
          f"r={1-best.value:.4f}  mse={best.user_attrs['mse']:.4g}  "
          f"bias={best.user_attrs['bias']:.4g}")
    print(f"  params: {best.params}")

    # Score on test
    pred_te = predict_from_static(static_te, best.params["k1"], best.params["D_low"],
                                    best.params["k2"], best.params["D_high"], fire_exp)
    m_te = weighted_metrics(pred_te, gfed_te, w3_te)
    print(f"\nHELD-OUT 2011..2016:  r={m_te['r']:.4f}  mse={m_te['mse']:.4g}  "
          f"bias={m_te['bias']:.4g}")
    print(f"  pred_mean={m_te['pred_mean']:.4g}  obs_mean={m_te['obs_mean']:.4g}")

    # Save
    new_params = dict(base_params)
    for k in ("k1", "D_low", "k2", "D_high"):
        new_params[k] = best.params[k]
    out_params = out_dir / "params.lei-refit.json"
    json.dump({
        "model": "Model C — retuned dryness against ED internal D_bar (Lei 2026-05-04)",
        "frozen": ["fire_exp", "P_half", "pre_dampen_half",
                    "gpp_af", "gpp_b", "gpp_d", "ign_k", "ign_c"],
        "retuned": ["k1", "D_low", "k2", "D_high"],
        "params": new_params,
    }, open(out_params, "w"), indent=2)
    print(f"\nwrote {out_params}")

    out_summary = out_dir / "refit_lei_summary.json"
    json.dump({
        "n_trials": args.n_trials,
        "train_window": [2001, args.train_end],
        "test_window": [args.train_end + 1, 2016],
        "best_trial": best.number,
        "train_loss_1minusR": best.value,
        "train_pearson_r": 1 - best.value,
        "train_mse": best.user_attrs["mse"],
        "train_bias": best.user_attrs["bias"],
        "test": m_te,
        "best_params": best.params,
    }, open(out_summary, "w"), indent=2)
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
