"""
Diagnostic maps for the current Model C refit (Lei inputs).

Loads:
  global_baseline_modelC_inputs_1997-2016.nc
  models/C/params.lei-full-refit.json
  data/gfed/GFED4.1s_*.hdf5

Writes (into NEW MAPS/):
  01_obs_mean_train.png         — GFED mean over 2001-2010
  02_pred_mean_train.png        — Model C mean over 2001-2010
  03_bias_train.png             — pred - obs (where the model under/over-shoots)
  04_obs_mean_test.png          — GFED mean over 2011-2016
  05_pred_mean_test.png         — Model C mean over 2011-2016
  06_bias_test.png              — pred - obs (held-out)
  07_per_cell_pearson_test.png  — per-cell time-series correlation on hold-out
"""
from __future__ import annotations
import json
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS"
OUT.mkdir(exist_ok=True)


# ---------- Model C primitives (must match refit_modelC_full.py) ----------
def sig(x, k, c):  return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))
def supp(x, k, c): return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))
def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def predict(d, p, fire_max_rate=5.0):
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


def load_inputs(ds):
    return dict(
        D=ds["D_bar"].values.astype(np.float32),
        T=ds["T_air"].values.astype(np.float32),
        Pa=ds["P_ann"].values.astype(np.float32),
        Pm=ds["P_month"].values.astype(np.float32),
        gpp=[ds[f"GPP_month_{lu}"].values.astype(np.float32)
             for lu in ("ntrl","scnd","past")],
        frac=[np.where(np.isfinite(ds[f"area_frac_{lu}"].values),
                       ds[f"area_frac_{lu}"].values, 0.0).astype(np.float32)
              for lu in ("ntrl","scnd","past")],
    )


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


# ---------- Plotting ----------
def make_map(arr2d, lat, lon, title, out_path, *,
             cmap="hot_r", vmin=None, vmax=None, divergent=False):
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, edgecolor="0.5")

    a = np.where(np.isfinite(arr2d), arr2d, np.nan)
    if divergent:
        v = np.nanmax(np.abs(a))
        if vmin is None: vmin = -v
        if vmax is None: vmax =  v
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        if vmin is None: vmin = 0.0
        if vmax is None: vmax = float(np.nanpercentile(a, 99))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    m = ax.pcolormesh(lon, lat, a, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading="auto")
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.03,
                       shrink=0.85, aspect=40)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def per_cell_pearson(pred, obs):
    """Time-series Pearson r per (lat, lon). Returns nan where insufficient."""
    pm = pred.mean(axis=0); om = obs.mean(axis=0)
    pa = pred - pm; oa = obs - om
    num = (pa * oa).sum(axis=0)
    den = np.sqrt((pa ** 2).sum(axis=0) * (oa ** 2).sum(axis=0)) + 1e-30
    r = num / den
    # Mask cells with no GFED activity
    no_obs = (obs.sum(axis=0) <= 0)
    r[no_obs] = np.nan
    return r


def main():
    print("Loading inputs ...")
    ds = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
    yr = np.array([d.year for d in ds["time"].values])
    train_mask = (yr >= 2001) & (yr <= 2010)
    test_mask  = (yr >= 2011) & (yr <= 2016)
    lat = ds["lat"].values; lon = ds["lon"].values

    params = json.load(open(REPO / "models" / "C" / "params.lei-full-refit.json"))["params"]
    print(f"  using params from params.lei-full-refit.json")

    print("Predicting train + test ...")
    d_tr = load_inputs(ds.isel(time=train_mask))
    d_te = load_inputs(ds.isel(time=test_mask))
    pred_tr = predict(d_tr, params)
    pred_te = predict(d_te, params)

    print("Loading GFED ...")
    obs_tr = load_gfed(range(2001, 2011))
    obs_te = load_gfed(range(2011, 2017))

    # Annualize: monthly fraction -> approximate yearly burned fraction by sum
    obs_tr_yr_mean  = obs_tr.reshape(10, 12, 360, 720).sum(axis=1).mean(axis=0)
    pred_tr_yr_mean = pred_tr.reshape(10, 12, 360, 720).sum(axis=1).mean(axis=0)
    obs_te_yr_mean  = obs_te.reshape(6, 12, 360, 720).sum(axis=1).mean(axis=0)
    pred_te_yr_mean = pred_te.reshape(6, 12, 360, 720).sum(axis=1).mean(axis=0)

    bias_tr = pred_tr_yr_mean - obs_tr_yr_mean
    bias_te = pred_te_yr_mean - obs_te_yr_mean

    # Mask: cells with any GFED activity in train
    land = (obs_tr.sum(axis=0) > 0) | (obs_te.sum(axis=0) > 0)
    def mask(a): return np.where(land, a, np.nan)

    print("Plotting ...")
    vmax_obs = float(np.nanpercentile(mask(obs_tr_yr_mean), 99))
    make_map(mask(obs_tr_yr_mean),  lat, lon,
             "GFED4.1s observed annual burned fraction  ·  mean 2001–2010",
             OUT / "01_obs_mean_train.png", vmax=vmax_obs)
    make_map(mask(pred_tr_yr_mean), lat, lon,
             "Model C (lei-full-refit) predicted annual burned fraction  ·  mean 2001–2010",
             OUT / "02_pred_mean_train.png", vmax=vmax_obs)
    vlim_b = float(np.nanpercentile(np.abs(mask(bias_tr)), 99))
    make_map(mask(bias_tr), lat, lon,
             "Bias (pred − obs)  ·  mean 2001–2010   [blue = under-predict, red = over-predict]",
             OUT / "03_bias_train.png", cmap="RdBu_r", divergent=True,
             vmin=-vlim_b, vmax=vlim_b)

    make_map(mask(obs_te_yr_mean),  lat, lon,
             "GFED4.1s observed annual burned fraction  ·  mean 2011–2016 (HOLD-OUT)",
             OUT / "04_obs_mean_test.png", vmax=vmax_obs)
    make_map(mask(pred_te_yr_mean), lat, lon,
             "Model C predicted annual burned fraction  ·  mean 2011–2016 (HOLD-OUT)",
             OUT / "05_pred_mean_test.png", vmax=vmax_obs)
    vlim_b = float(np.nanpercentile(np.abs(mask(bias_te)), 99))
    make_map(mask(bias_te), lat, lon,
             "Bias (pred − obs)  ·  mean 2011–2016 (HOLD-OUT)",
             OUT / "06_bias_test.png", cmap="RdBu_r", divergent=True,
             vmin=-vlim_b, vmax=vlim_b)

    print("Computing per-cell Pearson r on hold-out ...")
    r_te = per_cell_pearson(pred_te, obs_te)
    make_map(r_te, lat, lon,
             "Per-cell Pearson r  ·  monthly time series  ·  2011–2016 (HOLD-OUT)",
             OUT / "07_per_cell_pearson_test.png",
             cmap="RdBu_r", divergent=True, vmin=-1, vmax=1)

    print(f"\nAll maps in:  {OUT}")


if __name__ == "__main__":
    main()
