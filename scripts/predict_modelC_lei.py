"""
Predict Model C burned-area from Lei's hand-off file and score it against GFED4.1s.

Reads:
  global_baseline_modelC_inputs_1997-2016.nc   (Lei, 2026-05-04)
  data/gfed/GFED4.1s_{2001..2016}.hdf5
  models/C/params.json   OR  --params <path>

Writes (optional):
  out_lei/predicted_burntArea.nc

Reports: Pearson r, MSE, bias, normalized RMSE — global, on the 2001-2016 overlap window.

NOTE on D_bar scale:
  Lei's D_bar is ED's INTERNAL dryness accumulator (mean ~5e4 mm).
  The current models/C/params.json was tuned against the canonical OFFLINE
  Thornthwaite dbar (mean ~10^2 mm). So a baseline run with the current
  params will give near-zero fire — that's the diagnostic, not a bug.
  Use scripts/refit_modelC_lei.py to retune the 4 dbar params against ED's
  internal D_bar (README option 2).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]


# ---------- Model C formula (copied from reproduce_modelC.py, identical) ----------
def sig(x, k, c):  return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))
def supp(x, k, c): return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))
def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def fire_C_per_lu(D_bar, T_air, P_ann, P_month, GPP_lu, p):
    """Model C applied to one landuse tile. Returns annual fire rate (yr-1)."""
    onset    = sig(D_bar,  p["k1"], p["D_low"])
    suppress = supp(D_bar, p["k2"], p["D_high"])
    p_floor  = P_ann / (P_ann + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + P_month / (p["pre_dampen_half"] + 1e-12))
    gpp_mod  = hump(p["gpp_af"] * GPP_lu, p["gpp_b"], p["gpp_d"])
    ign_mod  = sig(T_air, p["ign_k"], p["ign_c"])
    product  = onset * suppress * p_floor * p_damp * gpp_mod * ign_mod
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def predict_global(ds, params, fire_max_rate=5.0):
    """
    Apply Model C per landuse, area-weight to a single grid-cell rate, then
    apply ED's saturation transform: monthly_frac = (1 - exp(-rate))/12.
    Returns (pred_monthly_frac [time,lat,lon], rate_yr [time,lat,lon]).
    """
    D = ds["D_bar"].values; T = ds["T_air"].values
    Pa = ds["P_ann"].values; Pm = ds["P_month"].values

    rate_total = np.zeros_like(D, dtype=np.float32)
    for lu in ("ntrl", "scnd", "past"):
        gpp = ds[f"GPP_month_{lu}"].values
        frac = ds[f"area_frac_{lu}"].values
        # Model C per landuse, broadcast scalars-as-arrays already aligned
        rate_lu = fire_C_per_lu(D, T, Pa, Pm, gpp, params)
        rate_total += np.where(np.isfinite(frac), frac, 0.0) * rate_lu

    rate_capped = np.minimum(rate_total, fire_max_rate)
    monthly = (1.0 - np.exp(-rate_capped)) / 12.0
    return monthly.astype(np.float32), rate_total.astype(np.float32)


# ---------- GFED loading at 0.5 deg, 2001-2016 ----------
def load_gfed_halfdeg(years):
    """Load GFED4.1s monthly burned fraction at 0.5 deg, oriented S->N."""
    out = np.zeros((len(years) * 12, 360, 720), dtype=np.float32)
    idx = 0
    for yr in years:
        path = REPO / "data" / "gfed" / f"GFED4.1s_{yr}.hdf5"
        with h5py.File(path, "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:]    # 720 x 1440, N->S
                arr = arr[::-1, :]                                     # S->N
                # 0.25 deg -> 0.5 deg
                out[idx] = arr.reshape(360, 2, 720, 2).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


# ---------- Metrics ----------
def metrics(pred, obs, weight):
    """Pearson r, MSE, bias on cells with finite values + nonzero weight."""
    m = np.isfinite(pred) & np.isfinite(obs) & (weight > 0)
    if not m.any():
        return dict(r=np.nan, mse=np.nan, bias=np.nan, n=0)
    p = pred[m].astype(np.float64)
    o = obs[m].astype(np.float64)
    w = weight[m].astype(np.float64)
    wsum = w.sum()
    pm = (p * w).sum() / wsum
    om = (o * w).sum() / wsum
    cov = (w * (p - pm) * (o - om)).sum() / wsum
    vp  = (w * (p - pm) ** 2).sum() / wsum
    vo  = (w * (o - om) ** 2).sum() / wsum
    r = cov / (np.sqrt(vp * vo) + 1e-30)
    mse = (w * (p - o) ** 2).sum() / wsum
    bias = pm - om
    return dict(r=float(r), mse=float(mse), bias=float(bias),
                pred_mean=float(pm), obs_mean=float(om), n=int(m.sum()))


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--params", default=str(REPO / "models" / "C" / "params.json"))
    ap.add_argument("--year-start", type=int, default=2001)
    ap.add_argument("--year-end", type=int, default=2016)
    ap.add_argument("--write", default=None,
                    help="Optional path to write predicted burntArea.nc")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    ds = xr.open_dataset(REPO / args.input)
    # Slice to overlap window
    t = ds["time"].values
    yr = np.array([d.year for d in t])
    mask_t = (yr >= args.year_start) & (yr <= args.year_end)
    ds_w = ds.isel(time=mask_t)
    print(f"  time window: {args.year_start}..{args.year_end}  "
          f"({mask_t.sum()} months)")

    print(f"Loading params from {args.params} ...")
    pj = json.load(open(args.params))
    params = pj["params"] if "params" in pj else pj
    print(f"  D_low={params['D_low']:.4g}  D_high={params['D_high']:.4g}  "
          f"k1={params['k1']:.4g}  k2={params['k2']:.4g}")

    print("Predicting ...")
    pred, rate = predict_global(ds_w, params)
    print(f"  rate_yr  mean={np.nanmean(rate):.4g}  max={np.nanmax(rate):.4g}")
    print(f"  monthly  mean={np.nanmean(pred):.4g}  max={np.nanmax(pred):.4g}")

    print("Loading GFED 4.1s ...")
    obs = load_gfed_halfdeg(range(args.year_start, args.year_end + 1))

    # Spatial weight: cos(lat). Lei's lat is S->N at -89.75..89.75
    lat = ds_w["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    w2 = np.broadcast_to(cos_lat[:, None], (360, 720))
    # Land-fire mask: any GFED activity over the window
    land_fire = (obs > 0).any(axis=0)
    w3 = np.broadcast_to((w2 * land_fire)[None, :, :], pred.shape)

    print("\nMetrics on cells with any GFED activity (weighted by cos lat):")
    M = metrics(pred, obs, w3)
    for k, v in M.items():
        print(f"  {k:10s} {v:.6g}" if isinstance(v, float) else f"  {k:10s} {v}")

    if args.write:
        out = REPO / args.write
        out.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"burntArea": (("time", "lat", "lon"), pred,
                           {"units": "1", "long_name": "Burnt Area Fraction"})},
            coords={"time": ds_w.time, "lat": ds_w.lat, "lon": ds_w.lon},
            attrs={"title": "Model C predicted burntArea (Lei inputs)",
                   "params_source": args.params}
        ).to_netcdf(out, encoding={"burntArea": {"zlib": True, "complevel": 4}})
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
