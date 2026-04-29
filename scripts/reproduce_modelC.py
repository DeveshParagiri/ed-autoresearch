"""Regenerate ED-ModelC-final/burntArea.nc from canonical inputs + current params.

Reads:
    data/crujra/{dbar,p_ann,p_month,t_air}_monthly.npy
    data/trendy_v14/EDv3_S3_gpp.nc
    models/C/params.json

Writes:
    ilamb/MODELS/ED-ModelC-final/burntArea.nc (0.5 deg, monthly, 2001-2016)

Logic (ED-coupled-consistent — 2026-04-28 retune):
  1. Apply fire_C formula on 1 deg drivers; output is interpreted as an
     ANNUAL fire rate (yr^-1).
  2. Mask to GFED burnable-cell land mask.
  3. Apply ED's saturating annual->monthly transform:
        rate_capped = min(rate, FIRE_MAX_RATE)
        annual_frac = 1 - exp(-rate_capped * 1 yr)
        monthly_frac = annual_frac / 12
     This matches what coupled ED writes to TRENDY-format burntArea.
  4. Un-coarsen 1 deg -> 0.5 deg by nearest-neighbor 2x2.
  5. Mask non-burnable cells to NaN.
  6. Write CF-compliant NetCDF.

FIRE_MAX_RATE defaults to 5.0 yr^-1 (offline tuning cap). Coupled ED needs
fire_max_disturbance_rate >= 1.0 in ED_params.defaults.cfg to reproduce this NC
bit-exact, since the optimizer pushes a few high-fire cells up to ~1.0 yr^-1.
"""
from __future__ import annotations
import gc, json, os
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MODELS = REPO / "models"
YEARS = list(range(2001, 2017))
N_MONTHS = 192
FIRE_MAX_RATE = float(os.environ.get("FIRE_MAX_RATE", 5.0))   # see module docstring


def coarsen(arr):
    return arr.reshape(*arr.shape[:-2], 180, 2, 360, 2).mean(axis=(-3, -1)).astype(np.float32)


def uncoarsen(arr):
    return np.repeat(np.repeat(arr, 2, axis=1), 2, axis=2).astype(np.float32)


def sig(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))


def supp(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip(k * (x - c), -50, 50)))


def hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def fire_C(d, p):
    """Model C: onset + suppress on dbar, precip floor+dampen, GPP hump, T_air sigmoid."""
    onset    = sig(d["dbar"],    p["k1"],  p["D_low"])
    suppress = supp(d["dbar"],   p["k2"],  p["D_high"])
    p_floor  = d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))
    gpp_mod  = hump(p["gpp_af"] * d["gpp_monthly"], p["gpp_b"], p["gpp_d"])
    ign_mod  = sig(d["t_air"], p["ign_k"], p["ign_c"])
    product  = onset * suppress * p_floor * p_damp * gpp_mod * ign_mod
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def load_drivers():
    cru = DATA / "crujra"
    d = {
        "dbar":    np.load(cru / "dbar_monthly.npy").astype(np.float32),
        "p_ann":   np.load(cru / "p_ann_monthly.npy").astype(np.float32),
        "p_month": np.load(cru / "p_month_monthly.npy").astype(np.float32),
        "t_air":   np.load(cru / "t_air_monthly.npy").astype(np.float32),
    }
    ds = xr.open_dataset(DATA / "trendy_v14" / "EDv3_S3_gpp.nc", decode_times=False)
    gpp = ds["gpp"].isel(time=slice(3612, 3804)).values.astype(np.float32) * 86400 * 365
    lat_v = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    if lat_v[0] > 0:
        gpp = gpp[:, ::-1, :]
    gpp = np.nan_to_num(gpp, nan=0.0)
    d["gpp_monthly"] = coarsen(gpp)
    ds.close()
    return d


def load_gfed_1deg():
    out = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
    idx = 0
    for yr in YEARS:
        with h5py.File(DATA / "gfed" / f"GFED4.1s_{yr}.hdf5", "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:]
                arr = arr[::-1, :]
                out[idx] = arr.reshape(180, 4, 360, 4).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


def add_cf_bounds(ds):
    times = ds.time.values
    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds = ds.assign(time_bounds=(("time", "nb"), tb))
    ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    lat = ds.lat.values; dlat = abs(float(lat[1] - lat[0]))
    ds = ds.assign(lat_bounds=(("lat", "nb"),
                               np.stack([lat - dlat/2, lat + dlat/2], axis=1)))
    ds.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north",
                         "standard_name": "latitude", "axis": "Y"})
    lon = ds.lon.values; dlon = abs(float(lon[1] - lon[0]))
    ds = ds.assign(lon_bounds=(("lon", "nb"),
                               np.stack([lon - dlon/2, lon + dlon/2], axis=1)))
    ds.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east",
                         "standard_name": "longitude", "axis": "X"})
    return ds


def main():
    params = json.load(open(MODELS / "C" / "params.json"))["params"]
    print(f"Using params: {params}")

    print("Loading drivers ...")
    d = load_drivers()
    obs = load_gfed_1deg()

    lat_1 = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
    cos_lat = np.cos(np.deg2rad(lat_1)).astype(np.float32)
    w3 = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360))

    # Land mask: cells where GFED has any fire at any time step
    land_mask = (obs > 0).any(axis=0)
    w3_land = w3 * land_mask[None, :, :]
    print(f"  land cells: {land_mask.sum()} / {land_mask.size}")

    print("Computing fire_C ...")
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(d, params)

    # Apply land mask
    rate = rate * land_mask[None, :, :]
    raw_lm = float((rate * w3_land).sum() / (w3_land.sum() + 1e-12))
    print(f"  raw rate land-mean (yr^-1): {raw_lm:.4g}  "
          f"max: {rate.max():.4g}  cap: {FIRE_MAX_RATE}")

    # ED-coupled-consistent transform:
    #   monthly_frac = (1 - exp(-min(rate, FIRE_MAX_RATE) * 1 yr)) / 12
    rate_capped = np.minimum(rate, FIRE_MAX_RATE)
    annual_frac = 1.0 - np.exp(-rate_capped)
    pred = (annual_frac / 12.0).astype(np.float32)

    # Land-mean diagnostic
    pred_lm = float((pred * w3_land).sum() / (w3_land.sum() + 1e-12))
    gfed_lm = float((obs  * w3_land).sum() / (w3_land.sum() + 1e-12))
    print(f"  ED-transformed land-mean:   {pred_lm:.6g}  GFED:  {gfed_lm:.6g}  "
          f"ratio: {pred_lm/gfed_lm:.3f}")

    # NaN out non-land cells (matches GFED reference fill value)
    pred_masked = np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32)

    # Un-coarsen 1 -> 0.5 deg
    pred_hd = uncoarsen(pred_masked)

    # Build dataset
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), pred_hd,
                       {"units": "1", "standard_name": "burnt_area_fraction",
                        "long_name": "Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (ED-coupled-consistent retune)",
               "Conventions": "CF-1.7",
               "transform": f"monthly_frac = (1 - exp(-min(rate_yr, {FIRE_MAX_RATE}) * 1yr)) / 12"})
    ds = add_cf_bounds(ds)

    # Write
    ILAMB_MODELS = REPO / "ilamb" / "MODELS" / "ED-ModelC-final"
    ILAMB_MODELS.mkdir(parents=True, exist_ok=True)
    dst = ILAMB_MODELS / "burntArea.nc"
    time_units = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = dst.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst)
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
