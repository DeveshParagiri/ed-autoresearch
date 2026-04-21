"""Regenerate burntArea NetCDFs for Models A, B, C from params.json + inputs.

This is a standalone reproducer. Given the inputs in data/ (TRENDY v14 ED +
CRUJRA + frozen-sim heights/fractions), it computes each model's monthly
prediction and writes to out/ED-Model{A,B,C}/burntArea.nc in ILAMB format.

Usage:
    python scripts/reproduce.py

Requires:
    numpy, xarray, cftime, netCDF4, h5py
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "out"
YEARS = list(range(2001, 2017))
N_YRS = len(YEARS)
N_MONTHS = 12 * N_YRS


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def coarsen_half_to_1deg(arr):
    """0.5 deg (360,720) -> 1 deg (180,360). Block-mean 2x2 on last 2 axes."""
    return arr.reshape(*arr.shape[:-2], 180, 2, 360, 2).mean(axis=(-3, -1)).astype(np.float32)


def uncoarsen_1deg_to_half(arr):
    """1-deg -> 0.5-deg by nearest-neighbor broadcast."""
    if arr.ndim == 3:
        return np.repeat(np.repeat(arr, 2, axis=1), 2, axis=2)
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


# -----------------------------------------------------------------------------
# Load drivers (monthly 1-deg arrays)
# -----------------------------------------------------------------------------

def load_trendy_v14_monthly():
    """TRENDY v14 ED GPP + cLeaf+cWood + cSoil, coarsened to 1-deg."""
    # GPP (monthly, kg m-2 s-1 -> kg m-2 yr-1)
    gpp_ds = xr.open_dataset(DATA / "trendy_v14" / "EDv3_S3_gpp.nc", decode_times=False)
    gpp = gpp_ds["gpp"].isel(time=slice(3612, 3804)).values  # 2001-01 to 2016-12
    gpp = gpp * 86400 * 365
    lat_vals = (gpp_ds.latitude.values if "latitude" in gpp_ds.coords
                else gpp_ds.lat.values)
    flip = lat_vals[0] > 0
    if flip:
        gpp = gpp[:, ::-1, :]
    gpp = np.nan_to_num(gpp, nan=0.0).astype(np.float32)
    gpp_1d = coarsen_half_to_1deg(gpp)
    gpp_ds.close(); del gpp

    # cLeaf + cWood (annual) -> tile to monthly
    cleaf_ds = xr.open_dataset(DATA / "trendy_v14" / "EDv3_S3_cLeaf.nc", decode_times=False)
    cwood_ds = xr.open_dataset(DATA / "trendy_v14" / "EDv3_S3_cWood.nc", decode_times=False)
    cleaf = cleaf_ds["cLeaf"].isel(time=slice(301, 317)).values
    cwood = cwood_ds["cWood"].isel(time=slice(301, 317)).values
    if flip:
        cleaf = cleaf[:, ::-1, :]
        cwood = cwood[:, ::-1, :]
    agb = np.nan_to_num(cleaf + cwood, nan=0.0).astype(np.float32)
    agb_1d = coarsen_half_to_1deg(agb)
    agb_month = np.repeat(agb_1d, 12, axis=0).astype(np.float32)
    cleaf_ds.close(); cwood_ds.close()
    del cleaf, cwood, agb, agb_1d

    # cSoil (annual) -> tile
    csoil_ds = xr.open_dataset(DATA / "trendy_v14" / "EDv3_S3_cSoil.nc", decode_times=False)
    csoil = csoil_ds["cSoil"].isel(time=slice(301, 317)).values
    if flip:
        csoil = csoil[:, ::-1, :]
    csoil = np.nan_to_num(csoil, nan=0.0).astype(np.float32)
    csoil_1d = coarsen_half_to_1deg(csoil)
    soil_C = np.repeat(csoil_1d, 12, axis=0).astype(np.float32)
    csoil_ds.close(); del csoil, csoil_1d

    return {"gpp_ed": gpp_1d, "agb": agb_month, "soil_C": soil_C}


def load_crujra_monthly():
    """CRUJRA monthly D_bar, T_deep, P_ann at 1-deg."""
    # Simplified: expect preprocessed CRUJRA monthly arrays
    # User should run their own CRUJRA preprocessing to produce:
    #   data/crujra/dbar_monthly.npy     (192, 180, 360)
    #   data/crujra/t_deep_monthly.npy   (192, 180, 360)
    #   data/crujra/p_ann_monthly.npy    (192, 180, 360)

    cru_dir = DATA / "crujra"
    dbar = np.load(cru_dir / "dbar_monthly.npy")
    t_deep = np.load(cru_dir / "t_deep_monthly.npy")
    p_ann = np.load(cru_dir / "p_ann_monthly.npy")
    return {"dbar": dbar, "t_deep": t_deep, "p_ann": p_ann}


def load_ed_static():
    """ED heights and fractions from frozen sim. Tiled to monthly at 1-deg."""
    ed_dir = DATA / "ed_static"
    return {
        "h_natr": np.load(ed_dir / "h_natr_monthly.npy"),
        "h_scnd": np.load(ed_dir / "h_scnd_monthly.npy"),
        "f_natr": np.load(ed_dir / "f_natr_monthly.npy"),
        "f_scnd": np.load(ed_dir / "f_scnd_monthly.npy"),
    }


# -----------------------------------------------------------------------------
# Formulas
# -----------------------------------------------------------------------------

def _sig(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))

def _supp(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip(k * (x - c), -50, 50)))

def _hump(x, b, dec):
    b = max(b, 1e-9); dec = max(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def fire_A(d, p):
    """Model A: full 7-mechanism formula."""
    dbar = d["dbar"]
    out = _sig(dbar, p["k1"], p["D_low"]) * _supp(dbar, p["k2"], p["D_high"])

    a = p["af"] * d["agb"]
    out *= _hump(a, p["fb"], p["fd"])

    g = d["gpp_ed"]
    out *= np.exp(-((g - p["gpp_opt"]) / max(p["gpp_sigma"], 1e-9)) ** 2)

    out *= _sig(d["t_deep"], p["ss2"], p["sc2"])
    out *= d["p_ann"] / (d["p_ann"] + max(p["P_half"], 1e-9))

    h = d["h_natr"]
    out *= 1.0 / (1.0 + np.exp(np.clip(p["h_k"] * (h - p["h_crit"]), -50, 50)))

    fs, hs = d["f_scnd"], d["h_scnd"]
    out *= (1.0 + p["lu_c"] * fs) * np.exp(-hs / max(p["lu_tau"], 1e-9))

    c = d["soil_C"]
    out *= 1.0 + (p["sc_max"] - 1.0) * c / (c + max(p["sc_half"], 1e-9))

    return np.power(np.clip(out, 0, None), p["fire_exp"]).astype(np.float32)


def fire_B(d, p):
    """Model B: fuel + soil_temp + precip + height."""
    dbar = d["dbar"]
    out = _sig(dbar, p["k1"], p["D_low"]) * _supp(dbar, p["k2"], p["D_high"])

    a = p["af"] * d["agb"]
    out *= _hump(a, p["fb"], p["fd"])

    out *= _sig(d["t_deep"], p["ss2"], p["sc2"])
    out *= d["p_ann"] / (d["p_ann"] + max(p["P_half"], 1e-9))

    h = d["h_natr"]
    out *= 1.0 / (1.0 + np.exp(np.clip(p["h_k"] * (h - p["h_crit"]), -50, 50)))

    return np.power(np.clip(out, 0, None), p["fire_exp"]).astype(np.float32)


def fire_C(d, p):
    """Model C: fuel + soil_temp (minimal)."""
    dbar = d["dbar"]
    out = _sig(dbar, p["k1"], p["D_low"]) * _supp(dbar, p["k2"], p["D_high"])

    a = p["af"] * d["agb"]
    out *= _hump(a, p["fb"], p["fd"])

    out *= _sig(d["t_deep"], p["ss2"], p["sc2"])

    return np.power(np.clip(out, 0, None), p["fire_exp"]).astype(np.float32)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def add_cf_bounds(ds):
    lat = ds.lat.values
    dlat = float(abs(lat[1] - lat[0]))
    ds = ds.assign(lat_bounds=(("lat", "nb"), np.stack([lat - dlat/2, lat + dlat/2], axis=1)))
    ds.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north",
                          "standard_name": "latitude", "axis": "Y"})
    lon = ds.lon.values
    dlon = float(abs(lon[1] - lon[0]))
    ds = ds.assign(lon_bounds=(("lon", "nb"), np.stack([lon - dlon/2, lon + dlon/2], axis=1)))
    ds.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east",
                          "standard_name": "longitude", "axis": "X"})
    times = ds.time.values
    t_bounds = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        t_bounds[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        t_bounds[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds = ds.assign(time_bounds=(("time", "nb"), t_bounds))
    ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    return ds


def write_model(path, ds):
    ref_time = min(ds.time.values)
    units = f"days since {ref_time.year:04d}-{ref_time.month:02d}-{ref_time.day:02d} 00:00:00"
    encoding = {
        "burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
        "time": {"units": units, "calendar": "noleap", "dtype": "float64"},
        "time_bounds": {"units": units, "calendar": "noleap", "dtype": "float64"},
    }
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=encoding, format="NETCDF4_CLASSIC")
    os.replace(tmp, path)


def build_and_write(name, fire_fn, drivers, params, obs_mean, cos_lat):
    w3 = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360))
    pred = fire_fn(drivers, params)
    pm = float((pred * w3).sum() / w3.sum())
    if pm > 0:
        pred = pred * (obs_mean / pm)

    pred_hd = uncoarsen_1deg_to_half(pred).astype(np.float32)
    target_times = np.array(
        [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    )
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(-179.75, 180.0, 0.5)

    ds = xr.Dataset(
        {
            "burntArea": (("time", "lat", "lon"), pred_hd, {
                "units": "1",
                "standard_name": "burnt_area_fraction",
                "long_name": "Burnt Area Fraction",
            }),
        },
        coords={
            "time": ("time", target_times),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs={"title": f"ED-Model{name} burnt area", "Conventions": "CF-1.7"},
    )
    ds = add_cf_bounds(ds)

    dst_dir = OUT / f"ED-Model{name}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "burntArea.nc"
    write_model(dst, ds)
    print(f"  wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")


def main():
    print("Loading drivers...")
    trendy = load_trendy_v14_monthly()
    crujra = load_crujra_monthly()
    ed_static = load_ed_static()
    drivers = {**trendy, **crujra, **ed_static}
    del trendy, crujra, ed_static
    gc.collect()

    # Area-weighted global mean for rescaling
    lat_1d = np.arange(-89.5, 90.0, 1.0)
    cos_lat = np.cos(np.deg2rad(lat_1d)).astype(np.float32)

    # Need obs_mean for rescaling. User should provide or we compute from GFED
    # For now, use the canonical GFED4.1s 2001-2016 monthly mean over land.
    obs_mean = float(os.environ.get("GFED_MONTHLY_MEAN", 0.000764))
    print(f"Rescaling target (GFED monthly mean): {obs_mean:.6f}")

    print("\nModel A (19 params, 7 mechanisms)...")
    params_A = json.load(open(REPO / "models/A/params.json"))["params"]
    build_and_write("A", fire_A, drivers, params_A, obs_mean, cos_lat)

    print("\nModel B (13 params, 4 mechanisms)...")
    params_B = json.load(open(REPO / "models/B/params.json"))["params"]
    build_and_write("B", fire_B, drivers, params_B, obs_mean, cos_lat)

    print("\nModel C (10 params, 2 mechanisms)...")
    params_C = json.load(open(REPO / "models/C/params.json"))["params"]
    build_and_write("C", fire_C, drivers, params_C, obs_mean, cos_lat)

    print("\nAll three models written to out/")


if __name__ == "__main__":
    main()
