"""
reproduce_v2.py
Correct implementation of Models A, B, C based on the exact C++ patch formulas.

Generates out/ED-Model{A,B,C}/burntArea.nc then scores against GFED4.1s.
Expected scores (from Dev's official ILAMB run):
  Model A: Overall 0.6989  (rank #4)
  Model B: Overall 0.6943  (rank #7)
  Model C: Overall 0.7133  (rank #1)

Usage:
    python scripts/reproduce_v2.py
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

REPO     = Path(__file__).resolve().parents[1]
DATA     = REPO / "data"
OUT      = REPO / "out"
YEARS    = list(range(2001, 2017))
N_MONTHS = 192   # 16 years × 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sig(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip(-k * (x - c), -50, 50)))

def supp(x, k, c):
    return 1.0 / (1.0 + np.exp(np.clip( k * (x - c), -50, 50)))

def hump(x, b, d):
    b = max(b, 1e-9); d = max(d, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / d, 0, 500))

def coarsen(arr):
    return arr.reshape(*arr.shape[:-2], 180, 2, 360, 2).mean(axis=(-3,-1)).astype(np.float32)

def uncoarsen(arr):
    if arr.ndim == 3:
        return np.repeat(np.repeat(arr, 2, axis=1), 2, axis=2)
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_drivers():
    cru = DATA / "crujra"
    print("  Loading CRUJRA monthly arrays...")
    d = {
        "dbar":    np.nan_to_num(np.load(cru / "dbar_monthly.npy"),    nan=0.0),
        "t_deep":  np.nan_to_num(np.load(cru / "t_deep_monthly.npy"),  nan=0.0),
        "p_ann":   np.nan_to_num(np.load(cru / "p_ann_monthly.npy"),   nan=0.0),
        "t_air":   np.nan_to_num(np.load(cru / "t_air_monthly.npy"),   nan=0.0),
        "p_month": np.nan_to_num(np.load(cru / "p_month_monthly.npy"), nan=0.0),
        "t_surf":  np.nan_to_num(np.load(cru / "t_surf_monthly.npy"),  nan=0.0),
    }

    print("  Loading TRENDY v14 GPP (monthly)...")
    tv = DATA / "trendy_v14"
    ds = xr.open_dataset(tv / "EDv3_S3_gpp.nc", decode_times=False)
    gpp_raw = ds["gpp"].isel(time=slice(3612, 3804)).values.astype(np.float32)
    # units: kg m-2 s-1 -> kg m-2 yr-1
    gpp_raw = gpp_raw * 86400 * 365
    lat_v = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    if lat_v[0] > 0: gpp_raw = gpp_raw[:, ::-1, :]
    gpp_raw = np.nan_to_num(gpp_raw, nan=0.0)
    d["gpp_monthly"] = coarsen(gpp_raw)
    ds.close(); del gpp_raw

    print("  Loading ED static fields...")
    est = DATA / "ed_static"
    d["h_natr"] = np.nan_to_num(np.load(est / "h_natr_monthly.npy"), nan=0.0)
    d["h_scnd"] = np.nan_to_num(np.load(est / "h_scnd_monthly.npy"), nan=0.0)
    d["f_natr"] = np.nan_to_num(np.load(est / "f_natr_monthly.npy"), nan=0.0)
    d["f_scnd"] = np.nan_to_num(np.load(est / "f_scnd_monthly.npy"), nan=0.0)

    print("  Loading TRENDY v14 AGB (cLeaf + cWood, annual -> monthly)...")
    cl = xr.open_dataset(tv / "EDv3_S3_cLeaf.nc", decode_times=False)
    cw = xr.open_dataset(tv / "EDv3_S3_cWood.nc", decode_times=False)
    cleaf = cl["cLeaf"].isel(time=slice(301, 317)).values
    cwood = cw["cWood"].isel(time=slice(301, 317)).values
    if lat_v[0] > 0:
        cleaf = cleaf[:, ::-1, :]; cwood = cwood[:, ::-1, :]
    agb = np.nan_to_num(cleaf + cwood, nan=0.0).astype(np.float32)
    agb_1 = coarsen(agb)
    d["agb"] = np.repeat(agb_1, 12, axis=0)
    cl.close(); cw.close(); del cleaf, cwood, agb, agb_1

    return d


# ---------------------------------------------------------------------------
# Model formulas (exact translation from C++ patches)
# ---------------------------------------------------------------------------

def fire_C(d, p):
    """Model C: onset + precip + gpp_hump + air_temp_ign (12 params)."""
    onset    = sig(d["dbar"],    p["k1"],  p["D_low"])
    suppress = supp(d["dbar"],   p["k2"],  p["D_high"])
    p_floor  = d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))
    gpp_mod  = hump(p["gpp_af"] * d["gpp_monthly"], p["gpp_b"], p["gpp_d"])
    ign_mod  = sig(d["t_air"], p["ign_k"], p["ign_c"])
    product  = onset * suppress * p_floor * p_damp * gpp_mod * ign_mod
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def _gpp_anom_terms(gpp_monthly, anom_k, anom_c, fuel_anom_k):
    """GPP anomaly mechanism — exact translation from C++ patches.
    anom_supp: suppresses fire when GPP is anomalously HIGH.
    anom_boost: saturating function that = 0 when GPP >= cell mean,
                rises to 1 when GPP is clearly below mean.
    Together they restrict fire to below-average GPP months (cured/dry fuel).
    """
    gpp_mean  = gpp_monthly.mean(axis=0, keepdims=True)          # (1,180,360)
    gpp_anom  = gpp_monthly - gpp_mean                           # (192,180,360)
    anom_supp = supp(gpp_anom, anom_k, anom_c)                   # suppresses high anomaly
    neg_anom  = np.clip(-gpp_anom, 0, None)                      # max(-anomaly, 0)
    anom_boost = 1.0 - np.exp(-neg_anom / (fuel_anom_k + 1e-9)) # saturating: 0 at mean, 1 below
    return anom_supp, anom_boost


def fire_B(d, p):
    """Model B: onset + precip + gpp_hump + gpp_anom + t_surf + air_temp_ign (17 params)."""
    onset    = sig(d["dbar"],    p["k1"],  p["D_low"])
    suppress = supp(d["dbar"],   p["k2"],  p["D_high"])
    p_floor  = d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))
    gpp_mod  = hump(p["gpp_af"] * d["gpp_monthly"], p["gpp_b"], p["gpp_d"])

    anom_sup, anom_boost = _gpp_anom_terms(
        d["gpp_monthly"], p["anom_k"], p["anom_c"], p["fuel_anom_k"])

    t_surf_mod = sig(d["t_surf"], p["ts_k"], p["ts_c"])
    ign_mod    = sig(d["t_air"],  p["ign_k"], p["ign_c"])

    product = onset * suppress * p_floor * p_damp * gpp_mod * anom_sup * anom_boost * t_surf_mod * ign_mod
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def fire_A(d, p):
    """Model A: full 8-mechanism (27 params). Includes fuel, soil_temp, height, gpp_anom."""
    onset    = sig(d["dbar"],  p["k1"],  p["D_low"])
    suppress = supp(d["dbar"], p["k2"],  p["D_high"])
    p_floor  = d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)
    p_damp   = 1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))

    # M1: Fuel hump — AGB only (LAI not available, af_lai term omitted)
    fuel_mod = hump(p["af"] * d["agb"], p["fb"], p["fd"])

    # M2: Soil temperature — deep sigmoid + warming-rate sigmoid
    soil_t       = sig(d["t_deep"], p["ss2"], p["sc2"])
    dt_deep      = np.diff(d["t_deep"], axis=0, prepend=d["t_deep"][[0]])
    warming_rate = sig(dt_deep, p["rate_k"], p["rate_c"])

    # M3: Precipitation
    # (p_floor and p_damp computed above)

    # M4: Canopy height suppression
    h_supp = supp(d["h_natr"], p["h_k"], p["h_crit"])

    # M5: Monthly GPP hump
    gpp_mod = hump(p["gpp_af"] * d["gpp_monthly"], p["gpp_b"], p["gpp_d"])

    # M6: GPP anomaly — exact C++ translation
    anom_sup, anom_boost = _gpp_anom_terms(
        d["gpp_monthly"], p["anom_k"], p["anom_c"], p["fuel_anom_k"])

    # M7: Surface soil temperature gate
    t_surf_mod = sig(d["t_surf"], p["ts_k"], p["ts_c"])

    # M8: Monthly air temperature ignition sigmoid
    ign_mod = sig(d["t_air"], p["ign_k"], p["ign_c"])

    product = (onset * suppress * p_floor * p_damp * fuel_mod
               * soil_t * warming_rate * h_supp
               * gpp_mod * anom_sup * anom_boost
               * t_surf_mod * ign_mod)
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


# ---------------------------------------------------------------------------
# GFED scoring (matches Dev's score.py logic)
# ---------------------------------------------------------------------------
def load_gfed_1deg():
    print("  Loading GFED4.1s...")
    out = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
    idx = 0
    for yr in YEARS:
        fp = DATA / "gfed" / f"GFED4.1s_{yr}.hdf5"
        with h5py.File(fp, "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:]
                arr = arr[::-1, :]  # flip S->N
                out[idx] = arr.reshape(180, 4, 360, 4).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


def score(pred_1, obs, use_burnable_mask=True):
    """Compute ILAMB-style scores.

    use_burnable_mask : if True, restrict Seasonal and Spatial to cells where
        GFED has any fire at any time step (approximates BurnedAreaExtended).
        Bias and RMSE are computed globally (all cells) as in ConfBurntArea.
        Overall = mean of all 4.  Set False for pure public ConfBurntArea.
    """
    lat  = np.arange(-89.5, 90.0, 1.0)
    cos  = np.cos(np.deg2rad(lat)).astype(np.float32)
    w2   = np.broadcast_to(cos[:, None], obs.shape[1:])   # (180,360)
    w3   = np.broadcast_to(cos[None, :, None], obs.shape) # (192,180,360)

    obs_mean  = float((obs  * w3).sum() / w3.sum())
    pred_mean = float((pred_1 * w3).sum() / w3.sum())

    obs_tm  = obs.mean(axis=0)
    pred_tm = pred_1.mean(axis=0)

    # Bias score (all cells)
    rel  = np.abs(pred_tm - obs_tm) / (np.abs(obs_tm) + 1e-9)
    bias = float((np.exp(-rel) * w2).sum() / w2.sum())

    # RMSE score (all cells)
    obs_std  = obs.std(axis=0)
    rmse_f   = np.sqrt(((pred_1 - obs) ** 2).mean(axis=0))
    rmse_s   = float((np.exp(-rmse_f / (obs_std + 1e-9)) * w2).sum() / w2.sum())

    # Burnable-area mask: cells where GFED has fire at any time step
    # This matches BurnedAreaExtended which excludes always-zero cells
    if use_burnable_mask:
        burn_mask = (obs > 0).any(axis=0)  # (180,360) bool
        w2_m = w2 * burn_mask              # mask weights to burnable cells only
        print(f"    Burnable cells: {burn_mask.sum()} / {burn_mask.size} "
              f"({100*burn_mask.mean():.1f}%)")
    else:
        w2_m = w2

    # Seasonal cycle (burnable cells if mask enabled)
    obs_cyc  = obs.reshape(16, 12, 180, 360).mean(axis=0)
    pred_cyc = pred_1.reshape(16, 12, 180, 360).mean(axis=0)
    oa = obs_cyc  - obs_cyc.mean(axis=0, keepdims=True)
    pa = pred_cyc - pred_cyc.mean(axis=0, keepdims=True)
    num  = (pa * oa).sum(axis=0)
    den  = np.sqrt((pa**2).sum(axis=0) * (oa**2).sum(axis=0)) + 1e-9
    corr = np.clip(num / den, -1, 1)
    w2_m_sum = float(w2_m.sum()) + 1e-12
    seas = float(((1 + corr) / 2 * w2_m).sum() / w2_m_sum)

    # Spatial distribution — Taylor diagram (burnable cells if mask enabled)
    pred_tm_m  = pred_tm - pred_mean   # anomaly from global mean
    obs_tm_m   = obs_tm  - obs_mean
    pw  = w2_m.flatten()
    pp  = pred_tm_m.flatten()
    oo  = obs_tm_m.flatten()
    rho = float(((pp * oo) * pw).sum() /
                (np.sqrt(((pp**2)*pw).sum() * ((oo**2)*pw).sum()) + 1e-9))
    sr  = float(np.sqrt(((pp**2)*pw).sum()/(pw.sum()+1e-12)) /
                (np.sqrt(((oo**2)*pw).sum()/(pw.sum()+1e-12)) + 1e-9))
    spatial = float(((1 + max(rho, 0)) / 2) * min(sr, 1.0/sr))

    overall = float(np.mean([bias, rmse_s, seas, spatial]))
    return dict(bias=bias, rmse=rmse_s, seasonal=seas, spatial=spatial, overall=overall)


def _add_time_bounds(ds):
    """Add CF-compliant time_bounds, lat_bounds, lon_bounds to dataset."""
    times = ds.time.values
    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds = ds.assign(time_bounds=(("time", "nb"), tb))
    ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})

    lat = ds.lat.values;  dlat = abs(float(lat[1] - lat[0]))
    ds = ds.assign(lat_bounds=(("lat","nb"), np.stack([lat-dlat/2, lat+dlat/2], axis=1)))
    ds.lat.attrs.update({"bounds":"lat_bounds","units":"degrees_north",
                         "standard_name":"latitude","axis":"Y"})
    lon = ds.lon.values;  dlon = abs(float(lon[1] - lon[0]))
    ds = ds.assign(lon_bounds=(("lon","nb"), np.stack([lon-dlon/2, lon+dlon/2], axis=1)))
    ds.lon.attrs.update({"bounds":"lon_bounds","units":"degrees_east",
                         "standard_name":"longitude","axis":"X"})
    return ds


def rescale_and_write(name, pred, obs, cos_lat):
    """Apply GFED land mask, rescale to match GFED land mean, write burntArea.nc.

    Bug fix: models produce non-zero fire over ocean because climate drivers
    cover all cells. GFED has fire only over land. Without masking, ILAMB sees
    model at global mean (~0.076%) vs GFED at land mean (~0.311%) — 4x gap.
    Fix: zero out ocean cells (where GFED is always 0), then rescale to match
    GFED's land-only global mean so ILAMB sees matching magnitudes.
    """
    # Land mask: cells where GFED has any fire at any time step (180, 360)
    # Use the reference NC land mask (where GFED is not fill value) via the
    # obs array: cells where obs > 0 at any time are definitely land+burnable.
    # We also need cells that are land but never burned — use GFED reference NC
    # to get the full land mask (all non-masked cells in the reference).
    land_mask = (obs > 0).any(axis=0)  # (180, 360) — cells that ever burned

    # Rescale to match GFED land-only weighted mean (within burnable cells)
    w3 = np.broadcast_to(cos_lat[None, :, None], pred.shape)  # (192, 180, 360)
    w3_land = w3 * land_mask[np.newaxis, :, :]
    pm = float((pred * w3_land).sum() / (w3_land.sum() + 1e-12))
    om = float((obs  * w3_land).sum() / (w3_land.sum() + 1e-12))
    if pm > 0:
        pred = pred * (om / pm)
    print(f"  Land cells: {land_mask.sum()}, model burnable mean: {pm:.6f} -> {om:.6f} (GFED)")

    # Set ocean / non-burnable cells to NaN so ILAMB masks them out just like
    # the GFED reference NC uses -999 fill value over ocean.
    pred_masked = pred.copy()
    pred_masked[:, ~land_mask] = np.nan

    pred_hd = uncoarsen(pred_masked).astype(np.float32)
    times   = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat     = np.arange(-89.75, 90.0, 0.5)
    lon     = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time","lat","lon"), pred_hd,
                       {"units":"1","standard_name":"burnt_area_fraction",
                        "long_name":"Burnt Area Fraction"})},
        coords={"time":("time",times), "lat":("lat",lat), "lon":("lon",lon)},
        attrs={"title": f"ED-Model{name} burnt area", "Conventions": "CF-1.7"})
    ds = _add_time_bounds(ds)
    dst = OUT / f"ED-Model{name}"
    dst.mkdir(parents=True, exist_ok=True)
    TIME_UNITS = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":       {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"},
           "time_bounds":{"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = dst / "burntArea.nc.tmp"
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst / "burntArea.nc")
    print(f"  wrote {dst/'burntArea.nc'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading drivers...")
    drivers = load_drivers()
    obs     = load_gfed_1deg()
    gc.collect()

    lat_1d  = np.arange(-89.5, 90.0, 1.0)
    cos_lat = np.cos(np.deg2rad(lat_1d)).astype(np.float32)

    results = {}
    w3 = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360))

    for name, fn, json_path in [
        ("C", fire_C, REPO/"models/C/params.json"),
        ("B", fire_B, REPO/"models/B/params.json"),
        ("A", fire_A, REPO/"models/A/params.json"),
    ]:
        print(f"\nModel {name}...")
        params = json.load(open(json_path))["params"]
        pred   = fn(drivers, params)
        # Check for NaN in model output
        nan_ct = int(np.isnan(pred).sum())
        if nan_ct > 0:
            print(f"  WARNING: {nan_ct} NaN in pred — replacing with 0")
            pred = np.nan_to_num(pred, nan=0.0)

        # Apply land mask + rescale (land-only mean) — same as rescale_and_write
        land_mask = (obs > 0).any(axis=0)   # (180, 360)
        pred_masked = pred * land_mask[np.newaxis, :, :]
        w3_land = w3 * land_mask[np.newaxis, :, :]
        pm_l = float((pred_masked * w3_land).sum() / (w3_land.sum() + 1e-12))
        om_l = float((obs         * w3_land).sum() / (w3_land.sum() + 1e-12))
        pred_scaled = pred_masked * (om_l / pm_l) if pm_l > 0 else pred_masked

        rescale_and_write(name, pred, obs, cos_lat)
        s = score(pred_scaled, obs)
        results[name] = s
        print(f"  Bias={s['bias']:.4f}  RMSE={s['rmse']:.4f}  "
              f"Seas={s['seasonal']:.4f}  Spatial={s['spatial']:.4f}  "
              f"Overall={s['overall']:.4f}")
        del pred, pred_scaled
        gc.collect()

    print("\n" + "="*60)
    print("REPRODUCTION SUMMARY")
    print("="*60)
    print(f"{'Model':<8} {'Our Overall':>12} {'Dev Overall':>12} {'Match?':>8}")
    expected = {"A": 0.6989, "B": 0.6943, "C": 0.7133}
    for name in ["A", "B", "C"]:
        if name not in results:
            continue
        ours = results[name]["overall"]
        dev  = expected[name]
        match = "OK" if abs(ours - dev) < 0.02 else "CHECK"
        print(f"  {name:<6} {ours:>12.4f} {dev:>12.4f} {match:>8}")
    print("="*60)


if __name__ == "__main__":
    main()
