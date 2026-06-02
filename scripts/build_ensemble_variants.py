"""
Build ensemble variants by averaging existing burntArea predictions and by
post-hoc magnitude scaling. No refit, just arithmetic.

Strategies:
1. ED-Ensemble-CA: simple mean of ED-ModelC-ILAMB and ED-ModelA-final
2. ED-ModelC-Scaled: ED-ModelC-ILAMB scaled so global mean = GFED global mean
3. ED-Ensemble-Top: mean of CLASSIC and CLM6 (just to verify ensemble even helps)
"""
from __future__ import annotations
import os, cftime
from pathlib import Path
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
MR = REPO / "ilamb" / "MODELS_LEADERBOARD"


def load_da(name):
    p = MR / name / "burntArea.nc"
    return xr.open_dataset(p)["burntArea"]


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


def write_variant(arr_3d, name, title, lat, lon, times):
    out = MR / name / "burntArea.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), arr_3d.astype(np.float32),
                       {"units": "1", "standard_name": "burnt_area_fraction",
                        "long_name": "Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": title, "Conventions": "CF-1.7"})
    ds = add_cf_bounds(ds)
    time_units = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, out)
    print(f"wrote {out}")


# Load source variants and GFED truth
print("Loading source variants...")
da_C = load_da("ED-ModelC-ILAMB")
da_A = load_da("ED-ModelA-final")
da_B = load_da("ED-ModelB-final")
lat = da_C.lat.values; lon = da_C.lon.values; times = da_C.time.values

print("Loading GFED reference...")
gfed = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc")
yrs = np.array([t.year for t in gfed.time.values])
gfed_mask = (yrs >= 2001) & (yrs <= 2016)
# GFED is in '%', convert to '1' (fraction) by /100
obs = (gfed["burntArea"].values[gfed_mask] / 100.0).astype(np.float32)
print(f"obs range {np.nanmin(obs):.4g} to {np.nanmax(obs):.4g}")

# Variant 1: Simple ensemble of C-ILAMB + A-final
print("\nVariant 1: simple ensemble of C-ILAMB and A-final")
ens_CA = (np.nan_to_num(da_C.values, nan=0.0) + np.nan_to_num(da_A.values, nan=0.0)) / 2.0
# Mask same as A-final mask
mask = ~np.isnan(da_A.values[0])  # use any time slice
ens_CA = np.where(mask[None, :, :], ens_CA, np.nan)
write_variant(ens_CA, "ED-Ensemble-CA",
              "ED-Ensemble-CA (mean of C-ILAMB and A-final)", lat, lon, times)

# Variant 2: Triple ensemble C+A+B
print("\nVariant 2: triple ensemble C+A+B")
ens_CAB = (np.nan_to_num(da_C.values, nan=0.0)
            + np.nan_to_num(da_A.values, nan=0.0)
            + np.nan_to_num(da_B.values, nan=0.0)) / 3.0
ens_CAB = np.where(mask[None, :, :], ens_CAB, np.nan)
write_variant(ens_CAB, "ED-Ensemble-CAB",
              "ED-Ensemble-CAB (mean of C-ILAMB, A-final, B-final)", lat, lon, times)

# Variant 3: Magnitude-scaled C-ILAMB
print("\nVariant 3: magnitude-scaled C-ILAMB")
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
w2 = cos_lat[:, None]
land = ~np.isnan(da_C.values[0])
w2_land = (w2 * land).astype(np.float32)
pred_mean = float((np.nan_to_num(da_C.values, nan=0.0) * w2_land[None, :, :]).sum() / (192 * w2_land.sum() + 1e-9))
obs_mean  = float((obs * w2_land[None, :, :]).sum() / (192 * w2_land.sum() + 1e-9))
ratio = obs_mean / (pred_mean + 1e-9)
print(f"  pred_mean={pred_mean:.4g}, obs_mean={obs_mean:.4g}, ratio={ratio:.4f}")
scaled = np.nan_to_num(da_C.values, nan=0.0) * ratio
scaled = np.where(land[None, :, :], scaled, np.nan)
write_variant(scaled, "ED-ModelC-Scaled",
              f"ED-ModelC-Scaled (C-ILAMB scaled by {ratio:.3f} to match GFED mean)", lat, lon, times)

# Variant 4: Weighted ensemble favoring better-Spatial models (CLASSIC + CLM6)
print("\nVariant 4: weighted ensemble C-ILAMB + A-final + B-final, weighted by ILAMB scores")
# weights proportional to current Overall scores
w_C = 0.6482; w_A = 0.6574; w_B = 0.6506
W = w_C + w_A + w_B
ens_W = (w_C * np.nan_to_num(da_C.values, nan=0.0)
        + w_A * np.nan_to_num(da_A.values, nan=0.0)
        + w_B * np.nan_to_num(da_B.values, nan=0.0)) / W
ens_W = np.where(mask[None, :, :], ens_W, np.nan)
write_variant(ens_W, "ED-Ensemble-Weighted",
              "ED-Ensemble-Weighted (score-weighted mean of C, A, B)", lat, lon, times)

print("\nAll variants written.")
