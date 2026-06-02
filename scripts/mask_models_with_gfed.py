"""
mask_models_with_gfed.py
Every model NC in models_full/<name>/burntArea.nc gets its ocean cells set to
NaN by matching the GFED4.1s reference land mask. This is what ILAMB needs for
continent outlines to actually show in the generated map PNGs (otherwise the
cmap paints 0 as cream over the ocean and hides the grey land/ocean backdrop).
"""
from pathlib import Path
import numpy as np
import xarray as xr
import cftime

REPO     = Path(__file__).resolve().parents[1]
GFED_REF = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc"
MODELS   = REPO / "models_full"

YEARS      = list(range(2001, 2017))
TIMES      = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
TIME_UNITS = "days since 2001-01-01 00:00:00"

def monthly_time_bounds():
    lo = [cftime.DatetimeNoLeap(y, m, 1) for y in YEARS for m in range(1, 13)]
    hi = []
    for y in YEARS:
        for m in range(1, 13):
            ny = y + (1 if m == 12 else 0)
            nm = 1 if m == 12 else m + 1
            hi.append(cftime.DatetimeNoLeap(ny, nm, 1))
    return np.array(list(zip(lo, hi)))
TBNDS = monthly_time_bounds()

# 1. Build GFED land mask on the reference 0.5-deg grid
print("Loading GFED reference mask...")
ds = xr.open_dataset(GFED_REF, decode_times=False)
ref = ds["burntArea"].values     # (240, 360, 720) or similar
ref_lat = ds["lat"].values
ref_lon = ds["lon"].values
ds.close()

# Land = any finite, non-negative value over time
land05 = np.any(np.isfinite(ref) & (ref >= 0), axis=0)   # (360, 720) bool
print(f"  GFED land cells: {land05.sum()}/{land05.size} ({100*land05.mean():.1f}%)")

# Helper: interpolate mask onto arbitrary lat/lon grid (nearest neighbour)
def regrid_mask(src_lat, src_lon, mask, dst_lat, dst_lon):
    i = np.clip(np.searchsorted(src_lat, dst_lat) - 0, 0, len(src_lat) - 1)
    # pick nearest
    i = np.array([np.argmin(np.abs(src_lat - v)) for v in dst_lat])
    j = np.array([np.argmin(np.abs(src_lon - v)) for v in dst_lon])
    return mask[np.ix_(i, j)]

# 2. Mask each model file
for d in sorted(MODELS.iterdir()):
    nc = d / "burntArea.nc"
    if not nc.exists(): continue
    ds = xr.open_dataset(nc, decode_times=False)
    arr = ds["burntArea"].values.astype(np.float32)
    lat = ds["lat"].values
    lon = ds["lon"].values

    mask = regrid_mask(ref_lat, ref_lon, land05, lat, lon)   # (nlat, nlon)
    if mask.shape != arr.shape[-2:]:
        print(f"  [skip] {d.name}: shape mismatch {mask.shape} vs {arr.shape}")
        ds.close(); continue

    before_nan = np.isnan(arr).sum()
    arr = np.where(mask[np.newaxis, :, :], arr, np.nan)
    after_nan  = np.isnan(arr).sum()

    # Preserve attrs
    var_attrs   = dict(ds["burntArea"].attrs)
    ds.close()

    out = xr.Dataset(
        {"burntArea":   (("time", "lat", "lon"), arr, var_attrs),
         "time_bounds": (("time", "nv"), TBNDS)},
        coords={"time": ("time", TIMES,
                          {"bounds": "time_bounds", "standard_name": "time", "axis": "T"}),
                "lat":  ("lat", lat.astype(np.float64)),
                "lon":  ("lon", lon.astype(np.float64))},
        attrs={"title": f"{d.name} burntArea (ocean-masked to match GFED land mask)",
               "Conventions": "CF-1.7"})
    enc = {"burntArea":   {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":        {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = nc.with_suffix(".nc.tmp")
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    import os; os.replace(tmp, nc)
    print(f"  [ok]  {d.name:20s}  NaNs: {before_nan} -> {after_nan}")

print("\nDone. Re-run ilamb-run on models_full/ to regenerate maps with continents.")
