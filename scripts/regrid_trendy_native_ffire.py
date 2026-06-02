"""Regrid native TRENDY v14 fFire files to 0.5° GFED5 grid, 2001-2016,
and replace the EF-derived fFire.nc in MODELS_LEADERBOARD_FFIRE_GFED5/.

The old EF-derived files are renamed to fFire.EF-derived.nc.bak first.
"""
from __future__ import annotations
import os, shutil
from pathlib import Path
import cftime, numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data" / "trendy_v14"
DST_BASE = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5"
TARGET_LAT = np.linspace(-89.75, 89.75, 360)
TARGET_LON = np.linspace(-179.75, 179.75, 720)

# Map: leaderboard model name → source NC filename
MODELS = {
    "CLASSIC":    "CLASSIC_S3_fFire.nc",
    "CLM-FATES":  "CLM-FATES_S3_fFire.nc",
    "CLM6":       "CLM6.0_S3_fFire.nc",
    "E3SM":       "E3SM_S3_fFire.nc",
    "ELM-FATES":  "ELM-FATES_S3_fFire.nc",
    "JSBACH":     "JSBACH_S3_fFire.nc",
    "SDGVM":      "SDGVM_S3_fFire.nc",
    "VISIT":      "VISIT_S3_fFire.nc",
    "EDv3":       "EDv3_S3_fFire.nc",
}


def decode_time_to_year_month(ds):
    """Return (years, months) arrays matching ds.time. Handles cftime, numpy
    datetime64, and decimal years / month-index encodings."""
    tv = ds.time.values
    sample = tv[0]
    # numpy datetime64
    if isinstance(sample, np.datetime64):
        dt = tv.astype("datetime64[s]").astype(object)
        years = np.array([d.year for d in dt])
        months = np.array([d.month for d in dt])
        return years, months
    # cftime / python datetime
    if hasattr(sample, "year"):
        years = np.array([t.year for t in tv])
        months = np.array([t.month for t in tv])
        return years, months
    # decimal years or month index
    tv_f = np.asarray(tv, dtype=float)
    if tv_f.max() < 3000 and tv_f.min() > 1500:
        years = np.floor(tv_f).astype(int)
        months = (np.round((tv_f - years) * 12).astype(int) % 12) + 1
    else:
        origin = 1860
        idx = np.floor(tv_f).astype(int)
        years = origin + idx // 12
        months = (idx % 12) + 1
    return years, months


def regrid_to_gfed5(da_3d, native_lat, native_lon):
    """Linear interpolate (T, lat, lon) to (T, 360, 720) on TARGET_LAT/LON."""
    # Wrap longitude to [-180, 180]
    lon_vals = np.asarray(native_lon, dtype=float)
    if lon_vals.max() > 180.0:
        lon_vals = np.where(lon_vals > 180.0, lon_vals - 360.0, lon_vals)
        order = np.argsort(lon_vals)
        lon_vals = lon_vals[order]
        da_3d = da_3d[..., order]
    # Flip lat to ascending if needed
    lat_vals = np.asarray(native_lat, dtype=float)
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        da_3d = da_3d[:, ::-1, :]
    # Build a temporary xarray with proper coords for interp
    tmp = xr.DataArray(da_3d, dims=("time", "lat", "lon"),
                       coords={"lat": lat_vals, "lon": lon_vals})
    out = tmp.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear",
                     kwargs={"fill_value": 0.0})
    return out.values.astype(np.float32)


def process(model_name, src_file):
    src = SRC_DIR / src_file
    print(f"=== {model_name}  ({src_file}) ===")
    try:
        ds = xr.open_dataset(src, decode_times=True)
    except Exception:
        ds = xr.open_dataset(src, decode_times=False)

    # Find lat / lon coord names
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    years, months = decode_time_to_year_month(ds)
    mask = (years >= 2001) & (years <= 2016)
    if mask.sum() == 0:
        print(f"  ERROR: no months in 2001-2016, time range {years.min()}-{years.max()}")
        ds.close()
        return False
    print(f"  slicing {mask.sum()} months  (year range {years[mask].min()}-{years[mask].max()})")

    arr = ds["fFire"].values[mask].astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    native_lat = ds[lat_name].values
    native_lon = ds[lon_name].values
    print(f"  native grid {arr.shape}  range [{arr.min():.3e}, {arr.max():.3e}]")

    regridded = regrid_to_gfed5(arr, native_lat, native_lon)
    print(f"  regridded   {regridded.shape}  range [{regridded.min():.3e}, {regridded.max():.3e}]")

    # Build CF-compliant out dataset on the target grid using monthly time
    yrs_train = years[mask]; mos_train = months[mask]
    times = np.array([cftime.DatetimeNoLeap(int(y), int(m), 15)
                      for y, m in zip(yrs_train, mos_train)])
    tb = np.empty((len(times), 2), dtype=object)
    for i, (y, m) in enumerate(zip(yrs_train, mos_train)):
        tb[i, 0] = cftime.DatetimeNoLeap(int(y), int(m), 1)
        tb[i, 1] = cftime.DatetimeNoLeap(int(y) + (m == 12), (m % 12) + 1, 1)

    out_ds = xr.Dataset(
        {"fFire": (("time", "lat", "lon"), regridded,
                    {"units": "kg m-2 s-1", "long_name": "Fire Carbon Flux",
                     "standard_name": "fire_carbon_flux"})},
        coords={"time": times, "lat": TARGET_LAT.astype(np.float64),
                "lon": TARGET_LON.astype(np.float64)},
        attrs={"title": f"{model_name} native fFire (TRENDY v14, GCB 2025, regridded to 0.5°)",
               "source_file": src_file,
               "method": "Native TRENDY S3 fFire, linear interp to GFED5 0.5° grid",
               "Conventions": "CF-1.7"})
    out_ds = out_ds.assign(time_bounds=(("time", "nb"), tb))
    out_ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})

    dst_dir = DST_BASE / model_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_p = dst_dir / "fFire.nc"
    # Archive old EF-derived file
    if dst_p.exists():
        bak = dst_dir / "fFire.EF-derived.nc.bak"
        if not bak.exists():
            shutil.copy(dst_p, bak)
            print(f"  backed up old EF-derived fFire to {bak.name}")

    enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": "days since 2001-01-01 00:00:00", "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": "days since 2001-01-01 00:00:00", "calendar": "noleap", "dtype": "float64"}}
    tmp_p = dst_p.with_suffix(".nc.tmp")
    out_ds.to_netcdf(tmp_p, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp_p, dst_p)
    print(f"  wrote {dst_p}")
    ds.close()
    return True


def main():
    results = []
    for name, src in MODELS.items():
        ok = process(name, src)
        results.append((name, ok))
        print()
    print("=== summary ===")
    for n, ok in results:
        print(f"  {n}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
