"""
Build an ILAMB-compatible GFED5 reference NetCDF.

Source: GFED5 monthly burned area files (BAYYYYMM.nc) from Zenodo 7668424.
Each source file:
  - 0.25° × 0.25° (lat 720, lon 1440)
  - Variable "Total" in km² per grid cell per month
  - Time coverage 2001-2020 at 0.25°
  - Coordinates lat -89.875 to 89.875, lon -179.875 to 179.875

Output (mirrors the GFED4.1S reference at ilamb_ref_official):
  - 0.5° × 0.5° (lat 360, lon 720)
  - Variable "burntArea" in % per month
  - Time coverage 2001-2020 (240 months) — but we cap at 2016 to match our refit window
  - Coordinates lat -89.75 to 89.75 (S→N), lon -179.75 to 179.75

Aggregation:
  1. Sum 2×2 0.25° cells into one 0.5° cell (km² is additive)
  2. Divide by grid-cell area to get fraction
  3. Multiply by 100 for percent
"""
from __future__ import annotations
from pathlib import Path
import os
import cftime
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data" / "gfed5"
OUT = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"

YEAR_START = 2001
YEAR_END = 2020   # GFED5 max at 0.25 deg

# 0.5 deg grid (matches GFED4.1S ILAMB ref + Lei's NC)
LAT_05 = np.arange(-89.75, 90.0, 0.5).astype(np.float64)  # 360 points S->N
LON_05 = np.arange(-179.75, 180.0, 0.5).astype(np.float64)  # 720 points


def grid_cell_area_km2(lat_05, lon_05):
    """Approximate 0.5° grid-cell area in km², varying with cos(lat).
    Uses spherical-cap formula.
    """
    R = 6371.0088   # Earth mean radius in km
    dlat = 0.5 * np.pi / 180.0
    dlon = 0.5 * np.pi / 180.0
    lat_rad = np.deg2rad(lat_05).astype(np.float64)
    # area = R^2 * dlon * (sin(lat+dlat/2) - sin(lat-dlat/2))
    area_per_lat = R * R * dlon * (np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2))
    # broadcast over lon
    return np.broadcast_to(area_per_lat[:, None], (len(lat_05), len(lon_05))).copy().astype(np.float64)


def aggregate_025_to_05(arr_025, lat_025):
    """Sum a 720x1440 grid (km²) into a 360x720 grid by 2x2 blocks."""
    # First reorient lat if needed. GFED5 lat is -89.875 to 89.875 (S->N).
    if lat_025[0] > lat_025[-1]:
        arr_025 = arr_025[::-1, :]
    # Now lat is S->N. Aggregate 2x2.
    out = arr_025.reshape(360, 2, 720, 2).sum(axis=(1, 3))
    return out


def main():
    print(f"Building GFED5 ILAMB reference at {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Build 0.5° grid cell area (km²) for fraction conversion
    area_km2 = grid_cell_area_km2(LAT_05, LON_05)
    print(f"  0.5° grid-cell area, equator-ish ~ {area_km2[180, 360]:.1f} km²,"
          f" pole ~ {area_km2[0, 0]:.1f} km²")

    months = list(range(1, 13))
    times = []
    burnt_list = []

    for yr in range(YEAR_START, YEAR_END + 1):
        for mo in months:
            fp = SRC_DIR / f"BA{yr:04d}{mo:02d}.nc"
            if not fp.exists():
                print(f"  MISSING {fp.name}, filling zeros")
                burnt_list.append(np.zeros((360, 720), dtype=np.float32))
                times.append(cftime.DatetimeNoLeap(yr, mo, 15))
                continue
            ds = xr.open_dataset(fp)
            total = ds["Total"].values  # (1, 720, 1440)
            if total.ndim == 3:
                total = total[0]
            # Aggregate 0.25° -> 0.5°
            total_05 = aggregate_025_to_05(np.nan_to_num(total, nan=0.0),
                                            ds["lat"].values)
            # km² -> fraction -> percent
            frac = total_05 / np.maximum(area_km2, 1e-9)
            pct = (frac * 100.0).astype(np.float32)
            burnt_list.append(pct)
            times.append(cftime.DatetimeNoLeap(yr, mo, 15))
            ds.close()
        print(f"  done {yr}")

    burnt = np.stack(burnt_list, axis=0)  # (n_months, 360, 720)
    print(f"\n  built array shape {burnt.shape}")
    print(f"  global mean = {float(np.nanmean(burnt)):.4g}% per month")
    print(f"  global max  = {float(np.nanmax(burnt)):.2f}% per month")

    # CF-compliant dataset (mirrors GFED4.1S ILAMB ref structure)
    ds_out = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), burnt,
                       {"units": "%",
                        "standard_name": "burned area fraction",
                        "long_name": "GFED5 burned area fraction"})},
        coords={"time": ("time", times),
                "lat": ("lat", LAT_05),
                "lon": ("lon", LON_05)},
        attrs={"title": "GFED5 burned area at 0.5deg (aggregated from 0.25deg)",
               "version": "5.0",
               "Conventions": "CF-1.7",
               "source": "Zenodo 7668424, Chen et al. (van der Werf group)",
               "comments": f"Time period 2001-01 through {YEAR_END}-12; aggregated by 2x2 sum from 0.25deg; converted km² to % using spherical-cap grid-cell area"})

    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds_out = ds_out.assign(time_bounds=(("time", "nb"), tb))
    ds_out.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    ds_out = ds_out.assign(lat_bounds=(("lat", "nb"),
                                        np.stack([LAT_05 - 0.25, LAT_05 + 0.25], axis=1)))
    ds_out.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north",
                              "standard_name": "latitude", "axis": "Y"})
    ds_out = ds_out.assign(lon_bounds=(("lon", "nb"),
                                        np.stack([LON_05 - 0.25, LON_05 + 0.25], axis=1)))
    ds_out.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east",
                              "standard_name": "longitude", "axis": "X"})

    time_units = f"days since {YEAR_START}-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = OUT.with_suffix(".nc.tmp")
    ds_out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
