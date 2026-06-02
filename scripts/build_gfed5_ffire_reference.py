"""
Build GFED5 fire-carbon-emission (fFire) ILAMB reference at 0.5°.

Source: GFED5.1 monthly emissions files (data/gfed5/emissions/GFED5.1_monthly_YYYY.nc),
each containing 12 monthly time steps. Variable C is in g C per month per 0.25° cell.

Output: ilamb_ref_official/DATA/fFire/GFED5/fFire.nc
  - 0.5° × 0.5° (lat 360, lon 720)
  - units kg m^-2 s^-1 (matches GFED4.1S fFire ref)
  - 2001-2020 (240 months) - matches our other GFED5 ref
"""
from __future__ import annotations
from pathlib import Path
import os
import cftime
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data" / "gfed5" / "emissions"
OUT = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"

YEAR_START = 2001
YEAR_END = 2020

LAT_05 = np.arange(-89.75, 90.0, 0.5).astype(np.float64)
LON_05 = np.arange(-179.75, 180.0, 0.5).astype(np.float64)

SEC_PER_MONTH = 365.25 / 12 * 86400.0
R = 6371.0088e3  # m
DLAT_05 = 0.5 * np.pi / 180.0
DLON_05 = 0.5 * np.pi / 180.0


def cell_area_m2_05deg():
    lat_rad = np.deg2rad(LAT_05)
    area = R * R * DLON_05 * (np.sin(lat_rad + DLAT_05 / 2) - np.sin(lat_rad - DLAT_05 / 2))
    return np.broadcast_to(area[:, None], (360, 720)).astype(np.float64)


def aggregate_025_to_05_sum(arr_025, lat_025):
    """Sum 2x2 0.25° cells into 0.5° cells. Reorient to S->N if needed."""
    if lat_025[0] > lat_025[-1]:
        arr_025 = arr_025[::-1, :]
    return arr_025.reshape(360, 2, 720, 2).sum(axis=(1, 3))


def main():
    print(f"Building GFED5 fFire reference at {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    area = cell_area_m2_05deg()
    print(f"  0.5° cell area equator ~ {area[180, 360]/1e6:.1f} km², pole ~ {area[0, 0]/1e6:.2f} km²")

    times = []
    flux_list = []

    for yr in range(YEAR_START, YEAR_END + 1):
        fp = SRC_DIR / f"GFED5.1_monthly_{yr:04d}.nc"
        if not fp.exists():
            print(f"  MISSING {fp.name}")
            continue
        ds = xr.open_dataset(fp, decode_times=False)
        c_yr = ds["C"].values  # (12, 720, 1440), g C per month per 0.25 cell
        lat_025 = ds["lat"].values
        for mo in range(12):
            c_025 = np.nan_to_num(c_yr[mo], nan=0.0)
            # Aggregate to 0.5° (sum of 4 cells)
            c_05 = aggregate_025_to_05_sum(c_025, lat_025)  # g C per month per 0.5° cell
            # Convert to kg / m^2 / s
            flux = (c_05 * 1e-3) / area / SEC_PER_MONTH
            flux_list.append(flux.astype(np.float32))
            times.append(cftime.DatetimeNoLeap(yr, mo + 1, 15))
        ds.close()
        print(f"  done {yr}")

    flux = np.stack(flux_list, axis=0)
    print(f"\n  shape {flux.shape}")
    print(f"  mean {float(np.nanmean(flux)):.4g} kg/m^2/s")
    print(f"  max  {float(np.nanmax(flux)):.4g}")

    ds_out = xr.Dataset(
        {"fFire": (("time", "lat", "lon"), flux,
                    {"units": "kg m-2 s-1",
                     "standard_name": "fire_carbon_flux",
                     "long_name": "GFED5 fire carbon emission flux"})},
        coords={"time": ("time", times),
                "lat": ("lat", LAT_05),
                "lon": ("lon", LON_05)},
        attrs={"title": "GFED5.1 fire carbon emissions at 0.5deg",
               "version": "5.1",
               "source": "Zenodo 16794692 GFED5.1_monthly bundle",
               "Conventions": "CF-1.7"})

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
    enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = OUT.with_suffix(".nc.tmp")
    ds_out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
