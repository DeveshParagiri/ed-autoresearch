#!/usr/bin/env python3
"""Build the ILAMB-ready GFED5.1 fire-carbon emissions reference.

The GFED5.1 monthly files store total grams of carbon emitted by each native
0.25 degree grid cell during a month. This converter sums each 2x2 block,
divides by the corresponding 0.5 degree cell area and exact month duration,
and writes the mean monthly flux in kg C m-2 s-1.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import xarray as xr


EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--burned-area-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2016)
    return parser.parse_args()


def cell_area(lat: np.ndarray, resolution: float) -> np.ndarray:
    half = 0.5 * resolution
    south = np.deg2rad(np.maximum(-90.0, lat - half))
    north = np.deg2rad(np.minimum(90.0, lat + half))
    width = np.deg2rad(resolution)
    return (EARTH_RADIUS_M**2 * width * (np.sin(north) - np.sin(south)))[:, None]


def month_bounds(year: int, month: int) -> tuple[np.datetime64, np.datetime64]:
    start = np.datetime64(f"{year:04d}-{month:02d}-01", "s")
    if month == 12:
        end = np.datetime64(f"{year + 1:04d}-01-01", "s")
    else:
        end = np.datetime64(f"{year:04d}-{month + 1:02d}-01", "s")
    return start, end


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start year must not exceed end year")

    years = range(args.start_year, args.end_year + 1)
    paths = [args.source / f"GFED5.1_monthly_{year}.nc" for year in years]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing GFED5.1 source files: {missing}")

    with xr.open_dataset(args.burned_area_mask) as mask_ds:
        mask_var = mask_ds["burntArea"]
        land = np.isfinite(mask_var.values).any(axis=0)
        mask_lat = mask_ds["lat"].values.astype(np.float64)
        mask_lon = mask_ds["lon"].values.astype(np.float64)

    lat = np.arange(-89.75, 90.0, 0.5, dtype=np.float64)
    lon = np.arange(-179.75, 180.0, 0.5, dtype=np.float64)
    if not np.allclose(lat, mask_lat) or not np.allclose(lon, mask_lon):
        raise ValueError("GFED5 burned-area mask does not use the expected 0.5 degree grid")

    area = cell_area(lat, 0.5)
    ntime = len(paths) * 12
    flux = np.empty((ntime, lat.size, lon.size), dtype=np.float32)
    times = np.empty(ntime, dtype="datetime64[s]")
    bounds = np.empty((ntime, 2), dtype="datetime64[s]")
    source_total_g = 0.0
    output_total_g = 0.0

    index = 0
    for year, path in zip(years, paths):
        with xr.open_dataset(path) as ds:
            carbon = ds["C"].values.astype(np.float64)
        if carbon.shape != (12, 720, 1440):
            raise ValueError(f"unexpected C shape in {path}: {carbon.shape}")

        carbon_05 = carbon.reshape(12, 360, 2, 720, 2).sum(axis=(2, 4))
        source_total_g += float(carbon.sum())

        for month in range(1, 13):
            start, end = month_bounds(year, month)
            seconds = calendar.monthrange(year, month)[1] * 86_400.0
            monthly = carbon_05[month - 1] * 1e-3 / (area * seconds)
            monthly[~land] = np.nan
            flux[index] = monthly.astype(np.float32)
            times[index] = start + (end - start) // 2
            bounds[index] = (start, end)
            output_total_g += float(
                np.nansum(monthly * area * seconds * 1e3, dtype=np.float64)
            )
            index += 1
        print(f"processed {year}: {path.name}")

    relative_error = abs(output_total_g - source_total_g) / source_total_g
    if relative_error > 1e-6:
        raise RuntimeError(f"carbon conservation check failed: relative error={relative_error:g}")

    lat_bounds = np.column_stack((lat - 0.25, lat + 0.25))
    lon_bounds = np.column_stack((lon - 0.25, lon + 0.25))
    dataset = xr.Dataset(
        data_vars={
            "fFire": (("time", "lat", "lon"), flux),
            "time_bounds": (("time", "bounds"), bounds),
            "lat_bounds": (("lat", "bounds"), lat_bounds),
            "lon_bounds": (("lon", "bounds"), lon_bounds),
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={
            "title": "GFED5.1 monthly fire carbon emissions at 0.5 degree",
            "institution": "Global Fire Emissions Database",
            "source": "GFED5.1 monthly files; doi:10.5281/zenodo.16794692",
            "history": "Aggregated conservatively from 0.25 to 0.5 degree",
            "Conventions": "CF-1.8",
            "temporal_coverage": f"{args.start_year}-01 through {args.end_year}-12",
        },
    )
    dataset["time"].attrs.update(long_name="time", bounds="time_bounds")
    dataset["lat"].attrs.update(
        units="degrees_north", standard_name="latitude", bounds="lat_bounds"
    )
    dataset["lon"].attrs.update(
        units="degrees_east", standard_name="longitude", bounds="lon_bounds"
    )
    dataset["fFire"].attrs.update(
        units="kg m-2 s-1",
        standard_name=(
            "surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_"
            "due_to_emission_from_fires"
        ),
        long_name="Fire carbon emissions",
        cell_methods="area: mean time: mean",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(
        args.output,
        encoding={
            "time": {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            },
            "time_bounds": {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            },
            "fFire": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "_FillValue": 1e20,
            }
        },
    )
    mean_pg_c_per_year = source_total_g / 1e15 / len(paths)
    print(f"wrote {args.output}")
    print(f"mean emissions: {mean_pg_c_per_year:.6f} Pg C yr-1")
    print(f"carbon conservation relative error: {relative_error:.3e}")


if __name__ == "__main__":
    main()
