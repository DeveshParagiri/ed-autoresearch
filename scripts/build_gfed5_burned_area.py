#!/usr/bin/env python3
"""Build the 0.5-degree ILAMB GFED5 burned-area reference."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import netCDF4
import numpy as np


EARTH_RADIUS_KM = 6371.0088
EXPECTED_MONTHS = tuple(
    f"{year}{month:02d}" for year in range(2001, 2021) for month in range(1, 13)
)


def cell_area_km2(latitudes: np.ndarray, resolution: float) -> np.ndarray:
    lower = np.deg2rad(latitudes - resolution / 2)
    upper = np.deg2rad(latitudes + resolution / 2)
    longitude_width = np.deg2rad(resolution)
    return EARTH_RADIUS_KM**2 * longitude_width * (np.sin(upper) - np.sin(lower))


def aggregate_total(source: np.ndarray) -> np.ndarray:
    if source.shape != (720, 1440):
        raise ValueError(f"expected a 720x1440 GFED5 grid, found {source.shape}")
    cells = source.reshape(360, 2, 720, 2)
    valid = np.isfinite(cells)
    values = np.where(valid, cells, 0.0).sum(axis=(1, 3))
    values[valid.sum(axis=(1, 3)) == 0] = np.nan
    return values


def expected_paths(input_dir: Path) -> list[Path]:
    by_month: dict[str, Path] = {}
    for path in input_dir.glob("BA??????.nc"):
        match = re.fullmatch(r"BA(\d{6})\.nc", path.name)
        if match:
            by_month[match.group(1)] = path
    missing = [month for month in EXPECTED_MONTHS if month not in by_month]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} GFED5 burned-area months; first missing: {missing[0]}"
        )
    return [by_month[month] for month in EXPECTED_MONTHS]


def noleap_time_axis() -> tuple[np.ndarray, np.ndarray]:
    month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    bounds = np.empty((len(EXPECTED_MONTHS), 2), dtype=np.float64)
    cursor = 0.0
    for index, month in enumerate(EXPECTED_MONTHS):
        month_number = int(month[-2:])
        bounds[index] = (cursor, cursor + month_lengths[month_number - 1])
        cursor = bounds[index, 1]
    return bounds.mean(axis=1), bounds


def create_output(path: Path) -> netCDF4.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    dataset = netCDF4.Dataset(partial, "w", format="NETCDF4")
    dataset.createDimension("time", len(EXPECTED_MONTHS))
    dataset.createDimension("lat", 360)
    dataset.createDimension("lon", 720)
    dataset.createDimension("nb", 2)

    time = dataset.createVariable("time", "f8", ("time",))
    latitude = dataset.createVariable("lat", "f8", ("lat",))
    longitude = dataset.createVariable("lon", "f8", ("lon",))
    time_bounds = dataset.createVariable("time_bounds", "f8", ("time", "nb"))
    latitude_bounds = dataset.createVariable("lat_bounds", "f8", ("lat", "nb"))
    longitude_bounds = dataset.createVariable("lon_bounds", "f8", ("lon", "nb"))
    burned_area = dataset.createVariable(
        "burntArea",
        "f4",
        ("time", "lat", "lon"),
        fill_value=np.float32(1e20),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=(1, 180, 360),
    )

    times, bounds = noleap_time_axis()
    latitudes = np.arange(-89.75, 90.0, 0.5, dtype=np.float64)
    longitudes = np.arange(-179.75, 180.0, 0.5, dtype=np.float64)
    time[:] = times
    time_bounds[:] = bounds
    latitude[:] = latitudes
    longitude[:] = longitudes
    latitude_bounds[:] = np.column_stack((latitudes - 0.25, latitudes + 0.25))
    longitude_bounds[:] = np.column_stack((longitudes - 0.25, longitudes + 0.25))

    time.units = "days since 2001-01-01"
    time.calendar = "noleap"
    time.bounds = "time_bounds"
    time.standard_name = "time"
    time.axis = "T"
    latitude.units = "degrees_north"
    latitude.standard_name = "latitude"
    latitude.axis = "Y"
    latitude.bounds = "lat_bounds"
    longitude.units = "degrees_east"
    longitude.standard_name = "longitude"
    longitude.axis = "X"
    longitude.bounds = "lon_bounds"
    burned_area.units = "%"
    burned_area.standard_name = "burned area fraction"
    burned_area.long_name = "GFED5 burned area fraction"

    dataset.title = "GFED5 burned area at 0.5 degree"
    dataset.version = "5.0"
    dataset.Conventions = "CF-1.7"
    dataset.source = "GFED5 burned area v0.1, Zenodo record 7668424"
    dataset.history = (
        "Aggregated monthly Total burned area from 0.25 to 0.5 degree by area sum; "
        "converted square kilometres to percent of spherical grid-cell area"
    )
    dataset.references = "https://doi.org/10.5281/zenodo.7668424"
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expected_paths(args.input_dir)
    output = create_output(args.output)
    partial = Path(output.filepath())
    latitudes = np.arange(-89.75, 90.0, 0.5, dtype=np.float64)
    areas = cell_area_km2(latitudes, 0.5)[:, None]
    source_total = 0.0
    reconstructed_total = 0.0
    try:
        target = output.variables["burntArea"]
        for index, path in enumerate(paths):
            with netCDF4.Dataset(path) as source:
                latitude = np.asarray(source.variables["lat"][:])
                longitude = np.asarray(source.variables["lon"][:])
                if not np.allclose(latitude[[0, -1]], (-89.875, 89.875)):
                    raise ValueError(f"unexpected latitude axis in {path}")
                if not np.allclose(longitude[[0, -1]], (-179.875, 179.875)):
                    raise ValueError(f"unexpected longitude axis in {path}")
                monthly = np.asarray(source.variables["Total"][0], dtype=np.float64)
            aggregated = aggregate_total(monthly)
            percent = aggregated / areas * 100.0
            target[index] = np.ma.masked_invalid(percent.astype(np.float32))
            source_total += float(np.nansum(monthly))
            reconstructed_total += float(np.nansum(percent / 100.0 * areas))
            if index % 12 == 11:
                print(f"processed {paths[index].name}")
        output.close()
    except BaseException:
        output.close()
        partial.unlink(missing_ok=True)
        raise

    relative_error = abs(reconstructed_total - source_total) / source_total
    if relative_error > 1e-7:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"burned-area conservation check failed: {relative_error:.3e}")
    os.replace(partial, args.output)
    print(f"wrote {args.output}")
    print(f"burned-area conservation relative error: {relative_error:.3e}")


if __name__ == "__main__":
    main()
