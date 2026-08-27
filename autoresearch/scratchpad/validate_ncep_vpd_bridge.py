"""Validate NCEP daily VPD duration as the 2016 continuation of 20CR."""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.validate_20cr_vpd import target_grid  # noqa: E402
from scripts.runtime import load_land_mask  # noqa: E402


def bilinear_target(
    values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Interpolate a regular global grid to 1-degree cell centers."""
    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        values = values[:, ::-1, :]
    normalized = (longitudes + 180.0) % 360.0 - 180.0
    order = np.argsort(normalized)
    normalized = normalized[order]
    values = values[:, :, order]
    extended_lon = np.concatenate(
        ([normalized[-1] - 360.0], normalized, [normalized[0] + 360.0])
    )
    values = np.concatenate((values[:, :, -1:], values, values[:, :, :1]), axis=2)

    target_lat = -89.5 + np.arange(180, dtype=np.float64)
    target_lon = -179.5 + np.arange(360, dtype=np.float64)
    lat_index = np.clip(np.searchsorted(latitudes, target_lat) - 1, 0, len(latitudes) - 2)
    lat_weight = (target_lat - latitudes[lat_index]) / (
        latitudes[lat_index + 1] - latitudes[lat_index]
    )
    latitude_values = (
        values[:, lat_index, :] * (1.0 - lat_weight[None, :, None])
        + values[:, lat_index + 1, :] * lat_weight[None, :, None]
    )
    lon_index = np.clip(
        np.searchsorted(extended_lon, target_lon) - 1, 0, len(extended_lon) - 2
    )
    lon_weight = (target_lon - extended_lon[lon_index]) / (
        extended_lon[lon_index + 1] - extended_lon[lon_index]
    )
    return (
        latitude_values[:, :, lon_index] * (1.0 - lon_weight[None, None, :])
        + latitude_values[:, :, lon_index + 1] * lon_weight[None, None, :]
    )


def monthly_fields(prefix: str, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = Path("/tmp/ed-fire-20cr-validation")
    if prefix == "ncep":
        air_path = sample / f"ncep-air.sig995.{year}.nc"
        humidity_path = sample / f"ncep-rhum.sig995.{year}.nc"
    else:
        air_path = sample / f"air.2m.{year}.nc"
        humidity_path = sample / f"rhum.2m.{year}.nc"
    with netCDF4.Dataset(air_path) as dataset:
        air = np.asarray(dataset.variables["air"][:], dtype=np.float64)
        latitudes = np.asarray(dataset.variables["lat"][:])
        longitudes = np.asarray(dataset.variables["lon"][:])
        time = dataset.variables["time"]
        dates = netCDF4.num2date(time[:], units=time.units)
    with netCDF4.Dataset(humidity_path) as dataset:
        humidity = np.asarray(dataset.variables["rhum"][:], dtype=np.float64)
    temperature = air - 273.15
    saturation = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    vpd = saturation * (1.0 - np.clip(humidity, 0.0, 100.0) / 100.0)
    mean = np.stack(
        [vpd[[date.month == month for date in dates]].mean(axis=0) for month in range(1, 13)]
    )
    duration = np.stack(
        [(vpd[[date.month == month for date in dates]] > 1.0).mean(axis=0) for month in range(1, 13)]
    )
    return mean, duration, latitudes, longitudes


def main() -> int:
    mean_20cr, duration_20cr, lat_20cr, lon_20cr = monthly_fields("20cr", 2001)
    mean_ncep, duration_ncep, lat_ncep, lon_ncep = monthly_fields("ncep", 2001)
    mean_20cr = target_grid(mean_20cr, lat_20cr, lon_20cr)
    duration_20cr = target_grid(duration_20cr, lat_20cr, lon_20cr)
    mean_ncep = bilinear_target(mean_ncep, lat_ncep, lon_ncep)
    duration_ncep = bilinear_target(duration_ncep, lat_ncep, lon_ncep)

    land = load_land_mask()
    area = np.cos(np.deg2rad(-89.5 + np.arange(180)))[:, None]
    weights = np.tile(np.broadcast_to(area, land.shape)[land], 12)
    for name, left, right in (
        ("mean_vpd", mean_20cr, mean_ncep),
        ("fraction_gt_1kpa", duration_20cr, duration_ncep),
    ):
        left = left[:, land].reshape(-1)
        right = right[:, land].reshape(-1)
        left_centered = left - np.average(left, weights=weights)
        right_centered = right - np.average(right, weights=weights)
        correlation = np.sum(weights * left_centered * right_centered) / np.sqrt(
            np.sum(weights * left_centered**2) * np.sum(weights * right_centered**2)
        )
        rmse = np.sqrt(np.average((left - right) ** 2, weights=weights))
        print(
            f"overlap_2001 {name} weighted_r={correlation:.6f} rmse={rmse:.6f} "
            f"mean_20cr={np.average(left, weights=weights):.6f} "
            f"mean_ncep={np.average(right, weights=weights):.6f}",
            flush=True,
        )

    mean_2016, duration_2016, lat_2016, lon_2016 = monthly_fields("ncep", 2016)
    mean_2016 = bilinear_target(mean_2016, lat_2016, lon_2016)
    duration_2016 = bilinear_target(duration_2016, lat_2016, lon_2016)
    print(
        f"ncep_2016 finite={np.isfinite(duration_2016).mean():.6f} "
        f"mean_vpd={np.average(mean_2016[:, land].reshape(-1), weights=weights):.6f} "
        f"mean_fraction_gt_1kpa="
        f"{np.average(duration_2016[:, land].reshape(-1), weights=weights):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
