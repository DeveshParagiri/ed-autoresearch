"""Validate a physical 20CR VPD bridge against the installed modern field."""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime import load_land_mask  # noqa: E402


def target_grid(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Bilinearly center a 1-degree corner grid on the model's cell centers."""
    normalized = (longitudes + 180.0) % 360.0 - 180.0
    order = np.argsort(normalized)
    south_to_north = values if latitudes[0] < latitudes[-1] else values[:, ::-1, :]
    south_to_north = south_to_north[:, :, order]
    latitude_centered = 0.5 * (
        south_to_north[:, :-1, :] + south_to_north[:, 1:, :]
    )
    return 0.5 * (
        latitude_centered + np.roll(latitude_centered, -1, axis=2)
    )


def main() -> int:
    sample = Path("/tmp/ed-fire-20cr-validation")
    with netCDF4.Dataset(sample / "air.2m.2001.nc") as air_dataset:
        air = np.asarray(air_dataset.variables["air"][:], dtype=np.float64)
        latitudes = np.asarray(air_dataset.variables["lat"][:])
        longitudes = np.asarray(air_dataset.variables["lon"][:])
        time = air_dataset.variables["time"]
        dates = netCDF4.num2date(time[:], units=time.units)
    with netCDF4.Dataset(sample / "rhum.2m.2001.nc") as humidity_dataset:
        humidity = np.asarray(
            humidity_dataset.variables["rhum"][:], dtype=np.float64
        )
    temperature_c = air - 273.15
    saturation = 0.6108 * np.exp(
        17.27 * temperature_c / (temperature_c + 237.3)
    )
    daily_vpd = saturation * (1.0 - np.clip(humidity, 0.0, 100.0) / 100.0)
    monthly = np.stack(
        [
            daily_vpd[[date.month == month for date in dates]].mean(axis=0)
            for month in range(1, 13)
        ]
    )
    twenty_cr = target_grid(monthly, latitudes, longitudes)

    with netCDF4.Dataset(ROOT / "autoresearch/inputs/climate.nc") as dataset:
        modern = np.asarray(
            dataset.variables["vapor_pressure_deficit_mean"][:12],
            dtype=np.float64,
        )
    land = load_land_mask()
    latitude_area = np.cos(
        np.deg2rad(-89.5 + np.arange(180, dtype=np.float64))
    )[:, None]
    weights = np.broadcast_to(latitude_area, land.shape)[land]

    observed = modern[:, land].reshape(-1)
    candidate = twenty_cr[:, land].reshape(-1)
    repeated_weights = np.tile(weights, 12)
    weight_sum = repeated_weights.sum()

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(values * repeated_weights) / weight_sum)

    mean_modern = weighted_mean(observed)
    mean_candidate = weighted_mean(candidate)
    centered_modern = observed - mean_modern
    centered_candidate = candidate - mean_candidate
    correlation = float(
        np.sum(repeated_weights * centered_modern * centered_candidate)
        / np.sqrt(
            np.sum(repeated_weights * centered_modern**2)
            * np.sum(repeated_weights * centered_candidate**2)
        )
    )
    ratio = mean_modern / (mean_candidate + 1e-12)
    scaled = candidate * ratio
    rmse = np.sqrt(weighted_mean((candidate - observed) ** 2))
    scaled_rmse = np.sqrt(weighted_mean((scaled - observed) ** 2))
    print(
        f"20CR_mean={mean_candidate:.6f} TerraClimate_mean={mean_modern:.6f} "
        f"ratio={ratio:.6f} weighted_r={correlation:.6f} "
        f"rmse={rmse:.6f} scaled_rmse={scaled_rmse:.6f}",
        flush=True,
    )
    for month in range(12):
        month_modern = modern[month, land]
        month_candidate = twenty_cr[month, land]
        print(
            f"month={month + 1:02d} "
            f"r={np.corrcoef(month_modern, month_candidate)[0, 1]:.6f} "
            f"mean_20cr={np.average(month_candidate, weights=weights):.6f} "
            f"mean_modern={np.average(month_modern, weights=weights):.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
