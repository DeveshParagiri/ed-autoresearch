"""Validate an 1850-capable route for daily precipitation event structure."""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.validate_20cr_vpd import target_grid  # noqa: E402
from autoresearch.scratchpad.validate_ncep_vpd_bridge import bilinear_target  # noqa: E402
from scripts.runtime import load_inputs, load_land_mask  # noqa: E402


SAMPLE = Path("/tmp/ed-fire-20cr-validation")


def monthly_events(
    path: Path, threshold_mm_day: float, ncep: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        precipitation = np.asarray(dataset.variables["prate"][:], dtype=np.float64)
        latitudes = np.asarray(dataset.variables["lat"][:])
        longitudes = np.asarray(dataset.variables["lon"][:])
        time = dataset.variables["time"]
        dates = netCDF4.num2date(time[:], units=time.units)
    millimetres = precipitation * 86_400.0
    wet_fraction: list[np.ndarray] = []
    maximum_dry_spell: list[np.ndarray] = []
    for month in range(1, 13):
        selected = millimetres[[date.month == month for date in dates]]
        wet = selected >= threshold_mm_day
        wet_fraction.append(wet.mean(axis=0))
        current = np.zeros(selected.shape[1:], dtype=np.float64)
        longest = np.zeros_like(current)
        for day_is_wet in wet:
            current = np.where(day_is_wet, 0.0, current + 1.0)
            longest = np.maximum(longest, current)
        maximum_dry_spell.append(longest)
    wet_array = np.stack(wet_fraction)
    dry_array = np.stack(maximum_dry_spell)
    interpolate = bilinear_target if ncep else target_grid
    return (
        interpolate(wet_array, latitudes, longitudes),
        interpolate(dry_array, latitudes, longitudes),
    )


def weighted_comparison(
    left: np.ndarray, right: np.ndarray, land: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    left = left[:, land].reshape(-1)
    right = right[:, land].reshape(-1)
    left_mean = np.average(left, weights=weights)
    right_mean = np.average(right, weights=weights)
    covariance = np.average(
        (left - left_mean) * (right - right_mean), weights=weights
    )
    correlation = covariance / np.sqrt(
        np.average((left - left_mean) ** 2, weights=weights)
        * np.average((right - right_mean) ** 2, weights=weights)
    )
    rmse = np.sqrt(np.average((left - right) ** 2, weights=weights))
    return float(correlation), float(rmse), float(left_mean), float(right_mean)


def main() -> int:
    installed = load_inputs(("wet_day_fraction", "maximum_consecutive_dry_days"))
    land = load_land_mask()
    area = np.cos(np.deg2rad(-89.5 + np.arange(180)))[:, None]
    weights = np.tile(np.broadcast_to(area, land.shape)[land], 12)
    for threshold in (0.5, 1.0, 2.0):
        wet_20cr, dry_20cr = monthly_events(
            SAMPLE / "prate.2001.nc", threshold
        )
        for name, candidate, reference in (
            ("wet_day_fraction", wet_20cr, installed["wet_day_fraction"][:12]),
            (
                "maximum_consecutive_dry_days",
                dry_20cr,
                installed["maximum_consecutive_dry_days"][:12],
            ),
        ):
            correlation, rmse, candidate_mean, reference_mean = weighted_comparison(
                candidate, reference, land, weights
            )
            print(
                f"20cr_vs_cpc_2001 threshold={threshold:.1f} {name} "
                f"weighted_r={correlation:.6f} rmse={rmse:.6f} "
                f"mean_20cr={candidate_mean:.6f} mean_cpc={reference_mean:.6f}",
                flush=True,
            )

    wet_20cr, dry_20cr = monthly_events(SAMPLE / "prate.2001.nc", 1.0)
    wet_ncep, dry_ncep = monthly_events(
        SAMPLE / "ncep-prate.2001.nc", 1.0, ncep=True
    )
    for name, left, right in (
        ("wet_day_fraction", wet_20cr, wet_ncep),
        ("maximum_consecutive_dry_days", dry_20cr, dry_ncep),
    ):
        correlation, rmse, mean_20cr, mean_ncep = weighted_comparison(
            left, right, land, weights
        )
        print(
            f"20cr_vs_ncep_2001 {name} weighted_r={correlation:.6f} "
            f"rmse={rmse:.6f} mean_20cr={mean_20cr:.6f} mean_ncep={mean_ncep:.6f}",
            flush=True,
        )

    wet_1850, dry_1850 = monthly_events(SAMPLE / "prate.1850.nc", 1.0)
    wet_2016, dry_2016 = monthly_events(
        SAMPLE / "ncep-prate.2016.nc", 1.0, ncep=True
    )
    print(
        f"coverage_1850 wet_finite={np.isfinite(wet_1850).mean():.6f} "
        f"dry_finite={np.isfinite(dry_1850).mean():.6f}",
        flush=True,
    )
    print(
        f"coverage_2016 wet_finite={np.isfinite(wet_2016).mean():.6f} "
        f"dry_finite={np.isfinite(dry_2016).mean():.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
