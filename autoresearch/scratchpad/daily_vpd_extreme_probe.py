"""Test whether daily VPD extremes add information beyond monthly mean VPD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.validate_20cr_vpd import target_grid  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


def weighted_residual(values: np.ndarray, controls: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(
        controls * root[:, None], values * root, rcond=None
    )[0]
    return values - controls @ coefficients


def weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    left = left - np.sum(left * weights) / total
    right = right - np.sum(right * weights) / total
    return float(
        np.sum(weights * left * right)
        / np.sqrt(np.sum(weights * left**2) * np.sum(weights * right**2) + 1e-30)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2001)
    parser.add_argument("--end", type=int, default=2001)
    args = parser.parse_args()
    sample = Path("/tmp/ed-fire-20cr-validation")
    collected: dict[str, list[np.ndarray]] = {}
    for year in range(args.start, args.end + 1):
        with netCDF4.Dataset(sample / f"air.2m.{year}.nc") as dataset:
            air = np.asarray(dataset.variables["air"][:], dtype=np.float64)
            latitudes = np.asarray(dataset.variables["lat"][:])
            longitudes = np.asarray(dataset.variables["lon"][:])
            time = dataset.variables["time"]
            dates = netCDF4.num2date(time[:], units=time.units)
        with netCDF4.Dataset(sample / f"rhum.2m.{year}.nc") as dataset:
            humidity = np.asarray(dataset.variables["rhum"][:], dtype=np.float64)

        temperature = air - 273.15
        saturation = 0.6108 * np.exp(
            17.27 * temperature / (temperature + 237.3)
        )
        daily = saturation * (1.0 - np.clip(humidity, 0.0, 100.0) / 100.0)
        for month in range(1, 13):
            selected = daily[[date.month == month for date in dates]]
            fields = {
                "mean": selected.mean(axis=0),
                "p90": np.quantile(selected, 0.90, axis=0),
                "p95": np.quantile(selected, 0.95, axis=0),
                "fraction_gt_1kpa": (selected > 1.0).mean(axis=0),
                "fraction_gt_2kpa": (selected > 2.0).mean(axis=0),
                "fraction_gt_3kpa": (selected > 3.0).mean(axis=0),
            }
            for name, values in fields.items():
                collected.setdefault(name, []).append(
                    target_grid(values[None, ...], latitudes, longitudes)[0]
                )
        print(f"processed year={year}", flush=True)
    summaries = {name: np.stack(values) for name, values in collected.items()}

    model = load_model()
    requested = tuple(dict.fromkeys(model.INPUTS + ("vapor_pressure_deficit_mean",)))
    data = load_inputs(requested)
    months_count = 12 * (args.end - args.start + 1)
    start_index = 12 * (args.start - 2001)
    stop_index = start_index + months_count
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))[
        start_index:stop_index
    ]
    with netCDF4.Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][start_index:stop_index])
    observed = reference.reshape(months_count, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0

    land = load_land_mask()
    rows, cols = np.where(land)
    month = np.tile(np.arange(months_count), rows.size)
    row = np.repeat(rows, months_count)
    col = np.repeat(cols, months_count)
    incumbent = prediction[month, row, col].astype(np.float64)
    target = np.log((observed[month, row, col] + 1e-4) / (incumbent + 1e-4))
    monthly_vpd = np.asarray(
        data["vapor_pressure_deficit_mean"][start_index:stop_index]
    )[month, row, col]
    area = np.cos(np.deg2rad(-89.5 + rows.astype(np.float64)))
    weights = np.repeat(area, months_count) * (observed[month, row, col] + 2e-4)
    controls = np.column_stack(
        (np.ones_like(incumbent), np.log(incumbent + 1e-4), monthly_vpd)
    )
    target_residual = weighted_residual(target, controls, weights)

    print("feature\tr_raw\tr_partial_given_incumbent_and_monthly_mean", flush=True)
    for name, values in summaries.items():
        feature = values[month, row, col]
        feature_residual = weighted_residual(feature, controls, weights)
        print(
            f"{name}\t{weighted_correlation(feature, target, weights):+.6f}\t"
            f"{weighted_correlation(feature_residual, target_residual, weights):+.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
