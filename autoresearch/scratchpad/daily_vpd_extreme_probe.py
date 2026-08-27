"""Test whether daily VPD extremes add information beyond monthly mean VPD."""

from __future__ import annotations

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
    sample = Path("/tmp/ed-fire-20cr-validation")
    with netCDF4.Dataset(sample / "air.2m.2001.nc") as dataset:
        air = np.asarray(dataset.variables["air"][:], dtype=np.float64)
        latitudes = np.asarray(dataset.variables["lat"][:])
        longitudes = np.asarray(dataset.variables["lon"][:])
        time = dataset.variables["time"]
        dates = netCDF4.num2date(time[:], units=time.units)
    with netCDF4.Dataset(sample / "rhum.2m.2001.nc") as dataset:
        humidity = np.asarray(dataset.variables["rhum"][:], dtype=np.float64)

    temperature = air - 273.15
    saturation = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    daily = saturation * (1.0 - np.clip(humidity, 0.0, 100.0) / 100.0)
    summaries: dict[str, np.ndarray] = {}
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
            summaries.setdefault(name, []).append(values)
    summaries = {
        name: target_grid(np.stack(values), latitudes, longitudes)
        for name, values in summaries.items()
    }

    model = load_model()
    requested = tuple(dict.fromkeys(model.INPUTS + ("vapor_pressure_deficit_mean",)))
    data = load_inputs(requested)
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))[:12]
    with netCDF4.Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:12])
    observed = reference.reshape(12, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0

    land = load_land_mask()
    rows, cols = np.where(land)
    month = np.tile(np.arange(12), rows.size)
    row = np.repeat(rows, 12)
    col = np.repeat(cols, 12)
    incumbent = prediction[month, row, col].astype(np.float64)
    target = np.log((observed[month, row, col] + 1e-4) / (incumbent + 1e-4))
    monthly_vpd = np.asarray(data["vapor_pressure_deficit_mean"][:12])[month, row, col]
    area = np.cos(np.deg2rad(-89.5 + rows.astype(np.float64)))
    weights = np.repeat(area, 12) * (observed[month, row, col] + 2e-4)
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
