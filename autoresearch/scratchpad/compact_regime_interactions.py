"""Distil ML-ranked seasonal interactions into compact smooth mechanisms.

This diagnostic contains only globally shared equations of local observable
state. Temperature, moisture, vegetation, and land-use regimes are overlapping
sigmoid memberships, never labels, coordinates, boxes, or if/else dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-value, -40.0, 40.0)))


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> float:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} "
        f"bias={score['bias_score']:.4f} rmse={score['rmse_score']:.4f} "
        f"seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}",
        flush=True,
    )
    return float(score["overall_score"])


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_annual = incumbent_cycle.sum(axis=0)
    incumbent_alloc = incumbent_cycle / (incumbent_annual[None, ...] + 1e-12)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    observed_cycle = observed.reshape(16, 12, 180, 360).mean(axis=0)
    observed_annual = observed_cycle.sum(axis=0)
    observed_alloc = observed_cycle / (observed_annual[None, ...] + 1e-12)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), cells.size)
    rows, cols = np.repeat(cell_rows, 12), np.repeat(cell_cols, 12)
    cycles = {
        name: np.asarray(values, dtype=np.float64)
        .reshape(16, 12, 180, 360)
        .mean(axis=0)
        for name, values in data.items()
    }
    means = {name: values.mean(axis=0) for name, values in cycles.items()}
    anomalies = {
        name: np.clip(
            (values - means[name][None, ...])
            / (values.std(axis=0)[None, ...] + 1e-6),
            -4.0,
            4.0,
        )
        for name, values in cycles.items()
    }

    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    current = incumbent_alloc[months, rows, cols]
    add("log_incumbent", np.log(current + 1e-6))
    add("sqrt_incumbent", np.sqrt(current))
    for threshold in (0.03, 0.07, 0.13, 0.22):
        add(f"incumbent_above_{threshold:.2f}", np.maximum(current - threshold, 0.0))

    angle = 2.0 * np.pi * months / 12.0
    harmonics: dict[str, np.ndarray] = {}
    for harmonic in (1, 2, 3):
        for label, function in (("sin", np.sin), ("cos", np.cos)):
            name = f"{label}{harmonic}"
            harmonics[name] = function(harmonic * angle)
            add(name, harmonics[name])

    dynamic_names = (
        "air_temperature",
        "monthly_precipitation",
        "dryness",
        "gpp",
        "leaf_area_index",
        "lightning_flash_rate",
    )
    current_anomaly: dict[str, np.ndarray] = {}
    previous_anomaly: dict[str, np.ndarray] = {}
    for name in dynamic_names:
        now = anomalies[name][months, rows, cols]
        previous = np.roll(anomalies[name], 1, axis=0)[months, rows, cols]
        current_anomaly[name] = now
        previous_anomaly[name] = previous
        for lag, value in (("current", now), ("previous", previous)):
            add(f"{name}_{lag}_anomaly", value)
            add(f"{name}_{lag}_positive", np.maximum(value, 0.0))
            add(f"{name}_{lag}_negative", np.minimum(value, 0.0))

    repeated_mean = {
        name: np.repeat(values[cell_rows, cell_cols], 12)
        for name, values in means.items()
    }
    mean_temperature = repeated_mean["air_temperature"]
    annual_rain = repeated_mean["annual_precipitation"]
    temperature_regimes = {
        "cold": sigmoid((5.0 - mean_temperature) / 3.0),
        "cool": sigmoid((mean_temperature - 0.0) / 3.0)
        * sigmoid((18.0 - mean_temperature) / 3.0),
        "warm": sigmoid((mean_temperature - 14.0) / 3.0)
        * sigmoid((28.0 - mean_temperature) / 3.0),
        "hot": sigmoid((mean_temperature - 24.0) / 3.0),
    }
    moisture_regimes = {
        "dry_climate": sigmoid((650.0 - annual_rain) / 180.0),
        "seasonal_climate": sigmoid((annual_rain - 400.0) / 150.0)
        * sigmoid((1700.0 - annual_rain) / 250.0),
        "humid_climate": sigmoid((annual_rain - 1300.0) / 250.0),
    }

    # EBM-ranked physics: thermal regime conditions weather response, ignition,
    # fuel phenology, and the phase of the locally recurring fire window.
    for regime_name, regime in temperature_regimes.items():
        add(f"thermal_{regime_name}", regime)
        for driver in dynamic_names:
            add(
                f"thermal_{regime_name}_x_{driver}_current",
                regime * current_anomaly[driver],
            )
            add(
                f"thermal_{regime_name}_x_{driver}_previous",
                regime * previous_anomaly[driver],
            )
        add(f"thermal_{regime_name}_x_sin1", regime * harmonics["sin1"])
        add(f"thermal_{regime_name}_x_cos1", regime * harmonics["cos1"])

    # Moisture regime changes whether rain grows fuel or quenches combustion.
    for regime_name, regime in moisture_regimes.items():
        add(f"moisture_{regime_name}", regime)
        for driver in (
            "monthly_precipitation",
            "dryness",
            "gpp",
            "leaf_area_index",
            "lightning_flash_rate",
        ):
            add(
                f"moisture_{regime_name}_x_{driver}_current",
                regime * current_anomaly[driver],
            )
            add(
                f"moisture_{regime_name}_x_{driver}_previous",
                regime * previous_anomaly[driver],
            )

    # Continuous land-state memberships condition ignition and fuel continuity.
    land_regimes = {
        "cropland": repeated_mean["luh2_cropland_fraction"],
        "pasture": repeated_mean["luh2_pasture_fraction"],
        "rangeland": repeated_mean["luh2_rangeland_fraction"],
        "natural": repeated_mean["natural_vegetation_fraction"],
    }
    for regime_name, regime in land_regimes.items():
        add(f"land_{regime_name}", regime)
        for driver in (
            "air_temperature",
            "dryness",
            "gpp",
            "leaf_area_index",
            "lightning_flash_rate",
        ):
            add(
                f"land_{regime_name}_x_{driver}_current",
                regime * current_anomaly[driver],
            )
            add(
                f"land_{regime_name}_x_{driver}_previous",
                regime * previous_anomaly[driver],
            )

    # Weather matters most where the mechanistic incumbent already exposes a
    # burnable window. Soft memberships avoid threshold branches.
    for threshold in (0.05, 0.12):
        opportunity = sigmoid((current - threshold) / 0.025)
        for driver in dynamic_names:
            add(
                f"opportunity_{threshold:.2f}_x_{driver}_current",
                opportunity * current_anomaly[driver],
            )
            add(
                f"opportunity_{threshold:.2f}_x_{driver}_previous",
                opportunity * previous_anomaly[driver],
            )

    x = np.column_stack(columns).astype(np.float64)
    y = observed_alloc[months, rows, cols].astype(np.float64)
    weights = np.repeat(
        observed_annual[cell_rows, cell_cols] + float(observed_annual.mean()) * 0.01,
        12,
    )
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    def candidate_from(values: np.ndarray, strength: float) -> np.ndarray:
        learned = np.zeros((12, 180, 360), dtype=np.float64)
        learned[months, rows, cols] = np.maximum(values, 1e-12)
        learned /= learned.sum(axis=0, keepdims=True) + 1e-12
        blended = np.power(incumbent_alloc + 1e-12, 1.0 - strength)
        blended *= np.power(learned + 1e-12, strength)
        blended /= blended.sum(axis=0, keepdims=True) + 1e-12
        prediction = incumbent_annual[None, ...] * blended
        return np.tile(prediction, (16, 1, 1)).astype(np.float32)

    rng = np.random.default_rng(419)
    cell_folds = rng.integers(0, 3, size=cells.size)
    folds = np.repeat(cell_folds, 12)
    for alpha in (0.03, 0.01, 0.003, 0.001):
        out_of_fold = np.zeros_like(y)
        fold_coefficients: list[np.ndarray] = []
        for fold in range(3):
            train = folds != fold
            held = ~train
            regressor = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-8)
            regressor.fit(standardized[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = regressor.predict(standardized[held])
            fold_coefficients.append(regressor.coef_)
        correlations = np.corrcoef(np.asarray(fold_coefficients))
        minimum_correlation = correlations[np.triu_indices(3, 1)].min()
        print(
            f"alpha={alpha} fold coefficient correlation min={minimum_correlation:.4f}",
            flush=True,
        )
        for strength in (0.5, 0.75, 1.0):
            report(
                evaluator,
                f"three-fold OOF alpha={alpha} strength={strength}",
                candidate_from(out_of_fold, strength),
            )

    regressor = PoissonRegressor(alpha=0.003, max_iter=2000, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    learned = regressor.predict(standardized)
    for strength in (0.5, 0.75, 1.0):
        report(
            evaluator,
            f"in-sample alpha=0.003 strength={strength}",
            candidate_from(learned, strength),
        )
    print("top standardized coefficients", flush=True)
    for index in np.argsort(np.abs(regressor.coef_))[::-1][:60]:
        print(f"{names[index]}\t{regressor.coef_[index]:+.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
