"""Interpretable global annual burned-area propensity diagnostic.

Every feature is a named continuous function of incumbent fire opportunity or
an annual/climatological physical input.  The equation has no coordinates,
regions, cell identifiers, or geographic masks.  It is diagnostic only; useful
response families are simplified before entering model.py.
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
    INPUTS_DIR,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


def input_names() -> list[str]:
    names: list[str] = []
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            names.extend(
                name
                for name, variable in dataset.variables.items()
                if variable.dimensions == ("time", "lat", "lon")
            )
    return names


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
    names = list(model.INPUTS) if "--model-inputs" in sys.argv else input_names()
    data = load_inputs(names)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_annual = incumbent_cycle.sum(axis=0)
    incumbent_shape = incumbent_cycle / (incumbent_annual[None, ...] + 1e-12)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0)
    obs_annual = obs_cycle.sum(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360
    current = incumbent_annual[rows, cols].astype(np.float64)

    feature_names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        feature_names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    log_current = np.log10(current + 1e-7)
    add("log10_incumbent_annual", log_current)
    add("sqrt_incumbent_annual", np.sqrt(current))
    for threshold in (-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5):
        add(f"log_incumbent_above_{threshold:+.1f}", np.maximum(log_current - threshold, 0.0))

    summaries: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        cycle = np.asarray(data[name], dtype=np.float64).reshape(16, 12, 180, 360).mean(axis=0)
        raw = {
            "mean": cycle.mean(axis=0)[rows, cols],
            "std": cycle.std(axis=0)[rows, cols],
            "p10": np.quantile(cycle, 0.10, axis=0)[rows, cols],
            "p90": np.quantile(cycle, 0.90, axis=0)[rows, cols],
        }
        if np.max(np.abs(raw["p90"] - raw["p10"])) < 1e-8:
            raw = {"mean": raw["mean"]}
        summaries[name] = {}
        for statistic, values in raw.items():
            median = np.median(values)
            iqr = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            z = np.clip((values - median) / (iqr + 1e-8), -5.0, 5.0)
            summaries[name][statistic] = z
            add(f"{name}:{statistic}:z", z)
            add(f"{name}:{statistic}:positive", np.maximum(z, 0.0))
            add(f"{name}:{statistic}:negative", np.minimum(z, 0.0))
            add(f"{name}:{statistic}:above_1", np.maximum(z - 1.0, 0.0))
            add(f"{name}:{statistic}:below_-1", np.minimum(z + 1.0, 0.0))

    # Residual fire occurrence changes smoothly with incumbent opportunity.
    # Interacting physical summaries with broad opportunity hinges lets the
    # same global process distinguish omitted rare-fire cells from saturation.
    for name in names:
        for statistic in summaries[name]:
            z = summaries[name][statistic]
            for threshold in (-4.5, -3.5, -2.5, -1.5):
                add(
                    f"{name}:{statistic}:z_x_log_incumbent_above_{threshold:+.1f}",
                    z * np.maximum(log_current - threshold, 0.0),
                )

    # Named compound fuel-climate controls motivated by annual residuals.
    for left, left_stat, right, right_stat in (
        ("monthly_precipitation", "std", "annual_precipitation", "mean"),
        ("monthly_precipitation", "std", "air_temperature", "p10"),
        ("vapor_pressure_deficit_mean", "std", "gpp", "mean"),
        ("vapor_pressure_deficit_mean", "p10", "aboveground_biomass", "p10"),
        ("maximum_consecutive_dry_days", "std", "gpp", "mean"),
        ("wind_speed_mean", "mean", "dryness", "mean"),
        ("lightning_flash_rate", "mean", "gpp", "mean"),
        ("luh2_cropland_fraction", "mean", "population_density", "mean"),
        ("luh2_rangeland_fraction", "mean", "aboveground_biomass", "mean"),
        ("luh2_primary_fraction", "mean", "leaf_area_index", "mean"),
        ("soil_carbon", "mean", "air_temperature", "p10"),
    ):
        if left not in summaries or right not in summaries:
            continue
        add(
            f"{left}:{left_stat}:z_x_{right}:{right_stat}:z",
            summaries[left][left_stat] * summaries[right][right_stat],
        )

    x = np.column_stack(columns).astype(np.float64)
    y = obs_annual[rows, cols].astype(np.float64)
    weight = y + float(y.mean()) * 0.02
    x_mean = np.average(x, axis=0, weights=weight)
    x_scale = np.sqrt(np.average(np.square(x - x_mean), axis=0, weights=weight)) + 1e-8
    xs = (x - x_mean) / x_scale
    print(f"cells={x.shape[0]} features={x.shape[1]}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    best: tuple[float, PoissonRegressor] | None = None
    for alpha in (0.03, 0.01, 0.003, 0.001):
        regressor = PoissonRegressor(alpha=alpha, max_iter=800, tol=1e-9)
        regressor.fit(xs, y, sample_weight=weight)
        learned = np.zeros((180, 360), dtype=np.float32)
        learned[rows, cols] = np.clip(regressor.predict(xs), 0.0, 1.0)
        for blend in (0.25, 0.50, 0.75, 1.0):
            annual = (1.0 - blend) * incumbent_annual + blend * learned
            candidate = np.tile(annual[None, ...] * incumbent_shape, (16, 1, 1)).astype(np.float32)
            score = report(evaluator, f"annual GLM alpha={alpha} blend={blend}", candidate)
            if best is None or score > best[0]:
                best = (score, regressor)

    assert best is not None
    coefficients = best[1].coef_ / x_scale
    order = np.argsort(np.abs(coefficients))[::-1]
    print("top annual physical coefficients", flush=True)
    for index in order[:160]:
        print(f"{feature_names[index]}\t{coefficients[index]:+.9f}", flush=True)

    print("reduced annual named models", flush=True)
    for count in (30, 50, 80, 120, 160, 240):
        selected = order[:count]
        reduced = PoissonRegressor(alpha=0.001, max_iter=1200, tol=1e-9)
        reduced.fit(xs[:, selected], y, sample_weight=weight)
        learned = np.zeros((180, 360), dtype=np.float32)
        learned[rows, cols] = np.clip(reduced.predict(xs[:, selected]), 0.0, 1.0)
        for blend in (0.50, 0.75, 1.0):
            annual = (1.0 - blend) * incumbent_annual + blend * learned
            candidate = np.tile(annual[None, ...] * incumbent_shape, (16, 1, 1)).astype(np.float32)
            report(evaluator, f"top-{count} annual GLM blend={blend}", candidate)
        if count in (80, 120):
            raw = reduced.coef_ / x_scale[selected]
            print(f"top-{count} annual refit coefficients", flush=True)
            for index, coefficient in zip(selected, raw, strict=True):
                print(f"{feature_names[index]}\t{coefficient:+.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
