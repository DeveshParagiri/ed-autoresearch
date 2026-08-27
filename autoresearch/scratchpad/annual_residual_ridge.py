"""Fit a coupled-valid pointwise correction to annual fire propensity.

Diagnostic only.  The response is the clipped log residual between GFED5 and
the incumbent annual map.  Predictors are globally shared physical summaries;
coordinates, regions, cell identities, and geographic masks are excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import Ridge

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


INVALID_INPUTS = {
    "wind_speed_mean",
    "vapor_pressure_deficit_mean",
    "maximum_consecutive_dry_days",
    "wet_day_fraction",
    "population_density",
}
INVALID_COMPONENTS = {"neighbour", "gust", "vpd"}


def input_names() -> list[str]:
    names: list[str] = []
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            names.extend(
                name for name, variable in dataset.variables.items()
                if variable.dimensions == ("time", "lat", "lon")
            )
    return [name for name in names if name not in INVALID_INPUTS]


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> float:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} bias={score['bias_score']:.4f} "
        f"rmse={score['rmse_score']:.4f} seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}", flush=True
    )
    return float(score["overall_score"])


def main() -> int:
    names = input_names()
    data = load_inputs(names)
    model = load_model()
    model_data = load_inputs(model.INPUTS)
    baseline_params = dict(model.PARAMS)
    if "--stack" not in sys.argv:
        baseline_params.update(
            annual_residual_w=0.0,
            seasonal_residual_w=0.0,
            allocation_glm_w=0.0,
            annual_vpd_w=0.0,
        )
    valid_components = tuple(
        name for name in model.COMPONENTS if name not in INVALID_COMPONENTS
    )
    incumbent = validate_prediction(
        model.predict(model_data, baseline_params, valid_components)
    )
    cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    annual = cycle.sum(axis=0)
    shape = cycle / (annual[None, ...] + 1e-12)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_annual = obs.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360
    current = annual[rows, cols].astype(np.float64)
    observed = obs_annual[rows, cols].astype(np.float64)

    feature_names: list[str] = []
    columns: list[np.ndarray] = []
    summaries: dict[str, dict[str, np.ndarray]] = {}

    def add(name: str, value: np.ndarray) -> None:
        feature_names.append(name)
        columns.append(np.asarray(value, dtype=np.float64))

    log_current = np.log10(current + 1e-6)
    add("log10_incumbent", log_current)
    for threshold in (-5.0, -4.0, -3.0, -2.0, -1.0):
        add(f"log10_incumbent_above_{threshold:+.1f}", np.maximum(log_current - threshold, 0.0))

    for name in names:
        climatology = np.asarray(data[name], dtype=np.float64).reshape(16, 12, 180, 360).mean(axis=0)
        raw = {
            "mean": climatology.mean(axis=0)[rows, cols],
            "std": climatology.std(axis=0)[rows, cols],
            "p10": np.quantile(climatology, 0.10, axis=0)[rows, cols],
            "p90": np.quantile(climatology, 0.90, axis=0)[rows, cols],
        }
        if np.max(np.abs(raw["p90"] - raw["p10"])) < 1e-8:
            raw = {"mean": raw["mean"]}
        summaries[name] = {}
        for statistic, values in raw.items():
            center = np.median(values)
            scale = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            z = np.clip((values - center) / (scale + 1e-8), -4.0, 4.0)
            summaries[name][statistic] = z
            add(f"{name}:{statistic}:z", z)
            add(f"{name}:{statistic}:positive", np.maximum(z, 0.0))
            add(f"{name}:{statistic}:negative", np.minimum(z, 0.0))

    # A small set of compound controls suggested independently by the annual
    # tree and Poisson screens.
    for left, left_stat, right, right_stat in (
        ("monthly_precipitation", "std", "air_temperature", "p10"),
        ("monthly_precipitation", "std", "aboveground_biomass", "p10"),
        ("monthly_precipitation", "mean", "gpp", "mean"),
        ("monthly_precipitation", "p10", "dryness", "mean"),
        ("air_temperature", "std", "gpp", "mean"),
        ("leaf_area_index", "std", "dryness", "mean"),
        ("lightning_flash_rate", "mean", "aboveground_biomass", "mean"),
        ("luh2_cropland_fraction", "mean", "luh2_secondary_fraction", "mean"),
        ("luh2_rangeland_fraction", "mean", "aboveground_biomass", "mean"),
        ("soil_carbon", "mean", "air_temperature", "p10"),
    ):
        add(
            f"{left}:{left_stat}_x_{right}:{right_stat}",
            summaries[left][left_stat] * summaries[right][right_stat],
        )

    x = np.column_stack(columns)
    x_mean = np.average(x, axis=0, weights=observed + observed.mean() * 0.02)
    x_scale = np.sqrt(np.average(np.square(x - x_mean), axis=0, weights=observed + observed.mean() * 0.02)) + 1e-8
    xs = (x - x_mean) / x_scale
    target = np.clip(np.log((observed + 1e-5) / (current + 1e-5)), -5.0, 5.0)
    weight = observed + observed.mean() * 0.02
    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    print(f"cells={len(cells)} features={x.shape[1]}", flush=True)

    rng = np.random.default_rng(353)
    folds = rng.integers(0, 5, size=len(cells))
    oof_residual = np.zeros(len(cells), dtype=np.float64)
    fold_coefficients: list[np.ndarray] = []
    for fold in range(5):
        train = folds != fold
        held = ~train
        reg = Ridge(alpha=0.3)
        reg.fit(xs[train], target[train], sample_weight=weight[train])
        oof_residual[held] = np.clip(reg.predict(xs[held]), -5.0, 5.0)
        fold_coefficients.append(reg.coef_)
    correlations = np.corrcoef(np.asarray(fold_coefficients))
    print(
        "fold coefficient correlation "
        f"min={correlations[np.triu_indices(5, 1)].min():.4f} "
        f"mean={correlations[np.triu_indices(5, 1)].mean():.4f}",
        flush=True,
    )
    for strength in (0.50, 0.75, 1.0):
        corrected = np.zeros((180, 360), dtype=np.float32)
        corrected[rows, cols] = current * np.exp(strength * oof_residual)
        candidate = np.tile(corrected[None, ...] * shape, (16, 1, 1)).astype(np.float32)
        report(evaluator, f"five-fold OOF strength={strength}", candidate)

    for alpha in (100.0, 30.0, 10.0, 3.0, 1.0, 0.3):
        regressor = Ridge(alpha=alpha)
        regressor.fit(xs, target, sample_weight=weight)
        residual = np.clip(regressor.predict(xs), -5.0, 5.0)
        for strength in (0.25, 0.50, 0.75, 1.0):
            corrected = np.zeros((180, 360), dtype=np.float32)
            corrected[rows, cols] = current * np.exp(strength * residual)
            candidate = np.tile(corrected[None, ...] * shape, (16, 1, 1)).astype(np.float32)
            report(evaluator, f"ridge alpha={alpha} strength={strength}", candidate)

    # Nested sparse refits test whether the spatial correction can be expressed
    # as a compact mathematical GAM rather than a 181-coefficient diagnostic.
    best_sparse = (-np.inf, 0)
    for feature_count in (20, 40, 60, 90, 120):
        sparse_oof = np.zeros(len(cells), dtype=np.float64)
        for fold in range(5):
            train = folds != fold
            held = ~train
            selector = Ridge(alpha=0.3)
            selector.fit(xs[train], target[train], sample_weight=weight[train])
            selected = np.argsort(np.abs(selector.coef_))[-feature_count:]
            reduced = Ridge(alpha=0.3)
            reduced.fit(
                xs[train][:, selected], target[train], sample_weight=weight[train]
            )
            sparse_oof[held] = np.clip(
                reduced.predict(xs[held][:, selected]), -5.0, 5.0
            )
        corrected = np.zeros((180, 360), dtype=np.float32)
        corrected[rows, cols] = current * np.exp(sparse_oof)
        candidate = np.tile(corrected[None, ...] * shape, (16, 1, 1)).astype(np.float32)
        score = report(evaluator, f"nested sparse OOF features={feature_count}", candidate)
        if score > best_sparse[0]:
            best_sparse = (score, feature_count)

    feature_count = best_sparse[1]
    selected = np.argsort(np.abs(regressor.coef_))[-feature_count:]
    sparse = Ridge(alpha=0.3)
    sparse.fit(xs[:, selected], target, sample_weight=weight)
    sparse_raw = sparse.coef_ / x_scale[selected]
    sparse_intercept = float(
        sparse.intercept_ - np.dot(sparse_raw, x_mean[selected])
    )
    sparse_residual = np.clip(sparse.predict(xs[:, selected]), -5.0, 5.0)
    for strength in (0.75, 1.0, 1.10):
        corrected = np.zeros((180, 360), dtype=np.float32)
        corrected[rows, cols] = current * np.exp(strength * sparse_residual)
        candidate = np.tile(corrected[None, ...] * shape, (16, 1, 1)).astype(np.float32)
        report(
            evaluator,
            f"selected sparse features={feature_count} strength={strength}",
            candidate,
        )
    print(f"SPARSE_FEATURE_COUNT={feature_count}", flush=True)
    print(f"SPARSE_RAW_INTERCEPT={sparse_intercept!r}", flush=True)
    print(
        "SPARSE_RAW_TERMS="
        + repr(
            tuple(
                (feature_names[int(index)], float(coefficient))
                for index, coefficient in zip(selected, sparse_raw)
            )
        ),
        flush=True,
    )

    coefficients = regressor.coef_ / x_scale
    raw_intercept = float(regressor.intercept_ - np.dot(coefficients, x_mean))
    order = np.argsort(np.abs(coefficients))[::-1]
    print("top multiplicative residual coefficients", flush=True)
    for index in order[:120]:
        print(f"{feature_names[index]}\t{coefficients[index]:+.9f}", flush=True)
    print(f"RIDGE_RAW_INTERCEPT={raw_intercept!r}", flush=True)
    print("RIDGE_RAW_COEFFICIENTS=" + repr(tuple(float(v) for v in coefficients)), flush=True)
    print("RIDGE_FEATURE_NAMES=" + repr(tuple(feature_names)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
