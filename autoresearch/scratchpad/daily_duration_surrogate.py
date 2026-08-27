"""Test whether coupled-valid monthly forcing can emulate daily VPD duration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.online_residual_glm import (  # noqa: E402
    load_daily_vpd_duration,
    running,
)
from scripts.runtime import load_inputs, load_land_mask  # noqa: E402


NAMES = (
    "temperature",
    "log_dryness",
    "log_monthly_rain",
    "log_annual_rain",
    "temperature_x_log_dryness",
    "temperature_x_dry_month",
    "rain_memory_3m",
    "rain_departure_3m",
    "dryness_memory_3m",
    "dryness_departure_3m",
    "temperature_departure_3m",
)
REDUCED_INDEX = (0, 3, 4, 5, 6)


def weighted_metrics(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    observed_mean = np.average(observed, weights=weights)
    predicted_mean = np.average(predicted, weights=weights)
    covariance = np.average(
        (observed - observed_mean) * (predicted - predicted_mean), weights=weights
    )
    correlation = covariance / np.sqrt(
        np.average((observed - observed_mean) ** 2, weights=weights)
        * np.average((predicted - predicted_mean) ** 2, weights=weights)
    )
    rmse = np.sqrt(np.average((observed - predicted) ** 2, weights=weights))
    r2 = 1.0 - np.average((observed - predicted) ** 2, weights=weights) / np.average(
        (observed - observed_mean) ** 2, weights=weights
    )
    return float(correlation), float(rmse), float(r2)


def main() -> int:
    data = load_inputs(
        ("air_temperature", "dryness", "monthly_precipitation", "annual_precipitation")
    )
    land = load_land_mask()
    cells = np.flatnonzero(land.ravel())
    rows, cols = cells // 360, cells % 360

    def extract(name: str) -> np.ndarray:
        return np.asarray(data[name][:, rows, cols].T, dtype=np.float64)

    temperature = extract("air_temperature")
    dryness = np.clip(extract("dryness"), 0.0, None)
    rain = np.clip(extract("monthly_precipitation"), 0.0, None)
    annual_rain = np.clip(extract("annual_precipitation"), 0.0, None)
    rain_3m = running(rain, 3.0)
    dryness_3m = running(dryness, 3.0)
    temperature_3m = running(temperature, 3.0)
    log_dryness = np.log1p(dryness)
    dry_month = 1.0 / (1.0 + rain / 30.0)
    fields = (
        temperature,
        log_dryness,
        np.log1p(rain),
        np.log1p(annual_rain),
        temperature * log_dryness,
        temperature * dry_month,
        np.log1p(rain_3m),
        (rain_3m - rain) / (rain_3m + rain + 10.0),
        np.log1p(dryness_3m),
        (dryness - dryness_3m) / (dryness + dryness_3m + 100.0),
        temperature - temperature_3m,
    )
    x = np.column_stack([field.reshape(-1) for field in fields]).astype(np.float32)
    target = np.asarray(
        load_daily_vpd_duration()[:, rows, cols].T, dtype=np.float32
    ).reshape(-1)
    area = np.cos(np.deg2rad(-89.5 + rows.astype(np.float64)))
    weights = np.repeat(area, 192)
    rng = np.random.default_rng(613)
    cell_folds = rng.integers(0, 3, size=cells.size)
    folds = np.repeat(cell_folds, 192)
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    ridge_oof = np.zeros_like(target)
    reduced_oof = np.zeros_like(target)
    tree_oof = np.zeros_like(target)
    importances: list[np.ndarray] = []
    transformed_target = np.log(
        np.clip(target, 1e-3, 1.0 - 1e-3)
        / np.clip(1.0 - target, 1e-3, 1.0)
    )
    for fold in range(3):
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if train.size > 600_000:
            train = rng.choice(train, size=600_000, replace=False)
        ridge = Ridge(alpha=10.0)
        ridge.fit(x[train], transformed_target[train], sample_weight=weights[train])
        linear = ridge.predict(x[held])
        ridge_oof[held] = 1.0 / (1.0 + np.exp(np.clip(-linear, -40.0, 40.0)))
        reduced = Ridge(alpha=10.0)
        reduced.fit(
            x[train][:, REDUCED_INDEX],
            transformed_target[train],
            sample_weight=weights[train],
        )
        reduced_linear = reduced.predict(x[held][:, REDUCED_INDEX])
        reduced_oof[held] = 1.0 / (
            1.0 + np.exp(np.clip(-reduced_linear, -40.0, 40.0))
        )
        tree = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=701 + fold,
        )
        tree.fit(x[train], target[train], sample_weight=weights[train])
        tree_oof[held] = np.clip(tree.predict(x[held]), 0.0, 1.0)
        sample = rng.choice(held, size=min(150_000, held.size), replace=False)
        base_error = np.average(
            (target[sample] - tree.predict(x[sample])) ** 2, weights=weights[sample]
        )
        fold_importance = np.empty(x.shape[1], dtype=np.float64)
        for feature in range(x.shape[1]):
            shuffled = x[sample].copy()
            rng.shuffle(shuffled[:, feature])
            error = np.average(
                (target[sample] - tree.predict(shuffled)) ** 2,
                weights=weights[sample],
            )
            fold_importance[feature] = error - base_error
        importances.append(fold_importance)
        print(f"completed fold={fold}", flush=True)

    for label, prediction in (
        ("ridge_logit", ridge_oof),
        ("reduced_ridge_logit", reduced_oof),
        ("histogram_tree", tree_oof),
    ):
        correlation, rmse, r2 = weighted_metrics(target, prediction, weights)
        print(f"{label} weighted_r={correlation:.6f} rmse={rmse:.6f} r2={r2:.6f}")
    mean_importance = np.asarray(importances).mean(axis=0)
    for index in np.argsort(mean_importance)[::-1]:
        print(f"importance {NAMES[index]}={mean_importance[index]:.8f}")
    reduced = Ridge(alpha=10.0)
    reduced.fit(
        x[:, REDUCED_INDEX], transformed_target, sample_weight=weights
    )
    print(f"REDUCED_NAMES={tuple(NAMES[index] for index in REDUCED_INDEX)!r}")
    print(f"REDUCED_INTERCEPT={reduced.intercept_!r}")
    print(f"REDUCED_COEFFICIENTS={tuple(reduced.coef_)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
