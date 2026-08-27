"""Fit the compact residual equation on actual causal monthly trajectories.

The earlier distillation averaged causal states into a 16-year monthly cycle
before fitting, then evaluated those coefficients at every online time step.
This diagnostic removes that semantic mismatch.  Every row is one actual
month at one independent land cell, folds hold out whole cells, and every
feature is constructed from current or prior local state only.
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


NAMES = (
    "dryness_departure_6m",
    "temperature_departure_6m_negative",
    "gpp_curing_6m",
    "gpp_curing_6m_positive",
    "lai_curing_6m",
    "lai_curing_6m_negative",
    "dryness_departure_12m_positive",
    "dryness_departure_24m",
    "dryness_departure_24m_positive",
    "gpp_curing_24m",
    "gpp_curing_24m_positive",
    "drying_x_fuel_6m",
    "drying_6m_x_humid_climate",
    "drying_x_fuel_12m",
    "drying_12m_x_temperate",
    "drying_12m_x_humid_climate",
    "drying_24m_x_rangeland",
    "opportunity_0.03",
    "opportunity_0.03_x_drying_12m",
    "opportunity_0.08",
)


def running(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[:, 0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[1]):
        state += alpha * (values[:, time] - state)
        output[:, time] = state
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -40.0, 40.0)))


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> None:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} "
        f"bias={score['bias_score']:.4f} rmse={score['rmse_score']:.4f} "
        f"seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}",
        flush=True,
    )


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360

    def extract(name: str) -> np.ndarray:
        return np.asarray(data[name][:, rows, cols].T, dtype=np.float64)

    rain = extract("monthly_precipitation")
    dryness = extract("dryness")
    temperature = extract("air_temperature")
    gpp = extract("gpp")
    lai = extract("leaf_area_index")
    annual_rain = extract("annual_precipitation")
    rangeland = extract("luh2_rangeland_fraction")
    baseline = np.asarray(incumbent[:, rows, cols].T, dtype=np.float64)

    rain_memory = {months: running(rain, months) for months in (6.0, 12.0, 24.0)}
    dryness_memory = {
        months: running(dryness, months) for months in (6.0, 12.0, 24.0)
    }
    gpp_memory = {months: running(gpp, months) for months in (6.0, 12.0, 24.0)}
    lai_memory_6m = running(lai, 6.0)
    temperature_memory_6m = running(temperature, 6.0)

    rain_departure = {
        months: (state - rain) / (state + rain + 10.0)
        for months, state in rain_memory.items()
    }
    dryness_departure = {
        months: (dryness - state)
        / (np.abs(state) + np.abs(dryness) + 100.0)
        for months, state in dryness_memory.items()
    }
    gpp_curing = {
        months: (state - gpp) / (state + gpp + 0.2)
        for months, state in gpp_memory.items()
    }
    lai_curing_6m = (lai_memory_6m - lai) / (lai_memory_6m + lai + 0.5)
    temperature_departure_6m = np.clip(
        (temperature - temperature_memory_6m) / 10.0, -2.0, 2.0
    )
    fuel_bank = {
        months: state / (state + 0.5) for months, state in gpp_memory.items()
    }
    humid_climate = sigmoid((annual_rain - 1300.0) / 250.0)
    temperate = sigmoid((temperature - 2.0) / 3.0) * sigmoid(
        (20.0 - temperature) / 3.0
    )

    cumulative = np.cumsum(baseline, axis=1)
    trailing = cumulative.copy()
    trailing[:, 12:] -= cumulative[:, :-12]
    exposure = np.minimum(np.arange(1, baseline.shape[1] + 1), 12)[None, :]
    trailing *= 12.0 / exposure
    share = baseline / (trailing + 1e-12)
    opportunity_003 = sigmoid((share - 0.03) / 0.025)
    opportunity_008 = sigmoid((share - 0.08) / 0.025)

    feature_fields = (
        dryness_departure[6.0],
        np.minimum(temperature_departure_6m, 0.0),
        gpp_curing[6.0],
        np.maximum(gpp_curing[6.0], 0.0),
        lai_curing_6m,
        np.minimum(lai_curing_6m, 0.0),
        np.maximum(dryness_departure[12.0], 0.0),
        dryness_departure[24.0],
        np.maximum(dryness_departure[24.0], 0.0),
        gpp_curing[24.0],
        np.maximum(gpp_curing[24.0], 0.0),
        rain_departure[6.0] * fuel_bank[6.0],
        rain_departure[6.0] * humid_climate,
        rain_departure[12.0] * fuel_bank[12.0],
        rain_departure[12.0] * temperate,
        rain_departure[12.0] * humid_climate,
        rain_departure[24.0] * rangeland,
        opportunity_003,
        opportunity_003 * rain_departure[12.0],
        opportunity_008,
    )
    x = np.column_stack([field.reshape(-1) for field in feature_fields]).astype(
        np.float32
    )
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    target = np.asarray(observed[:, rows, cols].T, dtype=np.float64).reshape(-1)
    offset = baseline.reshape(-1) + 1e-4
    y = target / offset
    weights = offset + float(offset.mean()) * 0.02
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    rng = np.random.default_rng(487)
    cell_folds = rng.integers(0, 3, size=cells.size)
    folds = np.repeat(cell_folds, baseline.shape[1])
    out_of_fold = np.zeros_like(y)
    coefficients: list[np.ndarray] = []
    for fold in range(3):
        train = folds != fold
        held = ~train
        regressor = PoissonRegressor(alpha=0.001, max_iter=1000, tol=1e-8)
        regressor.fit(
            standardized[train], y[train], sample_weight=weights[train]
        )
        out_of_fold[held] = regressor.predict(standardized[held])
        coefficients.append(regressor.coef_)
        print(f"completed fold={fold}", flush=True)
    correlations = np.corrcoef(np.asarray(coefficients))
    print(
        "fold coefficient correlation min="
        f"{correlations[np.triu_indices(3, 1)].min():.4f}",
        flush=True,
    )

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    for strength in (0.25, 0.50, 0.75, 1.0):
        corrected = baseline * np.power(np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6), strength)
        candidate = incumbent.copy()
        candidate[:, rows, cols] = corrected.T
        report(evaluator, f"online OOF strength={strength}", candidate)

    regressor = PoissonRegressor(alpha=0.001, max_iter=1500, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    print(f"ONLINE_GLM_NAMES={NAMES!r}", flush=True)
    print(f"ONLINE_GLM_INTERCEPT={regressor.intercept_!r}", flush=True)
    print(f"ONLINE_GLM_COEFFICIENTS={tuple(regressor.coef_)!r}", flush=True)
    print(f"ONLINE_GLM_CENTER={tuple(x_mean)!r}", flush=True)
    print(f"ONLINE_GLM_SCALE={tuple(x_scale)!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
