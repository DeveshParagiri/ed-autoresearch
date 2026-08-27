"""Distil a site-local VPD memory response from whole-cell holdouts.

The learner only selects coefficients for named fire-physics terms. It sees no
coordinates, region labels, neighbours, future climate, or cell identifiers.
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


def running_mean(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float32)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
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
    requested = tuple(dict.fromkeys(model.INPUTS + ("vapor_pressure_deficit_mean",)))
    data = load_inputs(requested)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    observed_cycle = observed.reshape(16, 12, 180, 360).mean(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), cells.size)
    rows, cols = np.repeat(cell_rows, 12), np.repeat(cell_cols, 12)

    def cycle(values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(16, 12, 180, 360).mean(axis=0)

    vpd = np.asarray(data["vapor_pressure_deficit_mean"], dtype=np.float64)
    gpp = np.asarray(data["gpp"], dtype=np.float64)
    vpd_cycle = cycle(vpd)
    current_vpd = vpd_cycle[months, rows, cols]
    vpd_memories = {
        scale: cycle(running_mean(vpd, scale))[months, rows, cols]
        for scale in (3.0, 12.0, 24.0)
    }
    departures = {
        scale: (current_vpd - memory) / (current_vpd + memory + 0.2)
        for scale, memory in vpd_memories.items()
    }
    background = vpd_memories[24.0] / (vpd_memories[24.0] + 0.7)
    fuel_memory = cycle(running_mean(gpp, 12.0))[months, rows, cols]
    fuel_bank = fuel_memory / (fuel_memory + 0.5)

    annual_rain = cycle(np.asarray(data["annual_precipitation"], dtype=np.float64))[
        months, rows, cols
    ]
    seasonal_climate = sigmoid((annual_rain - 400.0) / 150.0) * sigmoid(
        (1700.0 - annual_rain) / 250.0
    )
    humid_climate = sigmoid((annual_rain - 1300.0) / 250.0)
    primary = cycle(np.asarray(data["luh2_primary_fraction"], dtype=np.float64))[
        months, rows, cols
    ]
    cropland = cycle(np.asarray(data["luh2_cropland_fraction"], dtype=np.float64))[
        months, rows, cols
    ]

    incumbent_rows = incumbent_cycle[months, rows, cols]
    trailing_share = incumbent_rows / (
        incumbent_cycle.sum(axis=0)[rows, cols] + 1e-12
    )
    opportunity = sigmoid((trailing_share - 0.05) / 0.025)

    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float64))

    for scale in (3.0, 12.0, 24.0):
        departure = departures[scale]
        add(f"vpd_departure_{scale:g}m", departure)
        add(f"vpd_departure_{scale:g}m_positive", np.maximum(departure, 0.0))
        add(f"vpd_departure_{scale:g}m_negative", np.minimum(departure, 0.0))
    add("vpd_background_24m", background)
    add("vpd_pulse_3m_x_fuel_bank", np.maximum(departures[3.0], 0.0) * fuel_bank)
    add("vpd_pulse_3m_x_seasonal_climate", np.maximum(departures[3.0], 0.0) * seasonal_climate)
    add("vpd_pulse_3m_x_humid_climate", np.maximum(departures[3.0], 0.0) * humid_climate)
    add("vpd_pulse_3m_x_primary", np.maximum(departures[3.0], 0.0) * primary)
    add("vpd_pulse_3m_x_cropland", np.maximum(departures[3.0], 0.0) * cropland)
    add("vpd_pulse_3m_x_opportunity", np.maximum(departures[3.0], 0.0) * opportunity)
    add("vpd_background_x_fuel_bank", background * fuel_bank)

    x = np.column_stack(columns)
    offset = incumbent_rows + 1e-4
    y = observed_cycle[months, rows, cols] / offset
    weights = offset + float(offset.mean()) * 0.02
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average(np.square(x - center), axis=0, weights=weights)) + 1e-8
    standardized = (x - center) / scale

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    rng = np.random.default_rng(491)
    folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
    for alpha in (0.03, 0.01, 0.003, 0.001):
        out_of_fold = np.zeros_like(y)
        coefficients = []
        for fold in range(3):
            train = folds != fold
            held = ~train
            regressor = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-8)
            regressor.fit(standardized[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = regressor.predict(standardized[held])
            coefficients.append(regressor.coef_)
        correlations = np.corrcoef(np.asarray(coefficients))
        print(
            f"alpha={alpha} fold_coefficient_r="
            f"{correlations[np.triu_indices(3, 1)].min():.6f}",
            flush=True,
        )
        for strength in (0.25, 0.50, 0.75, 1.0):
            learned = np.zeros((12, 180, 360), dtype=np.float64)
            learned[months, rows, cols] = offset * np.power(
                np.clip(out_of_fold, 1e-6, 1e6), strength
            )
            candidate = np.tile(learned, (16, 1, 1)).astype(np.float32)
            report(evaluator, f"OOF alpha={alpha} strength={strength}", candidate)

    final = PoissonRegressor(alpha=0.01, max_iter=2000, tol=1e-8)
    final.fit(standardized, y, sample_weight=weights)
    print(f"VPD_GLM_INTERCEPT={final.intercept_!r}")
    print(f"VPD_GLM_NAMES={tuple(names)!r}")
    print(f"VPD_GLM_COEFFICIENTS={tuple(final.coef_)!r}")
    print(f"VPD_GLM_CENTER={tuple(center)!r}")
    print(f"VPD_GLM_SCALE={tuple(scale)!r}")
    print("ranked coefficients")
    for index in np.argsort(np.abs(final.coef_))[::-1]:
        print(f"{names[index]}\t{final.coef_[index]:+.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
