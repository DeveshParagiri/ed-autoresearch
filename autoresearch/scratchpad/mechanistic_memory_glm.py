"""Fit a compact causal moisture, fuel, and ignition memory correction.

The incumbent allocation is a fixed multiplicative offset. Learned terms are
globally shared smooth functions of running local reservoirs and named
ecological interactions; there is no future climatology or geography.
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


def running_mean(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float32)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


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
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_annual = incumbent_cycle.sum(axis=0)
    incumbent_share = np.empty_like(incumbent, dtype=np.float64)
    for time in range(incumbent.shape[0]):
        start = max(0, time - 11)
        trailing = incumbent[start:time + 1].sum(axis=0)
        trailing *= 12.0 / (time - start + 1)
        incumbent_share[time] = incumbent[time] / (trailing + 1e-12)
    incumbent_alloc = incumbent_share.reshape(16, 12, 180, 360).mean(axis=0)

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

    raw = {
        name: np.asarray(data[name], dtype=np.float64) for name in (
            "monthly_precipitation",
            "dryness",
            "air_temperature",
            "gpp",
            "leaf_area_index",
            "lightning_flash_rate",
        )
    }
    memories = {
        name: {
            timescale: running_mean(values, timescale)
            for timescale in (3.0, 6.0, 12.0, 24.0)
        }
        for name, values in raw.items()
    }

    def cycle(values: np.ndarray) -> np.ndarray:
        return values.reshape(16, 12, 180, 360).mean(axis=0)

    current = {name: cycle(values)[months, rows, cols] for name, values in raw.items()}
    memory = {
        name: {
            timescale: cycle(values)[months, rows, cols]
            for timescale, values in states.items()
        }
        for name, states in memories.items()
    }
    static = {
        name: cycle(np.asarray(data[name], dtype=np.float64))[months, rows, cols]
        for name in (
            "air_temperature",
            "annual_precipitation",
            "luh2_cropland_fraction",
            "luh2_pasture_fraction",
            "luh2_rangeland_fraction",
            "luh2_primary_fraction",
            "natural_vegetation_fraction",
        )
    }

    physical: dict[str, np.ndarray] = {}
    for timescale in (3.0, 6.0, 12.0, 24.0):
        precipitation_store = memory["monthly_precipitation"][timescale]
        physical[f"rain_departure_{timescale:g}m"] = (
            precipitation_store - current["monthly_precipitation"]
        ) / (precipitation_store + current["monthly_precipitation"] + 10.0)
        physical[f"rain_store_{timescale:g}m"] = precipitation_store / (
            precipitation_store + 30.0
        )
        dryness_store = memory["dryness"][timescale]
        physical[f"dryness_departure_{timescale:g}m"] = (
            current["dryness"] - dryness_store
        ) / (np.abs(dryness_store) + np.abs(current["dryness"]) + 100.0)
        temperature_store = memory["air_temperature"][timescale]
        physical[f"temperature_departure_{timescale:g}m"] = np.clip(
            (current["air_temperature"] - temperature_store) / 10.0,
            -2.0,
            2.0,
        )
        gpp_store = memory["gpp"][timescale]
        physical[f"fuel_bank_{timescale:g}m"] = gpp_store / (gpp_store + 0.5)
        physical[f"gpp_curing_{timescale:g}m"] = (
            gpp_store - current["gpp"]
        ) / (gpp_store + current["gpp"] + 0.2)
        lai_store = memory["leaf_area_index"][timescale]
        physical[f"lai_curing_{timescale:g}m"] = (
            lai_store - current["leaf_area_index"]
        ) / (lai_store + current["leaf_area_index"] + 0.5)
        lightning_store = memory["lightning_flash_rate"][timescale]
        physical[f"ignition_store_{timescale:g}m"] = lightning_store / (
            lightning_store + 0.01
        )

    temperature_regimes = {
        "cold": sigmoid((5.0 - static["air_temperature"]) / 3.0),
        "temperate": sigmoid((static["air_temperature"] - 2.0) / 3.0)
        * sigmoid((20.0 - static["air_temperature"]) / 3.0),
        "warm": sigmoid((static["air_temperature"] - 16.0) / 3.0),
    }
    land_regimes = {
        "crop": static["luh2_cropland_fraction"],
        "pasture": static["luh2_pasture_fraction"],
        "rangeland": static["luh2_rangeland_fraction"],
        "primary": static["luh2_primary_fraction"],
        "natural": static["natural_vegetation_fraction"],
    }
    rain_regimes = {
        "dry_climate": sigmoid((650.0 - static["annual_precipitation"]) / 180.0),
        "seasonal_climate": sigmoid(
            (static["annual_precipitation"] - 400.0) / 150.0
        )
        * sigmoid((1700.0 - static["annual_precipitation"]) / 250.0),
        "humid_climate": sigmoid(
            (static["annual_precipitation"] - 1300.0) / 250.0
        ),
    }

    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    for name, values in physical.items():
        add(name, values)
        add(f"{name}_positive", np.maximum(values, 0.0))
        add(f"{name}_negative", np.minimum(values, 0.0))

    # Named fire-triangle interactions found consistently by the memory HGB.
    for timescale in (6.0, 12.0, 24.0):
        drying = physical[f"rain_departure_{timescale:g}m"]
        fuel = physical[f"fuel_bank_{timescale:g}m"]
        ignition = physical[f"ignition_store_{timescale:g}m"]
        add(f"drying_x_fuel_{timescale:g}m", drying * fuel)
        add(f"drying_x_ignition_{timescale:g}m", drying * ignition)
        add(
            f"drying_x_warming_{timescale:g}m",
            drying * physical[f"temperature_departure_{timescale:g}m"],
        )
        add(
            f"drying_x_curing_{timescale:g}m",
            drying * physical[f"gpp_curing_{timescale:g}m"],
        )
        for regime_name, regime in temperature_regimes.items():
            add(f"drying_{timescale:g}m_x_{regime_name}", drying * regime)
        for regime_name, regime in land_regimes.items():
            add(f"drying_{timescale:g}m_x_{regime_name}", drying * regime)
        for regime_name, regime in rain_regimes.items():
            add(f"drying_{timescale:g}m_x_{regime_name}", drying * regime)

    incumbent_rows = incumbent_alloc[months, rows, cols]
    for center in (0.03, 0.08, 0.16):
        opportunity = sigmoid((incumbent_rows - center) / 0.025)
        add(f"opportunity_{center:.2f}", opportunity)
        for timescale in (6.0, 12.0):
            add(
                f"opportunity_{center:.2f}_x_drying_{timescale:g}m",
                opportunity * physical[f"rain_departure_{timescale:g}m"],
            )

    x = np.column_stack(columns).astype(np.float64)
    target_rows = observed_alloc[months, rows, cols]
    offset = incumbent_rows + 1e-4
    y = target_rows / offset
    weights = np.repeat(
        observed_annual[cell_rows, cell_cols] + float(observed_annual.mean()) * 0.01,
        12,
    ) * offset
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    def candidate_from(ratio: np.ndarray, strength: float) -> np.ndarray:
        learned = np.zeros((12, 180, 360), dtype=np.float64)
        learned[months, rows, cols] = offset * np.power(
            np.clip(ratio, 1e-6, 1e6), strength
        )
        learned /= learned.sum(axis=0, keepdims=True) + 1e-12
        prediction = np.tile(incumbent_annual[None, ...] * learned, (16, 1, 1))
        return prediction.astype(np.float32)

    rng = np.random.default_rng(479)
    folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
    alpha_values = (0.001,) if "--reduce" in sys.argv else (0.03, 0.01, 0.003, 0.001)
    for alpha in alpha_values:
        out_of_fold = np.zeros_like(y)
        coefficients: list[np.ndarray] = []
        for fold in range(3):
            train = folds != fold
            held = ~train
            regressor = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-8)
            regressor.fit(
                standardized[train], y[train], sample_weight=weights[train]
            )
            out_of_fold[held] = regressor.predict(standardized[held])
            coefficients.append(regressor.coef_)
        correlations = np.corrcoef(np.asarray(coefficients))
        print(
            f"alpha={alpha} coefficient correlation "
            f"min={correlations[np.triu_indices(3, 1)].min():.4f}",
            flush=True,
        )
        for strength in (0.25, 0.50, 0.75, 1.0):
            report(
                evaluator,
                f"three-fold OOF alpha={alpha} strength={strength}",
                candidate_from(out_of_fold, strength),
            )

    regressor = PoissonRegressor(alpha=0.001, max_iter=2000, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    print("top standardized coefficients", flush=True)
    for index in np.argsort(np.abs(regressor.coef_))[::-1][:60]:
        print(f"{names[index]}\t{regressor.coef_[index]:+.9f}", flush=True)
    if "--reduce" in sys.argv:
        ranking = np.argsort(np.abs(regressor.coef_))[::-1]
        for count in (20, 40, 80):
            selected = np.sort(ranking[:count])
            out_of_fold = np.zeros_like(y)
            for fold in range(3):
                train = folds != fold
                held = ~train
                reduced = PoissonRegressor(
                    alpha=0.001, max_iter=1500, tol=1e-8
                )
                reduced.fit(
                    standardized[train][:, selected],
                    y[train],
                    sample_weight=weights[train],
                )
                out_of_fold[held] = reduced.predict(
                    standardized[held][:, selected]
                )
            for strength in (0.5, 0.75):
                report(
                    evaluator,
                    f"three-fold OOF reduced={count} strength={strength}",
                    candidate_from(out_of_fold, strength),
                )
            if count == 20:
                reduced = PoissonRegressor(
                    alpha=0.001, max_iter=2000, tol=1e-8
                )
                reduced.fit(
                    standardized[:, selected], y, sample_weight=weights
                )
                print(
                    f"CAUSAL_GLM_NAMES={tuple(names[index] for index in selected)!r}",
                    flush=True,
                )
                print(f"CAUSAL_GLM_INTERCEPT={reduced.intercept_!r}", flush=True)
                print(
                    f"CAUSAL_GLM_COEFFICIENTS={tuple(reduced.coef_)!r}",
                    flush=True,
                )
                print(
                    f"CAUSAL_GLM_CENTER={tuple(x_mean[selected])!r}",
                    flush=True,
                )
                print(
                    f"CAUSAL_GLM_SCALE={tuple(x_scale[selected])!r}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
