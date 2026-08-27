"""Smooth annual fire-opportunity interactions distilled from held-out trees."""

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
    incumbent_alloc = incumbent_cycle / (incumbent_annual[None, ...] + 1e-12)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    observed_cycle = observed.reshape(16, 12, 180, 360).mean(axis=0)
    observed_annual = observed_cycle.sum(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360
    cycles = {
        name: np.asarray(values, dtype=np.float64)
        .reshape(16, 12, 180, 360)
        .mean(axis=0)
        for name, values in data.items()
    }
    summaries = {
        name: {
            "mean": values.mean(axis=0)[rows, cols],
            "std": values.std(axis=0)[rows, cols],
            "p10": np.quantile(values, 0.10, axis=0)[rows, cols],
            "p90": np.quantile(values, 0.90, axis=0)[rows, cols],
        }
        for name, values in cycles.items()
    }

    current = incumbent_annual[rows, cols]
    log_opportunity = np.log1p(10.0 * current)
    opportunity_regimes = {
        "rare": sigmoid((1.05 - log_opportunity) / 0.15),
        "low": sigmoid((log_opportunity - 0.95) / 0.15)
        * sigmoid((1.40 - log_opportunity) / 0.15),
        "medium": sigmoid((log_opportunity - 1.30) / 0.15)
        * sigmoid((1.75 - log_opportunity) / 0.15),
        "high": sigmoid((log_opportunity - 1.65) / 0.15),
    }
    state_gates = {
        "variable_lightning": sigmoid(
            (summaries["lightning_flash_rate"]["std"] - 0.03) / 0.008
        ),
        "persistent_lightning": sigmoid(
            (summaries["lightning_flash_rate"]["p10"] - 0.002) / 0.001
        ),
        "strong_lightning_peak": sigmoid(
            (summaries["lightning_flash_rate"]["p90"] - 0.08) / 0.02
        ),
        "seasonal_productivity": sigmoid(
            (summaries["gpp"]["std"] - 0.20) / 0.08
        ),
        "dry_season": sigmoid(
            (5.0 - summaries["monthly_precipitation"]["p10"]) / 2.0
        ),
        "seasonal_rain": sigmoid(
            (summaries["monthly_precipitation"]["p90"] - 70.0) / 20.0
        ),
        "moderate_rain_climate": sigmoid(
            (summaries["annual_precipitation"]["mean"] - 650.0) / 180.0
        ),
        "dynamic_canopy": sigmoid(
            (summaries["natural_canopy_height"]["std"] - 0.02) / 0.008
        ),
        "woody_canopy": sigmoid(
            (summaries["natural_canopy_height"]["p90"] - 5.0) / 2.0
        ),
        "warm_climate": sigmoid(
            (summaries["air_temperature"]["mean"] - 24.0) / 3.0
        ),
        "hot_peak": sigmoid(
            (summaries["air_temperature"]["p90"] - 28.0) / 2.0
        ),
        "high_biomass": sigmoid(
            (summaries["aboveground_biomass"]["p10"] - 0.20) / 0.05
        ),
        "managed_crop": sigmoid(
            (summaries["luh2_cropland_fraction"]["mean"] - 0.05) / 0.025
        ),
        "managed_pasture": sigmoid(
            (summaries["luh2_pasture_fraction"]["mean"] - 0.20) / 0.08
        ),
        "dryness_capacity": sigmoid(
            (summaries["dryness"]["mean"] - 330.0) / 120.0
        ),
    }

    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float32))

    add("log_opportunity", log_opportunity)
    add("sqrt_opportunity", np.sqrt(current))
    for center in (0.8, 1.1, 1.4, 1.7, 2.0):
        add(
            f"log_opportunity_above_{center}",
            np.maximum(log_opportunity - center, 0.0),
        )
    for regime_name, regime in opportunity_regimes.items():
        add(f"opportunity_{regime_name}", regime)
    for state_name, state in state_gates.items():
        add(state_name, state)
        for regime_name, regime in opportunity_regimes.items():
            add(f"{state_name}_x_opportunity_{regime_name}", state * regime)

    x = np.column_stack(columns).astype(np.float64)
    y = observed_annual[rows, cols].astype(np.float64)
    weights = y + float(y.mean()) * 0.02
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(f"cells={x.shape[0]} features={x.shape[1]}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    def candidate_from(values: np.ndarray, blend: float) -> np.ndarray:
        learned = np.zeros((180, 360), dtype=np.float64)
        learned[rows, cols] = np.clip(values, 0.0, 1.0)
        annual = (1.0 - blend) * incumbent_annual + blend * learned
        prediction = np.tile(annual[None, ...] * incumbent_alloc, (16, 1, 1))
        return prediction.astype(np.float32)

    rng = np.random.default_rng(443)
    folds = rng.integers(0, 5, size=cells.size)
    for alpha in (0.03, 0.01, 0.003, 0.001):
        out_of_fold = np.zeros_like(y)
        coefficients: list[np.ndarray] = []
        for fold in range(5):
            train = folds != fold
            held = ~train
            regressor = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-8)
            regressor.fit(standardized[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = regressor.predict(standardized[held])
            coefficients.append(regressor.coef_)
        correlations = np.corrcoef(np.asarray(coefficients))
        print(
            f"alpha={alpha} fold coefficient correlation "
            f"min={correlations[np.triu_indices(5, 1)].min():.4f}",
            flush=True,
        )
        for blend in (0.25, 0.50, 0.75, 1.0):
            report(
                evaluator,
                f"five-fold OOF alpha={alpha} blend={blend}",
                candidate_from(out_of_fold, blend),
            )

    regressor = PoissonRegressor(alpha=0.003, max_iter=2000, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    learned = regressor.predict(standardized)
    for blend in (0.25, 0.50, 0.75, 1.0):
        report(
            evaluator,
            f"in-sample alpha=0.003 blend={blend}",
            candidate_from(learned, blend),
        )
    print("top standardized coefficients", flush=True)
    for index in np.argsort(np.abs(regressor.coef_))[::-1][:50]:
        print(f"{names[index]}\t{regressor.coef_[index]:+.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
