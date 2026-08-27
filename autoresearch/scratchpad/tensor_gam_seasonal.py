"""Smooth tensor-GAM distillation of ML-ranked seasonal physics.

Only named ecological interactions are admitted. Every spline and tensor
coefficient is global, and deployment would freeze its knots so an ED site
depends only on its own state and running climatology.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.preprocessing import SplineTransformer

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

    def anomaly(name: str) -> np.ndarray:
        values = cycles[name]
        return np.clip(
            (values - means[name][None, ...])
            / (values.std(axis=0)[None, ...] + 1e-6),
            -4.0,
            4.0,
        )[months, rows, cols]

    variables = {
        "incumbent": incumbent_alloc[months, rows, cols],
        "temperature_anomaly": anomaly("air_temperature"),
        "precipitation_anomaly": anomaly("monthly_precipitation"),
        "previous_precipitation": np.roll(
            (cycles["monthly_precipitation"] - means["monthly_precipitation"][None, ...])
            / (cycles["monthly_precipitation"].std(axis=0)[None, ...] + 1e-6),
            1,
            axis=0,
        )[months, rows, cols],
        "dryness_anomaly": anomaly("dryness"),
        "previous_dryness": np.roll(
            (cycles["dryness"] - means["dryness"][None, ...])
            / (cycles["dryness"].std(axis=0)[None, ...] + 1e-6),
            1,
            axis=0,
        )[months, rows, cols],
        "gpp_anomaly": anomaly("gpp"),
        "lai_anomaly": anomaly("leaf_area_index"),
        "lightning_anomaly": anomaly("lightning_flash_rate"),
        "secondary_canopy_anomaly": anomaly("secondary_canopy_height"),
        "mean_temperature": np.repeat(
            means["air_temperature"][cell_rows, cell_cols], 12
        ),
        "mean_gpp": np.repeat(means["gpp"][cell_rows, cell_cols], 12),
        "annual_rain": np.repeat(
            means["annual_precipitation"][cell_rows, cell_cols], 12
        ),
        "cropland": np.repeat(
            means["luh2_cropland_fraction"][cell_rows, cell_cols], 12
        ),
        "pasture": np.repeat(
            means["luh2_pasture_fraction"][cell_rows, cell_cols], 12
        ),
        "primary": np.repeat(
            means["luh2_primary_fraction"][cell_rows, cell_cols], 12
        ),
    }
    splines: dict[str, np.ndarray] = {}
    transformers: dict[str, SplineTransformer] = {}
    for name, values in variables.items():
        transformer = SplineTransformer(
            n_knots=4,
            degree=3,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        splines[name] = transformer.fit_transform(
            np.asarray(values, dtype=np.float64)[:, None]
        ).astype(np.float32)
        transformers[name] = transformer

    interaction_pairs = (
        ("temperature_anomaly", "mean_temperature"),
        ("lightning_anomaly", "mean_temperature"),
        ("lai_anomaly", "mean_temperature"),
        ("precipitation_anomaly", "mean_temperature"),
        ("previous_precipitation", "mean_temperature"),
        ("gpp_anomaly", "mean_temperature"),
        ("incumbent", "dryness_anomaly"),
        ("incumbent", "previous_dryness"),
        ("incumbent", "secondary_canopy_anomaly"),
        ("temperature_anomaly", "cropland"),
        ("lightning_anomaly", "cropland"),
        ("lai_anomaly", "cropland"),
        ("dryness_anomaly", "cropland"),
        ("lightning_anomaly", "pasture"),
        ("dryness_anomaly", "primary"),
    )
    columns: list[np.ndarray] = []
    groups: list[tuple[str, slice]] = []

    def add_group(name: str, values: np.ndarray) -> None:
        start = sum(column.shape[1] for column in columns)
        columns.append(np.asarray(values, dtype=np.float32))
        groups.append((name, slice(start, start + values.shape[1])))

    for name, values in splines.items():
        add_group(f"main:{name}", values)
    for left, right in interaction_pairs:
        product = (
            splines[left][:, :, None] * splines[right][:, None, :]
        ).reshape(len(months), -1)
        add_group(f"tensor:{left}_x_{right}", product)

    angle = 2.0 * np.pi * months / 12.0
    for harmonic in (1, 2, 3):
        for wave_name, wave in (
            (f"sin{harmonic}", np.sin(harmonic * angle)),
            (f"cos{harmonic}", np.cos(harmonic * angle)),
        ):
            add_group(f"calendar:{wave_name}", wave[:, None])
            for state in ("mean_temperature", "mean_gpp", "incumbent"):
                add_group(
                    f"calendar:{wave_name}_x_{state}",
                    wave[:, None] * splines[state],
                )

    x = np.column_stack(columns).astype(np.float32)
    incumbent_rows = variables["incumbent"]
    target_rows = observed_alloc[months, rows, cols]
    y = np.log(target_rows + 1e-4) - np.log(incumbent_rows + 1e-4)
    weights = np.repeat(
        observed_annual[cell_rows, cell_cols] + float(observed_annual.mean()) * 0.01,
        12,
    )
    print(f"rows={x.shape[0]} features={x.shape[1]} groups={len(groups)}", flush=True)
    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    if "--poisson" in sys.argv:
        target = target_rows.astype(np.float64)
        x_mean = np.average(x, axis=0, weights=weights)
        x_scale = np.sqrt(
            np.average(np.square(x - x_mean), axis=0, weights=weights)
        ) + 1e-8
        standardized = (x - x_mean) / x_scale

        def poisson_candidate(values: np.ndarray, strength: float) -> np.ndarray:
            learned = np.zeros((12, 180, 360), dtype=np.float64)
            learned[months, rows, cols] = np.maximum(values, 1e-12)
            learned /= learned.sum(axis=0, keepdims=True) + 1e-12
            blended = np.power(incumbent_alloc + 1e-12, 1.0 - strength)
            blended *= np.power(learned + 1e-12, strength)
            blended /= blended.sum(axis=0, keepdims=True) + 1e-12
            prediction = np.tile(
                incumbent_annual[None, ...] * blended, (16, 1, 1)
            )
            return prediction.astype(np.float32)

        rng = np.random.default_rng(457)
        poisson_folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
        for alpha in (0.003, 0.001):
            out_of_fold = np.zeros_like(target)
            fold_coefficients: list[np.ndarray] = []
            for fold in range(3):
                train = poisson_folds != fold
                held = ~train
                regressor = PoissonRegressor(
                    alpha=alpha, max_iter=1500, tol=1e-8
                )
                regressor.fit(
                    standardized[train],
                    target[train],
                    sample_weight=weights[train],
                )
                out_of_fold[held] = regressor.predict(standardized[held])
                fold_coefficients.append(regressor.coef_)
            correlations = np.corrcoef(np.asarray(fold_coefficients))
            print(
                f"poisson alpha={alpha} coefficient correlation "
                f"min={correlations[np.triu_indices(3, 1)].min():.4f}",
                flush=True,
            )
            for strength in (0.25, 0.50, 0.75, 1.0):
                report(
                    evaluator,
                    f"poisson OOF alpha={alpha} strength={strength}",
                    poisson_candidate(out_of_fold, strength),
                )
        return 0

    def candidate_from(residual: np.ndarray, strength: float) -> np.ndarray:
        correction = np.zeros((12, 180, 360), dtype=np.float64)
        correction[months, rows, cols] = np.clip(residual, -8.0, 8.0)
        learned = incumbent_alloc * np.exp(strength * correction)
        learned /= learned.sum(axis=0, keepdims=True) + 1e-12
        prediction = np.tile(incumbent_annual[None, ...] * learned, (16, 1, 1))
        return prediction.astype(np.float32)

    rng = np.random.default_rng(457)
    folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
    for alpha in (10.0, 30.0, 100.0, 300.0):
        out_of_fold = np.zeros_like(y)
        fold_coefficients: list[np.ndarray] = []
        for fold in range(3):
            train = folds != fold
            held = ~train
            regressor = Ridge(alpha=alpha, max_iter=2000, tol=1e-6)
            regressor.fit(x[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = regressor.predict(x[held])
            fold_coefficients.append(regressor.coef_)
        correlations = np.corrcoef(np.asarray(fold_coefficients))
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

    regressor = Ridge(alpha=30.0, max_iter=2000, tol=1e-6)
    regressor.fit(x, y, sample_weight=weights)
    print("ranked coefficient group norms", flush=True)
    for name, section in sorted(
        groups,
        key=lambda item: np.linalg.norm(regressor.coef_[item[1]]),
        reverse=True,
    ):
        print(f"{name}\t{np.linalg.norm(regressor.coef_[section]):.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
