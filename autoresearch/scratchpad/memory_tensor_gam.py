"""Joint smooth GAM over causal local reservoir states.

The basis is restricted to ML-ranked moisture storage, fuel curing, ignition
memory, and thermal conditioning. Knots and coefficients are global; every
runtime feature can be updated sequentially at one independent ED site.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import PoissonRegressor
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

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    observed_cycle = observed.reshape(16, 12, 180, 360).mean(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), cells.size)
    rows, cols = np.repeat(cell_rows, 12), np.repeat(cell_cols, 12)

    selected_inputs = (
        "monthly_precipitation",
        "dryness",
        "air_temperature",
        "gpp",
        "leaf_area_index",
        "lightning_flash_rate",
    )
    raw = {
        name: np.asarray(data[name], dtype=np.float32) for name in selected_inputs
    }
    cycle = {
        name: values.reshape(16, 12, 180, 360).mean(axis=0)
        for name, values in raw.items()
    }
    values: dict[str, np.ndarray] = {
        "incumbent": incumbent_cycle[months, rows, cols]
    }
    for name in selected_inputs:
        values[f"{name}:current"] = cycle[name][months, rows, cols]
        values[f"{name}:previous"] = np.roll(cycle[name], 1, axis=0)[
            months, rows, cols
        ]
        for timescale in (3.0, 6.0, 12.0, 24.0):
            memory_cycle = running_mean(raw[name], timescale).reshape(
                16, 12, 180, 360
            ).mean(axis=0)
            memory_rows = memory_cycle[months, rows, cols]
            values[f"{name}:memory_{timescale:g}m"] = memory_rows
            values[f"{name}:departure_{timescale:g}m"] = (
                values[f"{name}:current"] - memory_rows
            )

    for name in (
        "aboveground_biomass",
        "natural_canopy_height",
        "secondary_canopy_height",
        "secondary_vegetation_fraction",
        "natural_vegetation_fraction",
        "soil_carbon",
        "annual_precipitation",
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_primary_fraction",
    ):
        state = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        values[f"{name}:current"] = state[months, rows, cols]

    # EBM-ranked conditional physics, retained as smooth tensor surfaces.
    pairs = (
        ("incumbent", "monthly_precipitation:departure_6m"),
        ("incumbent", "monthly_precipitation:departure_12m"),
        ("lightning_flash_rate:memory_3m", "monthly_precipitation:departure_6m"),
        ("lightning_flash_rate:memory_24m", "monthly_precipitation:departure_12m"),
        ("gpp:departure_3m", "monthly_precipitation:departure_6m"),
        ("leaf_area_index:departure_3m", "monthly_precipitation:departure_6m"),
        ("leaf_area_index:current", "air_temperature:memory_24m"),
        ("aboveground_biomass:current", "air_temperature:memory_6m"),
        ("aboveground_biomass:current", "air_temperature:memory_24m"),
        ("leaf_area_index:memory_3m", "air_temperature:memory_24m"),
        ("leaf_area_index:memory_24m", "air_temperature:memory_24m"),
        ("luh2_cropland_fraction:current", "monthly_precipitation:departure_12m"),
        ("luh2_pasture_fraction:current", "lightning_flash_rate:memory_3m"),
    )

    splines: dict[str, np.ndarray] = {}
    knots: dict[str, np.ndarray] = {}
    for name, state in values.items():
        state = np.asarray(state, dtype=np.float64)
        support = state[values["incumbent"] > 1e-5]
        nonzero = support[np.abs(support) > 1e-10]
        if nonzero.size >= max(100, support.size // 10):
            support = nonzero
        quantiles = (
            (0.02, 0.20, 0.50, 0.80, 0.98)
            if "--dense-main" in sys.argv
            else (0.10, 0.50, 0.90)
        )
        state_knots = np.quantile(support, quantiles)
        if np.any(np.diff(state_knots) <= 1e-10):
            low, high = np.quantile(support, (0.01, 0.99))
            if high <= low + 1e-10:
                high = low + 1.0
            state_knots = np.linspace(low, high, len(quantiles))
        transformer = SplineTransformer(
            degree=3 if "--dense-main" in sys.argv else 2,
            knots=state_knots[:, None],
            extrapolation="constant",
            include_bias=False,
        )
        splines[name] = transformer.fit_transform(
            np.asarray(state, dtype=np.float64)[:, None]
        ).astype(np.float32)
        knots[name] = state_knots

    columns: list[np.ndarray] = []
    groups: list[tuple[str, slice]] = []

    def add_group(name: str, state: np.ndarray) -> None:
        start = sum(item.shape[1] for item in columns)
        columns.append(np.asarray(state, dtype=np.float32))
        groups.append((name, slice(start, start + state.shape[1])))

    for name, state in splines.items():
        add_group(f"main:{name}", state)
    if "--dense-main" not in sys.argv:
        for left, right in pairs:
            tensor = (
                splines[left][:, :, None] * splines[right][:, None, :]
            ).reshape(len(months), -1)
            add_group(f"tensor:{left}_x_{right}", tensor)
    angle = 2.0 * np.pi * months / 12.0
    for harmonic in (1, 2):
        add_group(f"calendar:sin{harmonic}", np.sin(harmonic * angle)[:, None])
        add_group(f"calendar:cos{harmonic}", np.cos(harmonic * angle)[:, None])

    x = np.column_stack(columns).astype(np.float64)
    y = observed_cycle[months, rows, cols].astype(np.float64)
    weights = y + float(y.mean()) * 0.02
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(
        f"rows={x.shape[0]} features={x.shape[1]} groups={len(groups)}",
        flush=True,
    )

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    def candidate_from(prediction_rows: np.ndarray, blend: float) -> np.ndarray:
        learned = np.zeros((12, 180, 360), dtype=np.float64)
        learned[months, rows, cols] = np.clip(prediction_rows, 0.0, 1.0)
        candidate_cycle = (1.0 - blend) * incumbent_cycle + blend * learned
        return np.tile(candidate_cycle, (16, 1, 1)).astype(np.float32)

    rng = np.random.default_rng(487)
    folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
    alpha_values = (0.003, 0.001) if "--dense-main" in sys.argv else (0.01, 0.003, 0.001)
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
        for blend in (0.25, 0.50, 0.75, 1.0):
            report(
                evaluator,
                f"three-fold OOF alpha={alpha} blend={blend}",
                candidate_from(out_of_fold, blend),
            )

    final_alpha = 0.001 if "--dense-main" in sys.argv else 0.003
    regressor = PoissonRegressor(alpha=final_alpha, max_iter=2000, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    learned = regressor.predict(standardized)
    for blend in (0.25, 0.50, 0.75, 1.0):
        report(
            evaluator,
            f"in-sample alpha={final_alpha} blend={blend}",
            candidate_from(learned, blend),
        )
    ranked_groups = sorted(
        groups,
        key=lambda item: np.linalg.norm(regressor.coef_[item[1]]),
        reverse=True,
    )
    if "--reduce" in sys.argv:
        for count in (12, 20, 30, 40):
            chosen_groups = ranked_groups[:count]
            selected = np.concatenate(
                [np.arange(section.start, section.stop) for _, section in chosen_groups]
            )
            out_of_fold = np.zeros_like(y)
            fold_coefficients: list[np.ndarray] = []
            for fold in range(3):
                train = folds != fold
                held = ~train
                reduced = PoissonRegressor(alpha=0.001, max_iter=2000, tol=1e-8)
                reduced.fit(
                    standardized[train][:, selected],
                    y[train],
                    sample_weight=weights[train],
                )
                out_of_fold[held] = reduced.predict(
                    standardized[held][:, selected]
                )
                fold_coefficients.append(reduced.coef_)
            correlations = np.corrcoef(np.asarray(fold_coefficients))
            print(
                f"reduced groups={count} coefficient correlation "
                f"min={correlations[np.triu_indices(3, 1)].min():.4f}",
                flush=True,
            )
            report(
                evaluator,
                f"reduced OOF groups={count} blend=0.25",
                candidate_from(out_of_fold, 0.25),
            )
            reduced = PoissonRegressor(alpha=0.001, max_iter=2000, tol=1e-8)
            reduced.fit(standardized[:, selected], y, sample_weight=weights)
            report(
                evaluator,
                f"reduced in-sample groups={count} blend=0.25",
                candidate_from(reduced.predict(standardized[:, selected]), 0.25),
            )
            if count == 30:
                print(
                    "REDUCED_GROUPS=" + repr(tuple(name for name, _ in chosen_groups)),
                    flush=True,
                )
                print(f"REDUCED_INTERCEPT={reduced.intercept_!r}", flush=True)
                print(
                    "REDUCED_COEFFICIENTS="
                    + repr(tuple(float(value) for value in reduced.coef_)),
                    flush=True,
                )
                print(
                    "REDUCED_CENTER="
                    + repr(tuple(float(value) for value in x_mean[selected])),
                    flush=True,
                )
                print(
                    "REDUCED_SCALE="
                    + repr(tuple(float(value) for value in x_scale[selected])),
                    flush=True,
                )
        return 0
    print("ranked coefficient group norms", flush=True)
    for name, section in ranked_groups:
        print(f"{name}\t{np.linalg.norm(regressor.coef_[section]):.9f}", flush=True)
    print("spline knots", flush=True)
    for name, state_knots in knots.items():
        print(f"{name}\t{state_knots.tolist()}", flush=True)
    print(f"MEMORY_GAM_INTERCEPT={regressor.intercept_!r}", flush=True)
    print("MEMORY_GAM_COEFFICIENTS=" + repr(tuple(float(v) for v in regressor.coef_)), flush=True)
    print("MEMORY_GAM_CENTER=" + repr(tuple(float(v) for v in x_mean)), flush=True)
    print("MEMORY_GAM_SCALE=" + repr(tuple(float(v) for v in x_scale)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
