"""Test whether causal local running state closes the coupled-valid ceiling.

The diagnostic adds only exponentially decayed site memory available in a
sequential ED run. It contains no future climatology, spatial reductions,
coordinates, labels, or neighbour exchange and is never itself deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

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


MEMORY_INPUTS = (
    "monthly_precipitation",
    "dryness",
    "air_temperature",
    "gpp",
    "leaf_area_index",
    "lightning_flash_rate",
)

CANDIDATE_FORCINGS = (
    "vapor_pressure_deficit_mean",
    "wind_speed_mean",
    "wet_day_fraction",
    "maximum_consecutive_dry_days",
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
    candidate_forcings = "--candidate-forcings" in sys.argv
    requested_inputs = tuple(
        dict.fromkeys(
            model.INPUTS + (CANDIDATE_FORCINGS if candidate_forcings else ())
        )
    )
    data = load_inputs(requested_inputs)
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
    names = ["incumbent"]
    columns = [incumbent_cycle[months, rows, cols]]
    cycles: dict[str, np.ndarray] = {}
    for name in requested_inputs:
        cycle = np.asarray(data[name]).reshape(16, 12, 180, 360).mean(axis=0)
        cycles[name] = cycle
        names.extend((f"{name}:current", f"{name}:previous"))
        columns.extend(
            (
                cycle[months, rows, cols],
                np.roll(cycle, 1, axis=0)[months, rows, cols],
            )
        )

    memory_inputs = MEMORY_INPUTS + (
        CANDIDATE_FORCINGS if candidate_forcings else ()
    )
    for name in memory_inputs:
        raw = np.asarray(data[name], dtype=np.float32)
        for timescale in (3.0, 6.0, 12.0, 24.0):
            memory = running_mean(raw, timescale).reshape(
                16, 12, 180, 360
            ).mean(axis=0)
            departure = cycles[name] - memory
            names.extend(
                (
                    f"{name}:memory_{timescale:g}m",
                    f"{name}:departure_{timescale:g}m",
                )
            )
            columns.extend(
                (
                    memory[months, rows, cols],
                    departure[months, rows, cols],
                )
            )
    angle = 2.0 * np.pi * months / 12.0
    names.extend(("sin_month", "cos_month", "sin_2month", "cos_2month"))
    columns.extend(
        (np.sin(angle), np.cos(angle), np.sin(2.0 * angle), np.cos(2.0 * angle))
    )
    x = np.column_stack(columns).astype(np.float32)
    y = observed_cycle[months, rows, cols].astype(np.float32)
    weights = y + float(y.mean()) * 0.02
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    if "--ebm" in sys.argv:
        from interpret.glassbox import ExplainableBoostingRegressor

        learner = ExplainableBoostingRegressor(
            feature_names=names,
            max_bins=64,
            max_interaction_bins=32,
            interactions=40,
            validation_size=0.15,
            outer_bags=4,
            inner_bags=0,
            learning_rate=0.04,
            smoothing_rounds=200,
            interaction_smoothing_rounds=100,
            max_rounds=4000,
            early_stopping_rounds=100,
            min_samples_leaf=20,
            max_leaves=3,
            objective="rmse",
            n_jobs=-2,
            random_state=473,
        )
        learner.fit(x, y, sample_weight=weights)
        learned = np.zeros((12, 180, 360), dtype=np.float32)
        learned[months, rows, cols] = np.clip(learner.predict(x), 0.0, 1.0)
        for blend in (0.25, 0.50, 0.75, 1.0):
            candidate_cycle = (1.0 - blend) * incumbent_cycle + blend * learned
            candidate = np.tile(candidate_cycle, (16, 1, 1)).astype(np.float32)
            report(evaluator, f"running-memory EBM blend={blend:.2f}", candidate)
        print("ranked running-memory EBM terms", flush=True)
        importances = learner.term_importances()
        for index in np.argsort(importances)[::-1]:
            print(
                f"{learner.term_names_[index]}\t{importances[index]:.10f}",
                flush=True,
            )
        return 0
    if "--oof" in sys.argv:
        rng = np.random.default_rng(471)
        folds = np.repeat(rng.integers(0, 3, size=cells.size), 12)
        out_of_fold = np.zeros_like(y)
        for fold in range(3):
            train = folds != fold
            held = ~train
            learner = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.05,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=471 + fold,
            )
            learner.fit(x[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = learner.predict(x[held])
            print(f"completed fold={fold}", flush=True)
        learned = np.zeros((12, 180, 360), dtype=np.float32)
        learned[months, rows, cols] = np.clip(out_of_fold, 0.0, 1.0)
        for blend in (0.25, 0.50, 0.75, 1.0):
            candidate_cycle = (1.0 - blend) * incumbent_cycle + blend * learned
            candidate = np.tile(candidate_cycle, (16, 1, 1)).astype(np.float32)
            report(
                evaluator,
                f"three-fold OOF running-memory HGB blend={blend:.2f}",
                candidate,
            )
        return 0

    learner = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=467,
    )
    learner.fit(x, y, sample_weight=weights)
    learned = np.zeros((12, 180, 360), dtype=np.float32)
    learned[months, rows, cols] = np.clip(learner.predict(x), 0.0, 1.0)
    for blend in (0.25, 0.50, 0.75, 1.0):
        candidate_cycle = (1.0 - blend) * incumbent_cycle + blend * learned
        candidate = np.tile(candidate_cycle, (16, 1, 1)).astype(np.float32)
        report(evaluator, f"running-memory HGB blend={blend:.2f}", candidate)

    rng = np.random.default_rng(469)
    importance_rows = rng.choice(
        x.shape[0], size=min(30000, x.shape[0]), replace=False
    )
    importance = permutation_importance(
        learner,
        x[importance_rows],
        y[importance_rows],
        scoring="neg_mean_squared_error",
        n_repeats=2,
        random_state=469,
        sample_weight=weights[importance_rows],
    )
    print("top running-memory features", flush=True)
    for index in np.argsort(importance.importances_mean)[::-1][:40]:
        print(f"{names[index]}\t{importance.importances_mean[index]:.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
