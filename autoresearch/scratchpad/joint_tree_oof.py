"""Spatial holdout test for the no-geography nonlinear information ceiling.

Trees are diagnostic only. The purpose is to identify the minimum interaction
complexity worth translating into smooth ecological equations; tree rules or
benchmark-fitted leaves are never promoted into model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.tree import DecisionTreeRegressor, export_text

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
    angle = 2.0 * np.pi * months / 12.0
    seasonal_names = [
        "incumbent_allocation",
        "sin_month",
        "cos_month",
        "sin_2month",
        "cos_2month",
        "sin_3month",
        "cos_3month",
    ]
    seasonal_columns = [
        incumbent_alloc[months, rows, cols],
        np.sin(angle),
        np.cos(angle),
        np.sin(2.0 * angle),
        np.cos(2.0 * angle),
        np.sin(3.0 * angle),
        np.cos(3.0 * angle),
    ]
    annual_names = ["incumbent_annual", "log_incumbent_annual"]
    annual_columns = [
        incumbent_annual[cell_rows, cell_cols],
        np.log1p(10.0 * incumbent_annual[cell_rows, cell_cols]),
    ]
    for name in model.INPUTS:
        cycle = np.asarray(data[name]).reshape(16, 12, 180, 360).mean(axis=0)
        mean = cycle.mean(axis=0)
        std = cycle.std(axis=0)
        low = np.quantile(cycle, 0.1, axis=0)
        high = np.quantile(cycle, 0.9, axis=0)
        anomaly = (cycle - mean[None, ...]) / (std[None, ...] + 1e-6)
        seasonal_names.extend(
            (
                f"{name}:cycle",
                f"{name}:anomaly",
                f"{name}:previous",
                f"{name}:mean",
            )
        )
        seasonal_columns.extend(
            (
                cycle[months, rows, cols],
                anomaly[months, rows, cols],
                np.roll(anomaly, 1, axis=0)[months, rows, cols],
                mean[rows, cols],
            )
        )
        annual_names.extend(
            (f"{name}:mean", f"{name}:std", f"{name}:p10", f"{name}:p90")
        )
        annual_columns.extend(
            (
                mean[cell_rows, cell_cols],
                std[cell_rows, cell_cols],
                low[cell_rows, cell_cols],
                high[cell_rows, cell_cols],
            )
        )
    seasonal_x = np.column_stack(seasonal_columns).astype(np.float32)
    annual_x = np.column_stack(annual_columns).astype(np.float32)
    seasonal_y = observed_alloc[months, rows, cols].astype(np.float32)
    annual_y = observed_annual[cell_rows, cell_cols].astype(np.float32)
    seasonal_weight = (
        observed_annual[rows, cols] + float(observed_annual.mean()) * 0.01
    )
    annual_weight = annual_y + float(annual_y.mean()) * 0.02
    print(
        f"cells={cells.size} seasonal_features={seasonal_x.shape[1]} "
        f"annual_features={annual_x.shape[1]}",
        flush=True,
    )

    rng = np.random.default_rng(431)
    cell_folds = rng.integers(0, 3, size=cells.size)
    seasonal_folds = np.repeat(cell_folds, 12)
    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    if "--rules-only" in sys.argv:
        for leaves in (16, 32):
            seasonal_tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves, min_samples_leaf=40, random_state=431
            )
            seasonal_tree.fit(
                seasonal_x, seasonal_y, sample_weight=seasonal_weight
            )
            print(f"seasonal rules leaves={leaves}", flush=True)
            print(
                export_text(
                    seasonal_tree,
                    feature_names=seasonal_names,
                    max_depth=12,
                ),
                flush=True,
            )
            annual_tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves, min_samples_leaf=25, random_state=441
            )
            annual_tree.fit(annual_x, annual_y, sample_weight=annual_weight)
            print(f"annual rules leaves={leaves}", flush=True)
            print(
                export_text(
                    annual_tree,
                    feature_names=annual_names,
                    max_depth=12,
                ),
                flush=True,
            )
        return 0

    for leaves in (16, 32, 64, 128, 256):
        seasonal_oof = np.zeros_like(seasonal_y)
        annual_oof = np.zeros_like(annual_y)
        for fold in range(3):
            annual_train = cell_folds != fold
            annual_held = ~annual_train
            seasonal_train = seasonal_folds != fold
            seasonal_held = ~seasonal_train
            seasonal_tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves,
                min_samples_leaf=40,
                random_state=431 + fold,
            )
            seasonal_tree.fit(
                seasonal_x[seasonal_train],
                seasonal_y[seasonal_train],
                sample_weight=seasonal_weight[seasonal_train],
            )
            seasonal_oof[seasonal_held] = seasonal_tree.predict(
                seasonal_x[seasonal_held]
            )
            annual_tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves,
                min_samples_leaf=25,
                random_state=441 + fold,
            )
            annual_tree.fit(
                annual_x[annual_train],
                annual_y[annual_train],
                sample_weight=annual_weight[annual_train],
            )
            annual_oof[annual_held] = annual_tree.predict(annual_x[annual_held])

        learned_alloc = np.zeros((12, 180, 360), dtype=np.float64)
        learned_alloc[months, rows, cols] = np.maximum(seasonal_oof, 1e-12)
        learned_alloc /= learned_alloc.sum(axis=0, keepdims=True) + 1e-12
        learned_annual = np.zeros((180, 360), dtype=np.float64)
        learned_annual[cell_rows, cell_cols] = np.clip(annual_oof, 0.0, 1.0)
        for annual_blend in (0.5, 0.75, 1.0):
            annual = (
                (1.0 - annual_blend) * incumbent_annual
                + annual_blend * learned_annual
            )
            candidate = np.tile(annual[None, ...] * learned_alloc, (16, 1, 1))
            report(
                evaluator,
                f"three-fold OOF leaves={leaves} annual_blend={annual_blend}",
                candidate.astype(np.float32),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
