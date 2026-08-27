"""Rank nonlinear physical interactions with an explainable boosting model.

Diagnostic only. Geography is excluded, and no EBM object or lookup table is
allowed into model.py; stable named interactions are distilled into equations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask, load_model, validate_prediction  # noqa: E402


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    annual = cycle.sum(axis=0)
    allocation = cycle / (annual[None, ...] + 1e-12)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    obs_allocation = obs_cycle / (obs_annual[None, ...] + 1e-12)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), len(cells))
    rows, cols = np.repeat(cell_rows, 12), np.repeat(cell_cols, 12)
    angle = 2.0 * np.pi * months / 12.0
    names = ["incumbent_allocation", "sin_month", "cos_month", "sin_2month", "cos_2month"]
    columns = [allocation[months, rows, cols], np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)]
    for name in model.INPUTS:
        values = data[name].reshape(16, 12, 180, 360).mean(axis=0)
        mean = values.mean(axis=0)
        anomaly = (values - mean[None, ...]) / (values.std(axis=0)[None, ...] + 1e-6)
        names.extend((f"{name}:current", f"{name}:anomaly", f"{name}:previous", f"{name}:mean"))
        columns.extend((values[months, rows, cols], anomaly[months, rows, cols], np.roll(anomaly, 1, axis=0)[months, rows, cols], mean[rows, cols]))
    x = np.column_stack(columns).astype(np.float32)
    y = obs_allocation[months, rows, cols].astype(np.float32)
    weight = obs_annual[rows, cols] + float(obs_annual.mean()) * 0.01
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    learner = ExplainableBoostingRegressor(
        feature_names=names,
        max_bins=64,
        max_interaction_bins=32,
        interactions=30,
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
        random_state=419,
    )
    learner.fit(x, y, sample_weight=weight)
    learned = np.zeros((12, 180, 360), dtype=np.float64)
    learned[months, rows, cols] = np.clip(learner.predict(x), 1e-9, None)
    learned /= learned.sum(axis=0, keepdims=True) + 1e-12
    candidate = np.tile(annual[None, ...] * learned, (16, 1, 1)).astype(np.float32)
    evaluator = GFED5Evaluator(GFED5_PATH)
    for label, prediction in (("incumbent", incumbent), ("fixed-map EBM", candidate)):
        score = evaluator.score(prediction)["global"]
        print(f"{label} overall={score['overall_score']:.4f} rmse={score['rmse_score']:.4f} seasonal={score['seasonal_cycle_score']:.4f}", flush=True)

    print("ranked EBM terms", flush=True)
    importances = learner.term_importances()
    for index in np.argsort(importances)[::-1]:
        print(f"{learner.term_names_[index]}\t{importances[index]:.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
