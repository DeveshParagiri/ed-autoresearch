"""Ceiling for a genuinely local, online monthly ED fire equation.

Only the incumbent monthly fire opportunity, current local input values, and
their one-month lags are exposed. There are no coordinates, region labels,
cross-cell summaries, future climatological anomalies, or cell identifiers.
The booster is diagnostic only and is never copied into ``model.py``.
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


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> float:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} bias={score['bias_score']:.4f} "
        f"rmse={score['rmse_score']:.4f} seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}",
        flush=True,
    )
    return float(score["overall_score"])


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), len(cells))
    rows = np.repeat(cell_rows, 12)
    cols = np.repeat(cell_cols, 12)
    names = ["incumbent"]
    columns = [incumbent_cycle[months, rows, cols]]
    for name in model.INPUTS:
        cycle = data[name].reshape(16, 12, 180, 360).mean(axis=0)
        names.extend((f"{name}:current", f"{name}:previous"))
        columns.extend(
            (
                cycle[months, rows, cols],
                np.roll(cycle, 1, axis=0)[months, rows, cols],
            )
        )
    angle = 2.0 * np.pi * months / 12.0
    names.extend(("sin_month", "cos_month"))
    columns.extend((np.sin(angle), np.cos(angle)))
    x = np.column_stack(columns).astype(np.float32)
    y = obs_cycle[months, rows, cols].astype(np.float32)
    weight = y + float(y.mean()) * 0.02
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    learner = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=389,
    )
    learner.fit(x, y, sample_weight=weight)
    learned = np.zeros((12, 180, 360), dtype=np.float32)
    learned[months, rows, cols] = np.clip(learner.predict(x), 0.0, 1.0)
    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    for blend in (0.25, 0.50, 0.75, 1.0):
        candidate_cycle = (1.0 - blend) * incumbent_cycle + blend * learned
        candidate = np.tile(candidate_cycle, (16, 1, 1)).astype(np.float32)
        report(evaluator, f"local monthly HGB blend={blend:.2f}", candidate)

    importance = permutation_importance(
        learner,
        x,
        y,
        scoring="neg_mean_squared_error",
        n_repeats=3,
        random_state=397,
        sample_weight=weight,
    )
    print("top local monthly features", flush=True)
    for index in np.argsort(importance.importances_mean)[::-1][:30]:
        print(f"{names[index]}\t{importance.importances_mean[index]:.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
