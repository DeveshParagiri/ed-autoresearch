"""Learn the missing seasonal allocation with the incumbent annual map fixed.

Diagnostic only.  The learner sees globally shared physical inputs and calendar
harmonics, never coordinates, regions, cell IDs, or geographic masks.  Its
output is normalised to a twelve-month allocation in each cell and multiplied
by the incumbent annual burned area, so the annual map is conserved exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    INPUTS_DIR,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


INVALID_INPUTS = {
    "wind_speed_mean",
    "vapor_pressure_deficit_mean",
    "maximum_consecutive_dry_days",
    "wet_day_fraction",
    "population_density",
}


def input_names() -> list[str]:
    names: list[str] = []
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            names.extend(
                name
                for name, variable in dataset.variables.items()
                if variable.dimensions == ("time", "lat", "lon")
            )
    return [name for name in names if name not in INVALID_INPUTS]


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
    names = input_names()
    data = load_inputs(names)
    model = load_model()
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_annual = incumbent_cycle.sum(axis=0)
    incumbent_allocation = incumbent_cycle / (incumbent_annual[None, ...] + 1e-12)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0).astype(np.float32)
    obs_annual = obs_cycle.sum(axis=0)
    obs_allocation = obs_cycle / (obs_annual[None, ...] + 1e-12)

    land_cells = np.flatnonzero(load_land_mask().ravel())
    rows = land_cells // 360
    cols = land_cells % 360
    months = np.tile(np.arange(12, dtype=np.int64), land_cells.size)
    train_rows = np.repeat(rows, 12)
    train_cols = np.repeat(cols, 12)
    angle = 2.0 * np.pi * months / 12.0

    feature_names = [
        "incumbent_allocation",
        "sin_month",
        "cos_month",
        "sin_2month",
        "cos_2month",
        "sin_3month",
        "cos_3month",
    ]
    columns = [
        incumbent_allocation[months, train_rows, train_cols],
        np.sin(angle),
        np.cos(angle),
        np.sin(2.0 * angle),
        np.cos(2.0 * angle),
        np.sin(3.0 * angle),
        np.cos(3.0 * angle),
    ]
    for name in names:
        cycle = data[name].reshape(16, 12, 180, 360).mean(axis=0)
        mean = cycle.mean(axis=0)
        scale = cycle.std(axis=0)
        anomaly = (cycle - mean[None, ...]) / (scale[None, ...] + 1e-6)
        previous = np.roll(anomaly, 1, axis=0)
        feature_names.extend(
            (f"{name}:cycle", f"{name}:anomaly", f"{name}:previous", f"{name}:mean")
        )
        columns.extend(
            (
                cycle[months, train_rows, train_cols],
                anomaly[months, train_rows, train_cols],
                previous[months, train_rows, train_cols],
                mean[train_rows, train_cols],
            )
        )

    x = np.column_stack(columns).astype(np.float32)
    y = obs_allocation[months, train_rows, train_cols].astype(np.float32)
    cell_weight = obs_annual[train_rows, train_cols]
    weight = cell_weight + float(obs_annual.mean()) * 0.01
    print(f"rows={x.shape[0]} features={x.shape[1]} inputs={len(names)}", flush=True)

    evaluator = GFED5Evaluator(GFED5_PATH)

    def allocation_prediction(values: np.ndarray) -> np.ndarray:
        grid = np.zeros((12, 180, 360), dtype=np.float32)
        grid[months, train_rows, train_cols] = np.clip(values, 1e-9, None)
        grid /= grid.sum(axis=0, keepdims=True) + 1e-12
        cycle = incumbent_annual[None, ...] * grid
        return np.tile(cycle, (16, 1, 1)).astype(np.float32)

    best_tree: DecisionTreeRegressor | None = None
    best_score = -np.inf
    for leaves in (16, 32, 64, 128, 256):
        tree = DecisionTreeRegressor(
            max_leaf_nodes=leaves,
            min_samples_leaf=40,
            random_state=321,
        )
        tree.fit(x, y, sample_weight=weight)
        candidate = allocation_prediction(tree.predict(x))
        score = evaluator.score(candidate)["global"]["overall_score"]
        print(f"tree leaves={leaves} overall={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_tree = tree
    assert best_tree is not None
    print("inspectable tree rules", flush=True)
    print(export_text(best_tree, feature_names=feature_names, max_depth=6), flush=True)
    if "--tree-only" in sys.argv:
        return 0

    # Depth one is a nonlinear additive response model; depths two and three
    # quantify how much is gained by a small number of physical interactions.
    for depth in (1, 2, 3, 4):
        shallow = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.06,
            max_iter=400,
            max_depth=depth,
            max_leaf_nodes=None,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=370 + depth,
        )
        shallow.fit(x, y, sample_weight=weight)
        report(
            evaluator,
            f"fixed-map boosted depth={depth}",
            allocation_prediction(shallow.predict(x)),
        )

    learner = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=500,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=317,
        verbose=1,
    )
    learner.fit(x, y, sample_weight=weight)
    learned = np.clip(learner.predict(x), 1e-9, None)

    learned_grid = np.zeros((12, 180, 360), dtype=np.float32)
    learned_grid[months, train_rows, train_cols] = learned
    learned_grid /= learned_grid.sum(axis=0, keepdims=True) + 1e-12
    hybrid_cycle = incumbent_annual[None, ...] * learned_grid
    hybrid = np.tile(hybrid_cycle, (16, 1, 1)).astype(np.float32)

    report(evaluator, "incumbent", incumbent)
    report(evaluator, "fixed-map learned allocation", hybrid)

    # Combine the independently learned monthly allocation with a separate
    # no-geography annual propensity head to measure the joint information
    # ceiling before spending effort on mechanistic distillation.
    annual_feature_names = ["incumbent_annual", "log_incumbent_annual"]
    annual_columns = [
        incumbent_annual[rows, cols],
        np.log1p(10.0 * incumbent_annual[rows, cols]),
    ]
    for name in names:
        cycle = data[name].reshape(16, 12, 180, 360).mean(axis=0)
        annual_feature_names.extend(
            (f"{name}:mean", f"{name}:std", f"{name}:p10", f"{name}:p90")
        )
        annual_columns.extend(
            (
                cycle.mean(axis=0)[rows, cols],
                cycle.std(axis=0)[rows, cols],
                np.quantile(cycle, 0.1, axis=0)[rows, cols],
                np.quantile(cycle, 0.9, axis=0)[rows, cols],
            )
        )
    annual_x = np.column_stack(annual_columns).astype(np.float32)
    annual_y = obs_annual[rows, cols].astype(np.float32)
    annual_weight = annual_y + float(annual_y.mean()) * 0.02
    for label, annual_learner in (
        (
            "tree128",
            DecisionTreeRegressor(
                max_leaf_nodes=128, min_samples_leaf=25, random_state=341
            ),
        ),
        (
            "hgb63",
            HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.05,
                max_iter=500,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=331,
            ),
        ),
    ):
        annual_learner.fit(annual_x, annual_y, sample_weight=annual_weight)
        annual_learned = np.zeros((180, 360), dtype=np.float32)
        annual_learned[rows, cols] = np.clip(annual_learner.predict(annual_x), 0.0, 1.0)
        for blend in (0.50, 0.75, 1.0):
            joint_annual = (1.0 - blend) * incumbent_annual + blend * annual_learned
            joint = np.tile(joint_annual[None, ...] * learned_grid, (16, 1, 1))
            report(evaluator, f"joint {label} blend={blend:.2f}", joint.astype(np.float32))

    rng = np.random.default_rng(319)
    subset = rng.choice(x.shape[0], size=min(35_000, x.shape[0]), replace=False)
    importance = permutation_importance(
        learner,
        x[subset],
        y[subset],
        scoring="neg_mean_squared_error",
        n_repeats=2,
        random_state=319,
        sample_weight=weight[subset],
    )
    ranked = np.argsort(importance.importances_mean)[::-1][:24]
    print("top permutation features", flush=True)
    for index in ranked:
        print(
            f"{feature_names[index]}\t{importance.importances_mean[index]:.8f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
