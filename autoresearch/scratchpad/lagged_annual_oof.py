"""Test a causal lagged-year annual fire-capacity learner.

The diagnostic holds out complete land cells.  For each year after 2001 it
predicts the observed annual burned fraction from only the model output and
local environmental state accumulated during the preceding year.  The tree is
diagnostic: stable interactions are candidates for smooth process equations,
but fitted leaves are not promoted into ``model.py``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
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
    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360

    baseline = np.asarray(incumbent[:, rows, cols].T, dtype=np.float64)
    baseline_year = baseline.reshape(cells.size, 16, 12).sum(axis=2)
    prior_baseline = np.column_stack(
        (baseline_year[:, 0], baseline_year[:, :-1])
    )
    names = ["prior_model_annual", "log_prior_model_annual"]
    columns = [prior_baseline, np.log1p(10.0 * prior_baseline)]

    for name in model.INPUTS:
        values = np.asarray(data[name][:, rows, cols].T, dtype=np.float64)
        yearly = values.reshape(cells.size, 16, 12)
        prior = np.concatenate((yearly[:, :1], yearly[:, :-1]), axis=1)
        names.extend(
            (
                f"{name}:prior_mean",
                f"{name}:prior_std",
                f"{name}:prior_p10",
                f"{name}:prior_p90",
            )
        )
        columns.extend(
            (
                prior.mean(axis=2),
                prior.std(axis=2),
                np.quantile(prior, 0.1, axis=2),
                np.quantile(prior, 0.9, axis=2),
            )
        )

    x = np.column_stack([column.reshape(-1) for column in columns]).astype(
        np.float32
    )
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    observed_cells = np.asarray(observed[:, rows, cols].T, dtype=np.float64)
    observed_year = observed_cells.reshape(cells.size, 16, 12).sum(axis=2)
    target = (observed_year / (prior_baseline + 1e-4)).reshape(-1)
    weights = (prior_baseline + float(prior_baseline.mean()) * 0.02).reshape(-1)

    years = np.tile(np.arange(16), cells.size)
    usable = years > 0
    rng = np.random.default_rng(937)
    if "--bin-top" in sys.argv:
        name_to_index = {name: index for index, name in enumerate(names)}
        sample_pool = np.flatnonzero(usable)
        sample = rng.choice(
            sample_pool, size=min(250_000, sample_pool.size), replace=False
        )

        def ratio_matrix(left_name: str, right_name: str) -> None:
            left = x[sample, name_to_index[left_name]]
            right = x[sample, name_to_index[right_name]]
            quantiles = np.linspace(0.0, 1.0, 7)
            left_edges = np.unique(np.quantile(left, quantiles))
            right_edges = np.unique(np.quantile(right, quantiles))
            left_bins = np.clip(
                np.searchsorted(left_edges, left, side="right") - 1,
                0,
                left_edges.size - 2,
            )
            right_bins = np.clip(
                np.searchsorted(right_edges, right, side="right") - 1,
                0,
                right_edges.size - 2,
            )
            matrix = np.full(
                (left_edges.size - 1, right_edges.size - 1), np.nan
            )
            for left_bin in range(matrix.shape[0]):
                for right_bin in range(matrix.shape[1]):
                    selected = (
                        (left_bins == left_bin) & (right_bins == right_bin)
                    )
                    if selected.sum() < 100:
                        continue
                    selected_rows = sample[selected]
                    matrix[left_bin, right_bin] = np.average(
                        target[selected_rows], weights=weights[selected_rows]
                    )
            print(f"ratio matrix {left_name} x {right_name}", flush=True)
            print("left_edges=" + np.array2string(left_edges, precision=6), flush=True)
            print("right_edges=" + np.array2string(right_edges, precision=6), flush=True)
            print(np.array2string(matrix, precision=3), flush=True)

        for pair in (
            ("prior_model_annual", "luh2_cropland_fraction:prior_mean"),
            ("prior_model_annual", "lightning_flash_rate:prior_std"),
            ("prior_model_annual", "annual_precipitation:prior_mean"),
            ("prior_model_annual", "natural_vegetation_fraction:prior_p10"),
            ("annual_precipitation:prior_mean", "lightning_flash_rate:prior_std"),
            ("air_temperature:prior_mean", "air_temperature:prior_std"),
            ("aboveground_biomass:prior_mean", "leaf_area_index:prior_p90"),
        ):
            ratio_matrix(*pair)
        return 0

    cell_folds = rng.integers(0, 3, size=cells.size)
    folds = np.repeat(cell_folds, 16)
    out_of_fold = np.ones_like(target)
    trained: HistGradientBoostingRegressor | None = None
    held_for_importance: np.ndarray | None = None
    for fold in range(3):
        train = np.flatnonzero((folds != fold) & usable)
        held = np.flatnonzero((folds == fold) & usable)
        learner = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=16,
            min_samples_leaf=100,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=941 + fold,
        )
        learner.fit(x[train], target[train], sample_weight=weights[train])
        out_of_fold[held] = learner.predict(x[held])
        trained = learner
        held_for_importance = held
        print(
            f"completed fold={fold} train={train.size} held={held.size} "
            f"iterations={learner.n_iter_}",
            flush=True,
        )

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    factors = np.clip(out_of_fold.reshape(cells.size, 16), 0.1, 10.0)
    for strength in (0.10, 0.25, 0.50, 0.75, 1.0):
        corrected = baseline.reshape(cells.size, 16, 12) * np.power(
            factors[:, :, None], strength
        )
        candidate = incumbent.copy()
        candidate[:, rows, cols] = np.clip(
            corrected.reshape(cells.size, 192).T, 0.0, 1.0
        )
        report(evaluator, f"lagged-year HGB OOF strength={strength}", candidate)

    assert trained is not None and held_for_importance is not None
    importance_rows = rng.choice(
        held_for_importance,
        size=min(80_000, held_for_importance.size),
        replace=False,
    )
    importance = permutation_importance(
        trained,
        x[importance_rows],
        target[importance_rows],
        scoring="neg_mean_poisson_deviance",
        n_repeats=3,
        random_state=947,
        sample_weight=weights[importance_rows],
    )
    print("lagged-year HGB permutation importance", flush=True)
    for index in np.argsort(importance.importances_mean)[::-1][:30]:
        print(f"{names[index]}\t{importance.importances_mean[index]:+.9f}", flush=True)

    pair_gain: dict[tuple[int, int], float] = defaultdict(float)
    for stage in trained._predictors:
        nodes = stage[0].nodes
        splits = nodes[nodes["is_leaf"] == 0]
        features = sorted(set(int(index) for index in splits["feature_idx"]))
        gain = float(np.maximum(splits["gain"], 0.0).sum())
        for left, right in combinations(features, 2):
            pair_gain[(left, right)] += gain
    print("lagged-year HGB recurring feature pairs", flush=True)
    for (left, right), gain in sorted(
        pair_gain.items(), key=lambda item: item[1], reverse=True
    )[:30]:
        print(f"{names[left]} x {names[right]}\t{gain:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
