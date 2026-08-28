"""Strict whole-cell diagnosis of per-cell anomaly-amplitude deficits.

The high-fire population and training weights are selected from the pinned
incumbent only.  GFED supplies the amplitude-ratio label and a separate
evaluation weight, never a carrier mask, feature, fold, or population choice.
All learner features are current or point-local prefix-causal inputs/states.
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "autoresearch" / "scratchpad"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

import deep_reverse_ml_121c83c as deep  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "2dd6d61"
CYCLE_DAYS = np.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64
)
MONTH_DAYS = np.tile(CYCLE_DAYS, 16)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}_amplitude")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def weighted_mean(values, weights) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_rmse(values, weights) -> float:
    return float(np.sqrt(weighted_mean(np.square(values), weights)))


def weighted_r2(target, prediction, weights) -> float:
    center = weighted_mean(target, weights)
    denominator = np.sum(weights * np.square(target - center))
    return float(1.0 - np.sum(weights * np.square(target - prediction)) / denominator)


def amplitude(cycle):
    anomaly = cycle - cycle.mean(axis=0, keepdims=True)
    return np.sqrt(
        np.sum(CYCLE_DAYS[:, None] * np.square(anomaly), axis=0)
        / CYCLE_DAYS.sum()
    )


def high_fire_cells(incumbent, area, fraction):
    annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    mass = np.maximum(annual, 0.0) * area
    order = np.argsort(mass.ravel())[::-1]
    cumulative = np.cumsum(mass.ravel()[order]) / mass.sum()
    return order[: int(np.searchsorted(cumulative, fraction) + 1)]


def unique_features(data, incumbent, rows, cols):
    names, matrix = deep.build_features(data, incumbent, rows, cols)
    keep = []
    seen = set()
    removed = []
    for index, name in enumerate(names):
        if name in seen:
            removed.append(name)
            continue
        seen.add(name)
        keep.append(index)
    return tuple(names[index] for index in keep), matrix[:, keep], tuple(removed)


def tree_counts(regressor, names):
    features = Counter()
    pairs = Counter()
    for stage in regressor._predictors:
        nodes = stage[0].nodes
        stack = [0]
        while stack:
            node = stack.pop()
            if nodes["is_leaf"][node]:
                continue
            feature = names[int(nodes["feature_idx"][node])]
            features[feature] += 1
            for child_name in ("left", "right"):
                child = int(nodes[child_name][node])
                if not nodes["is_leaf"][child]:
                    child_feature = names[int(nodes["feature_idx"][child])]
                    if child_feature != feature:
                        pairs[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)
    return features, pairs


def stable_main_effects(matrix, weights, row_folds, names, models, feature_counts):
    common = set(feature_counts[0])
    for counts in feature_counts[1:]:
        common &= set(counts)
    ranked = sorted(
        common,
        key=lambda name: sum(counts[name] for counts in feature_counts),
        reverse=True,
    )
    rng = np.random.default_rng(28082026)
    print(f"STABLE_FEATURES count={len(ranked)}")
    for rank, name in enumerate(ranked[:18], start=1):
        column = names.index(name)
        effects = []
        curves = []
        for fold, learner in enumerate(models):
            train = row_folds != fold
            eligible = np.flatnonzero(row_folds == fold)
            probability = weights[eligible].astype(np.float64)
            probability /= probability.sum()
            sample_index = rng.choice(
                eligible,
                size=min(1800, eligible.size),
                replace=False,
                p=probability,
            )
            sample = matrix[sample_index].copy()
            grid = np.quantile(matrix[train, column], (0.10, 0.50, 0.90))
            values = []
            for setting in grid:
                probe = sample.copy()
                probe[:, column] = setting
                values.append(float(np.mean(learner.predict(probe))))
            effects.append(values[2] - values[0])
            curves.append(values)
        stable = min(effects) > 0.0 or max(effects) < 0.0
        print(
            f"FEATURE rank={rank} name={name} "
            f"counts={','.join(str(counts[name]) for counts in feature_counts)} "
            f"high_minus_low={','.join(f'{value:+.5f}' for value in effects)} stable={stable} "
            f"curves_p10_p50_p90={';'.join(','.join(f'{value:+.5f}' for value in curve) for curve in curves)}"
        )


def stable_interactions(matrix, weights, row_folds, names, models, pair_counts):
    common = set(pair_counts[0])
    for counts in pair_counts[1:]:
        common &= set(counts)
    ranked = sorted(
        common,
        key=lambda pair: sum(counts[pair] for counts in pair_counts),
        reverse=True,
    )
    rng = np.random.default_rng(882026)
    print(f"STABLE_TREE_PAIRS count={len(ranked)}")
    for rank, pair in enumerate(ranked[:20], start=1):
        left = names.index(pair[0])
        right = names.index(pair[1])
        contrasts = []
        for fold, learner in enumerate(models):
            train = row_folds != fold
            eligible = np.flatnonzero(row_folds == fold)
            probability = weights[eligible].astype(np.float64)
            probability /= probability.sum()
            sample_index = rng.choice(
                eligible,
                size=min(1800, eligible.size),
                replace=False,
                p=probability,
            )
            sample = matrix[sample_index].copy()
            left_low, left_high = np.quantile(matrix[train, left], (0.25, 0.75))
            right_low, right_high = np.quantile(matrix[train, right], (0.25, 0.75))
            predictions = {}
            for left_key, left_value in (("l", left_low), ("h", left_high)):
                for right_key, right_value in (("l", right_low), ("h", right_high)):
                    probe = sample.copy()
                    probe[:, left] = left_value
                    probe[:, right] = right_value
                    predictions[left_key + right_key] = learner.predict(probe)
            contrasts.append(
                float(
                    np.mean(
                        predictions["hh"]
                        - predictions["hl"]
                        - predictions["lh"]
                        + predictions["ll"]
                    )
                )
            )
        stable = min(contrasts) > 0.0 or max(contrasts) < 0.0
        print(
            f"PAIR rank={rank} features={pair[0]}*{pair[1]} "
            f"counts={','.join(str(counts[pair]) for counts in pair_counts)} "
            f"interaction={','.join(f'{value:+.5f}' for value in contrasts)} stable={stable}"
        )


def intensity_strata(target, prediction, model_annual, model_mass, reference_mass, folds):
    boundaries = np.quantile(np.log10(model_annual + 1.0e-8), (0.0, 0.25, 0.5, 0.75, 1.0))
    intensity = np.log10(model_annual + 1.0e-8)
    for index in range(4):
        if index == 3:
            selected = (intensity >= boundaries[index]) & (intensity <= boundaries[index + 1])
        else:
            selected = (intensity >= boundaries[index]) & (intensity < boundaries[index + 1])
        fold_r2 = []
        for fold in range(4):
            held = selected & (folds == fold)
            fold_r2.append(weighted_r2(target[held], prediction[held], model_mass[held]))
        print(
            f"INTENSITY_STRATUM quartile={index+1} cells={int(selected.sum())} "
            f"log10_model_annual={boundaries[index]:+.5f},{boundaries[index+1]:+.5f} "
            f"model_weighted_target={weighted_mean(target[selected], model_mass[selected]):+.7f} "
            f"reference_weighted_target={weighted_mean(target[selected], reference_mass[selected]):+.7f} "
            f"model_r2={weighted_r2(target[selected], prediction[selected], model_mass[selected]):+.7f} "
            f"fold_r2={','.join(f'{value:+.7f}' for value in fold_r2)}"
        )


def main():
    global model
    model = load_pinned()
    deep.model = model
    current_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pinned_blob = subprocess.run(
        ["git", "rev-parse", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != pinned_blob:
        raise RuntimeError(f"current model blob {current_blob} differs from {PINNED} blob {pinned_blob}")

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))

    cells = high_fire_cells(incumbent, area, 0.85)
    rows, cols = cells // 360, cells % 360
    cell_folds = ((rows // 15) + 3 * (cols // 15)) % 4
    row_folds = np.tile(cell_folds, 192)
    selected_model = np.asarray(incumbent[:, rows, cols], dtype=np.float64)
    selected_observed = np.asarray(observed[:, rows, cols], dtype=np.float64)
    model_annual = np.average(selected_model, axis=0, weights=MONTH_DAYS)
    reference_annual = np.average(selected_observed, axis=0, weights=MONTH_DAYS)
    model_mass = area[rows, cols] * model_annual
    reference_mass = area[rows, cols] * reference_annual
    model_cycle = selected_model.reshape(16, 12, -1).mean(axis=0)
    reference_cycle = selected_observed.reshape(16, 12, -1).mean(axis=0)
    model_amplitude = amplitude(model_cycle)
    reference_amplitude = amplitude(reference_cycle)

    epsilon = 1.0e-5
    raw_target = np.log(
        (reference_amplitude + epsilon) / (model_amplitude + epsilon)
    )
    cell_target = np.clip(raw_target, -3.0, 3.0).astype(np.float32)
    row_target = np.tile(cell_target, 192)
    names, matrix, removed = unique_features(data, incumbent, rows, cols)
    training_mode = "reference" if "--reference-training" in sys.argv else "model"
    training_cell_weight = reference_mass if training_mode == "reference" else model_mass
    row_weight = np.tile(training_cell_weight, 192).astype(np.float64)
    row_weight *= np.repeat(MONTH_DAYS, cells.size)
    row_weight /= row_weight.mean()

    all_model_mass = area * np.average(incumbent, axis=0, weights=MONTH_DAYS)
    all_reference_mass = area * np.average(observed, axis=0, weights=MONTH_DAYS)
    print(
        f"IDENTITY pinned={PINNED} model_blob={current_blob} cells={cells.size} rows={matrix.shape[0]} "
        f"features={matrix.shape[1]} removed_duplicate={','.join(removed) or 'none'} "
        f"model_mass_coverage={model_mass.sum()/all_model_mass.sum():.7f} "
        f"reference_mass_coverage={reference_mass.sum()/all_reference_mass.sum():.7f} "
        f"fold_cells={','.join(str(int(np.sum(cell_folds == fold))) for fold in range(4))} "
        f"training_weight={training_mode} target_selected_population=0 "
        "target_selected_carrier=0 coordinate_features=0"
    )
    print(
        f"TARGET model_weighted_mean_log_ratio={weighted_mean(raw_target, model_mass):+.7f} "
        f"reference_weighted_mean_log_ratio={weighted_mean(raw_target, reference_mass):+.7f} "
        f"model_weighted_required_geomean={np.exp(weighted_mean(raw_target, model_mass)):.7f} "
        f"reference_weighted_required_geomean={np.exp(weighted_mean(raw_target, reference_mass)):.7f} "
        f"model_underamplitude_weight={np.sum(model_mass[raw_target>0])/np.sum(model_mass):.7f} "
        f"reference_underamplitude_weight={np.sum(reference_mass[raw_target>0])/np.sum(reference_mass):.7f} "
        f"clipped_cells={int(np.sum((raw_target < -3.0) | (raw_target > 3.0)))} "
        f"zero_reference_amplitude={int(np.sum(reference_amplitude < 1e-8))}"
    )

    oof_rows = np.empty_like(row_target, dtype=np.float32)
    models = []
    feature_counts = []
    pair_counts = []
    for fold in range(4):
        train = row_folds != fold
        held = ~train
        learner = HistGradientBoostingRegressor(
            max_depth=4,
            max_iter=160,
            learning_rate=0.05,
            min_samples_leaf=250,
            l2_regularization=3.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=8282026 + 31 * fold,
        )
        learner.fit(matrix[train], row_target[train], sample_weight=row_weight[train])
        oof_rows[held] = learner.predict(matrix[held]).astype(np.float32)
        features, pairs = tree_counts(learner, names)
        models.append(learner)
        feature_counts.append(features)
        pair_counts.append(pairs)
        print(
            f"FOLD_ROW fold={fold} iterations={learner.n_iter_} "
            f"r2={weighted_r2(row_target[held], oof_rows[held], row_weight[held]):+.7f} "
            f"top_features={','.join(name for name, _ in features.most_common(10))}"
        )

    oof_monthly = oof_rows.reshape(192, cells.size)
    oof_cell = np.average(oof_monthly, axis=0, weights=MONTH_DAYS)
    for label, weights in (("MODEL", model_mass), ("REFERENCE", reference_mass)):
        fold_values = []
        for fold in range(4):
            held = cell_folds == fold
            fold_values.append(weighted_r2(cell_target[held], oof_cell[held], weights[held]))
        print(
            f"CELL_OOF weighting={label} r2={weighted_r2(cell_target, oof_cell, weights):+.7f} "
            f"baseline_rmse={weighted_rmse(cell_target, weights):.7f} "
            f"oof_rmse={weighted_rmse(cell_target-oof_cell, weights):.7f} "
            f"fold_r2={','.join(f'{value:+.7f}' for value in fold_values)}"
        )
    intensity_strata(
        cell_target,
        oof_cell,
        model_annual,
        model_mass,
        reference_mass,
        cell_folds,
    )
    stable_main_effects(matrix, row_weight, row_folds, names, models, feature_counts)
    stable_interactions(matrix, row_weight, row_folds, names, models, pair_counts)

    for strength in (0.25, 0.50, 1.00):
        residual = cell_target - strength * oof_cell
        fold_gain = []
        for fold in range(4):
            held = cell_folds == fold
            fold_gain.append(
                weighted_rmse(cell_target[held], reference_mass[held])
                - weighted_rmse(residual[held], reference_mass[held])
            )
        print(
            f"AMPLITUDE_RATIO_HEADROOM strength={strength:.2f} "
            f"reference_rmse_gain={weighted_rmse(cell_target, reference_mass)-weighted_rmse(residual, reference_mass):+.7f} "
            f"fold_rmse_gain={','.join(f'{value:+.7f}' for value in fold_gain)}"
        )


if __name__ == "__main__":
    main()
