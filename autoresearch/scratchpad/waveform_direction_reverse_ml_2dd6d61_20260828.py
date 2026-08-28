"""Whole-cell reverse-ML diagnosis of monthly waveform-direction residuals.

This script is deliberately diagnostic.  It pins the incumbent to 2dd6d61,
selects the high-fire footprint using incumbent fire mass alone, and uses GFED
only as the supervised residual label and evaluation weight.  Learner features
contain no coordinates, region, cell identity, calendar harmonics, future
summary, or benchmark-derived value.  Coordinates are used only for disjoint
15-degree whole-cell folds.  Incumbent-cycle phase labels are used only after
prediction to stratify onset, peak, and recession diagnostics.
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
ROLE_NAMES = ("onset", "peak", "recession")


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weighted_mean(np.square(values), weights)))


def weighted_r2(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    center = weighted_mean(target, weights)
    denominator = np.sum(weights * np.square(target - center))
    return float(1.0 - np.sum(weights * np.square(target - prediction)) / denominator)


def direction(cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluator-aligned direction: unweighted centering, day-weighted RMS."""
    mean = cycle.mean(axis=0)
    anomaly = cycle - mean[None, :]
    amplitude = np.sqrt(
        np.sum(CYCLE_DAYS[:, None] * np.square(anomaly), axis=0) / CYCLE_DAYS.sum()
    )
    unit = anomaly / np.maximum(amplitude[None, :], 1.0e-8)
    return unit, amplitude, mean


def incumbent_roles(unit_model: np.ndarray) -> np.ndarray:
    """Target-free retrospective strata; never passed to the learner."""
    peak = np.argmax(unit_model, axis=0)
    trough = np.argmin(unit_model, axis=0)
    month = np.arange(12)[:, None]
    from_trough = (month - trough[None, :]) % 12
    peak_distance = (peak - trough) % 12
    roles = np.full(unit_model.shape, 2, dtype=np.int8)
    roles[(from_trough > 0) & (from_trough < peak_distance[None, :])] = 0
    roles[month == peak[None, :]] = 1
    return roles


def high_fire_cells(incumbent: np.ndarray, area: np.ndarray, fraction: float) -> np.ndarray:
    annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    mass = np.maximum(annual, 0.0) * area
    order = np.argsort(mass.ravel())[::-1]
    cumulative = np.cumsum(mass.ravel()[order]) / mass.sum()
    return order[: int(np.searchsorted(cumulative, fraction) + 1)]


def unique_features(data, incumbent, rows, cols):
    """Reuse the audited causal feature builder, removing its duplicated column."""
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


def tree_counts(regressor, names: tuple[str, ...]):
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


def role_metrics(label, target, prediction, weights, roles, folds):
    for role_code, role_name in enumerate(ROLE_NAMES):
        selected = roles == role_code
        baseline = weighted_rmse(target[selected], weights[selected])
        residual = weighted_rmse(target[selected] - prediction[selected], weights[selected])
        values = []
        for fold in range(4):
            held = selected & (folds == fold)
            values.append(weighted_r2(target[held], prediction[held], weights[held]))
        print(
            f"{label}_ROLE role={role_name} rows={int(selected.sum())} "
            f"target_mean={weighted_mean(target[selected], weights[selected]):+.7f} "
            f"baseline_rmse={baseline:.7f} oof_rmse={residual:.7f} "
            f"r2={weighted_r2(target[selected], prediction[selected], weights[selected]):+.7f} "
            f"fold_r2={','.join(f'{value:+.7f}' for value in values)}"
        )


def role_partial_interactions(
    matrix,
    target_free_weight,
    row_folds,
    row_roles,
    names,
    models,
    pair_counts,
):
    common = set(pair_counts[0])
    for counts in pair_counts[1:]:
        common &= set(counts)
    ranked = sorted(
        common,
        key=lambda pair: sum(counts[pair] for counts in pair_counts),
        reverse=True,
    )
    print(f"STABLE_TREE_PAIRS count={len(ranked)}")
    rng = np.random.default_rng(22082026)
    for rank, pair in enumerate(ranked[:12], start=1):
        left = names.index(pair[0])
        right = names.index(pair[1])
        role_output = []
        for role_code, role_name in enumerate(ROLE_NAMES):
            contrasts = []
            for fold, learner in enumerate(models):
                train = row_folds != fold
                eligible = np.flatnonzero((row_folds == fold) & (row_roles == role_code))
                probability = target_free_weight[eligible].astype(np.float64)
                probability /= probability.sum()
                sample_index = rng.choice(
                    eligible,
                    size=min(1200, eligible.size),
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
            role_output.append(
                f"{role_name}={','.join(f'{value:+.5f}' for value in contrasts)}:{stable}"
            )
        print(
            f"PAIR rank={rank} features={pair[0]}*{pair[1]} "
            f"counts={','.join(str(counts[pair]) for counts in pair_counts)} "
            + " ".join(role_output)
        )


def normalized_headroom(
    unit_observed,
    unit_model,
    oof,
    cell_folds,
    cycle_roles,
    reference_cell_weight,
):
    cells = unit_model.shape[1]
    correction = oof.reshape(16, 12, cells).mean(axis=0)
    baseline_error = unit_observed - unit_model
    for strength in (0.25, 0.50, 1.00):
        trial = unit_model + strength * correction
        trial -= trial.mean(axis=0, keepdims=True)
        trial /= np.maximum(
            np.sqrt(
                np.sum(CYCLE_DAYS[:, None] * np.square(trial), axis=0)
                / CYCLE_DAYS.sum()
            )[None, :],
            1.0e-8,
        )
        error = unit_observed - trial
        total_weight = CYCLE_DAYS[:, None] * reference_cell_weight[None, :]
        total_gain = weighted_rmse(baseline_error, total_weight) - weighted_rmse(error, total_weight)
        fold_gain = []
        for fold in range(4):
            held = cell_folds == fold
            fold_gain.append(
                weighted_rmse(baseline_error[:, held], total_weight[:, held])
                - weighted_rmse(error[:, held], total_weight[:, held])
            )
        role_gain = []
        for role_code, role_name in enumerate(ROLE_NAMES):
            selected = cycle_roles == role_code
            role_gain.append(
                f"{role_name}:{weighted_rmse(baseline_error[selected], total_weight[selected]) - weighted_rmse(error[selected], total_weight[selected]):+.7f}"
            )
        print(
            f"NORMALIZED_DIRECTION_HEADROOM strength={strength:.2f} total_rmse_gain={total_gain:+.7f} "
            f"fold_rmse_gain={','.join(f'{value:+.7f}' for value in fold_gain)} "
            f"role_rmse_gain={','.join(role_gain)}"
        )


def main() -> None:
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
    observed_annual = np.average(selected_observed, axis=0, weights=MONTH_DAYS)
    model_mass = area[rows, cols] * model_annual
    reference_mass = area[rows, cols] * observed_annual

    model_cycle = selected_model.reshape(16, 12, -1).mean(axis=0)
    observed_cycle = selected_observed.reshape(16, 12, -1).mean(axis=0)
    unit_model, amplitude_model, _ = direction(model_cycle)
    unit_observed, amplitude_observed, _ = direction(observed_cycle)
    cycle_target = unit_observed - unit_model
    target = np.tile(cycle_target, (16, 1)).reshape(-1).astype(np.float32)
    cycle_roles = incumbent_roles(unit_model)
    row_roles = np.tile(cycle_roles, (16, 1)).reshape(-1)

    names, matrix, removed = unique_features(data, incumbent, rows, cols)
    target_free_weight = np.tile(CYCLE_DAYS[:, None] * model_mass[None, :], (16, 1, 1)).reshape(-1)
    reference_weight = np.tile(CYCLE_DAYS[:, None] * reference_mass[None, :], (16, 1, 1)).reshape(-1)
    target_free_weight /= target_free_weight.mean()
    reference_weight /= reference_weight.mean()
    all_model_mass = area * np.average(incumbent, axis=0, weights=MONTH_DAYS)
    all_reference_mass = area * np.average(observed, axis=0, weights=MONTH_DAYS)
    print(
        f"IDENTITY pinned={PINNED} model_blob={current_blob} cells={cells.size} rows={matrix.shape[0]} "
        f"features={matrix.shape[1]} removed_duplicate={','.join(removed) or 'none'} "
        f"model_mass_coverage={model_mass.sum()/all_model_mass.sum():.7f} "
        f"reference_mass_coverage={reference_mass.sum()/all_reference_mass.sum():.7f} "
        f"fold_cells={','.join(str(int(np.sum(cell_folds == fold))) for fold in range(4))}"
    )
    print(
        f"AMPLITUDE model_weighted_mean={weighted_mean(amplitude_model, model_mass):.8f} "
        f"observed_model_weighted_mean={weighted_mean(amplitude_observed, model_mass):.8f} "
        f"observed_reference_weighted_mean={weighted_mean(amplitude_observed, reference_mass):.8f} "
        f"observed_amp_lt_1e-8={int(np.sum(amplitude_observed < 1e-8))} "
        f"model_amp_lt_1e-8={int(np.sum(amplitude_model < 1e-8))}"
    )

    oof = np.empty_like(target, dtype=np.float32)
    models = []
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
            random_state=2282026 + 31 * fold,
        )
        learner.fit(matrix[train], target[train], sample_weight=target_free_weight[train])
        oof[held] = learner.predict(matrix[held]).astype(np.float32)
        features, pairs = tree_counts(learner, names)
        models.append(learner)
        pair_counts.append(pairs)
        print(
            f"FOLD fold={fold} iterations={learner.n_iter_} "
            f"target_free_r2={weighted_r2(target[held], oof[held], target_free_weight[held]):+.7f} "
            f"reference_r2={weighted_r2(target[held], oof[held], reference_weight[held]):+.7f} "
            f"top_features={','.join(name for name, _ in features.most_common(10))}"
        )
    print(
        f"OOF target_free_r2={weighted_r2(target, oof, target_free_weight):+.7f} "
        f"reference_r2={weighted_r2(target, oof, reference_weight):+.7f} "
        f"baseline_reference_rmse={weighted_rmse(target, reference_weight):.7f} "
        f"oof_reference_rmse={weighted_rmse(target-oof, reference_weight):.7f}"
    )
    role_metrics("TARGET_FREE", target, oof, target_free_weight, row_roles, row_folds)
    role_metrics("REFERENCE", target, oof, reference_weight, row_roles, row_folds)
    role_partial_interactions(
        matrix,
        target_free_weight,
        row_folds,
        row_roles,
        names,
        models,
        pair_counts,
    )
    normalized_headroom(
        unit_observed,
        unit_model,
        oof,
        cell_folds,
        cycle_roles,
        reference_mass,
    )


if __name__ == "__main__":
    main()
