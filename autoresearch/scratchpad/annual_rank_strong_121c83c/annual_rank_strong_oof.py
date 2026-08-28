"""Strong, strict spatial-block reverse-ML audit of annual cell ranking.

This is a diagnostic against the canonical ``121c83c`` model, not a candidate
model.  The learned target is the annual log residual, but no target-derived
field, incumbent prediction, coordinate, region, cell identity, calendar label,
completed-record summary, or future value is a learner input.  Predictor rows
contain only clean coupled-valid exogenous fields and point-local current or
prefix-causal summaries.  Coordinates assign disjoint whole-cell folds only.

Unlike the earlier target-selected 4,463-cell audit, OOF correction is emitted
for every cell in the fixed input-derived land mask.  Every corrected land cell
is predicted by a learner that saw no row from that cell's spatial fold.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "121c83c"
EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
EXPECTED_INCUMBENT = 0.719892388
CLEAN_INPUTS = (
    "dryness",
    "monthly_precipitation",
    "air_temperature",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_primary_fraction",
    "luh2_secondary_fraction",
    "luh2_urban_fraction",
    "lightning_flash_rate",
)
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0
FIT_MONTHS = np.arange(1, 192, 4, dtype=np.int64)
STRENGTHS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


def load_pinned():
    source = subprocess.run(
        ("git", "show", f"{PINNED}:autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    blob = subprocess.run(
        ("git", "hash-object", "--stdin"),
        cwd=ROOT,
        input=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"unexpected pinned model blob {blob}")
    module = types.ModuleType(f"model_{PINNED}_annual_rank_strong")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, blob


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    """Point-local causal EMA initialized at the first coupled timestep."""
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(values[0], dtype=np.float32).copy()
    output = np.empty_like(values, dtype=np.float32)
    for step in range(values.shape[0]):
        state += alpha * (values[step] - state)
        output[step] = state
    return output


def build_features(data, rows: np.ndarray, columns: np.ndarray):
    """Build only clean, geography-free current or prefix-causal features."""
    names: list[str] = []
    feature_columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        expected = (192, rows.size)
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape}, expected {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} is not finite")
        names.append(name)
        feature_columns.append(values.reshape(-1))

    selected = {
        name: np.asarray(data[name][:, rows, columns], dtype=np.float32)
        for name in CLEAN_INPUTS
    }
    rain = np.clip(selected["monthly_precipitation"], 0.0, None)
    dryness = np.clip(selected["dryness"], 0.0, None)
    temperature = selected["air_temperature"]
    lightning = np.clip(selected["lightning_flash_rate"], 0.0, None)

    add("log_rain", np.log1p(rain))
    add("log_dryness", np.log1p(dryness))
    add("temperature", temperature)
    add("log_lightning", np.log1p(1000.0 * lightning))

    memories: dict[tuple[str, int], np.ndarray] = {}
    for name, values, offset, multiplier in (
        ("rain", rain, 10.0, 1.0),
        ("dryness", dryness, 100.0, 1.0),
        ("temperature", temperature, 10.0, 1.0),
        ("lightning", lightning, 0.002, 1000.0),
    ):
        for months in (3, 12, 24):
            memory = antecedent(values, float(months))
            memories[(name, months)] = memory
            if name == "temperature":
                add(f"temperature_ema{months}", memory)
            else:
                add(f"log_{name}_ema{months}", np.log1p(multiplier * np.clip(memory, 0.0, None)))
            add(
                f"{name}_departure{months}",
                (values - memory) / (np.abs(values) + np.abs(memory) + offset),
            )

    for name in (
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_primary_fraction",
        "luh2_secondary_fraction",
        "luh2_urban_fraction",
    ):
        add(name, np.clip(selected[name], 0.0, None))

    rain3 = memories[("rain", 3)]
    rain12 = memories[("rain", 12)]
    dry3 = memories[("dryness", 3)]
    temp3 = memories[("temperature", 3)]
    temp12 = memories[("temperature", 12)]
    light3 = memories[("lightning", 3)]
    light12 = memories[("lightning", 12)]
    annualized_rain = 12.0 * rain12
    crop = np.clip(selected["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(selected["luh2_urban_fraction"], 0.0, 1.0)

    add("log_causal_annualized_rain", np.log1p(annualized_rain))
    add("rain_deficit3", np.maximum((rain3 - rain) / (rain3 + rain + 10.0), 0.0))
    add("rain_deficit12", np.maximum((rain12 - rain) / (rain12 + rain + 10.0), 0.0))
    add("rain_wet_anomaly12", np.maximum((rain - rain12) / (rain + rain12 + 10.0), 0.0))
    add("dryness_rise3", np.maximum((dryness - dry3) / (dryness + dry3 + 100.0), 0.0))
    add("warming3", temperature - temp3)
    add("warming12", temperature - temp12)
    add(
        "temperature_variability12",
        np.sqrt(
            np.maximum(
                antecedent(np.square(temperature), 12.0) - np.square(temp12),
                0.0,
            )
        ),
    )
    add(
        "lightning_pulse3",
        np.maximum((lightning - light3) / (lightning + light3 + 0.002), 0.0),
    )
    add(
        "lightning_coherence12",
        light12 / (light12 + 0.02)
        * 4.0 / (4.0 + np.sqrt(np.maximum(
            antecedent(np.square(temperature), 12.0) - np.square(temp12), 0.0
        ))),
    )
    add(
        "annual_fuel_climate",
        annualized_rain / (annualized_rain + 260.0) * np.exp(-annualized_rain / 3200.0),
    )
    add(
        "combustion_opportunity",
        dryness / (dryness + 110.0) / (1.0 + rain / 42.0),
    )
    add("fragmentation_pressure", np.clip(crop + 4.0 * urban, 0.0, 4.0))

    matrix = np.column_stack(feature_columns).astype(np.float32, copy=False)
    return tuple(names), matrix


def generic_tree_counts(regressor, names: tuple[str, ...]):
    feature_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    def walk(children_left, children_right, features) -> None:
        stack = [0]
        while stack:
            node = stack.pop()
            if children_left[node] == children_right[node]:
                continue
            feature = names[int(features[node])]
            feature_counts[feature] += 1
            for child in (int(children_left[node]), int(children_right[node])):
                if children_left[child] != children_right[child]:
                    child_feature = names[int(features[child])]
                    if child_feature != feature:
                        pair_counts[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)

    if isinstance(regressor, HistGradientBoostingRegressor):
        for stage in regressor._predictors:
            nodes = stage[0].nodes
            stack = [0]
            while stack:
                node = stack.pop()
                if nodes["is_leaf"][node]:
                    continue
                feature = names[int(nodes["feature_idx"][node])]
                feature_counts[feature] += 1
                for key in ("left", "right"):
                    child = int(nodes[key][node])
                    if not nodes["is_leaf"][child]:
                        child_feature = names[int(nodes["feature_idx"][child])]
                        if child_feature != feature:
                            pair_counts[tuple(sorted((feature, child_feature)))] += 1
                    stack.append(child)
    else:
        for estimator in regressor.estimators_:
            tree = estimator.tree_
            walk(tree.children_left, tree.children_right, tree.feature)
    return feature_counts, pair_counts


def make_learner(label: str, seed: int):
    if label == "deep_hgb":
        return HistGradientBoostingRegressor(
            max_iter=280,
            learning_rate=0.04,
            max_leaf_nodes=63,
            min_samples_leaf=120,
            l2_regularization=3.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    if label == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=96,
            max_depth=18,
            min_samples_leaf=80,
            max_features=0.8,
            bootstrap=False,
            n_jobs=-1,
            random_state=seed,
        )
    if label == "random_forest":
        return RandomForestRegressor(
            n_estimators=72,
            max_depth=16,
            min_samples_leaf=100,
            max_features=0.75,
            bootstrap=True,
            max_samples=0.75,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(label)


def weighted_r2(target, prediction, weight) -> float:
    return float(r2_score(target, prediction, sample_weight=weight))


def held_losses(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    observed_annual: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    weight = area * observed_annual
    observed_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    predicted_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    observed_allocation = observed_cycle / (observed_cycle.sum(axis=0, keepdims=True) + 1e-12)
    predicted_allocation = predicted_cycle / (predicted_cycle.sum(axis=0, keepdims=True) + 1e-12)
    annual, allocation, raw_cycle = [], [], []
    for fold in range(4):
        held = folds == fold
        held_weight = weight[held]
        denominator = np.sum(held_weight) + 1e-15
        annual.append(np.sqrt(np.sum(
            held_weight * np.square(
                np.log(observed_annual[held] + 1e-5)
                - np.log(predicted_annual[held] + 1e-5)
            )
        ) / denominator))
        allocation.append(np.sqrt(np.sum(
            held_weight[None, :] * np.square(
                observed_allocation[:, held] - predicted_allocation[:, held]
            )
        ) / (12.0 * denominator)))
        raw_cycle.append(np.sqrt(np.sum(
            held_weight[None, :] * np.square(
                observed_cycle[:, held] - predicted_cycle[:, held]
            )
        ) / (12.0 * denominator)))
    return np.asarray(annual), np.asarray(allocation), np.asarray(raw_cycle)


def apply_correction(prediction: np.ndarray, residual: np.ndarray, strength: float) -> np.ndarray:
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    factor = np.exp(np.clip(strength * residual, -1.0, 1.0))
    return np.asarray(-np.expm1(-np.clip(hazard * factor, 0.0, 50.0)), dtype=np.float32)


def quantile_from_rows(matrix, row_mask, column, rng):
    candidates = np.flatnonzero(row_mask)
    if candidates.size > 200_000:
        candidates = rng.choice(candidates, size=200_000, replace=False)
    return np.quantile(matrix[candidates, column], (0.25, 0.75))


def dependence_report(
    label: str,
    matrix: np.ndarray,
    weights: np.ndarray,
    row_folds: np.ndarray,
    names: tuple[str, ...],
    models,
    feature_counts,
    pair_counts,
) -> dict:
    stable_features = set(names)
    stable_pairs = set(pair_counts[0])
    for counts in feature_counts:
        stable_features &= set(counts)
    for counts in pair_counts[1:]:
        stable_pairs &= set(counts)
    ranked_features = sorted(
        stable_features,
        key=lambda name: min(
            counts[name] / max(sum(counts.values()), 1) for counts in feature_counts
        ),
        reverse=True,
    )
    ranked_pairs = sorted(
        stable_pairs,
        key=lambda pair: min(
            counts[pair] / max(sum(counts.values()), 1) for counts in pair_counts
        ),
        reverse=True,
    )
    rng = np.random.default_rng(121083)
    output = {"features": [], "pairs": []}
    print(f"{label}_STABLE_FEATURES count={len(ranked_features)}", flush=True)
    for name in ranked_features[:18]:
        column = names.index(name)
        effects = []
        grids = []
        for fold, model in enumerate(models):
            train = row_folds != fold
            held_candidates = np.flatnonzero(row_folds == fold)
            if held_candidates.size > 20_000:
                held_candidates = rng.choice(held_candidates, size=20_000, replace=False)
            probabilities = np.maximum(weights[held_candidates], 1e-12).astype(np.float64)
            probabilities /= probabilities.sum()
            sample = rng.choice(
                held_candidates,
                size=min(2500, held_candidates.size),
                replace=False,
                p=probabilities,
            )
            low, high = quantile_from_rows(matrix, train, column, rng)
            low_probe = matrix[sample].copy()
            high_probe = matrix[sample].copy()
            low_probe[:, column] = low
            high_probe[:, column] = high
            effects.append(float(np.average(
                model.predict(high_probe) - model.predict(low_probe),
                weights=np.maximum(weights[sample], 1e-12),
            )))
            grids.append((float(low), float(high)))
        record = {
            "name": name,
            "counts": [int(counts[name]) for counts in feature_counts],
            "effects": effects,
            "q25_q75": grids,
        }
        output["features"].append(record)
        print(
            f"FEATURE {name} counts={','.join(map(str, record['counts']))} "
            f"effects={','.join(f'{value:+.6f}' for value in effects)} "
            f"stable_sign={int(min(effects) > 0.0 or max(effects) < 0.0)}",
            flush=True,
        )

    print(f"{label}_STABLE_PAIRS count={len(ranked_pairs)}", flush=True)
    for left_name, right_name in ranked_pairs[:18]:
        left = names.index(left_name)
        right = names.index(right_name)
        contrasts, left_effects, right_effects, grids = [], [], [], []
        for fold, model in enumerate(models):
            train = row_folds != fold
            held_candidates = np.flatnonzero(row_folds == fold)
            if held_candidates.size > 20_000:
                held_candidates = rng.choice(held_candidates, size=20_000, replace=False)
            probabilities = np.maximum(weights[held_candidates], 1e-12).astype(np.float64)
            probabilities /= probabilities.sum()
            sample = rng.choice(
                held_candidates,
                size=min(2500, held_candidates.size),
                replace=False,
                p=probabilities,
            )
            left_low, left_high = quantile_from_rows(matrix, train, left, rng)
            right_low, right_high = quantile_from_rows(matrix, train, right, rng)
            values = {}
            for left_key, left_value in (("l", left_low), ("h", left_high)):
                for right_key, right_value in (("l", right_low), ("h", right_high)):
                    probe = matrix[sample].copy()
                    probe[:, left] = left_value
                    probe[:, right] = right_value
                    values[left_key + right_key] = model.predict(probe)
            sample_weight = np.maximum(weights[sample], 1e-12)
            contrasts.append(float(np.average(
                values["hh"] - values["hl"] - values["lh"] + values["ll"],
                weights=sample_weight,
            )))
            left_effects.append(float(np.average(
                0.5 * (values["hl"] + values["hh"] - values["ll"] - values["lh"]),
                weights=sample_weight,
            )))
            right_effects.append(float(np.average(
                0.5 * (values["lh"] + values["hh"] - values["ll"] - values["hl"]),
                weights=sample_weight,
            )))
            grids.append((float(left_low), float(left_high), float(right_low), float(right_high)))
        pair = (left_name, right_name)
        record = {
            "pair": pair,
            "counts": [int(counts[pair]) for counts in pair_counts],
            "contrasts": contrasts,
            "left_effects": left_effects,
            "right_effects": right_effects,
            "q25_q75": grids,
        }
        output["pairs"].append(record)
        print(
            f"PAIR {left_name}*{right_name} counts={','.join(map(str, record['counts']))} "
            f"interaction={','.join(f'{value:+.6f}' for value in contrasts)} "
            f"stable_sign={int(min(contrasts) > 0.0 or max(contrasts) < 0.0)} "
            f"left_effect={','.join(f'{value:+.6f}' for value in left_effects)} "
            f"right_effect={','.join(f'{value:+.6f}' for value in right_effects)}",
            flush=True,
        )
    return output


def score_text(score: dict[str, float]) -> str:
    return (
        f"overall={score['overall_score']:.9f} bias={score['bias_score']:.9f} "
        f"rmse={score['rmse_score']:.9f} seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}"
    )


def main() -> int:
    start = time.monotonic()
    model, blob = load_pinned()
    model_data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(model_data, dict(model.PARAMS), None))
    del model_data
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_score = evaluator.score(incumbent)["global"]
    if abs(base_score["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"pinned incumbent drift {base_score['overall_score']:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192], dtype=np.float32)
    observed_grid = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / np.float32(100.0)
    del fine
    land = load_land_mask()
    cells = np.flatnonzero(land.ravel())
    rows, columns = cells // 360, cells % 360
    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    row_folds = np.tile(cell_folds, 192)
    clean_data = load_inputs(CLEAN_INPUTS)
    names, matrix = build_features(clean_data, rows, columns)
    del clean_data

    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float32)
    selected_observed = np.asarray(observed_grid[:, rows, columns], dtype=np.float32)
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    selected_area = area_grid[rows, columns]
    observed_annual = np.average(selected_observed, axis=0, weights=MONTH_DAYS)
    predicted_annual = np.average(selected_incumbent, axis=0, weights=MONTH_DAYS)
    target_cell = np.clip(
        np.log((observed_annual + 1e-5) / (predicted_annual + 1e-5)),
        -3.0,
        3.0,
    ).astype(np.float32)
    target = np.tile(target_cell, 192)
    cell_weight = selected_area * (
        observed_annual + np.maximum(predicted_annual - observed_annual, 0.0)
    )
    positive = cell_weight[cell_weight > 0]
    floor = np.quantile(positive, 0.02) * 0.01
    cell_weight = np.maximum(cell_weight, floor)
    row_weight = np.tile(cell_weight, 192).astype(np.float64)
    row_weight /= row_weight.mean()
    fit_rows = np.flatnonzero(np.isin(np.arange(192), FIT_MONTHS).repeat(cells.size))

    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        observed_annual,
        cell_folds,
    )
    print(
        f"IDENTITY pinned={PINNED} model_blob={blob} incumbent={base_score['overall_score']:.9f} "
        f"land_cells={cells.size} rows={matrix.shape[0]} fit_rows={fit_rows.size} "
        f"features={len(names)} fold_cells="
        + ",".join(str(int(np.sum(cell_folds == fold))) for fold in range(4)),
        flush=True,
    )
    print("FEATURE_NAMES " + ",".join(names), flush=True)
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )

    summary: dict[str, object] = {
        "identity": {
            "pinned": PINNED,
            "model_blob": blob,
            "incumbent": float(base_score["overall_score"]),
            "land_cells": int(cells.size),
            "features": list(names),
            "fit_months": FIT_MONTHS.tolist(),
            "fold_cells": [int(np.sum(cell_folds == fold)) for fold in range(4)],
        },
        "learners": {},
    }
    for algorithm_index, label in enumerate(("deep_hgb", "extra_trees", "random_forest")):
        oof = np.empty(matrix.shape[0], dtype=np.float32)
        models = []
        feature_counts = []
        pair_counts = []
        folds_summary = []
        print(f"ALGORITHM_START label={label} elapsed={time.monotonic()-start:.1f}", flush=True)
        for fold in range(4):
            train = fit_rows[row_folds[fit_rows] != fold]
            held = row_folds == fold
            learner = make_learner(label, 121083 + 1000 * algorithm_index + 37 * fold)
            fold_start = time.monotonic()
            learner.fit(matrix[train], target[train], sample_weight=row_weight[train])
            held_indices = np.flatnonzero(held)
            for chunk_start in range(0, held_indices.size, 150_000):
                chunk = held_indices[chunk_start : chunk_start + 150_000]
                oof[chunk] = learner.predict(matrix[chunk]).astype(np.float32)
            counts, pairs = generic_tree_counts(learner, names)
            fold_r2 = weighted_r2(target[held], oof[held], row_weight[held])
            fold_summary = {
                "fold": fold,
                "r2": fold_r2,
                "seconds": time.monotonic() - fold_start,
                "top_features": [name for name, _ in counts.most_common(10)],
            }
            folds_summary.append(fold_summary)
            models.append(learner)
            feature_counts.append(counts)
            pair_counts.append(pairs)
            print(
                f"FOLD label={label} fold={fold} r2={fold_r2:.9f} "
                f"seconds={fold_summary['seconds']:.1f} top="
                + ",".join(fold_summary["top_features"]),
                flush=True,
            )
        joint_r2 = weighted_r2(target, oof, row_weight)
        print(f"OOF label={label} r2={joint_r2:.9f}", flush=True)
        dependence = dependence_report(
            label,
            matrix,
            row_weight,
            row_folds,
            names,
            models,
            feature_counts,
            pair_counts,
        )

        brackets = []
        residual = np.clip(oof.reshape(192, cells.size), -3.0, 3.0)
        for strength in STRENGTHS:
            corrected = apply_correction(selected_incumbent, residual, strength)
            losses = held_losses(
                corrected,
                selected_observed,
                selected_area,
                observed_annual,
                cell_folds,
            )
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            trial = incumbent.copy()
            trial[:, rows, columns] = corrected
            score = evaluator.score(validate_prediction(trial))["global"]
            record = {
                "strength": strength,
                "score": {key: float(value) for key, value in score.items()},
                "annual_gain": gains[0].tolist(),
                "allocation_gain": gains[1].tolist(),
                "raw_cycle_gain": gains[2].tolist(),
                "all_gates": bool(all(np.all(gain > 0.0) for gain in gains)),
            }
            brackets.append(record)
            print(
                f"HEADROOM label={label} strength={strength:.2f} {score_text(score)} "
                f"delta={score['overall_score']-base_score['overall_score']:+.9f} "
                f"all_gates={int(record['all_gates'])} annual_gain="
                + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2]),
                flush=True,
            )
            del corrected, trial
            gc.collect()
        summary["learners"][label] = {
            "joint_r2": joint_r2,
            "folds": folds_summary,
            "dependence": dependence,
            "brackets": brackets,
        }
        del oof, models, feature_counts, pair_counts
        gc.collect()

    output_path = Path(__file__).with_name("annual_rank_strong_oof.json")
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"DONE output={output_path} elapsed={time.monotonic()-start:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
