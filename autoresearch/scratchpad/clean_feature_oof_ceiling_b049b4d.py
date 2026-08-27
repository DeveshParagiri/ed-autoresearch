"""Held-block clean-input information ceiling for ED-Fire.

Diagnostic only.  Every learner sees only certified exogenous coupled-valid
climate, LUH2, and lightning state.  Coordinates are used only to construct
15-degree spatial validation blocks.  No coordinate, region, calendar,
benchmark-derived, ED-state, modern-weather, future, or completed-record
climatology is a feature.  Learned surfaces never enter ``model.py`` or the
official evaluator ledger.
"""

from __future__ import annotations

import gc
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad import clean_landuse_pathway_rebuild2 as clean_model  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_model,
    validate_prediction,
)


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


def metric_text(score: dict[str, float]) -> str:
    return " ".join(
        f"{name}={score[key]:.8f}"
        for name, key in (
            ("overall", "overall_score"),
            ("bias", "bias_score"),
            ("rmse", "rmse_score"),
            ("seasonal", "seasonal_cycle_score"),
            ("spatial", "spatial_distribution_score"),
        )
    )


def antecedent(series: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(series[0], dtype=np.float32).copy()
    result = np.empty_like(series, dtype=np.float32)
    for step in range(series.shape[0]):
        state += alpha * (series[step] - state)
        result[step] = state
    return result


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
                    pairs[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)
    return features, pairs


def score_selected(
    evaluator: GFED5Evaluator,
    baseline: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, float]:
    candidate = baseline.copy()
    candidate[:, rows, cols] = np.clip(corrected, 0.0, 1.0)
    score = evaluator.score(validate_prediction(candidate))["global"]
    del candidate
    gc.collect()
    return score


def select_cells(evaluator: GFED5Evaluator, observed: np.ndarray):
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    reference_mean = np.average(observed, axis=0, weights=MONTH_DAYS)
    cell_weight = area * reference_mean
    order = np.argsort(cell_weight.ravel())[::-1]
    cumulative = np.cumsum(cell_weight.ravel()[order]) / cell_weight.sum()
    count = int(np.searchsorted(cumulative, 0.85) + 1)
    cells = order[:count]
    rows, cols = cells // 360, cells % 360
    retained = float(cell_weight.ravel()[cells].sum() / cell_weight.sum())
    return rows, cols, cell_weight[rows, cols], retained


def build_features(data, rows: np.ndarray, cols: np.ndarray):
    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (192, rows.size):
            raise ValueError(f"feature {name} has unexpected shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"feature {name} is not finite")
        names.append(name)
        columns.append(values.reshape(-1))

    selected = {
        name: np.asarray(data[name][:, rows, cols], dtype=np.float32)
        for name in CLEAN_INPUTS
    }
    rain = selected["monthly_precipitation"]
    dry = selected["dryness"]
    temp = selected["air_temperature"]
    flash = selected["lightning_flash_rate"]

    add("log1p_rain", np.log1p(np.clip(rain, 0.0, None)))
    add("log1p_dryness", np.log1p(np.clip(dry, 0.0, None)))
    add("temperature", temp)
    add("log1p_lightning", np.log1p(1000.0 * np.clip(flash, 0.0, None)))

    memories: dict[tuple[str, int], np.ndarray] = {}
    for name, values, offset in (
        ("rain", rain, 10.0),
        ("dryness", dry, 100.0),
        ("temperature", temp, 10.0),
        ("lightning", flash, 0.002),
    ):
        for months in (3, 6, 12, 24):
            memory = antecedent(values, months)
            memories[(name, months)] = memory
            if name in ("rain", "dryness", "lightning"):
                multiplier = 1000.0 if name == "lightning" else 1.0
                add(f"log1p_{name}_ema{months}", np.log1p(multiplier * np.clip(memory, 0.0, None)))
            else:
                add(f"temperature_ema{months}", memory)
            add(
                f"{name}_departure{months}",
                (values - memory) / (np.abs(values) + np.abs(memory) + offset),
            )

    # The prepared annual_precipitation field is the completed calendar-year
    # total repeated into all twelve months, so it leaks future rainfall.  The
    # only annual-scale moisture feature here is a prefix-causal annualisation
    # of the twelve-month EMA of monthly precipitation.
    annual = 12.0 * memories[("rain", 12)]
    add("log1p_causal_annualized_rain", np.log1p(np.clip(annual, 0.0, None)))

    for name in (
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_primary_fraction",
        "luh2_secondary_fraction",
        "luh2_urban_fraction",
    ):
        add(name, selected[name])

    crop = selected["luh2_cropland_fraction"]
    pasture = selected["luh2_pasture_fraction"]
    rangeland = selected["luh2_rangeland_fraction"]
    primary = selected["luh2_primary_fraction"]
    secondary = selected["luh2_secondary_fraction"]
    urban = selected["luh2_urban_fraction"]
    add("managed_open", np.clip(pasture + rangeland, 0.0, 1.0))
    add("woody_land", np.clip(primary + secondary, 0.0, 1.0))
    add("managed_fragmentation", np.clip(crop + 4.0 * urban, 0.0, 4.0))
    add("rain_deficit3", np.maximum(
        (memories[("rain", 3)] - rain) /
        (memories[("rain", 3)] + rain + 10.0), 0.0,
    ))
    add("rain_deficit12", np.maximum(
        (memories[("rain", 12)] - rain) /
        (memories[("rain", 12)] + rain + 10.0), 0.0,
    ))
    add("rain_wet_anomaly12", np.maximum(
        (rain - memories[("rain", 12)]) /
        (memories[("rain", 12)] + rain + 10.0), 0.0,
    ))
    add("dryness_rise3", np.maximum(
        (dry - memories[("dryness", 3)]) /
        (dry + memories[("dryness", 3)] + 100.0), 0.0,
    ))
    add("warming3", temp - memories[("temperature", 3)])
    add("warming12", temp - memories[("temperature", 12)])
    add("lightning_pulse12", np.maximum(
        (flash - memories[("lightning", 12)]) /
        (flash + memories[("lightning", 12)] + 0.002), 0.0,
    ))
    add("annual_fuel_support", (
        annual / (annual + 260.0) * np.exp(-annual / 3200.0)
    ))
    add("combustion_opportunity", (
        dry / (dry + 110.0) / (1.0 + rain / 42.0)
    ))

    x = np.column_stack(columns).astype(np.float32, copy=False)
    return tuple(names), x


def clean_baseline(data, evaluator: GFED5Evaluator):
    records = []
    best_prediction = None
    best_score = -np.inf
    best_label = ""
    best_params = None
    for scale in (0.45, 0.75, 1.0, 1.4, 2.0, 3.0):
        params = dict(clean_model.BASE)
        params["global_scale"] *= scale
        prediction = validate_prediction(clean_model.predict(data, params))
        score = evaluator.score(prediction)["global"]
        label = f"clean_scale={scale:g}"
        print(f"CLEAN_BRACKET {label} {metric_text(score)}", flush=True)
        records.append((float(score["overall_score"]), label, params, dict(score)))
        if score["overall_score"] > best_score:
            best_prediction = prediction
            best_score = float(score["overall_score"])
            best_label = label
            best_params = params
        else:
            del prediction
        gc.collect()
    # Reproduce the documented clean-line endpoint exactly.  This is the one
    # bounded ecological refinement in the source experiment, not a new fit:
    # it strengthens humid-primary, grazing, and fragmentation suppression and
    # reduces the weak cold pathway after the scale bracket closes at 3x.
    params = dict(clean_model.BASE)
    params["global_scale"] *= 3.0
    params.update({
        "wet_primary_brake": 7.0,
        "grazing_strength": 4.0,
        "fragmentation_strength": 3.0,
        "crop_fragmentation": 3.0,
        "cold_scale": 0.0005,
    })
    prediction = validate_prediction(clean_model.predict(data, params))
    score = evaluator.score(prediction)["global"]
    label = "clean_ecology_suppressed"
    print(f"CLEAN_BRACKET {label} {metric_text(score)}", flush=True)
    records.append((float(score["overall_score"]), label, params, dict(score)))
    if score["overall_score"] > best_score:
        del best_prediction
        best_prediction = prediction
        best_score = float(score["overall_score"])
        best_label = label
        best_params = params
    else:
        del prediction
    gc.collect()
    assert best_prediction is not None and best_params is not None
    print(f"CLEAN_BEST {best_label} {metric_text(records[np.argmax([r[0] for r in records])][3])}", flush=True)
    return best_prediction, best_label, best_params


def targets_for(baseline_selected: np.ndarray, observed_selected: np.ndarray):
    eps = np.float32(1e-5)
    annual_base = np.average(baseline_selected, axis=0, weights=MONTH_DAYS)
    annual_obs = np.average(observed_selected, axis=0, weights=MONTH_DAYS)
    map_target = np.clip(np.log((annual_obs + eps) / (annual_base + eps)), -4.0, 4.0)
    map_target = np.tile(map_target, 192).astype(np.float32)

    base_cycle = baseline_selected.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed_selected.reshape(16, 12, -1).mean(axis=0)
    base_alloc = base_cycle / (base_cycle.sum(axis=0, keepdims=True) + eps)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + eps)
    cycle_residual = np.clip(obs_alloc - base_alloc, -0.5, 0.5)
    cycle_target = np.tile(cycle_residual, (16, 1, 1)).reshape(-1).astype(np.float32)

    joint_target = np.clip(
        np.log((observed_selected + eps) / (baseline_selected + eps)), -4.0, 4.0
    ).reshape(-1).astype(np.float32)
    return map_target, cycle_target, joint_target, base_alloc


def score_oof(
    label: str,
    evaluator: GFED5Evaluator,
    baseline: np.ndarray,
    observed: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    predictions: dict[str, np.ndarray],
    base_alloc: np.ndarray,
):
    selected = baseline[:, rows, cols].astype(np.float32)
    count = rows.size
    map_prediction = predictions["map"].reshape(192, count).mean(axis=0)
    cycle_prediction = predictions["cycle"].reshape(16, 12, count).mean(axis=0)
    joint_prediction = predictions["joint"].reshape(192, count)
    month_index = np.arange(192) % 12
    results = []

    for blend in (0.10, 0.25, 0.50, 0.75, 1.0):
        corrected = selected * np.exp(np.clip(blend * map_prediction[None, :], -3.0, 3.0))
        score = score_selected(evaluator, baseline, rows, cols, corrected)
        tag = f"map={blend:g}"
        print(f"SCORE {label} {tag} {metric_text(score)}", flush=True)
        results.append((float(score["overall_score"]), tag, dict(score)))

    cycle_candidates = {}
    for blend in (0.25, 0.50, 0.75, 1.0):
        allocation = np.clip(base_alloc + blend * cycle_prediction, 1e-6, None)
        allocation /= allocation.sum(axis=0, keepdims=True)
        ratio = allocation / (base_alloc + 1e-6)
        corrected = selected * ratio[month_index]
        score = score_selected(evaluator, baseline, rows, cols, corrected)
        tag = f"cycle={blend:g}"
        print(f"SCORE {label} {tag} {metric_text(score)}", flush=True)
        results.append((float(score["overall_score"]), tag, dict(score)))
        cycle_candidates[blend] = ratio

    for blend in (0.10, 0.25, 0.50, 0.75, 1.0):
        corrected = selected * np.exp(np.clip(blend * joint_prediction, -3.0, 3.0))
        score = score_selected(evaluator, baseline, rows, cols, corrected)
        tag = f"monthly_joint={blend:g}"
        print(f"SCORE {label} {tag} {metric_text(score)}", flush=True)
        results.append((float(score["overall_score"]), tag, dict(score)))

    for map_blend, cycle_blend in ((0.25, 0.5), (0.25, 1.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)):
        corrected = (
            selected
            * np.exp(np.clip(map_blend * map_prediction[None, :], -3.0, 3.0))
            * cycle_candidates[cycle_blend][month_index]
        )
        score = score_selected(evaluator, baseline, rows, cols, corrected)
        tag = f"separate_joint_map={map_blend:g}_cycle={cycle_blend:g}"
        print(f"SCORE {label} {tag} {metric_text(score)}", flush=True)
        results.append((float(score["overall_score"]), tag, dict(score)))
    best = max(results, key=lambda row: row[0])
    print(f"OOF_BEST {label} {best[1]} {metric_text(best[2])}", flush=True)
    return results


def partial_dependence(
    label: str,
    target_name: str,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    names: tuple[str, ...],
    feature_order: list[str],
) -> None:
    learner = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=60,
        learning_rate=0.08,
        l2_regularization=1.0,
        min_samples_leaf=300,
        early_stopping=False,
        random_state=49001 + len(label) + len(target_name),
    )
    learner.fit(x, target, sample_weight=weights)
    rng = np.random.default_rng(4900)
    sample_index = rng.choice(x.shape[0], size=min(30000, x.shape[0]), replace=False)
    sample = x[sample_index].copy()
    for feature in feature_order[:6]:
        index = names.index(feature)
        grid = np.quantile(x[:, index], (0.10, 0.25, 0.50, 0.75, 0.90))
        values = []
        original = sample[:, index].copy()
        for point in grid:
            sample[:, index] = point
            values.append(float(learner.predict(sample).mean()))
        sample[:, index] = original
        center = values[2]
        print(
            f"PDP {label} target={target_name} feature={feature} "
            f"grid={','.join(f'{v:.6g}' for v in grid)} "
            f"effect={','.join(f'{v - center:+.6g}' for v in values)}",
            flush=True,
        )
    del learner, sample
    gc.collect()


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    del reference
    rows, cols, cell_weight, retained = select_cells(evaluator, observed)
    count = rows.size
    cell_folds = ((rows // 15) + 3 * (cols // 15)) % 4
    folds = np.tile(cell_folds, 192)
    weights = np.tile(cell_weight, 192).astype(np.float64)
    weights /= weights.mean()
    print(
        f"DESIGN cells={count} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(cell_folds == fold))) for fold in range(4))}",
        flush=True,
    )

    clean_data = load_inputs(CLEAN_INPUTS)
    clean_data["annual_precipitation"] = 12.0 * antecedent(
        clean_data["monthly_precipitation"], 12.0
    )
    clean_prediction, clean_label, clean_params = clean_baseline(clean_data, evaluator)
    names, x = build_features(clean_data, rows, cols)
    print(f"FEATURE_MATRIX rows={x.shape[0]} columns={x.shape[1]} bytes={x.nbytes}", flush=True)
    del clean_data
    gc.collect()

    current_model = load_model()
    current_data = load_inputs(current_model.INPUTS)
    current_original = validate_prediction(
        current_model.predict(current_data, dict(current_model.PARAMS), None)
    )
    current_original_score = evaluator.score(current_original)["global"]
    print(f"CURRENT_ORIGINAL_FUTURE_LEAKING {metric_text(current_original_score)}", flush=True)
    current_data["annual_precipitation"] = 12.0 * antecedent(
        current_data["monthly_precipitation"], 12.0
    )
    current_prediction = validate_prediction(
        current_model.predict(current_data, dict(current_model.PARAMS), None)
    )
    current_score = evaluator.score(current_prediction)["global"]
    print(f"CURRENT_CAUSAL_ANNUALIZED {metric_text(current_score)}", flush=True)
    del current_original
    del current_data, current_model
    gc.collect()

    baselines = ((clean_label, clean_prediction), ("current_causal", current_prediction))
    stability_features = defaultdict(lambda: defaultdict(set))
    stability_pairs = defaultdict(lambda: defaultdict(set))
    pdp_requests = []
    for baseline_index, (baseline_label, baseline) in enumerate(baselines):
        baseline_selected = baseline[:, rows, cols].astype(np.float32)
        observed_selected = observed[:, rows, cols].astype(np.float32)
        map_target, cycle_target, joint_target, base_alloc = targets_for(
            baseline_selected, observed_selected
        )
        targets = {"map": map_target, "cycle": cycle_target, "joint": joint_target}
        for depth in (2, 3):
            predictions = {
                target_name: np.empty_like(target, dtype=np.float32)
                for target_name, target in targets.items()
            }
            for fold in range(4):
                train = folds != fold
                held = ~train
                for target_index, (target_name, target) in enumerate(targets.items()):
                    learner = HistGradientBoostingRegressor(
                        max_depth=depth,
                        max_iter=60,
                        learning_rate=0.08,
                        l2_regularization=1.0,
                        min_samples_leaf=300,
                        early_stopping=False,
                        random_state=490000 + 1000 * baseline_index + 100 * depth + 10 * fold + target_index,
                    )
                    learner.fit(x[train], target[train], sample_weight=weights[train])
                    predictions[target_name][held] = learner.predict(x[held]).astype(np.float32)
                    feature_counts, pair_counts = tree_counts(learner, names)
                    for feature, _ in feature_counts.most_common(15):
                        stability_features[(baseline_label, depth, target_name)][feature].add(fold)
                    for pair, _ in pair_counts.most_common(20):
                        stability_pairs[(baseline_label, depth, target_name)][pair].add(fold)
                    del learner
                    gc.collect()
            for target_name, target in targets.items():
                value = r2_score(target, predictions[target_name], sample_weight=weights)
                print(
                    f"OOF_R2 baseline={baseline_label} depth={depth} target={target_name} value={value:.8f}",
                    flush=True,
                )
            score_oof(
                f"{baseline_label}:depth={depth}", evaluator, baseline, observed,
                rows, cols, predictions, base_alloc,
            )
            if depth == 3:
                for target_name in ("map", "cycle", "joint"):
                    key = (baseline_label, depth, target_name)
                    stable = sorted(
                        (
                            (len(seen), feature)
                            for feature, seen in stability_features[key].items()
                        ),
                        reverse=True,
                    )
                    feature_order = [feature for _, feature in stable]
                    pdp_requests.append((baseline_label, target_name, targets[target_name], feature_order))
            del predictions
            gc.collect()
        del baseline_selected, observed_selected, map_target, cycle_target, joint_target
        gc.collect()

    for key in sorted(stability_features):
        stable = sorted(
            ((len(seen), feature) for feature, seen in stability_features[key].items()),
            reverse=True,
        )
        print(
            f"STABLE_FEATURES baseline={key[0]} depth={key[1]} target={key[2]} "
            + ",".join(f"{feature}:{count}/4" for count, feature in stable[:15]),
            flush=True,
        )
        pairs = sorted(
            ((len(seen), pair) for pair, seen in stability_pairs[key].items()),
            reverse=True,
        )
        print(
            f"STABLE_PAIRS baseline={key[0]} depth={key[1]} target={key[2]} "
            + ",".join(
                f"{left}*{right}:{count}/4" for count, (left, right) in pairs[:15]
            ),
            flush=True,
        )

    for baseline_label, target_name, target, feature_order in pdp_requests:
        partial_dependence(
            baseline_label, target_name, x, target, weights, names, feature_order
        )

    print(f"CLEAN_PARAMS {clean_params!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
