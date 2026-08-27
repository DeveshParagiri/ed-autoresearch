"""Causal whole-cell held-block screen for missing annual fire interactions.

Diagnostic only.  Each row describes one land cell at the end of a year and
predicts the incumbent log annual-fire residual in the following year.  Every
feature is therefore known before the target year begins.  Spatial coordinates
construct held blocks only; they are never learner features.  The script uses
all coupled-valid inputs declared by the pinned canonical model, but excludes
the completed-year precipitation field, targets-as-features, regions,
neighbours, modern-only weather, and benchmark-derived runtime tables.

The learned correction is applied out of fold only to measure exact diagnostic
headroom.  It must never be copied into ``model.py`` or officially evaluated.
Tree pairs locate interactions; a candidate mechanism is released only as a
smooth, globally shared ecological equation after separate testing.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
import types
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    CACHE,
    EXPECTED_BASE,
    EXPECTED_MODEL_BLOB,
    load_observation,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask, validate_prediction  # noqa: E402


EPS = np.float32(1e-6)
PREVIOUS_DECEMBER = np.arange(11, 180, 12)
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    output = np.empty_like(values, dtype=np.float32)
    state = np.asarray(values[0], dtype=np.float32).copy()
    for index in range(values.shape[0]):
        state += alpha * (values[index] - state)
        output[index] = state
    return output


def metric_text(score: Mapping[str, float]) -> str:
    return " ".join(
        f"{name}={score[key]:.9f}"
        for name, key in (
            ("overall", "overall_score"),
            ("bias", "bias_score"),
            ("rmse", "rmse_score"),
            ("seasonal", "seasonal_cycle_score"),
            ("spatial", "spatial_distribution_score"),
        )
    )


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.abs(values) * weights) / np.sum(weights))


def pinned_model():
    source = subprocess.run(
        ("git", "cat-file", "blob", EXPECTED_MODEL_BLOB),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("ed_fire_pinned_39ee93e")
    module.__file__ = f"git-blob:{EXPECTED_MODEL_BLOB}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def tree_counts(regressor, names: tuple[str, ...]):
    features = Counter()
    pairs = Counter()
    directions: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for stage in regressor._predictors:
        nodes = stage[0].nodes
        stack = [0]
        while stack:
            node = stack.pop()
            if nodes["is_leaf"][node]:
                continue
            feature = names[int(nodes["feature_idx"][node])]
            features[feature] += 1
            threshold = float(nodes["num_threshold"][node])
            for child_name in ("left", "right"):
                child = int(nodes[child_name][node])
                if not nodes["is_leaf"][child]:
                    child_feature = names[int(nodes["feature_idx"][child])]
                    pair = tuple(sorted((feature, child_feature)))
                    pairs[pair] += 1
                    directions.setdefault(pair, []).append(
                        (threshold, float(nodes["num_threshold"][child]), float(child_name == "right"))
                    )
                stack.append(child)
    return features, pairs, directions


def ecology_masks(mean: dict[str, np.ndarray], land_count: int) -> dict[str, np.ndarray]:
    rain = 12.0 * mean["monthly_precipitation"]
    temperature = mean["air_temperature"]
    lai = mean["leaf_area_index"]
    canopy = mean["natural_canopy_height"]
    biomass = mean["aboveground_biomass"]
    natural = mean["natural_vegetation_fraction"]
    primary = mean["luh2_primary_fraction"]
    crop = mean["luh2_cropland_fraction"]
    rangeland = mean["luh2_rangeland_fraction"]
    masks = {
        "intact_tropical_closed": (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0) & (biomass >= 0.2),
        "crop": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }
    if any(mask.shape != (land_count,) for mask in masks.values()):
        raise ValueError("ecology mask shape mismatch")
    return masks


def ecology_ratio(
    values: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    model_annual = values.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    output = {}
    for name, mask in masks.items():
        output[name] = float(np.sum(model_annual[mask] * area[mask])) / max(
            float(np.sum(obs_annual[mask] * area[mask])), 1e-12
        )
    return output


def main() -> int:
    started = time.perf_counter()
    if not CACHE.exists():
        raise RuntimeError(f"missing pinned {EXPECTED_MODEL_BLOB[:8]} cache")

    blob = EXPECTED_MODEL_BLOB
    model = pinned_model()
    land = load_land_mask()
    flat = np.flatnonzero(land.ravel())
    rows, columns = flat // 360, flat % 360
    count = flat.size
    baseline_grid = np.load(CACHE, mmap_mode="r")
    baseline = baseline_grid[:, rows, columns].astype(np.float32)
    observation_grid = load_observation()
    observation = observation_grid[:, rows, columns].astype(np.float32)
    del observation_grid
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_score = evaluator.score(validate_prediction(baseline_grid))["global"]
    if abs(float(base_score["overall_score"]) - EXPECTED_BASE) > 5e-9:
        raise RuntimeError(f"baseline mismatch {metric_text(base_score)}")
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    area = area_grid[rows, columns].astype(np.float64)
    print(
        f"DESIGN model_blob={blob} land_cells={count} samples={15 * count} "
        f"year_ahead=1 coordinates_features=0 {metric_text(base_score)}",
        flush=True,
    )

    names: list[str] = []
    columns_out: list[np.ndarray] = []
    previous_means: dict[str, np.ndarray] = {}
    climatology: dict[str, np.ndarray] = {}

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (15, count) or not np.isfinite(values).all():
            raise ValueError(f"bad feature {name}: {values.shape}")
        names.append(name)
        columns_out.append(values.reshape(-1))

    for input_name in model.INPUTS:
        values = np.asarray(selected_input(input_name, rows, columns)[:, 0, :], dtype=np.float32)
        yearly = values.reshape(16, 12, count)
        previous_mean = yearly[:15].mean(axis=1)
        previous_std = yearly[:15].std(axis=1)
        previous_end = values[PREVIOUS_DECEMBER]
        previous_start = yearly[:15, 0, :]
        previous_trend = (previous_end - previous_start) / (
            np.abs(previous_end) + np.abs(previous_start) + np.float32(1e-3)
        )
        add(f"{input_name}_prior_mean", previous_mean)
        add(f"{input_name}_prior_std", previous_std)
        add(f"{input_name}_prior_trend", previous_trend)
        previous_means[input_name] = previous_mean
        climatology[input_name] = values.mean(axis=0)
        del values, yearly, previous_std, previous_end, previous_start, previous_trend
        gc.collect()

    rain = np.clip(previous_means["monthly_precipitation"], 0.0, None)
    dryness = np.clip(previous_means["dryness"], 0.0, None)
    temperature = previous_means["air_temperature"]
    gpp = np.clip(previous_means["gpp"], 0.0, None)
    biomass = np.clip(previous_means["aboveground_biomass"], 0.0, None)
    soil = np.clip(previous_means["soil_carbon"], 0.0, None)
    lai = np.clip(previous_means["leaf_area_index"], 0.0, None)
    canopy = np.clip(previous_means["natural_canopy_height"], 0.0, None)
    second_canopy = np.clip(previous_means["secondary_canopy_height"], 0.0, None)
    natural = np.clip(previous_means["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(previous_means["secondary_vegetation_fraction"], 0.0, 1.0)
    crop = np.clip(previous_means["luh2_cropland_fraction"], 0.0, 1.0)
    pasture = np.clip(previous_means["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(previous_means["luh2_rangeland_fraction"], 0.0, 1.0)
    primary = np.clip(previous_means["luh2_primary_fraction"], 0.0, 1.0)
    urban = np.clip(previous_means["luh2_urban_fraction"], 0.0, 1.0)
    lightning = np.clip(previous_means["lightning_flash_rate"], 0.0, None)

    annual_rain = 12.0 * rain
    fine_fuel = gpp / (gpp + 0.35)
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (second_canopy + 8.0)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    woody_fuel = natural * canopy / (canopy + 8.0) * biomass / (biomass + 2.0)
    fragmentation = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    rain_built = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(-annual_rain / 3000.0)
    cold = sigmoid((5.0 - temperature) / 3.0)
    warm = sigmoid((temperature - 18.0) / 3.0)
    ignition = lightning / (lightning + 0.02)
    rooted_storage = soil / (soil + 8.0) * biomass / (biomass + 2.0)
    pathway_features = {
        "fine_fuel": fine_fuel,
        "combustion": combustion,
        "natural_open": natural_open,
        "secondary_open": secondary_open,
        "managed_open": managed_open,
        "woody_fuel": woody_fuel,
        "continuity": fragmentation,
        "rain_built_fuel": rain_built * fine_fuel,
        "connected_surface_fuel": (natural_open + secondary_open + managed_open) * fine_fuel * fragmentation,
        "connected_surface_combustion": (natural_open + secondary_open + managed_open) * fine_fuel * fragmentation * combustion,
        "cold_organic_fuel": cold * rooted_storage,
        "cold_organic_combustion": cold * rooted_storage * combustion,
        "warm_woody_humidity": warm * woody_fuel * rain / (rain + 80.0),
        "primary_woody_fuel": primary * woody_fuel,
        "secondary_open_fuel": secondary_open * fine_fuel,
        "managed_open_fuel": managed_open * fine_fuel,
        "crop_residue_fuel": crop * fine_fuel,
        "managed_ignition_access": managed_open * (0.25 + 0.75 * ignition),
        "natural_ignition_access": natural_open * ignition,
        "arid_fuel_limit": rain_built * (natural_open + managed_open),
        "woody_root_storage": woody_fuel * rooted_storage,
        "fragmented_fine_fuel": (crop + managed_open) * fine_fuel * (1.0 - fragmentation),
    }
    for feature_name, values in pathway_features.items():
        add(feature_name, values)

    x = np.column_stack(columns_out).astype(np.float32, copy=False)
    feature_names = tuple(names)
    del columns_out, previous_means, pathway_features
    gc.collect()
    print(f"FEATURES columns={x.shape[1]} bytes={x.nbytes}", flush=True)

    baseline_year = baseline.reshape(16, 12, count)[1:].sum(axis=1)
    observed_year = observation.reshape(16, 12, count)[1:].sum(axis=1)
    target = np.clip(np.log((observed_year + EPS) / (baseline_year + EPS)), -4.0, 4.0).reshape(-1).astype(np.float32)
    observed_floor = float(np.sum(observed_year * area[None, :])) / (
        15.0 * float(np.sum(area))
    )
    target_weight = area[None, :] * (observed_year + 0.02 * observed_floor)
    weights = target_weight.reshape(-1).astype(np.float64)
    weights /= weights.mean()
    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    folds = np.tile(cell_folds, (15, 1)).reshape(-1)

    oof_by_depth: dict[int, np.ndarray] = {}
    pairs_by_depth: dict[int, list[Counter]] = {}
    for depth in (2,):
        oof = np.empty_like(target)
        feature_counts: list[Counter] = []
        pair_counts: list[Counter] = []
        for fold in range(4):
            train, held = folds != fold, folds == fold
            learner = HistGradientBoostingRegressor(
                max_depth=depth,
                max_iter=70,
                learning_rate=0.06,
                l2_regularization=2.0,
                min_samples_leaf=450,
                early_stopping=False,
                random_state=7310 + 100 * depth + fold,
            )
            learner.fit(x[train], target[train], sample_weight=weights[train])
            oof[held] = learner.predict(x[held]).astype(np.float32)
            features, pairs, _ = tree_counts(learner, feature_names)
            feature_counts.append(features)
            pair_counts.append(pairs)
            before = weighted_mae(target[held], weights[held])
            after = weighted_mae(target[held] - oof[held], weights[held])
            signed = float(np.sum(oof[held] * weights[held]) / np.sum(weights[held]))
            print(
                f"FOLD depth={depth} fold={fold} baseline_mae={before:.9f} "
                f"corrected_mae={after:.9f} delta={after-before:+.9f} "
                f"mean_correction={signed:+.9f}",
                flush=True,
            )
            del learner
            gc.collect()
        oof_by_depth[depth] = oof
        pairs_by_depth[depth] = pair_counts
        feature_total = sum(feature_counts, Counter())
        pair_total = sum(pair_counts, Counter())
        for feature, split_count in feature_total.most_common(20):
            stability = sum(feature in {name for name, _ in fold.most_common(20)} for fold in feature_counts)
            print(
                f"FEATURE depth={depth} name={feature} splits={split_count} stable_folds={stability}/4",
                flush=True,
            )
        for pair, split_count in pair_total.most_common(30):
            stability = sum(pair in {name for name, _ in fold.most_common(30)} for fold in pair_counts)
            print(
                f"PAIR depth={depth} left={pair[0]} right={pair[1]} "
                f"splits={split_count} stable_folds={stability}/4",
                flush=True,
            )

    del x
    gc.collect()
    all_scores = []
    ecology = ecology_masks(climatology, count)
    base_ecology = ecology_ratio(baseline, observation, area, ecology)
    month_grid = np.arange(12, 192).reshape(15, 12)
    best_candidate = None
    best_label = ""
    best_score = None
    for depth, oof in oof_by_depth.items():
        annual_correction = oof.reshape(15, count)
        for blend in (0.10, 0.25, 0.50, 0.75, 1.0):
            candidate_grid = baseline_grid.copy()
            factor = np.exp(np.clip(blend * annual_correction, -2.0, 2.0)).astype(np.float32)
            for year in range(15):
                for month in month_grid[year]:
                    candidate_grid[month, rows, columns] *= factor[year]
            np.clip(candidate_grid, 0.0, 1.0, out=candidate_grid)
            score = evaluator.score(validate_prediction(candidate_grid))["global"]
            label = f"depth={depth}:blend={blend:g}"
            all_scores.append((float(score["overall_score"]), label, dict(score)))
            print(f"EXACT_OOF {label} delta={float(score['overall_score'])-EXPECTED_BASE:+.9f} {metric_text(score)}", flush=True)
            if best_score is None or float(score["overall_score"]) > float(best_score["overall_score"]):
                best_candidate = candidate_grid
                best_label = label
                best_score = dict(score)
            else:
                del candidate_grid
            gc.collect()

    if best_candidate is None or best_score is None:
        raise RuntimeError("no candidate scores")
    best_selected = best_candidate[:, rows, columns]
    best_ecology = ecology_ratio(best_selected, observation, area, ecology)
    for name in ecology:
        print(
            f"ECOLOGY best={best_label} name={name} baseline={base_ecology[name]:.9f} "
            f"oof={best_ecology[name]:.9f} delta={best_ecology[name]-base_ecology[name]:+.9f}",
            flush=True,
        )

    print("TOP_EXACT", flush=True)
    for overall, label, score in sorted(all_scores, reverse=True)[:10]:
        print(
            f"TOP label={label} delta={overall-EXPECTED_BASE:+.9f} {metric_text(score)}",
            flush=True,
        )
    peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        f"DONE wall_seconds={time.perf_counter()-started:.3f} "
        f"peak_rss_gib={peak_raw/(1024.0**3):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
