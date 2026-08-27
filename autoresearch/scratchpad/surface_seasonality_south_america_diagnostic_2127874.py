"""Diagnose the South America trade in warm surface-seasonality capacity.

This is a diagnostic intermediate for the official 0.719 candidate at
``2127874``.  It samples high-weight whole cells from NHAF, CEAM, NHSA, and
SHSA, then asks whether coupled-valid local ecological states distinguish
places that still need additional fire from places where the new positive
capacity worsens an overprediction.  Region bounds and coordinates are used
only to build the audit sample and held spatial folds; neither enters the
learner feature matrix.

The shallow learned residual surface is diagnostic only.  It must never be
copied into ``model.py`` or evaluated officially.  The intended output is one
stable interaction that can be translated into a globally shared physical
equation and tested separately.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
import types
from collections import Counter
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
from scripts.fast_ilamb import GFED_REGIONS  # noqa: E402
from scripts.runtime import load_land_mask  # noqa: E402


CURRENT_COMMIT = "2127874ce757af418aeab4f5a5d93b765030ff57"
CURRENT_MODEL_BLOB = "ca6848f2db28af24a06cd9f06e3adcdecaf7fcc0"
REGIONS = ("nhaf", "ceam", "nhsa", "shsa")
EPS = np.float32(1e-6)


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


def trailing_std(values: np.ndarray, months: int = 12) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        start = max(0, index - months + 1)
        output[index] = values[start : index + 1].std(axis=0)
    return output


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_mean(np.abs(values), weights)


def pinned_current_model():
    source = subprocess.run(
        ("git", "show", f"{CURRENT_COMMIT}:autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    observed_blob = subprocess.run(
        ("git", "hash-object", "--stdin"),
        cwd=ROOT,
        input=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_blob != CURRENT_MODEL_BLOB:
        raise RuntimeError(f"unexpected current model blob {observed_blob}")
    module = types.ModuleType("ed_fire_pinned_2127874")
    module.__file__ = f"git:{CURRENT_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def region_inside(
    name: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    south, north, west, east = GFED_REGIONS[name]
    return (
        (latitude > south)
        & (latitude <= north)
        & (longitude > west)
        & (longitude <= east)
    )


def select_audit_cells(
    rows: np.ndarray,
    columns: np.ndarray,
    observation_annual: np.ndarray,
    baseline_annual: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latitude = -89.5 + rows.astype(np.float64)
    longitude = -179.5 + columns.astype(np.float64)
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    importance = area * (observation_annual + baseline_annual + 1e-7)
    selected: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for region_index, name in enumerate(REGIONS):
        inside = region_inside(name, latitude, longitude)
        for fold in range(4):
            candidates = np.flatnonzero(inside & (folds == fold))
            order = np.argsort(importance[candidates])[::-1]
            chosen = candidates[order[:96]]
            selected.append(chosen)
            labels.append(np.full(chosen.size, region_index, dtype=np.int8))
            print(
                f"SAMPLE region={name} fold={fold} available={candidates.size} "
                f"selected={chosen.size} weight={float(importance[chosen].sum()):.9e}",
                flush=True,
            )
    return np.concatenate(selected), np.concatenate(labels)


def tree_counts(regressor, names: tuple[str, ...]):
    feature_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for stage in regressor._predictors:
        nodes = stage[0].nodes
        stack = [0]
        while stack:
            node = stack.pop()
            if nodes["is_leaf"][node]:
                continue
            feature = names[int(nodes["feature_idx"][node])]
            feature_counts[feature] += 1
            for child_name in ("left", "right"):
                child = int(nodes[child_name][node])
                if not nodes["is_leaf"][child]:
                    child_feature = names[int(nodes["feature_idx"][child])]
                    pair_counts[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)
    return feature_counts, pair_counts


def main() -> int:
    started = time.perf_counter()
    if not CACHE.exists():
        raise RuntimeError(f"missing pinned baseline cache {CACHE}")
    if EXPECTED_MODEL_BLOB != "39ee93ebf1155af9ae9d70e05847b9c3f086887d":
        raise RuntimeError("unexpected pre-capacity baseline")
    if abs(EXPECTED_BASE - 0.718363408) > 5e-10:
        raise RuntimeError("unexpected pre-capacity exact score")

    model = pinned_current_model()
    land = load_land_mask()
    flat = np.flatnonzero(land.ravel())
    all_rows, all_columns = flat // 360, flat % 360
    baseline_grid = np.load(CACHE, mmap_mode="r")
    baseline_all = np.asarray(
        baseline_grid[:, all_rows, all_columns], dtype=np.float32
    )
    observation_grid = load_observation()
    observation_all = np.asarray(
        observation_grid[:, all_rows, all_columns], dtype=np.float32
    )
    del observation_grid
    observation_annual_all = observation_all.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    baseline_annual_all = baseline_all.reshape(16, 12, -1).mean(axis=0).sum(axis=0)

    radius = 6.371e6
    bounds = np.deg2rad(np.arange(-90.0, 91.0, 1.0))
    latitude_area = radius**2 * np.diff(np.sin(bounds)) * np.deg2rad(1.0)
    area_grid = np.repeat(latitude_area[:, None], 360, axis=1)
    area_all = area_grid[all_rows, all_columns]
    chosen, region_labels = select_audit_cells(
        all_rows,
        all_columns,
        observation_annual_all,
        baseline_annual_all,
        area_all,
    )
    rows, columns = all_rows[chosen], all_columns[chosen]
    baseline = baseline_all[:, chosen]
    observation = observation_all[:, chosen]
    area = area_all[chosen]
    del baseline_all, observation_all, observation_annual_all, baseline_annual_all
    gc.collect()

    input_names = (
        "monthly_precipitation",
        "air_temperature",
        "dryness",
        "gpp",
        "aboveground_biomass",
        "soil_carbon",
        "leaf_area_index",
        "natural_canopy_height",
        "secondary_canopy_height",
        "natural_vegetation_fraction",
        "secondary_vegetation_fraction",
        "luh2_primary_fraction",
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
    )
    data: dict[str, np.ndarray] = {}
    for name in input_names:
        data[name] = np.asarray(
            selected_input(name, rows, columns)[:, 0, :], dtype=np.float32
        )
    rain = np.clip(data["monthly_precipitation"], 0.0, None)
    data["annual_precipitation"] = 12.0 * ema(rain, 12.0)

    enabled = set(model.COMPONENTS)
    current = model._surface_seasonality_capacity(
        baseline[:, None, :],
        {name: values[:, None, :] for name, values in data.items()},
        dict(model.PARAMS),
        enabled,
    )[:, 0, :]
    baseline_annual = baseline.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    current_annual = current.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    observed_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    target = np.clip(
        np.log((observed_annual + EPS) / (current_annual + EPS)), -3.0, 3.0
    ).astype(np.float32)
    increment = np.maximum(current_annual - baseline_annual, 0.0)
    floor = float(np.sum(observed_annual * area) / np.sum(area))
    weights = area * (observed_annual + 0.02 * floor)
    impact_scale = float(np.quantile(increment[increment > 0.0], 0.75))
    weights *= 0.20 + np.minimum(increment / max(impact_scale, 1e-9), 2.0)
    weights /= weights.mean()

    temperature = data["air_temperature"]
    dryness = np.clip(data["dryness"], 0.0, None)
    gpp = np.clip(data["gpp"], 0.0, None)
    temperature_12 = ema(temperature, 12.0)
    temperature_std = trailing_std(temperature)
    annual_rain = data["annual_precipitation"]
    rain_support = (
        sigmoid((annual_rain - 350.0) / 100.0)
        * annual_rain / (annual_rain + 500.0)
        * np.exp(-annual_rain / 3000.0)
    )
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    fine_fuel = ema(gpp, 12.0) / (ema(gpp, 12.0) + 0.35)
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(data["secondary_canopy_height"], 0.0, None)
    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    open_total = natural_open + secondary_open + managed_open
    woody_capacity = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * np.clip(data["aboveground_biomass"], 0.0, None) / (
        np.clip(data["aboveground_biomass"], 0.0, None) + 1.0
    )
    surface_capacity = open_total * fine_fuel
    surface_share = surface_capacity / (0.05 + surface_capacity + woody_capacity)
    natural_surface_share = natural_open / (0.05 + open_total)
    secondary_surface_share = secondary_open / (0.05 + open_total)
    managed_surface_share = managed_open / (0.05 + open_total)
    thermal_limit = 1.0 - sigmoid((temperature - 20.0))
    thermal_headroom = np.sum(combustion * thermal_limit, axis=0) / (
        np.sum(combustion, axis=0) + 1e-6
    )
    combustion_temperature_alignment = np.sum(
        combustion * (temperature - temperature.mean(axis=0)), axis=0
    ) / (np.sum(combustion, axis=0) + 1e-6)
    combustion_fuel_overlap = np.mean(combustion * fine_fuel, axis=0) / (
        np.mean(combustion, axis=0) * np.mean(fine_fuel, axis=0) + 1e-6
    )
    surface_modifier = np.clip(
        open_total
        * fine_fuel
        * continuity
        * combustion
        * sigmoid((temperature_12 - 15.0) / 3.0)
        * rain_support
        * temperature_std / (temperature_std + 4.0),
        0.0,
        1.0,
    )

    feature_map = {
        "rain_mean": rain.mean(axis=0),
        "rain_cv": rain.std(axis=0) / (rain.mean(axis=0) + 1.0),
        "dryness_mean": dryness.mean(axis=0),
        "dryness_cv": dryness.std(axis=0) / (dryness.mean(axis=0) + 1.0),
        "temperature_mean": temperature.mean(axis=0),
        "temperature_std": temperature.std(axis=0),
        "gpp_mean": gpp.mean(axis=0),
        "gpp_cv": gpp.std(axis=0) / (gpp.mean(axis=0) + 0.01),
        "biomass": data["aboveground_biomass"].mean(axis=0),
        "soil_carbon": data["soil_carbon"].mean(axis=0),
        "lai": data["leaf_area_index"].mean(axis=0),
        "canopy": canopy.mean(axis=0),
        "secondary_canopy": secondary_canopy.mean(axis=0),
        "primary": data["luh2_primary_fraction"].mean(axis=0),
        "crop": crop.mean(axis=0),
        "managed_open": managed_open.mean(axis=0),
        "natural_surface_share": natural_surface_share.mean(axis=0),
        "secondary_surface_share": secondary_surface_share.mean(axis=0),
        "managed_surface_share": managed_surface_share.mean(axis=0),
        "surface_share": surface_share.mean(axis=0),
        "woody_capacity": woody_capacity.mean(axis=0),
        "continuity": continuity.mean(axis=0),
        "rain_support": rain_support.mean(axis=0),
        "combustion": combustion.mean(axis=0),
        "thermal_headroom": thermal_headroom,
        "combustion_temperature_alignment": combustion_temperature_alignment,
        "combustion_fuel_overlap": combustion_fuel_overlap,
        "surface_modifier": surface_modifier.mean(axis=0),
    }
    feature_names = tuple(feature_map)
    features = np.column_stack([feature_map[name] for name in feature_names]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("non-finite feature matrix")

    cell_folds = ((rows // 12) + 3 * (columns // 12)) % 4
    oof = np.empty_like(target)
    fold_features: list[Counter[str]] = []
    fold_pairs: list[Counter[tuple[str, str]]] = []
    for fold in range(4):
        train = cell_folds != fold
        held = cell_folds == fold
        learner = HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=3.0,
            min_samples_leaf=45,
            early_stopping=False,
            random_state=2127 + fold,
        )
        learner.fit(features[train], target[train], sample_weight=weights[train])
        oof[held] = learner.predict(features[held]).astype(np.float32)
        counts, pairs = tree_counts(learner, feature_names)
        fold_features.append(counts)
        fold_pairs.append(pairs)
        before = weighted_mae(target[held], weights[held])
        after = weighted_mae(target[held] - oof[held], weights[held])
        print(
            f"FOLD fold={fold} cells={int(held.sum())} baseline_mae={before:.9f} "
            f"corrected_mae={after:.9f} delta={after-before:+.9f} "
            f"target_mean={weighted_mean(target[held], weights[held]):+.9f} "
            f"oof_mean={weighted_mean(oof[held], weights[held]):+.9f}",
            flush=True,
        )
        del learner

    feature_total = sum(fold_features, Counter())
    pair_total = sum(fold_pairs, Counter())
    for name, splits in feature_total.most_common(18):
        stable = sum(name in counts for counts in fold_features)
        print(
            f"FEATURE name={name} splits={splits} stable_folds={stable}/4",
            flush=True,
        )
    for pair, splits in pair_total.most_common(20):
        stable = sum(pair in counts for counts in fold_pairs)
        print(
            f"PAIR left={pair[0]} right={pair[1]} splits={splits} "
            f"stable_folds={stable}/4",
            flush=True,
        )

    for region_index, name in enumerate(REGIONS):
        mask = region_labels == region_index
        region_weights = weights[mask]
        baseline_ratio = float(np.sum(baseline_annual[mask] * area[mask])) / max(
            float(np.sum(observed_annual[mask] * area[mask])), 1e-12
        )
        current_ratio = float(np.sum(current_annual[mask] * area[mask])) / max(
            float(np.sum(observed_annual[mask] * area[mask])), 1e-12
        )
        print(
            f"REGION name={name} cells={int(mask.sum())} baseline_ratio={baseline_ratio:.9f} "
            f"current_ratio={current_ratio:.9f} "
            f"target_mean={weighted_mean(target[mask],region_weights):+.9f} "
            f"oof_mean={weighted_mean(oof[mask],region_weights):+.9f}",
            flush=True,
        )
        for feature_name in (
            "rain_mean",
            "dryness_mean",
            "dryness_cv",
            "temperature_std",
            "rain_cv",
            "gpp_mean",
            "lai",
            "primary",
            "managed_surface_share",
            "natural_surface_share",
            "secondary_surface_share",
            "surface_share",
            "woody_capacity",
            "thermal_headroom",
            "combustion_temperature_alignment",
            "surface_modifier",
        ):
            print(
                f"REGION_FEATURE name={name} feature={feature_name} "
                f"mean={weighted_mean(feature_map[feature_name][mask],region_weights):.9f}",
                flush=True,
            )

    positive = target > 0.0
    for feature_name in feature_names:
        positive_mean = weighted_mean(feature_map[feature_name][positive], weights[positive])
        negative_mean = weighted_mean(feature_map[feature_name][~positive], weights[~positive])
        scale = float(np.std(feature_map[feature_name])) + 1e-9
        separation = (positive_mean - negative_mean) / scale
        print(
            f"DIRECTION feature={feature_name} need_more={positive_mean:.9f} "
            f"need_less={negative_mean:.9f} standardized={separation:+.9f}",
            flush=True,
        )

    print(
        f"SUMMARY cells={target.size} base_mae={weighted_mae(target,weights):.9f} "
        f"oof_mae={weighted_mae(target-oof,weights):.9f} "
        f"delta={weighted_mae(target-oof,weights)-weighted_mae(target,weights):+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
