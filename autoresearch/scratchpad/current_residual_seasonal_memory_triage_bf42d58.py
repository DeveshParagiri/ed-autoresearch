"""Whole-cell OOF residual triage for the exact bf42d58 incumbent.

This is an ML diagnostic, never a candidate model.  It asks whether a site's
own prefix-causal seasonal memory adds missing fire information beyond the
ordinary short memories already represented by the mechanistic model.  The
new states are twelve independent month-of-year memories updated only after a
month has been predicted, plus finite trailing co-occurrence statistics.

Coordinates define four held spatial blocks only.  No coordinate, region,
neighbour, future value, completed-record climatology, invalid forcing, or
benchmark-derived runtime field is exposed to a learner.  Annual factors use
only the previous completed year.  Seasonal corrections are centered with a
causal twelve-month EMA rather than a future annual normalization.
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
from autoresearch.scratchpad.stable_annual_residual_rules_39ee93e import (  # noqa: E402
    ecology_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    metric_text,
    select_high_weight,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask, validate_prediction  # noqa: E402


MODEL_COMMIT = "bf42d58"
MODEL_BLOB = "a1966275d22874d1c71c45c7b8a8f5c8e473358d"
EXPECTED_OVERALL = 0.718995365
PRECAP_CACHE = ROOT / "autoresearch/scratchpad/canonical_39ee93eb_chunked.npy"
CURRENT_CACHE = ROOT / f"autoresearch/scratchpad/canonical_{MODEL_BLOB[:8]}_chunked.npy"
EPS = np.float32(1e-6)


def pinned_model():
    resolved = subprocess.check_output(
        ("git", "rev-parse", f"{MODEL_COMMIT}:autoresearch/model.py"),
        cwd=ROOT,
        text=True,
    ).strip()
    if resolved != MODEL_BLOB:
        raise RuntimeError(f"model pin changed: {resolved}")
    source = subprocess.check_output(
        ("git", "cat-file", "blob", MODEL_BLOB), cwd=ROOT
    )
    module = types.ModuleType("ed_fire_bf42d58")
    module.__file__ = f"git-blob:{MODEL_BLOB}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def build_current_cache(model, land: np.ndarray) -> np.ndarray:
    """Reconstruct current from the pinned pre-capacity cache in chunks."""
    if CURRENT_CACHE.exists():
        output = np.load(CURRENT_CACHE, mmap_mode="r")
        if output.shape != (192, 180, 360) or output.dtype != np.float32:
            raise ValueError(f"bad cache {CURRENT_CACHE}: {output.shape} {output.dtype}")
        print(f"CACHE reuse={CURRENT_CACHE} bytes={CURRENT_CACHE.stat().st_size}", flush=True)
        return output
    if not PRECAP_CACHE.exists():
        raise RuntimeError(f"missing pinned pre-capacity cache {PRECAP_CACHE}")

    predecessor = np.load(PRECAP_CACHE, mmap_mode="r")
    output = np.zeros((192, 180, 360), dtype=np.float32)
    rows, columns = np.nonzero(land)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    for start in range(0, rows.size, 1536):
        stop = min(start + 1536, rows.size)
        chunk_rows, chunk_columns = rows[start:stop], columns[start:stop]
        data = {
            name: selected_input(name, chunk_rows, chunk_columns)
            for name in model.INPUTS
        }
        data["annual_precipitation"] = 12.0 * model._antecedent(
            np.asarray(data["monthly_precipitation"], dtype=np.float32),
            alpha_12,
        )
        base = np.asarray(
            predecessor[:, chunk_rows, chunk_columns], dtype=np.float32
        )[:, None, :]
        prediction = model._surface_seasonality_capacity(
            base, data, dict(model.PARAMS), set(model.COMPONENTS)
        )[:, 0, :]
        output[:, chunk_rows, chunk_columns] = prediction
        print(f"CACHE_CHUNK {start}:{stop}/{rows.size}", flush=True)
        del data, base, prediction
        gc.collect()
    np.save(CURRENT_CACHE, output, allow_pickle=False)
    print(f"CACHE created={CURRENT_CACHE} bytes={CURRENT_CACHE.stat().st_size}", flush=True)
    return np.load(CURRENT_CACHE, mmap_mode="r")


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(values[0], dtype=np.float32).copy()
    output = np.empty_like(values, dtype=np.float32)
    for time_index in range(values.shape[0]):
        state += alpha * (values[time_index] - state)
        output[time_index] = state
    return output


def causal_seasonal_normal(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return prior same-month mean and last-year value without future use."""
    output = np.empty_like(values, dtype=np.float32)
    previous = np.empty_like(values, dtype=np.float32)
    state = np.zeros((12, values.shape[1]), dtype=np.float64)
    count = np.zeros(12, dtype=np.int64)
    for time_index in range(values.shape[0]):
        month = time_index % 12
        if count[month] == 0:
            output[time_index] = values[time_index]
            previous[time_index] = values[time_index]
        else:
            output[time_index] = state[month] / count[month]
            previous[time_index] = values[time_index - 12]
        state[month] += values[time_index]
        count[month] += 1
    return output, previous


def cosine_overlap(left: np.ndarray, right: np.ndarray, months: float = 12.0) -> np.ndarray:
    numerator = ema(left * right, months)
    denominator = np.sqrt(
        np.maximum(ema(np.square(left), months) * ema(np.square(right), months), 1e-12)
    )
    return np.clip(numerator / denominator, -1.0, 1.0)


def causal_center(values: np.ndarray, months: float = 12.0) -> np.ndarray:
    """Remove slow learned magnitude without a future year normalization."""
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.zeros(values.shape[1], dtype=np.float32)
    output = np.empty_like(values, dtype=np.float32)
    for time_index in range(values.shape[0]):
        output[time_index] = values[time_index] - state
        state += alpha * (values[time_index] - state)
    return output


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.abs(values) * weights) / max(float(np.sum(weights)), 1e-12))


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


def print_ranks(label: str, feature_folds, pair_folds) -> None:
    total_features = sum(feature_folds, Counter())
    total_pairs = sum(pair_folds, Counter())
    print(f"RANK label={label} kind=feature", flush=True)
    for name, count in total_features.most_common(30):
        stability = sum(
            name in {item for item, _ in fold.most_common(30)}
            for fold in feature_folds
        )
        print(
            f"FEATURE label={label} name={name} splits={count} stable={stability}/4",
            flush=True,
        )
    print(f"RANK label={label} kind=pair", flush=True)
    for pair, count in total_pairs.most_common(40):
        stability = sum(
            pair in {item for item, _ in fold.most_common(40)}
            for fold in pair_folds
        )
        print(
            f"PAIR label={label} left={pair[0]} right={pair[1]} "
            f"splits={count} stable={stability}/4",
            flush=True,
        )


def fit_oof(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    names: tuple[str, ...],
    label: str,
    min_samples_leaf: int,
) -> np.ndarray:
    output = np.empty_like(target, dtype=np.float32)
    feature_folds: list[Counter] = []
    pair_folds: list[Counter] = []
    for fold in range(4):
        train, held = folds != fold, folds == fold
        learner = HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=100,
            learning_rate=0.055,
            l2_regularization=2.0,
            min_samples_leaf=min_samples_leaf,
            early_stopping=False,
            random_state=28100 + 100 * ("season" in label) + fold,
        )
        learner.fit(x[train], target[train], sample_weight=weights[train])
        output[held] = learner.predict(x[held]).astype(np.float32)
        features, pairs = tree_counts(learner, names)
        feature_folds.append(features)
        pair_folds.append(pairs)
        before = weighted_mae(target[held], weights[held])
        after = weighted_mae(target[held] - output[held], weights[held])
        print(
            f"FOLD label={label} fold={fold} baseline_mae={before:.9f} "
            f"corrected_mae={after:.9f} delta={after-before:+.9f}",
            flush=True,
        )
        del learner
        gc.collect()
    print_ranks(label, feature_folds, pair_folds)
    return output


def selected_data(model, rows: np.ndarray, columns: np.ndarray):
    return {
        name: np.asarray(selected_input(name, rows, columns)[:, 0, :], dtype=np.float32)
        for name in model.INPUTS
    }


def build_monthly_features(
    data: Mapping[str, np.ndarray], prediction: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, int, dict[str, np.ndarray]]:
    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != prediction.shape or not np.isfinite(array).all():
            raise ValueError(f"bad monthly feature {name}: {array.shape}")
        names.append(name)
        columns.append(array.reshape(-1))

    dynamic_sources = {
        "rain": np.clip(data["monthly_precipitation"], 0.0, None),
        "dry": np.clip(data["dryness"], 0.0, None),
        "temp": data["air_temperature"],
        "gpp": np.clip(data["gpp"], 0.0, None),
        "lai": np.clip(data["leaf_area_index"], 0.0, None),
        "flash": np.clip(data["lightning_flash_rate"], 0.0, None),
    }
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    add("base:log_hazard", np.log(hazard + EPS))
    add("base:hazard_ema12", ema(hazard, 12.0))
    add("base:hazard_ema36", ema(hazard, 36.0))
    memories: dict[str, dict[int, np.ndarray]] = {}
    for short, values in dynamic_sources.items():
        add(f"base:{short}:current", values)
        memories[short] = {}
        for months in (3, 12):
            memory = ema(values, months)
            memories[short][months] = memory
            add(f"base:{short}:ema{months}", memory)
            scale = 10.0 if short == "rain" else 100.0 if short == "dry" else 0.1
            add(
                f"base:{short}:departure{months}",
                (values - memory) / (np.abs(values) + np.abs(memory) + scale),
            )

    cover_sources = {
        "natural": np.clip(data["natural_vegetation_fraction"], 0.0, 1.0),
        "secondary": np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0),
        "primary": np.clip(data["luh2_primary_fraction"], 0.0, 1.0),
        "crop": np.clip(data["luh2_cropland_fraction"], 0.0, 1.0),
        "pasture": np.clip(data["luh2_pasture_fraction"], 0.0, 1.0),
        "range": np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0),
        "urban": np.clip(data["luh2_urban_fraction"], 0.0, 1.0),
        "biomass": np.clip(data["aboveground_biomass"], 0.0, None),
        "soil": np.clip(data["soil_carbon"], 0.0, None),
        "canopy": np.clip(data["natural_canopy_height"], 0.0, None),
        "secondary_canopy": np.clip(data["secondary_canopy_height"], 0.0, None),
    }
    for short, values in cover_sources.items():
        add(f"base:{short}", values)

    rain, dry = dynamic_sources["rain"], dynamic_sources["dry"]
    temp, gpp = dynamic_sources["temp"], dynamic_sources["gpp"]
    flash = dynamic_sources["flash"]
    combustion = dry / (dry + 250.0) / (1.0 + rain / 35.0)
    fuel = memories["gpp"][12] / (memories["gpp"][12] + 0.35)
    ignition = memories["flash"][12] / (memories["flash"][12] + 0.02)
    managed = np.clip(cover_sources["pasture"] + cover_sources["range"], 0.0, 1.0)
    natural_open = cover_sources["natural"] * 8.0 / (cover_sources["canopy"] + 8.0)
    secondary_open = cover_sources["secondary"] * 8.0 / (
        cover_sources["secondary_canopy"] + 8.0
    )
    open_cover = np.clip(natural_open + secondary_open + managed, 0.0, 2.0)
    woody = cover_sources["natural"] * cover_sources["canopy"] / (
        cover_sources["canopy"] + 8.0
    ) * cover_sources["biomass"] / (cover_sources["biomass"] + 2.0)
    continuity = 1.0 / (
        1.0 + 2.0 * np.power(cover_sources["crop"], 1.5) + 5.0 * cover_sources["urban"]
    )
    add("base:path:combustion", combustion)
    add("base:path:fuel", fuel)
    add("base:path:ignition", ignition)
    add("base:path:connected_surface", open_cover * fuel * continuity * combustion)
    add("base:path:managed_event", managed * fuel * combustion * (0.25 + 0.75 * ignition))
    add("base:path:woody_event", woody * combustion * ignition)
    base_count = len(names)

    normals: dict[str, np.ndarray] = {}
    previous: dict[str, np.ndarray] = {}
    for short, values in dynamic_sources.items():
        normal, last_year = causal_seasonal_normal(values)
        normals[short] = normal
        previous[short] = last_year
        scale = 10.0 if short == "rain" else 100.0 if short == "dry" else 0.1
        add(f"novel:{short}:seasonal_normal", normal)
        add(
            f"novel:{short}:seasonal_anomaly",
            (values - normal) / (np.abs(values) + np.abs(normal) + scale),
        )
        add(
            f"novel:{short}:year_on_year",
            (values - last_year) / (np.abs(values) + np.abs(last_year) + scale),
        )

    normal_combustion = normals["dry"] / (normals["dry"] + 250.0) / (
        1.0 + normals["rain"] / 35.0
    )
    normal_fuel = normals["gpp"] / (normals["gpp"] + 0.35)
    normal_ignition = normals["flash"] / (normals["flash"] + 0.02)
    expected_window = normal_combustion * normal_fuel * (0.25 + 0.75 * normal_ignition)
    dry_surprise = np.maximum(
        (combustion - normal_combustion) / (combustion + normal_combustion + 0.05), 0.0
    )
    curing = np.maximum(
        (normals["gpp"] - gpp) / (normals["gpp"] + gpp + 0.1), 0.0
    )
    add("novel:path:expected_window", expected_window)
    add("novel:path:dry_surprise", dry_surprise)
    add("novel:path:seasonal_curing", curing)
    add("novel:path:open_expected_window", open_cover * continuity * expected_window)
    add("novel:path:managed_expected_window", managed * expected_window)
    add("novel:path:primary_dry_surprise", cover_sources["primary"] * dry_surprise)
    add("novel:path:curing_release", fuel * curing * combustion)
    add("novel:path:managed_curing_release", managed * fuel * curing * combustion)
    add("novel:path:fuel_combustion_overlap", cosine_overlap(fuel, combustion))
    add("novel:path:ignition_combustion_overlap", cosine_overlap(ignition, combustion))
    add("novel:path:temperature_combustion_overlap", cosine_overlap(temp + 40.0, combustion))
    add(
        "novel:path:open_fuel_combustion_overlap",
        open_cover * continuity * cosine_overlap(fuel, combustion),
    )
    add(
        "novel:path:managed_ignition_overlap",
        managed * cosine_overlap(ignition, combustion),
    )

    auxiliary = {
        "combustion": combustion,
        "fuel": fuel,
        "ignition": ignition,
        "managed": managed,
        "open_cover": open_cover,
        "woody": woody,
        "expected_window": expected_window,
        "dry_surprise": dry_surprise,
        "curing": curing,
    }
    x = np.column_stack(columns).astype(np.float32, copy=False)
    return tuple(names), x, base_count, auxiliary


def build_annual_features(
    data: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    monthly_names: tuple[str, ...],
    monthly_x: np.ndarray,
    monthly_base_count: int,
) -> tuple[tuple[str, ...], np.ndarray, int]:
    count = prediction.shape[1]
    month_cube = monthly_x.reshape(192, count, -1)
    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (15, count) or not np.isfinite(array).all():
            raise ValueError(f"bad annual feature {name}: {array.shape}")
        names.append(name)
        columns.append(array.reshape(-1))

    pred_year = prediction.reshape(16, 12, count).sum(axis=1)
    add("base:previous_fire", pred_year[:-1])
    add("base:previous_fire_log", np.log(pred_year[:-1] + EPS))
    # Prior-year summaries contain no information from the target year.
    for index, name in enumerate(monthly_names[:monthly_base_count]):
        values = month_cube[:, :, index].reshape(16, 12, count)[:-1]
        add(f"{name}:prior_mean", values.mean(axis=1))
        if name in (
            "base:rain:current", "base:dry:current", "base:temp:current",
            "base:gpp:current", "base:flash:current", "base:path:combustion",
            "base:path:connected_surface", "base:path:managed_event",
        ):
            add(f"{name}:prior_std", values.std(axis=1))
    base_count = len(names)

    # The novel columns are themselves prefix-causal.  Their prior-year means
    # and variability summarize an established seasonal normal and departures
    # from it at the start of the following year.
    for index, name in enumerate(monthly_names[monthly_base_count:], start=monthly_base_count):
        values = month_cube[:, :, index].reshape(16, 12, count)[:-1]
        add(f"{name}:prior_mean", values.mean(axis=1))
        if any(token in name for token in ("anomaly", "year_on_year", "expected_window", "overlap", "curing_release")):
            add(f"{name}:prior_std", values.std(axis=1))
    x = np.column_stack(columns).astype(np.float32, copy=False)
    return tuple(names), x, base_count


def score_variant(
    evaluator: GFED5Evaluator,
    baseline_grid: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, dict[str, float]]:
    candidate = np.array(baseline_grid, dtype=np.float32, copy=True)
    candidate[:, rows, columns] = np.clip(corrected, 0.0, 1.0)
    scores = evaluator.score(validate_prediction(candidate))
    del candidate
    gc.collect()
    return scores


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    pred_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    return {
        name: float(np.sum(pred_annual[mask] * area[mask]))
        / max(float(np.sum(obs_annual[mask] * area[mask])), 1e-12)
        for name, mask in masks.items()
    }


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline_grid = build_current_cache(model, land)
    base_scores = evaluator.score(validate_prediction(baseline_grid))
    base_global = base_scores["global"]
    if abs(float(base_global["overall_score"]) - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(f"current mismatch: {metric_text(base_global)}")
    print(f"BASE {metric_text(base_global)}", flush=True)

    observation_grid = load_observation()
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    rows, columns, _, retained = select_high_weight(observation_grid, area_grid)
    count = rows.size
    print(
        f"SELECTION cells={count} observed_fire_weight={retained:.9f} "
        "coordinates_features=0 spatial_folds=4",
        flush=True,
    )
    data = selected_data(model, rows, columns)
    baseline = np.asarray(baseline_grid[:, rows, columns], dtype=np.float32)
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float32)
    direct = np.asarray(model.predict({name: values[:, None, :] for name, values in data.items()}, dict(model.PARAMS), None), dtype=np.float32)[:, 0, :]
    direct_delta = float(np.max(np.abs(direct - baseline)))
    if direct_delta > 1e-7:
        raise RuntimeError(f"selected current mismatch: {direct_delta}")
    print(f"BASE_CHECK max_abs={direct_delta:.12g}", flush=True)
    del direct

    monthly_names, monthly_x, monthly_base_count, _ = build_monthly_features(data, baseline)
    annual_names, annual_x, annual_base_count = build_annual_features(
        data, baseline, monthly_names, monthly_x, monthly_base_count
    )
    print(
        f"FEATURES annual_rows={annual_x.shape[0]} annual_base={annual_base_count} "
        f"annual_full={annual_x.shape[1]} monthly_rows={monthly_x.shape[0]} "
        f"monthly_base={monthly_base_count} monthly_full={monthly_x.shape[1]}",
        flush=True,
    )

    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    annual_folds = np.tile(cell_folds, (15, 1)).reshape(-1)
    monthly_folds = np.tile(cell_folds, (180, 1)).reshape(-1)
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    pred_year = baseline.reshape(16, 12, count).sum(axis=1)
    obs_year = observation.reshape(16, 12, count).sum(axis=1)
    annual_target = np.clip(
        np.log((obs_year[1:] + EPS) / (pred_year[1:] + EPS)), -3.0, 3.0
    ).reshape(-1).astype(np.float32)
    obs_floor = float(np.sum(obs_year[1:] * area[None, :])) / (
        15.0 * float(np.sum(area))
    )
    annual_weights = (
        area[None, :] * (obs_year[1:] + 0.02 * obs_floor)
    ).reshape(-1)
    annual_weights /= annual_weights.mean()

    base_alloc = baseline.reshape(16, 12, count) / (pred_year[:, None, :] + EPS)
    obs_alloc = observation.reshape(16, 12, count) / (obs_year[:, None, :] + EPS)
    seasonal_target = np.clip(
        np.log((obs_alloc[1:] + EPS) / (base_alloc[1:] + EPS)), -3.0, 3.0
    ).reshape(-1).astype(np.float32)
    seasonal_weights = (
        area[None, None, :]
        * (obs_year[1:, None, :] + 0.02 * obs_floor)
        * (obs_alloc[1:] + 0.02 / 12.0)
    ).reshape(-1)
    seasonal_weights /= seasonal_weights.mean()

    annual_oof = {}
    for label, x, names in (
        ("annual_base", annual_x[:, :annual_base_count], annual_names[:annual_base_count]),
        ("annual_seasonal_memory", annual_x, annual_names),
    ):
        annual_oof[label] = fit_oof(
            x, annual_target, annual_weights, annual_folds, names, label, 180
        ).reshape(15, count)

    monthly_used = monthly_x.reshape(192, count, -1)[12:].reshape(180 * count, -1)
    seasonal_oof = {}
    for label, x, names in (
        ("season_base", monthly_used[:, :monthly_base_count], monthly_names[:monthly_base_count]),
        ("season_seasonal_memory", monthly_used, monthly_names),
    ):
        raw = fit_oof(
            x, seasonal_target, seasonal_weights, monthly_folds, names, label, 800
        ).reshape(180, count)
        seasonal_oof[label] = causal_center(raw)
    del annual_x, monthly_x, monthly_used
    gc.collect()

    variants: list[tuple[float, str, dict[str, dict[str, float]], np.ndarray]] = []
    annual_corrected = {}
    for label, correction in annual_oof.items():
        for blend in (0.25, 0.50, 1.0):
            corrected = baseline.copy()
            factor = np.exp(np.clip(blend * correction, -2.0, 2.0)).astype(np.float32)
            corrected[12:] *= np.repeat(factor[:, None, :], 12, axis=1).reshape(180, count)
            scores = score_variant(evaluator, baseline_grid, rows, columns, corrected)
            full_label = f"{label}:blend={blend:g}"
            variants.append((float(scores["global"]["overall_score"]), full_label, scores, corrected))
            annual_corrected[(label, blend)] = corrected
            print(f"SCORE label={full_label} {metric_text(scores['global'])}", flush=True)

    seasonal_corrected = {}
    for label, correction in seasonal_oof.items():
        for blend in (0.10, 0.25, 0.50):
            corrected = baseline.copy()
            corrected[12:] *= np.exp(np.clip(blend * correction, -2.0, 2.0)).astype(np.float32)
            scores = score_variant(evaluator, baseline_grid, rows, columns, corrected)
            full_label = f"{label}:blend={blend:g}"
            variants.append((float(scores["global"]["overall_score"]), full_label, scores, corrected))
            seasonal_corrected[(label, blend)] = corrected
            print(f"SCORE label={full_label} {metric_text(scores['global'])}", flush=True)

    # Combine only the enriched annual and seasonal heads.  This is an OOF
    # information ceiling, not a model proposal or a parameter search.
    for annual_blend in (0.25, 0.50):
        for seasonal_blend in (0.10, 0.25):
            corrected = annual_corrected[("annual_seasonal_memory", annual_blend)].copy()
            corrected[12:] *= np.exp(
                np.clip(
                    seasonal_blend * seasonal_oof["season_seasonal_memory"], -2.0, 2.0
                )
            ).astype(np.float32)
            scores = score_variant(evaluator, baseline_grid, rows, columns, corrected)
            label = f"joint_memory:annual={annual_blend:g}:season={seasonal_blend:g}"
            variants.append((float(scores["global"]["overall_score"]), label, scores, corrected))
            print(f"SCORE label={label} {metric_text(scores['global'])}", flush=True)

    variants.sort(key=lambda item: item[0], reverse=True)
    print("TOP_EXACT_OOF", flush=True)
    for overall, label, scores, _ in variants[:12]:
        print(
            f"TOP label={label} delta={overall-float(base_global['overall_score']):+.9f} "
            f"{metric_text(scores['global'])}",
            flush=True,
        )

    best_overall, best_label, best_scores, best_selected = variants[0]
    full_rows, full_columns = np.nonzero(land)
    full_data = {
        name: np.asarray(selected_input(name, full_rows, full_columns)[:, 0, :], dtype=np.float32)
        for name in (
            "monthly_precipitation", "air_temperature", "leaf_area_index",
            "natural_canopy_height", "aboveground_biomass",
            "natural_vegetation_fraction", "luh2_primary_fraction",
            "luh2_cropland_fraction", "luh2_rangeland_fraction",
        )
    }
    full_mean = {name: values.mean(axis=0) for name, values in full_data.items()}
    masks = ecology_masks(full_mean, full_rows.size)
    baseline_land = np.asarray(baseline_grid[:, full_rows, full_columns], dtype=np.float32)
    best_grid = np.array(baseline_grid, dtype=np.float32, copy=True)
    best_grid[:, rows, columns] = np.clip(best_selected, 0.0, 1.0)
    best_land = best_grid[:, full_rows, full_columns]
    obs_land = observation_grid[:, full_rows, full_columns]
    land_area = area_grid[full_rows, full_columns]
    baseline_ecology = ecology_ratios(baseline_land, obs_land, land_area, masks)
    best_ecology = ecology_ratios(best_land, obs_land, land_area, masks)
    for name in masks:
        print(
            f"ECOLOGY best={best_label} name={name} baseline={baseline_ecology[name]:.9f} "
            f"oof={best_ecology[name]:.9f} delta={best_ecology[name]-baseline_ecology[name]:+.9f}",
            flush=True,
        )
    positive = 0
    for name in sorted(key for key in best_scores if key != "global"):
        delta = float(best_scores[name]["overall_score"] - base_scores[name]["overall_score"])
        positive += int(delta > 0.0)
        print(
            f"REGION best={best_label} name={name} delta={delta:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH best={best_label} positive={positive}/14", flush=True)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        f"DONE best={best_label} best_overall={best_overall:.9f} "
        f"delta={best_overall-float(base_global['overall_score']):+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f} "
        f"peak_rss_gib={peak/(1024.0**3):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
