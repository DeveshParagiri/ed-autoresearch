"""Exact held-block annual-map and normalized-cycle headroom audit.

This diagnostic is pinned to the pruned causal canonical model.  It constructs
the full incumbent prediction in small land-cell chunks solely so held-out
corrections on the highest-reference-weight cells can be scored by the exact
GFED5 evaluator while every untouched cell remains canonical.  Learned
surfaces are diagnostic only and never enter ``model.py`` or the ledger.

Every predictor is a globally shared ecological pathway interaction built from
current local inputs or prefix-causal memories.  Coordinates assign held
spatial blocks only.  Regions, calendar indices, targets, future values,
completed-year precipitation, neighbours, and invalid modern forcings are not
features.  LUH2 transition terms are causal departures from trailing land-use
state, not fitted geographic labels.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_land_mask,
    load_model,
    validate_prediction,
)


EXPECTED_MODEL_BLOB = "39ee93ebf1155af9ae9d70e05847b9c3f086887d"
EXPECTED_BASE = 0.718363408
CACHE = Path(__file__).with_name(
    f"canonical_{EXPECTED_MODEL_BLOB[:8]}_chunked.npy"
)
MONTH_DAYS = np.tile(
    np.asarray(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
        dtype=np.float64,
    ),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    output = np.empty_like(values, dtype=np.float32)
    state = np.asarray(values[0], dtype=np.float32).copy()
    for time_index in range(values.shape[0]):
        state += alpha * (values[time_index] - state)
        output[time_index] = state
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


def load_observation() -> np.ndarray:
    output = np.empty((192, 180, 360), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for row in range(180):
            slab = np.ma.asarray(variable[:192, 2 * row: 2 * row + 2, :])
            if np.ma.getmaskarray(slab).any():
                raise ValueError("masked GFED observation")
            output[:, row, :] = np.asarray(slab, dtype=np.float32).reshape(
                192, 2, 360, 2
            ).mean(axis=(1, 3)) / 100.0
    return output


def selected_inputs(model, rows: np.ndarray, columns: np.ndarray):
    return {
        name: selected_input(name, rows, columns)
        for name in model.INPUTS
    }


def chunked_incumbent(model, land: np.ndarray) -> np.ndarray:
    if CACHE.exists():
        cached = np.load(CACHE)
        if cached.shape != (192, 180, 360) or cached.dtype != np.float32:
            raise ValueError(f"invalid cache {CACHE}: {cached.shape} {cached.dtype}")
        print(f"BASE_CACHE reused={CACHE} bytes={CACHE.stat().st_size}", flush=True)
        return cached

    output = np.zeros((192, 180, 360), dtype=np.float32)
    rows, columns = np.nonzero(land)
    chunk_size = 1536
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        data = selected_inputs(model, rows[start:stop], columns[start:stop])
        prediction = np.asarray(
            model.predict(data, dict(model.PARAMS), None), dtype=np.float32
        )[:, 0, :]
        output[:, rows[start:stop], columns[start:stop]] = prediction
        print(
            f"BASE_CHUNK start={start} stop={stop} cells={rows.size}",
            flush=True,
        )
        del data, prediction
        gc.collect()
    np.save(CACHE, output, allow_pickle=False)
    print(f"BASE_CACHE created={CACHE} bytes={CACHE.stat().st_size}", flush=True)
    return output


def select_high_weight(
    observation: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    annual = observation.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    weight = area * annual
    order = np.argsort(weight.ravel())[::-1]
    cumulative = np.cumsum(weight.ravel()[order]) / max(float(weight.sum()), 1e-12)
    count = int(np.searchsorted(cumulative, 0.85) + 1)
    selected = order[:count]
    retained = float(weight.ravel()[selected].sum() / weight.sum())
    return selected // 360, selected % 360, weight.ravel()[selected], retained


def build_pathway_features(
    data: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    def field(name: str) -> np.ndarray:
        return np.asarray(data[name][:, 0, :], dtype=np.float32)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    temperature = field("air_temperature")
    dryness = np.clip(field("dryness"), 0.0, None)
    gpp = np.clip(field("gpp"), 0.0, None)
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    biomass = np.clip(field("aboveground_biomass"), 0.0, None)
    natural = np.clip(field("natural_vegetation_fraction"), 0.0, 1.0)
    canopy = np.clip(field("natural_canopy_height"), 0.0, None)
    secondary = np.clip(field("secondary_vegetation_fraction"), 0.0, 1.0)
    secondary_canopy = np.clip(field("secondary_canopy_height"), 0.0, None)
    primary = np.clip(field("luh2_primary_fraction"), 0.0, 1.0)
    crop = np.clip(field("luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field("luh2_urban_fraction"), 0.0, 1.0)

    rain3, rain12 = ema(rain, 3.0), ema(rain, 12.0)
    temperature3, temperature24 = ema(temperature, 3.0), ema(temperature, 24.0)
    dryness3, dryness12 = ema(dryness, 3.0), ema(dryness, 12.0)
    gpp3, gpp12 = ema(gpp, 3.0), ema(gpp, 12.0)
    lightning3, lightning12 = ema(lightning, 3.0), ema(lightning, 12.0)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    hazard12 = ema(hazard, 12.0)

    annual_rain = 12.0 * rain12
    rain_built = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
        -annual_rain / 3000.0
    )
    drying = np.maximum((dryness - dryness3) / (dryness + dryness3 + 100.0), 0.0)
    rain_deficit = np.maximum((rain12 - rain) / (rain12 + rain + 10.0), 0.0)
    gpp_decline = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.1), 0.0)
    warming = sigmoid((temperature - temperature3 - 0.5) / 1.5)
    lightning_arrival = np.maximum(
        (lightning - lightning3) / (lightning + lightning3 + 0.002), 0.0
    )
    ignition = lightning12 / (lightning12 + 0.02) * (
        0.35 + 0.65 * sigmoid((lightning_arrival - 0.05) / 0.10)
    )
    combustion = np.sqrt(
        dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    )
    fuel = gpp12 / (gpp12 + 0.35)
    open_natural = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    woody = natural * canopy / (canopy + 8.0) * biomass / (biomass + 2.0)
    recurrence = hazard12 / (hazard12 + 0.01)
    opportunity = 1.0 - recurrence
    cold = sigmoid((5.0 - temperature24) / 3.0)
    warm = sigmoid((temperature24 - 18.0) / 3.0)

    luh = {
        "primary": primary,
        "crop": crop,
        "pasture": pasture,
        "rangeland": rangeland,
        "urban": urban,
    }
    luh_memory = {name: ema(values, 12.0) for name, values in luh.items()}
    gain = {
        name: np.maximum(values - luh_memory[name], 0.0)
        for name, values in luh.items()
    }
    loss = {
        name: np.maximum(luh_memory[name] - values, 0.0)
        for name, values in luh.items()
    }
    managed_gain = np.clip(gain["crop"] + gain["pasture"] + gain["rangeland"], 0.0, 1.0)
    managed_loss = np.clip(loss["crop"] + loss["pasture"] + loss["rangeland"], 0.0, 1.0)
    turnover = sum(np.abs(luh[name] - luh_memory[name]) for name in luh)

    pathways = {
        "rain_built_drydown": rain_built * fuel * rain_deficit * combustion,
        "warming_drying_ignition": warming * drying * combustion * ignition,
        "natural_open_event": open_natural * fuel * combustion * ignition,
        "secondary_open_event": secondary_open * fuel * combustion * ignition,
        "managed_open_event": managed_open * fuel * combustion * (0.25 + 0.75 * ignition),
        "crop_event": crop * fuel * combustion * warming,
        "woody_rare_event": woody * opportunity * combustion * ignition,
        "recurrent_surface_event": recurrence * (open_natural + secondary_open) * combustion,
        "humid_woody_shield": woody * rain12 / (rain12 + 80.0) * (1.0 - combustion),
        "cold_open_thaw": cold * (open_natural + managed_open) * warming * combustion * ignition,
        "warm_open_recurrence": warm * (open_natural + managed_open) * recurrence * combustion,
        "live_to_dead_transition": fuel * gpp_decline * drying * combustion,
        "primary_conversion_residue": loss["primary"] * managed_gain * biomass / (biomass + 0.5) * combustion,
        "crop_expansion_combustion": gain["crop"] * fuel * combustion * (0.25 + 0.75 * ignition),
        "grazing_expansion_combustion": (gain["pasture"] + gain["rangeland"]) * fuel * combustion,
        "managed_abandonment_regrowth": managed_loss * secondary_open * rain_built * drying,
        "urban_expansion_fragmentation": gain["urban"] * recurrence * (crop + managed_open),
        "landuse_turnover_combustion": turnover * fuel * combustion,
    }
    transition_diagnostics = {
        "primary_loss": loss["primary"],
        "managed_gain": managed_gain,
        "crop_gain": gain["crop"],
        "grazing_gain": gain["pasture"] + gain["rangeland"],
        "managed_loss": managed_loss,
        "urban_gain": gain["urban"],
        "turnover": turnover,
    }
    names = tuple(pathways)
    x = np.column_stack([pathways[name].reshape(-1) for name in names]).astype(
        np.float32, copy=False
    )
    return names, x, transition_diagnostics


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


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def fit_oof(
    x: np.ndarray,
    target: np.ndarray,
    row_weights: np.ndarray,
    folds: np.ndarray,
    names: tuple[str, ...],
    label: str,
    depth: int,
):
    output = np.empty_like(target, dtype=np.float32)
    feature_counts: list[Counter] = []
    pair_counts: list[Counter] = []
    for fold in range(4):
        train, held = folds != fold, folds == fold
        learner = HistGradientBoostingRegressor(
            max_depth=depth,
            max_iter=70,
            learning_rate=0.06,
            l2_regularization=1.5,
            min_samples_leaf=180,
            early_stopping=False,
            random_state=2710 + 100 * depth + 10 * fold + (label == "cycle"),
        )
        learner.fit(x[train], target[train], sample_weight=row_weights[train])
        output[held] = learner.predict(x[held]).astype(np.float32)
        features, pairs = tree_counts(learner, names)
        feature_counts.append(features)
        pair_counts.append(pairs)
        before = weighted_mean(np.abs(target[held]), row_weights[held])
        after = weighted_mean(np.abs(target[held] - output[held]), row_weights[held])
        print(
            f"FOLD target={label} depth={depth} fold={fold} "
            f"baseline_mae={before:.9f} corrected_mae={after:.9f} "
            f"delta={after - before:+.9f}",
            flush=True,
        )
    return output, feature_counts, pair_counts


def print_rank(label: str, names: tuple[str, ...], feature_counts, pair_counts):
    feature_total = Counter()
    pair_total = Counter()
    for counts in feature_counts:
        feature_total.update(counts)
    for counts in pair_counts:
        pair_total.update(counts)
    print(f"PATHWAY_RANK target={label}", flush=True)
    for name, count in feature_total.most_common():
        stability = sum(name in fold.most_common(8) for fold in feature_counts)
        # Counter.most_common returns pairs; compare names explicitly.
        stability = sum(
            name in {item for item, _ in fold.most_common(8)}
            for fold in feature_counts
        )
        print(
            f"PATHWAY target={label} name={name} splits={count} top8_folds={stability}",
            flush=True,
        )
    print(f"PAIR_RANK target={label}", flush=True)
    for (left, right), count in pair_total.most_common(20):
        stability = sum(
            (left, right) in {item for item, _ in fold.most_common(12)}
            for fold in pair_counts
        )
        print(
            f"PAIR target={label} left={left} right={right} "
            f"splits={count} top12_folds={stability}",
            flush=True,
        )


def score_selected(
    evaluator: GFED5Evaluator,
    baseline: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    corrected: np.ndarray,
):
    candidate = baseline.copy()
    candidate[:, rows, columns] = np.clip(corrected, 0.0, 1.0)
    score = evaluator.score(validate_prediction(candidate))["global"]
    del candidate
    gc.collect()
    return score


def main() -> int:
    started = time.perf_counter()
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"model blob changed: {blob}")
    model = load_model()
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline = chunked_incumbent(model, land)
    base_score = evaluator.score(validate_prediction(baseline))["global"]
    if abs(float(base_score["overall_score"]) - EXPECTED_BASE) > 5e-9:
        raise RuntimeError(f"baseline mismatch: {metric_text(base_score)}")
    print(f"BASE model_blob={blob} {metric_text(base_score)}", flush=True)

    observation = load_observation()
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    rows, columns, cell_weight, retained = select_high_weight(observation, area)
    count = rows.size
    print(
        f"SELECTION cells={count} observed_fire_weight={retained:.9f} "
        "coordinates_used_only_for_folds=1",
        flush=True,
    )
    data = selected_inputs(model, rows, columns)
    selected_base = baseline[:, rows, columns]
    selected_check = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    check_delta = float(np.max(np.abs(selected_base - selected_check)))
    if check_delta > 1e-7:
        raise RuntimeError(f"chunked/selective mismatch: {check_delta}")
    print(f"BASE_CHECK max_abs={check_delta:.12g}", flush=True)
    selected_obs = observation[:, rows, columns]
    del observation, land
    gc.collect()

    names, x, transition_diagnostics = build_pathway_features(data, selected_base)
    for name, values in transition_diagnostics.items():
        positive = values[values > 1e-8]
        print(
            f"TRANSITION name={name} nonzero_fraction={float(np.mean(values > 1e-8)):.9f} "
            f"p95={float(np.quantile(values, .95)):.9g} "
            f"positive_p95={float(np.quantile(positive, .95)) if positive.size else 0.0:.9g} "
            f"max={float(values.max()):.9g}",
            flush=True,
        )
    del transition_diagnostics, data
    gc.collect()

    eps = np.float32(1e-6)
    base_cycle = selected_base.reshape(16, 12, count).mean(axis=0)
    obs_cycle = selected_obs.reshape(16, 12, count).mean(axis=0)
    base_annual = base_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    map_target_cell = np.clip(
        np.log((obs_annual + eps) / (base_annual + eps)), -4.0, 4.0
    ).astype(np.float32)
    map_target = np.tile(map_target_cell, 192)
    base_alloc = base_cycle / (base_annual[None, :] + eps)
    obs_alloc = obs_cycle / (obs_annual[None, :] + eps)
    cycle_cell_month = np.clip(obs_alloc - base_alloc, -0.5, 0.5).astype(np.float32)
    cycle_target = np.tile(cycle_cell_month, (16, 1, 1)).reshape(-1)

    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    folds = np.tile(cell_folds, 192)
    row_weights = np.tile(cell_weight, 192).astype(np.float64)
    row_weights /= row_weights.mean()
    month_index = np.arange(192) % 12
    all_results: list[tuple[float, str, dict[str, float]]] = []

    for depth in (2, 3):
        map_oof, map_features, map_pairs = fit_oof(
            x, map_target, row_weights, folds, names, "map", depth
        )
        cycle_oof, cycle_features, cycle_pairs = fit_oof(
            x, cycle_target, row_weights, folds, names, "cycle", depth
        )
        print_rank(f"map_depth{depth}", names, map_features, map_pairs)
        print_rank(f"cycle_depth{depth}", names, cycle_features, cycle_pairs)

        map_prediction = map_oof.reshape(192, count).mean(axis=0)
        cycle_prediction = cycle_oof.reshape(16, 12, count).mean(axis=0)
        for blend in (0.1, 0.25, 0.5, 1.0):
            factor = np.exp(np.clip(blend * map_prediction, -2.0, 2.0))
            corrected = selected_base * factor[None, :]
            score = score_selected(evaluator, baseline, rows, columns, corrected)
            label = f"depth={depth}:map={blend:g}"
            all_results.append((float(score["overall_score"]), label, dict(score)))
            print(f"SCORE {label} {metric_text(score)}", flush=True)

        cycle_corrected: dict[float, np.ndarray] = {}
        for blend in (0.25, 0.5, 1.0):
            allocation = np.clip(base_alloc + blend * cycle_prediction, 1e-6, None)
            allocation /= np.maximum(allocation.sum(axis=0, keepdims=True), 1e-12)
            ratio = np.clip(allocation / (base_alloc + 1e-6), 0.1, 10.0)
            corrected = selected_base * ratio[month_index]
            cycle_corrected[blend] = corrected
            score = score_selected(evaluator, baseline, rows, columns, corrected)
            label = f"depth={depth}:cycle={blend:g}"
            all_results.append((float(score["overall_score"]), label, dict(score)))
            print(f"SCORE {label} {metric_text(score)}", flush=True)

        for map_blend, cycle_blend in (
            (0.1, 0.5),
            (0.1, 1.0),
            (0.25, 0.5),
            (0.25, 1.0),
            (0.5, 0.5),
            (0.5, 1.0),
        ):
            factor = np.exp(np.clip(map_blend * map_prediction, -2.0, 2.0))
            corrected = cycle_corrected[cycle_blend] * factor[None, :]
            score = score_selected(evaluator, baseline, rows, columns, corrected)
            label = f"depth={depth}:joint_map={map_blend:g}:cycle={cycle_blend:g}"
            all_results.append((float(score["overall_score"]), label, dict(score)))
            print(f"SCORE {label} {metric_text(score)}", flush=True)
        del map_oof, cycle_oof, cycle_corrected
        gc.collect()

    print("TOP_EXACT", flush=True)
    for overall, label, score in sorted(all_results, reverse=True)[:16]:
        print(
            f"TOP label={label} delta={overall - float(base_score['overall_score']):+.9f} "
            f"{metric_text(score)}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        f"RESOURCES wall_seconds={elapsed:.6f} peak_rss_raw={peak_raw} "
        f"peak_rss_gib={peak_raw / (1024.0**3):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
