"""Held-cell falsification of persistent drying for surface event capacity.

The official 0.719 thermal-range term raises event capacity from instantaneous
combustion weather.  The South America residual diagnostic found that its gain
occurs under sustained atmospheric dryness, whereas intermittent seasonal dry
pulses in productive primary mosaics are already overburned.  This script tests
one globally shared physical refinement:

    h' = h * (1 + k * Q * S)
    S = EMA12(D) / (EMA12(D) + 500)

Here ``Q`` is the incumbent warm, rain-supported connected-surface modifier and
``S`` requires persistent desiccation before thermal range can enlarge a fine-
fuel fire footprint.  Coordinates and region bounds select balanced held audit
cells only.  No geography, target, neighbour, future value, or learned surface
enters the equation.  Diagnostic only; no canonical edit or official score.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_south_america_diagnostic_2127874 import (  # noqa: E402
    CURRENT_COMMIT,
    CURRENT_MODEL_BLOB,
    ema,
    pinned_current_model,
    sigmoid,
    trailing_std,
    weighted_mean,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    CACHE,
    EXPECTED_BASE,
    EXPECTED_MODEL_BLOB,
    load_observation,
)
from scripts.fast_ilamb import GFED_REGIONS  # noqa: E402
from scripts.runtime import load_land_mask  # noqa: E402


AUDIT_REGIONS = tuple(name for name in GFED_REGIONS if name != "global")
EPS = np.float32(1e-6)


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


def select_balanced_cells(
    rows: np.ndarray,
    columns: np.ndarray,
    observed_annual: np.ndarray,
    baseline_annual: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latitude = -89.5 + rows.astype(np.float64)
    longitude = -179.5 + columns.astype(np.float64)
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    importance = area * (observed_annual + baseline_annual + 1e-7)
    selected: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for region_index, name in enumerate(AUDIT_REGIONS):
        inside = region_inside(name, latitude, longitude)
        for fold in range(4):
            candidates = np.flatnonzero(inside & (folds == fold))
            order = np.argsort(importance[candidates])[::-1]
            chosen = candidates[order[:48]]
            selected.append(chosen)
            labels.append(np.full(chosen.size, region_index, dtype=np.int8))
    return np.concatenate(selected), np.concatenate(labels)


def annual_loss(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, tuple[float, ...]]:
    model_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    floor = float(np.sum(obs_annual * area) / np.sum(area)) * 0.02
    weights = area * (obs_annual + floor)
    error = np.abs(np.log((model_annual + EPS) / (obs_annual + EPS)))
    total = weighted_mean(error, weights)
    held = tuple(
        weighted_mean(error[folds == fold], weights[folds == fold])
        for fold in range(4)
    )
    return total, held


def cycle_loss(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, tuple[float, ...]]:
    model_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation.reshape(16, 12, -1).mean(axis=0)

    def score(mask: np.ndarray) -> float:
        model = model_cycle[:, mask] * area[mask][None, :]
        obs = obs_cycle[:, mask] * area[mask][None, :]
        model /= np.maximum(model.sum(axis=0, keepdims=True), 1e-12)
        obs /= np.maximum(obs.sum(axis=0, keepdims=True), 1e-12)
        cell_weight = obs_cycle[:, mask].sum(axis=0) * area[mask]
        per_cell = np.sqrt(np.mean(np.square(model - obs), axis=0))
        return weighted_mean(per_cell, cell_weight)

    total_mask = np.ones(folds.shape, dtype=bool)
    return score(total_mask), tuple(score(folds == fold) for fold in range(4))


def ecological_masks(mean: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    annual_rain = 12.0 * mean["monthly_precipitation"]
    temperature = mean["air_temperature"]
    lai = mean["leaf_area_index"]
    canopy = mean["natural_canopy_height"]
    biomass = mean["aboveground_biomass"]
    natural = mean["natural_vegetation_fraction"]
    primary = mean["luh2_primary_fraction"]
    crop = mean["luh2_cropland_fraction"]
    rangeland = mean["luh2_rangeland_fraction"]
    return {
        "intact_tropical_closed": (temperature >= 20.0) & (annual_rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (temperature >= 20.0) & (annual_rain >= 500.0) & (annual_rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (annual_rain >= 250.0) & (annual_rain < 1500.0) & (biomass >= 0.2),
        "crop": crop >= 0.5,
        "arid_low_fuel": (annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def area_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray,
) -> float:
    model_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    return float(np.sum(model_annual[mask] * area[mask])) / max(
        float(np.sum(obs_annual[mask] * area[mask])), 1e-12
    )


def apply_capacity(
    baseline: np.ndarray,
    modifier: np.ndarray,
    persistent: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    adjusted = hazard * (1.0 + strength * modifier * persistent)
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def main() -> int:
    started = time.perf_counter()
    if CURRENT_COMMIT != "2127874ce757af418aeab4f5a5d93b765030ff57":
        raise RuntimeError("unexpected official model pin")
    if CURRENT_MODEL_BLOB != "ca6848f2db28af24a06cd9f06e3adcdecaf7fcc0":
        raise RuntimeError("unexpected official model blob")
    if EXPECTED_MODEL_BLOB != "39ee93ebf1155af9ae9d70e05847b9c3f086887d":
        raise RuntimeError("unexpected pre-capacity model pin")
    if abs(EXPECTED_BASE - 0.718363408) > 5e-10 or not CACHE.exists():
        raise RuntimeError("unexpected or missing pre-capacity baseline")

    model = pinned_current_model()
    land = load_land_mask()
    flat = np.flatnonzero(land.ravel())
    all_rows, all_columns = flat // 360, flat % 360
    baseline_grid = np.load(CACHE, mmap_mode="r")
    baseline_all = np.asarray(baseline_grid[:, all_rows, all_columns], dtype=np.float32)
    observation_grid = load_observation()
    observation_all = np.asarray(observation_grid[:, all_rows, all_columns], dtype=np.float32)
    del observation_grid
    observed_annual_all = observation_all.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    baseline_annual_all = baseline_all.reshape(16, 12, -1).mean(axis=0).sum(axis=0)

    radius = 6.371e6
    bounds = np.deg2rad(np.arange(-90.0, 91.0, 1.0))
    latitude_area = radius**2 * np.diff(np.sin(bounds)) * np.deg2rad(1.0)
    area_grid = np.repeat(latitude_area[:, None], 360, axis=1)
    area_all = area_grid[all_rows, all_columns]
    chosen, region_labels = select_balanced_cells(
        all_rows, all_columns, observed_annual_all, baseline_annual_all, area_all
    )
    rows, columns = all_rows[chosen], all_columns[chosen]
    baseline = baseline_all[:, chosen]
    observation = observation_all[:, chosen]
    area = area_all[chosen]
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    del baseline_all, observation_all, observed_annual_all, baseline_annual_all
    gc.collect()

    input_names = (
        "monthly_precipitation",
        "air_temperature",
        "dryness",
        "gpp",
        "aboveground_biomass",
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
    means: dict[str, np.ndarray] = {}
    for name in input_names:
        values = np.asarray(selected_input(name, rows, columns)[:, 0, :], dtype=np.float32)
        data[name] = values
        means[name] = values.mean(axis=0)

    rain = np.clip(data["monthly_precipitation"], 0.0, None)
    annual_rain = 12.0 * ema(rain, 12.0)
    temperature = data["air_temperature"]
    temperature_12 = ema(temperature, 12.0)
    seasonality = trailing_std(temperature) / (trailing_std(temperature) + 4.0)
    warm = sigmoid((temperature_12 - 15.0) / 3.0)
    dryness = np.clip(data["dryness"], 0.0, None)
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    gpp = np.clip(data["gpp"], 0.0, None)
    fine_fuel = ema(gpp, 12.0) / (ema(gpp, 12.0) + 0.35)
    rain_support = (
        sigmoid((annual_rain - 350.0) / 100.0)
        * annual_rain / (annual_rain + 500.0)
        * np.exp(-annual_rain / 3000.0)
    )
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    secondary = np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0)
    secondary_canopy = np.clip(data["secondary_canopy_height"], 0.0, None)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    open_cover = (
        natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0)
        + np.clip(pasture + rangeland, 0.0, 1.0)
    )
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    modifier = np.clip(
        np.clip(open_cover, 0.0, 2.0)
        * fine_fuel
        * continuity
        * combustion
        * warm
        * rain_support
        * seasonality,
        0.0,
        1.0,
    )
    persistent = ema(dryness, 12.0) / (ema(dryness, 12.0) + 500.0)

    prepared = {name: values[:, None, :] for name, values in data.items()}
    prepared["annual_precipitation"] = annual_rain[:, None, :]
    current = model._surface_seasonality_capacity(
        baseline[:, None, :], prepared, dict(model.PARAMS), set(model.COMPONENTS)
    )[:, 0, :]
    manual_current = apply_capacity(baseline, modifier, np.ones_like(persistent), 2.0)
    max_difference = float(np.max(np.abs(current - manual_current)))
    if max_difference > 2e-7:
        raise RuntimeError(f"failed incumbent reproduction {max_difference}")

    variants: dict[str, np.ndarray] = {
        "pre_capacity": baseline,
        "official_current": current,
    }
    for strength in (2.0, 3.0, 4.0):
        variants[f"persistent_k{strength:g}"] = apply_capacity(
            baseline, modifier, persistent, strength
        )

    annual_results = {}
    cycle_results = {}
    for label, prediction in variants.items():
        annual_results[label] = annual_loss(prediction, observation, area, folds)
        cycle_results[label] = cycle_loss(prediction, observation, area, folds)
        annual_total, annual_held = annual_results[label]
        cycle_total, cycle_held = cycle_results[label]
        print(
            f"VARIANT label={label} annual={annual_total:.9f} cycle={cycle_total:.9f} "
            + " ".join(
                f"fold{fold}_annual={annual_held[fold]:.9f} fold{fold}_cycle={cycle_held[fold]:.9f}"
                for fold in range(4)
            ),
            flush=True,
        )

    ecology = ecological_masks(means)
    for label, prediction in variants.items():
        for name, mask in ecology.items():
            if int(mask.sum()) == 0:
                continue
            print(
                f"ECOLOGY label={label} name={name} cells={int(mask.sum())} "
                f"ratio={area_ratio(prediction,observation,area,mask):.9f}",
                flush=True,
            )

    for region_index, name in enumerate(AUDIT_REGIONS):
        mask = region_labels == region_index
        if not mask.any():
            continue
        print(f"REGION name={name} cells={int(mask.sum())}", flush=True)
        for label, prediction in variants.items():
            print(
                f"REGION_RATIO name={name} label={label} "
                f"ratio={area_ratio(prediction,observation,area,mask):.9f}",
                flush=True,
            )

    # Direct operator prefix check: future forcing changes cannot alter the
    # first half because both modifier and persistent drying are causal states.
    perturbed_dryness = dryness.copy()
    perturbed_dryness[96:] *= 1.5
    perturbed_persistent = ema(perturbed_dryness, 12.0) / (
        ema(perturbed_dryness, 12.0) + 500.0
    )
    original_prefix = apply_capacity(baseline, modifier, persistent, 3.0)[:96]
    perturbed_prefix = apply_capacity(
        baseline, modifier, perturbed_persistent, 3.0
    )[:96]
    print(
        f"PREFIX max_abs={float(np.max(np.abs(original_prefix-perturbed_prefix))):.9e}",
        flush=True,
    )

    reference = annual_results["official_current"]
    for strength in (2.0, 3.0, 4.0):
        label = f"persistent_k{strength:g}"
        annual = annual_results[label]
        cycle = cycle_results[label]
        print(
            f"DELTA label={label} annual={annual[0]-reference[0]:+.9f} "
            f"cycle={cycle[0]-cycle_results['official_current'][0]:+.9f} "
            f"annual_folds_positive={sum(annual[1][fold] < reference[1][fold] for fold in range(4))}/4 "
            f"cycle_folds_positive={sum(cycle[1][fold] < cycle_results['official_current'][1][fold] for fold in range(4))}/4",
            flush=True,
        )
    print(
        f"DONE cells={chosen.size} current_reproduction_max={max_difference:.9e} "
        f"wall_seconds={time.perf_counter()-started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
