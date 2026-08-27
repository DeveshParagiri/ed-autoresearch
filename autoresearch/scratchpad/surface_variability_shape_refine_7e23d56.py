"""Test causal variability shapes for the persistent-desiccation capacity.

The canonical equation at ``7e23d56`` uses a fixed strength of four and gates
the warm surface-seasonality capacity by the trailing mean dryness.  A shallow
held-cell diagnostic repeatedly paired dryness variability with combustion-
temperature alignment, primary cover, and rain variability.  This script
translates those interactions into a deliberately small bracket of smooth,
globally shared shapes.  All variability statistics are trailing causal
states; geography is used only for balanced audit sampling and reporting.

Diagnostic only.  It does not edit the canonical model or run official ILAMB.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    AUDIT_REGIONS,
    annual_loss,
    apply_capacity,
    area_ratio,
    cycle_loss,
    ecological_masks,
    select_balanced_cells,
)
from autoresearch.scratchpad.surface_seasonality_south_america_diagnostic_2127874 import (  # noqa: E402
    ema,
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
from scripts.runtime import load_land_mask  # noqa: E402


CURRENT_COMMIT = "7e23d56dae60b56373d5d34ff41299f37e8aa4fc"
CURRENT_MODEL_BLOB = "c56b96a1cbd57e4342b14f4cc13ea541830703e7"


def committed_model():
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
        raise RuntimeError(f"unexpected incumbent model blob {observed_blob}")
    module = types.ModuleType("ed_fire_pinned_7e23d56")
    module.__file__ = f"git:{CURRENT_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def causal_cv(values: np.ndarray, months: float = 12.0, floor: float = 1.0) -> np.ndarray:
    mean = ema(values, months)
    variance = np.maximum(ema(np.square(values), months) - np.square(mean), 0.0)
    return np.sqrt(variance) / (mean + floor)


def causal_combustion_temperature_alignment(
    combustion: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    weighted_temperature = ema(combustion * temperature, 12.0) / (
        ema(combustion, 12.0) + 1e-6
    )
    temperature_mean = ema(temperature, 12.0)
    temperature_range = trailing_std(temperature)
    return (weighted_temperature - temperature_mean) / (temperature_range + 1.0)


def main() -> int:
    started = time.perf_counter()
    if EXPECTED_MODEL_BLOB != "39ee93ebf1155af9ae9d70e05847b9c3f086887d":
        raise RuntimeError("unexpected pre-capacity model pin")
    if abs(EXPECTED_BASE - 0.718363408) > 5e-10 or not CACHE.exists():
        raise RuntimeError("unexpected or missing pre-capacity baseline")

    model = committed_model()
    land = load_land_mask()
    flat = np.flatnonzero(land.ravel())
    all_rows, all_columns = flat // 360, flat % 360
    baseline_grid = np.load(CACHE, mmap_mode="r")
    baseline_all = np.asarray(baseline_grid[:, all_rows, all_columns], dtype=np.float32)
    observation_grid = load_observation()
    observation_all = np.asarray(
        observation_grid[:, all_rows, all_columns], dtype=np.float32
    )
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
    temperature = data["air_temperature"]
    dryness = np.clip(data["dryness"], 0.0, None)
    annual_rain = 12.0 * ema(rain, 12.0)
    temperature_12 = ema(temperature, 12.0)
    seasonality = trailing_std(temperature) / (trailing_std(temperature) + 4.0)
    warm = sigmoid((temperature_12 - 15.0) / 3.0)
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
    base_modifier = np.clip(
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
    dryness_12 = ema(dryness, 12.0)
    persistent = dryness_12 / (dryness_12 + 500.0)
    dryness_cv = causal_cv(dryness)
    rain_cv = causal_cv(rain)
    alignment = causal_combustion_temperature_alignment(combustion, temperature)
    primary = np.clip(data["luh2_primary_fraction"], 0.0, 1.0)

    dry_linear = np.clip(2.0 / (1.0 + dryness_cv), 0.5, 1.5)
    dry_quadratic = np.clip(2.0 / (1.0 + np.square(dryness_cv)), 0.5, 1.5)
    dry_rain = np.clip(
        4.0 / ((1.0 + dryness_cv) * (1.0 + rain_cv)), 0.5, 1.5
    )
    thermal_overlap = 2.0 * sigmoid(alignment)
    dry_alignment = np.clip(dry_linear * thermal_overlap, 0.5, 1.5)
    primary_continuity = 1.5 / (1.0 + primary * np.square(dryness_cv))
    dry_primary = np.clip(dry_linear * primary_continuity, 0.5, 1.5)
    coherent_surface = np.clip(
        dry_linear
        * thermal_overlap
        / np.sqrt(1.0 + primary * np.square(dryness_cv)),
        0.5,
        1.5,
    )

    prepared = {name: values[:, None, :] for name, values in data.items()}
    prepared["annual_precipitation"] = annual_rain[:, None, :]
    incumbent = model._surface_seasonality_capacity(
        baseline[:, None, :], prepared, dict(model.PARAMS), set(model.COMPONENTS)
    )[:, 0, :]
    manual_incumbent = apply_capacity(baseline, base_modifier, persistent, 4.0)
    max_difference = float(np.max(np.abs(incumbent - manual_incumbent)))
    if max_difference > 2e-7:
        raise RuntimeError(f"failed incumbent reproduction {max_difference}")

    shapes = {
        "dry_cv_linear": dry_linear,
        "dry_cv_quadratic": dry_quadratic,
        "dry_rain_cv": dry_rain,
        "dry_cv_alignment": dry_alignment,
        "dry_cv_primary": dry_primary,
        "coherent_surface": coherent_surface,
    }
    variants = {"incumbent": incumbent}
    for label, shape in shapes.items():
        variants[label] = apply_capacity(
            baseline, base_modifier, persistent * shape, 4.0
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
                f"fold{fold}_annual={annual_held[fold]:.9f} "
                f"fold{fold}_cycle={cycle_held[fold]:.9f}"
                for fold in range(4)
            ),
            flush=True,
        )

    ecology = ecological_masks(means)
    for label, prediction in variants.items():
        for name, mask in ecology.items():
            if mask.any():
                print(
                    f"ECOLOGY label={label} name={name} cells={int(mask.sum())} "
                    f"ratio={area_ratio(prediction,observation,area,mask):.9f}",
                    flush=True,
                )

    for region_index, name in enumerate(AUDIT_REGIONS):
        mask = region_labels == region_index
        cell_weights = np.broadcast_to(area[mask][None, :], dryness_cv[:, mask].shape)
        print(
            f"REGION_STATE name={name} "
            f"dry_cv={weighted_mean(dryness_cv[:,mask],cell_weights):.9f} "
            f"rain_cv={weighted_mean(rain_cv[:,mask],cell_weights):.9f} "
            f"alignment={weighted_mean(alignment[:,mask],cell_weights):+.9f} "
            f"primary={weighted_mean(primary[:,mask],cell_weights):.9f}",
            flush=True,
        )
        for label, prediction in variants.items():
            print(
                f"REGION name={name} label={label} cells={int(mask.sum())} "
                f"ratio={area_ratio(prediction,observation,area,mask):.9f}",
                flush=True,
            )

    reference_annual = annual_results["incumbent"]
    reference_cycle = cycle_results["incumbent"]
    for label, shape in shapes.items():
        annual = annual_results[label]
        cycle = cycle_results[label]
        print(
            f"DELTA label={label} annual={annual[0]-reference_annual[0]:+.9f} "
            f"cycle={cycle[0]-reference_cycle[0]:+.9f} "
            f"annual_folds_positive={sum(annual[1][fold] < reference_annual[1][fold] for fold in range(4))}/4 "
            f"cycle_folds_positive={sum(cycle[1][fold] < reference_cycle[1][fold] for fold in range(4))}/4 "
            f"shape_mean={weighted_mean(shape,np.ones_like(shape)):.9f}",
            flush=True,
        )

    perturbed_dryness = dryness.copy()
    perturbed_dryness[96:] *= 1.5
    perturbed_cv = causal_cv(perturbed_dryness)
    original_prefix = dry_linear[:96]
    perturbed_prefix = np.clip(2.0 / (1.0 + perturbed_cv), 0.5, 1.5)[:96]
    print(
        f"PREFIX max_abs={float(np.max(np.abs(original_prefix-perturbed_prefix))):.9e}",
        flush=True,
    )
    print(
        f"DONE cells={chosen.size} incumbent_reproduction_max={max_difference:.9e} "
        f"wall_seconds={time.perf_counter()-started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
