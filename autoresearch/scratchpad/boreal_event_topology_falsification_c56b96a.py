"""Exact proxy falsification of two cold-event spatial-allocation equations.

Both equations are globally shared, smooth, causal, site-local, and use only
coupled-valid model inputs.  The first suppresses diffuse fire in cold, wet,
organic closed fuel.  The second enlarges events in cold managed residue only
when thaw, combustion, and fuel continuity coincide.  Region and ecological
masks are post-prediction audits only.  Nothing here edits the canonical model
or invokes the official evaluator.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.boreal_spatial_residual_diagnostic_c56b96a import (  # noqa: E402
    CACHE,
    EXPECTED_OVERALL,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


SHIELD_STRENGTHS = (0.5, 1.0)
MANAGED_STRENGTHS = (0.5, 1.0, 2.0)


def rising(values: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-k * (values - center), -50.0, 50.0)))


def falling(values: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(k * (values - center), -50.0, 50.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def states(rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temperature = selected_input("air_temperature", rows, columns)[:, 0, :]
    rain = np.clip(
        selected_input("monthly_precipitation", rows, columns)[:, 0, :],
        0.0,
        None,
    )
    dryness = np.clip(selected_input("dryness", rows, columns)[:, 0, :], 0.0, None)
    gpp = np.clip(selected_input("gpp", rows, columns)[:, 0, :], 0.0, None)
    lightning = np.clip(
        selected_input("lightning_flash_rate", rows, columns)[:, 0, :],
        0.0,
        None,
    )
    soil = np.clip(selected_input("soil_carbon", rows, columns)[:, 0, :], 0.0, None)
    natural = np.clip(
        selected_input("natural_vegetation_fraction", rows, columns)[:, 0, :],
        0.0,
        1.0,
    )
    canopy = np.clip(
        selected_input("natural_canopy_height", rows, columns)[:, 0, :],
        0.0,
        None,
    )
    crop = np.clip(
        selected_input("luh2_cropland_fraction", rows, columns)[:, 0, :],
        0.0,
        1.0,
    )
    pasture = np.clip(
        selected_input("luh2_pasture_fraction", rows, columns)[:, 0, :],
        0.0,
        1.0,
    )
    rangeland = np.clip(
        selected_input("luh2_rangeland_fraction", rows, columns)[:, 0, :],
        0.0,
        1.0,
    )

    temperature_3 = ema(temperature, 3.0)
    temperature_24 = ema(temperature, 24.0)
    rain_6 = ema(rain, 6.0)
    lightning_12 = ema(lightning, 12.0)
    gpp_12 = ema(gpp, 12.0)
    cold = falling(temperature_24, 1.0 / 3.0, 7.0)
    thaw = rising(temperature, 1.0 / 3.0, 1.0) * rising(
        temperature - temperature_3, 0.5, 2.0
    )
    rain_deficit = np.maximum((rain_6 - rain) / (rain_6 + rain + 10.0), 0.0)
    combustion = dryness / (dryness + 250.0) * (0.35 + 0.65 * rain_deficit)
    ignition = lightning_12 / (lightning_12 + 0.004)
    event_readiness = np.sqrt(np.clip(thaw * combustion * ignition, 0.0, 1.0))

    forest = natural * canopy / (canopy + 8.0)
    organic = soil / (soil + 4.0)
    wet_organic_shield = np.clip(
        cold * forest * organic * (1.0 - event_readiness), 0.0, 1.0
    )

    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    managed = np.clip(crop + 0.25 * pasture + 0.25 * rangeland, 0.0, 1.0)
    open_access = 1.0 - forest
    managed_residue_event = np.clip(
        cold
        * managed
        * open_access
        * fine_fuel
        * event_readiness,
        0.0,
        1.0,
    )
    return wet_organic_shield, managed_residue_event


def ecological_masks(rows: np.ndarray, columns: np.ndarray) -> dict[str, np.ndarray]:
    names = (
        "monthly_precipitation",
        "air_temperature",
        "leaf_area_index",
        "natural_canopy_height",
        "aboveground_biomass",
        "natural_vegetation_fraction",
        "luh2_primary_fraction",
        "luh2_cropland_fraction",
        "luh2_rangeland_fraction",
    )
    means = {
        name: selected_input(name, rows, columns)[:, 0, :].mean(axis=0)
        for name in names
    }
    rain = 12.0 * means["monthly_precipitation"]
    temperature = means["air_temperature"]
    lai = means["leaf_area_index"]
    canopy = means["natural_canopy_height"]
    biomass = means["aboveground_biomass"]
    natural = means["natural_vegetation_fraction"]
    primary = means["luh2_primary_fraction"]
    crop = means["luh2_cropland_fraction"]
    rangeland = means["luh2_rangeland_fraction"]
    return {
        "intact_tropical_closed": (
            (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0)
            & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5)
        ),
        "temperate_closed": (
            (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0)
            & (lai >= 2.5) & (natural >= 0.6)
        ),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (
            (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0)
            & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5)
        ),
        "productive_rangeland": (
            (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0)
            & (biomass >= 0.2)
        ),
        "crop": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    pred_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    return {
        name: float(np.sum(pred_annual[mask] * area[mask]))
        / max(float(np.sum(obs_annual[mask] * area[mask])), 1e-12)
        for name, mask in masks.items()
    }


def fold_losses(
    baseline: np.ndarray,
    candidate: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> str:
    pred = baseline.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    cand = candidate.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    positive = obs[obs > 0.0]
    floor = 0.02 * float(np.median(positive))
    target = np.log((obs + floor) / (pred + floor))
    changed = np.log((obs + floor) / (cand + floor))
    weights = area * (obs + floor)
    folds = (rows // 5 + 2 * (columns // 8)) % 4
    output = []
    for fold in range(4):
        mask = folds == fold
        before = float(np.sum(np.abs(target[mask]) * weights[mask]) / np.sum(weights[mask]))
        after = float(np.sum(np.abs(changed[mask]) * weights[mask]) / np.sum(weights[mask]))
        output.append(f"{after - before:+.6f}")
    return ",".join(output)


def main() -> int:
    if not CACHE.exists():
        raise RuntimeError("run boreal_spatial_residual_diagnostic_c56b96a.py first")
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    incumbent_grid = np.load(CACHE, mmap_mode="r")
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_scores = evaluator.score(incumbent_grid)
    if abs(base_scores["global"]["overall_score"] - EXPECTED_OVERALL) > 5e-8:
        raise RuntimeError("pinned incumbent cache mismatch")

    labels = [*(f"shield_{value:g}" for value in SHIELD_STRENGTHS), *(f"managed_{value:g}" for value in MANAGED_STRENGTHS)]
    variants = {
        label: np.asarray(incumbent_grid, dtype=np.float32).copy()
        for label in labels
    }
    for start in range(0, rows.size, 1536):
        stop = min(start + 1536, rows.size)
        chunk_rows = rows[start:stop]
        chunk_columns = columns[start:stop]
        baseline = np.asarray(
            incumbent_grid[:, chunk_rows, chunk_columns], dtype=np.float64
        )
        hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
        shield, managed = states(chunk_rows, chunk_columns)
        for strength in SHIELD_STRENGTHS:
            adjusted = hazard * np.exp(-strength * shield)
            variants[f"shield_{strength:g}"][:, chunk_rows, chunk_columns] = np.asarray(
                1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
            )
        for strength in MANAGED_STRENGTHS:
            adjusted = hazard * (1.0 + strength * managed)
            variants[f"managed_{strength:g}"][:, chunk_rows, chunk_columns] = np.asarray(
                1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
            )
        print(f"APPLY {start}:{stop}/{rows.size}", flush=True)
        del baseline, hazard, shield, managed, adjusted
        gc.collect()

    observation_grid = load_observation()
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float32)
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    masks = ecological_masks(rows, columns)
    base_ecology = ecology_ratios(
        np.asarray(incumbent_grid[:, rows, columns]), observation, area, masks
    )
    inside_boas = ~evaluator.regions["boas"].reshape(180, 2, 360, 2).all(
        axis=(1, 3)
    )
    boas_selector = inside_boas[rows, columns]
    boas_rows = rows[boas_selector]
    boas_columns = columns[boas_selector]
    boas_base = np.asarray(incumbent_grid[:, boas_rows, boas_columns])
    boas_obs = observation[:, boas_selector]
    boas_area = area[boas_selector]

    for label, grid in variants.items():
        scores = evaluator.score(grid)
        ecology = ecology_ratios(grid[:, rows, columns], observation, area, masks)
        global_score = scores["global"]
        boas = scores["boas"]
        regional = sorted(
            (
                float(scores[name]["overall_score"])
                - float(base_scores[name]["overall_score"]),
                name,
            )
            for name in scores
            if name != "global"
        )
        fold_text = fold_losses(
            boas_base,
            grid[:, boas_rows, boas_columns],
            boas_obs,
            boas_area,
            boas_rows,
            boas_columns,
        )
        eco_text = ",".join(
            f"{name}:{base_ecology[name]:.4f}->{ecology[name]:.4f}"
            for name in base_ecology
        )
        print(
            f"RESULT {label} overall={global_score['overall_score']:.9f} "
            f"delta={global_score['overall_score'] - EXPECTED_OVERALL:+.9f} "
            f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
            f"seasonal={global_score['seasonal_cycle_score']:.9f} "
            f"spatial={global_score['spatial_distribution_score']:.9f} "
            f"boas_overall={boas['overall_score']:.9f} "
            f"boas_spatial={boas['spatial_distribution_score']:.9f} "
            f"fold_logmae_deltas={fold_text} "
            f"regions={sum(delta > 0 for delta, _ in regional)}/14 "
            f"worst={regional[0][1]}:{regional[0][0]:+.6f} "
            f"best={regional[-1][1]}:{regional[-1][0]:+.6f} ecology={eco_text}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
