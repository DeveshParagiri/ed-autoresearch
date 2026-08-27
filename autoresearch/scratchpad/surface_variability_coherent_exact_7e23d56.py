"""Exact proxy audit of the coherent surface-variability shape.

The candidate is the single held-block survivor from
``surface_variability_shape_refine_7e23d56.py``.  It remains a diagnostic
candidate: this script reconstructs it from the pinned pre-capacity prediction
in land-cell chunks and runs the exact local GFED5 scorer, without editing the
canonical model or invoking official evaluation.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    apply_capacity,
    ecological_masks,
)
from autoresearch.scratchpad.surface_seasonality_south_america_diagnostic_2127874 import (  # noqa: E402
    ema,
    sigmoid,
    trailing_std,
)
from autoresearch.scratchpad.surface_variability_shape_refine_7e23d56 import (  # noqa: E402
    causal_combustion_temperature_alignment,
    causal_cv,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    CACHE,
    EXPECTED_BASE,
    EXPECTED_MODEL_BLOB,
    load_observation,
    metric_text,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


EXPECTED_INCUMBENT = 0.718882578


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def capacity_terms(data: dict[str, np.ndarray]):
    rain = np.clip(data["monthly_precipitation"], 0.0, None)
    temperature = data["air_temperature"]
    dryness = np.clip(data["dryness"], 0.0, None)
    annual_rain = 12.0 * ema(rain, 12.0)
    temperature_12 = ema(temperature, 12.0)
    temperature_std = trailing_std(temperature)
    seasonality = temperature_std / (temperature_std + 4.0)
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
    alignment = causal_combustion_temperature_alignment(combustion, temperature)
    primary = np.clip(data["luh2_primary_fraction"], 0.0, 1.0)
    dry_linear = np.clip(2.0 / (1.0 + dryness_cv), 0.5, 1.5)
    thermal_overlap = 2.0 * sigmoid(alignment)
    coherent_surface = np.clip(
        dry_linear
        * thermal_overlap
        / np.sqrt(1.0 + primary * np.square(dryness_cv)),
        0.5,
        1.5,
    )
    return base_modifier, persistent, coherent_surface


def main() -> int:
    started = time.perf_counter()
    if EXPECTED_MODEL_BLOB != "39ee93ebf1155af9ae9d70e05847b9c3f086887d":
        raise RuntimeError("unexpected pre-capacity model pin")
    if abs(EXPECTED_BASE - 0.718363408) > 5e-10 or not CACHE.exists():
        raise RuntimeError("unexpected or missing pre-capacity baseline")

    land = load_land_mask()
    rows, columns = np.nonzero(land)
    base = np.load(CACHE, mmap_mode="r")
    if base.shape != (192, 180, 360) or base.dtype != np.float32:
        raise ValueError("invalid pre-capacity cache")
    observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent = np.zeros_like(observation)
    candidate = np.zeros_like(observation)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    ecology_sums: dict[str, list[float]] = {}

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
    chunk_size = 1536
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        chunk_rows, chunk_columns = rows[start:stop], columns[start:stop]
        data = {
            name: np.asarray(
                selected_input(name, chunk_rows, chunk_columns)[:, 0, :],
                dtype=np.float32,
            )
            for name in input_names
        }
        base_chunk = np.asarray(base[:, chunk_rows, chunk_columns], dtype=np.float32)
        modifier, persistent, shape = capacity_terms(data)
        incumbent_chunk = apply_capacity(base_chunk, modifier, persistent, 4.0)
        candidate_chunk = apply_capacity(
            base_chunk, modifier, persistent * shape, 4.0
        )
        incumbent[:, chunk_rows, chunk_columns] = incumbent_chunk
        candidate[:, chunk_rows, chunk_columns] = candidate_chunk

        chunk_observation = observation[:, chunk_rows, chunk_columns]
        chunk_area = area[chunk_rows, chunk_columns]
        obs_annual = chunk_observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        incumbent_annual = incumbent_chunk.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        candidate_annual = candidate_chunk.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        means = {name: values.mean(axis=0) for name, values in data.items()}
        for name, mask in ecological_masks(means).items():
            totals = ecology_sums.setdefault(name, [0.0, 0.0, 0.0, 0.0])
            totals[0] += float(np.sum(incumbent_annual[mask] * chunk_area[mask]))
            totals[1] += float(np.sum(candidate_annual[mask] * chunk_area[mask]))
            totals[2] += float(np.sum(obs_annual[mask] * chunk_area[mask]))
            totals[3] += int(mask.sum())
        print(
            f"CHUNK start={start} stop={stop} cells={rows.size} rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del data, base_chunk, modifier, persistent, shape
        del incumbent_chunk, candidate_chunk, chunk_observation
        gc.collect()

    incumbent_scores = evaluator.score(incumbent)
    candidate_scores = evaluator.score(candidate)
    incumbent_global = incumbent_scores["global"]
    candidate_global = candidate_scores["global"]
    print(f"INCUMBENT {metric_text(incumbent_global)}", flush=True)
    print(
        f"CANDIDATE {metric_text(candidate_global)} "
        f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f}",
        flush=True,
    )
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError("failed exact incumbent reproduction")
    positive = 0
    for name in sorted(key for key in candidate_scores if key != "global"):
        old = incumbent_scores[name]["overall_score"]
        new = candidate_scores[name]["overall_score"]
        positive += int(new > old)
        print(
            f"REGION name={name} incumbent={old:.9f} candidate={new:.9f} "
            f"delta={new-old:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH positive={positive}/14", flush=True)
    for name, (old_fire, new_fire, obs_fire, cells) in ecology_sums.items():
        print(
            f"ECOLOGY name={name} cells={int(cells)} "
            f"incumbent={old_fire/max(obs_fire,1e-12):.9f} "
            f"candidate={new_fire/max(obs_fire,1e-12):.9f}",
            flush=True,
        )
    print(
        f"DONE wall_seconds={time.perf_counter()-started:.3f} peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
