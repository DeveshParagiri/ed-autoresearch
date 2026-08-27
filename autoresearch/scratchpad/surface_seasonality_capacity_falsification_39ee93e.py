"""Bounded mechanistic falsification of warm surface-seasonality capacity.

The diagnostic at ``stable_annual_residual_rules_39ee93e.py`` repeatedly found
temperature seasonality interacting with connected surface combustion.  This
script tests only the distilled physical equation, never the learned surface:

    h' = h * (1 + k * Q)
    Q = C_surface * W(T12) * F(P12) * sigma_T/(sigma_T + 4 K)

``C_surface`` is connected open fine fuel under current combustion weather,
``W`` excludes cold climates, and ``F`` requires rain-supported fuel while
falling again in very wet climates.  Every state is local and prefix-causal.
Coordinates construct held spatial folds only.  No region, neighbour, future
input, completed-year precipitation, or learned coefficient enters Q.

Diagnostic only: this file does not alter ``model.py`` or the official ledger.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np


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


def ecology_masks(mean: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rain = 12.0 * mean["monthly_precipitation"]
    temperature = mean["air_temperature"]
    lai = mean["leaf_area_index"]
    canopy = mean["natural_canopy_height"]
    biomass = mean["aboveground_biomass"]
    natural = mean["natural_vegetation_fraction"]
    primary = mean["luh2_primary_fraction"]
    crop = mean["luh2_cropland_fraction"]
    rangeland = mean["luh2_rangeland_fraction"]
    return {
        "intact_tropical_closed": (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0) & (biomass >= 0.2),
        "crop": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    model_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    return {
        name: float(np.sum(model_annual[mask] * area[mask]))
        / max(float(np.sum(obs_annual[mask] * area[mask])), 1e-12)
        for name, mask in masks.items()
    }


def proxy_loss(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    fold: np.ndarray,
) -> tuple[float, tuple[float, ...]]:
    model_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    floor = float(np.sum(obs_annual * area) / np.sum(area)) * 0.02
    weights = area * (obs_annual + floor)
    residual = np.abs(np.log((obs_annual + 1e-6) / (model_annual + 1e-6)))
    total = float(np.sum(weights * residual) / np.sum(weights))
    held = tuple(
        float(np.sum(weights[fold == index] * residual[fold == index]) / np.sum(weights[fold == index]))
        for index in range(4)
    )
    return total, held


def main() -> int:
    started = time.perf_counter()
    if not CACHE.exists():
        raise RuntimeError(f"missing cache for {EXPECTED_MODEL_BLOB[:8]}")
    # The working tree may host another agent's candidate.  This falsification
    # is deliberately pinned to the immutable cache and model blob instead.
    subprocess.run(
        ("git", "cat-file", "-e", EXPECTED_MODEL_BLOB),
        cwd=ROOT,
        check=True,
    )
    land = load_land_mask()
    flat = np.flatnonzero(land.ravel())
    rows, columns = flat // 360, flat % 360
    count = flat.size
    baseline_grid = np.load(CACHE, mmap_mode="r")
    baseline = np.asarray(baseline_grid[:, rows, columns], dtype=np.float32)
    observation_grid = load_observation()
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float32)
    del observation_grid
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_score = evaluator.score(validate_prediction(baseline_grid))["global"]
    if abs(float(base_score["overall_score"]) - EXPECTED_BASE) > 5e-9:
        raise RuntimeError(f"baseline mismatch: {metric_text(base_score)}")
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    area = area_grid[rows, columns].astype(np.float64)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4

    means: dict[str, np.ndarray] = {}
    for name in (
        "leaf_area_index",
        "aboveground_biomass",
        "luh2_primary_fraction",
    ):
        values = selected_input(name, rows, columns)[:, 0, :]
        means[name] = values.mean(axis=0)
        del values
        gc.collect()

    rain = np.clip(selected_input("monthly_precipitation", rows, columns)[:, 0, :], 0.0, None)
    means["monthly_precipitation"] = rain.mean(axis=0)
    rain12 = ema(rain, 12.0)
    annual_rain = 12.0 * rain12
    rain_support = (
        sigmoid((annual_rain - 350.0) / 100.0)
        * annual_rain / (annual_rain + 500.0)
        * np.exp(-annual_rain / 3000.0)
    )

    temperature = selected_input("air_temperature", rows, columns)[:, 0, :]
    means["air_temperature"] = temperature.mean(axis=0)
    temperature12 = ema(temperature, 12.0)
    temperature_std = trailing_std(temperature, 12)
    seasonality = temperature_std / (temperature_std + 4.0)
    warm = sigmoid((temperature12 - 15.0) / 3.0)
    del temperature, temperature12, temperature_std
    gc.collect()

    gpp = np.clip(selected_input("gpp", rows, columns)[:, 0, :], 0.0, None)
    fine_fuel = ema(gpp, 12.0) / (ema(gpp, 12.0) + 0.35)
    del gpp

    dryness = np.clip(selected_input("dryness", rows, columns)[:, 0, :], 0.0, None)
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    del dryness, rain, rain12, annual_rain
    gc.collect()

    canopy = np.clip(selected_input("natural_canopy_height", rows, columns)[:, 0, :], 0.0, None)
    means["natural_canopy_height"] = canopy.mean(axis=0)
    natural = np.clip(selected_input("natural_vegetation_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    means["natural_vegetation_fraction"] = natural.mean(axis=0)
    open_cover = natural * 8.0 / (canopy + 8.0)
    del natural, canopy

    second_canopy = np.clip(selected_input("secondary_canopy_height", rows, columns)[:, 0, :], 0.0, None)
    secondary = np.clip(selected_input("secondary_vegetation_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    open_cover += secondary * 8.0 / (second_canopy + 8.0)
    del secondary, second_canopy

    pasture = np.clip(selected_input("luh2_pasture_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    rangeland = np.clip(selected_input("luh2_rangeland_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    means["luh2_rangeland_fraction"] = rangeland.mean(axis=0)
    open_cover += np.clip(pasture + rangeland, 0.0, 1.0)
    del pasture, rangeland

    crop = np.clip(selected_input("luh2_cropland_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    means["luh2_cropland_fraction"] = crop.mean(axis=0)
    urban = np.clip(selected_input("luh2_urban_fraction", rows, columns)[:, 0, :], 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    del crop, urban

    connected_surface = np.clip(open_cover, 0.0, 2.0) * fine_fuel * continuity * combustion
    modifier = np.clip(connected_surface * warm * rain_support * seasonality, 0.0, 1.0).astype(np.float32)
    del open_cover, fine_fuel, continuity, combustion, warm, rain_support, seasonality, connected_surface
    gc.collect()
    print(
        f"DESIGN blob={EXPECTED_MODEL_BLOB} cells={count} prefix_causal=1 "
        f"modifier_mean={float(modifier.mean()):.9f} p95={float(np.quantile(modifier,.95)):.9f} "
        f"max={float(modifier.max()):.9f} {metric_text(base_score)}",
        flush=True,
    )

    masks = ecology_masks(means)
    base_proxy, base_folds = proxy_loss(baseline, observation, area, folds)
    base_ecology = ecology_ratios(baseline, observation, area, masks)
    print(
        f"BASE_PROXY loss={base_proxy:.9f} "
        + " ".join(f"fold{index}={value:.9f}" for index, value in enumerate(base_folds)),
        flush=True,
    )

    best = None
    for strength in (0.25, 0.50, 1.0, 2.0, 4.0):
        hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
        candidate = 1.0 - np.exp(-hazard * (1.0 + strength * modifier))
        candidate = np.asarray(np.clip(candidate, 0.0, 1.0), dtype=np.float32)
        loss, held = proxy_loss(candidate, observation, area, folds)
        print(
            f"PROXY strength={strength:g} loss={loss:.9f} delta={loss-base_proxy:+.9f} "
            + " ".join(
                f"fold{index}={value:.9f} fold{index}_delta={value-base_folds[index]:+.9f}"
                for index, value in enumerate(held)
            ),
            flush=True,
        )

        candidate_grid = np.asarray(baseline_grid).copy()
        candidate_grid[:, rows, columns] = candidate
        score = evaluator.score(validate_prediction(candidate_grid))["global"]
        print(
            f"EXACT_PROXY strength={strength:g} delta={float(score['overall_score'])-EXPECTED_BASE:+.9f} "
            f"{metric_text(score)}",
            flush=True,
        )
        ratios = ecology_ratios(candidate, observation, area, masks)
        for name in masks:
            print(
                f"ECOLOGY strength={strength:g} name={name} baseline={base_ecology[name]:.9f} "
                f"candidate={ratios[name]:.9f} delta={ratios[name]-base_ecology[name]:+.9f}",
                flush=True,
            )
        record = (float(score["overall_score"]), strength, loss, held, dict(score), ratios)
        if best is None or record[0] > best[0]:
            best = record
        del hazard, candidate, candidate_grid
        gc.collect()

    assert best is not None
    print(
        f"BEST strength={best[1]:g} exact_delta={best[0]-EXPECTED_BASE:+.9f} "
        f"proxy_delta={best[2]-base_proxy:+.9f}",
        flush=True,
    )
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        f"DONE wall_seconds={time.perf_counter()-started:.3f} peak_rss_gib={peak/(1024.0**3):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
