"""Partition observed fire outside the established clean ecology masks.

This is a scratch-only diagnostic.  It compares the committed canonical model
with the best ecology-aware climate-structure reconstruction, but never exposes
the masks, observations, coordinates, or region labels to either prediction.

The primary seven masks reproduce the long-running structure-aware ecological
audit: intact tropical closed canopy, temperate closed canopy, boreal forest,
tropical open woodland, productive rangeland, cropland, and arid low fuel.
ED secondary-dominant vegetation is added only as an eighth coverage
sensitivity.  These established masks are used after prediction only.  Their
true complement is assigned exactly once to nine regimes defined solely by
clean climate and LUH2 state so weights and residual contributions add cleanly.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad import climate_structure_equilibrium_b049b4d as climate  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def cycle_and_annual(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cycle = np.asarray(values, dtype=np.float64).reshape(16, 12, 180, 360).mean(axis=0)
    return cycle, cycle.sum(axis=0)


def causal_mean_states(data: Mapping[str, np.ndarray], model) -> dict[str, np.ndarray]:
    """Return diagnostic means of states whose monthly histories are causal."""
    alpha12 = 1.0 - np.exp(-1.0 / 12.0)
    monthly_rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None)
    causal_annual = climate.causal_annual_rain(data)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    dryness = np.clip(np.asarray(data["dryness"], dtype=np.float64), 0.0, None)
    lightning = np.clip(np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None)
    rain_memory = model._antecedent(monthly_rain, alpha12)
    rain_departure = np.abs(monthly_rain - rain_memory) / (rain_memory + 25.0)
    rain_seasonality = rain_departure.mean(axis=0)
    del rain_departure, rain_memory
    temperature_mean = model._antecedent(temperature, alpha12).mean(axis=0)
    dryness_mean = model._antecedent(dryness, alpha12).mean(axis=0)
    lightning_mean = model._antecedent(lightning, alpha12).mean(axis=0)

    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)

    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    pasture = mean("luh2_pasture_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    urban = mean("luh2_urban_fraction")
    # The installed LUH2 secondary layer is saturated at one over most land
    # and is therefore not a usable ecological discriminator.  Exclude it from
    # the partition rather than letting it absorb almost the whole complement.
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    return {
        "annual_rain": np.asarray(causal_annual, dtype=np.float64).mean(axis=0),
        "temperature": temperature_mean,
        "dryness": dryness_mean,
        "dry_fraction": (dryness / (dryness + 500.0)).mean(axis=0),
        "rain_seasonality": rain_seasonality,
        "lightning": lightning_mean,
        "primary": primary,
        "crop": crop,
        "pasture": pasture,
        "rangeland": rangeland,
        "urban": urban,
        "managed": np.clip(crop + pasture + rangeland + urban, 0.0, 1.0),
        "woody": np.clip(primary, 0.0, 1.0),
        "open": np.clip(pasture + rangeland + residual, 0.0, 1.0),
    }


def established_masks(
    data: Mapping[str, np.ndarray],
    states: Mapping[str, np.ndarray],
    land: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reproduce seven established masks plus ED-secondary sensitivity."""
    rain = states["annual_rain"]
    temperature = states["temperature"]
    primary = states["primary"]
    crop = states["crop"]
    rangeland = states["rangeland"]

    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)

    canopy = mean("natural_canopy_height")
    lai = mean("leaf_area_index")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    secondary = mean("secondary_vegetation_fraction")
    masks = {
        "intact_tropical_closed_canopy": (
            (temperature >= 20.0)
            & (rain >= 1200.0)
            & (canopy >= 20.0)
            & (lai >= 3.0)
            & (natural >= 0.7)
            & (primary >= 0.5)
        ),
        "temperate_closed_canopy": (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (canopy >= 15.0)
            & (lai >= 2.5)
            & (natural >= 0.6)
        ),
        "boreal_forest": (
            (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6)
        ),
        "tropical_open_woodland": (
            (temperature >= 20.0)
            & (rain >= 500.0)
            & (rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        ),
        "productive_rangeland": (
            (rangeland >= 0.4)
            & (rain >= 250.0)
            & (rain < 1500.0)
            & (biomass >= 0.2)
        ),
        "cropland_dominant": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
        "secondary_dominant": secondary >= 0.5,
    }
    return {name: np.asarray(mask & land, dtype=bool) for name, mask in masks.items()}


def remainder_partition(
    states: Mapping[str, np.ndarray],
    complement: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign every complement cell to one clean, interpretable regime."""
    rain = states["annual_rain"]
    temperature = states["temperature"]
    dry_fraction = states["dry_fraction"]
    open_cover = states["open"]
    woody = states["woody"]
    managed = states["managed"]

    conditions = (
        (
            "warm_seasonal_open_carrier",
            (temperature >= 18.0)
            & (rain >= 400.0)
            & (rain < 1600.0)
            & (dry_fraction >= 0.18)
            & (open_cover >= 0.20),
        ),
        (
            "warm_seasonal_primary_mosaic",
            (temperature >= 18.0)
            & (rain >= 600.0)
            & (rain < 1800.0)
            & (dry_fraction >= 0.18)
            & (woody >= 0.25),
        ),
        (
            "warm_humid_primary_mosaic",
            (temperature >= 18.0) & (rain >= 1200.0) & (woody >= 0.25),
        ),
        (
            "warm_managed_mosaic",
            (temperature >= 18.0) & (managed >= 0.25),
        ),
        (
            "warm_dry_sparse_mosaic",
            (temperature >= 18.0) & (rain < 600.0),
        ),
        (
            "cool_managed_mosaic",
            (temperature >= 5.0) & (temperature < 18.0) & (managed >= 0.25),
        ),
        (
            "cool_primary_mosaic",
            (temperature >= 5.0) & (temperature < 18.0) & (woody >= 0.25),
        ),
        ("cold_mixed", temperature < 5.0),
        ("other_mixed_land", np.ones_like(complement, dtype=bool)),
    )
    unassigned = np.asarray(complement, dtype=bool).copy()
    result: dict[str, np.ndarray] = {}
    for name, condition in conditions:
        selected = unassigned & condition
        result[name] = selected
        unassigned &= ~selected
    if np.any(unassigned):
        raise RuntimeError("remainder partition is not exhaustive")
    stacked = np.stack(tuple(result.values()), axis=0).sum(axis=0)
    if not np.array_equal(stacked, complement.astype(np.int64)):
        raise RuntimeError("remainder partition overlaps or misses cells")
    return result


def coverage(
    masks: Mapping[str, np.ndarray],
    obs_annual: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    stack = np.stack(tuple(masks.values()), axis=0)
    union = np.any(stack, axis=0)
    obs_weight = obs_annual * area
    total = float(np.sum(obs_weight))
    union_share = float(np.sum(obs_weight * union)) / total
    raw_sum_share = sum(float(np.sum(obs_weight * mask)) / total for mask in masks.values())
    return union, union_share, raw_sum_share - union_share


def regime_statistics(
    mask: np.ndarray,
    pred_cycle: np.ndarray,
    pred_annual: np.ndarray,
    obs_cycle: np.ndarray,
    obs_annual: np.ndarray,
    area: np.ndarray,
    global_obs_fire: float,
    global_abs_residual: float,
    global_log_residual: float,
) -> dict[str, float | int | str]:
    weight = area * mask
    pred_fire = float(np.sum(pred_annual * weight))
    obs_fire = float(np.sum(obs_annual * weight))
    abs_residual = float(np.sum(np.abs(pred_annual - obs_annual) * weight))
    valid = (obs_annual > 1e-8) & mask
    log_residual = float(
        np.sum(
            obs_annual[valid]
            * area[valid]
            * np.abs(np.log((pred_annual[valid] + 1e-6) / (obs_annual[valid] + 1e-6)))
        )
    )
    pred_monthly = np.sum(pred_cycle * weight[None, ...], axis=(1, 2))
    obs_monthly = np.sum(obs_cycle * weight[None, ...], axis=(1, 2))
    pred_norm = pred_monthly / max(float(pred_monthly.sum()), 1e-12)
    obs_norm = obs_monthly / max(float(obs_monthly.sum()), 1e-12)
    seasonal_l1 = 0.5 * float(np.sum(np.abs(pred_norm - obs_norm)))
    pred_peak = int(np.argmax(pred_monthly))
    obs_peak = int(np.argmax(obs_monthly))
    phase = min(abs(pred_peak - obs_peak), 12 - abs(pred_peak - obs_peak))
    return {
        "cells": int(np.count_nonzero(mask)),
        "obs_weight_pct": 100.0 * obs_fire / global_obs_fire,
        "ratio": pred_fire / obs_fire if obs_fire > 1e-12 else float("nan"),
        "abs_resid_pct": 100.0 * abs_residual / global_abs_residual,
        "log_resid_pct": 100.0 * log_residual / global_log_residual,
        "seasonal_l1": seasonal_l1,
        "model_peak": MONTHS[pred_peak],
        "obs_peak": MONTHS[obs_peak],
        "phase_months": phase,
    }


def print_model_table(
    label: str,
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    regimes: Mapping[str, np.ndarray],
) -> None:
    pred_cycle, pred_annual = cycle_and_annual(prediction)
    obs_cycle, obs_annual = cycle_and_annual(observation)
    global_obs_fire = float(np.sum(obs_annual * area))
    global_abs_residual = float(np.sum(np.abs(pred_annual - obs_annual) * area))
    valid = obs_annual > 1e-8
    global_log_residual = float(
        np.sum(
            obs_annual[valid]
            * area[valid]
            * np.abs(np.log((pred_annual[valid] + 1e-6) / (obs_annual[valid] + 1e-6)))
        )
    )
    print(f"MODEL {label}", flush=True)
    print(
        "regime\tcells\tobs_weight_pct\tratio\tabs_resid_pct\tlog_resid_pct"
        "\tseasonal_l1\tmodel_peak\tobs_peak\tphase_months",
        flush=True,
    )
    for name, mask in regimes.items():
        values = regime_statistics(
            mask,
            pred_cycle,
            pred_annual,
            obs_cycle,
            obs_annual,
            area,
            global_obs_fire,
            global_abs_residual,
            global_log_residual,
        )
        print(
            f"{name}\t{values['cells']}\t{values['obs_weight_pct']:.6f}"
            f"\t{values['ratio']:.6f}\t{values['abs_resid_pct']:.6f}"
            f"\t{values['log_resid_pct']:.6f}\t{values['seasonal_l1']:.6f}"
            f"\t{values['model_peak']}\t{values['obs_peak']}\t{values['phase_months']}",
            flush=True,
        )


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    del reference

    states = causal_mean_states(data, model)
    masks = established_masks(data, states, land)
    primary_masks = {name: mask for name, mask in masks.items() if name != "secondary_dominant"}
    primary_union, primary_share, primary_overlap = coverage(
        primary_masks, cycle_and_annual(observation)[1], area
    )
    sensitivity_union, sensitivity_share, sensitivity_overlap = coverage(
        masks, cycle_and_annual(observation)[1], area
    )
    print(
        f"COVERAGE primary7_union_obs_pct={100.0 * primary_share:.6f} "
        f"primary7_overlap_obs_pct={100.0 * primary_overlap:.6f} "
        f"plus_secondary_union_obs_pct={100.0 * sensitivity_share:.6f} "
        f"plus_secondary_overlap_obs_pct={100.0 * sensitivity_overlap:.6f}",
        flush=True,
    )
    print("PRIMARY_MASK_WEIGHTS", flush=True)
    obs_annual = cycle_and_annual(observation)[1]
    total_obs = float(np.sum(obs_annual * area))
    for name, mask in masks.items():
        share = float(np.sum(obs_annual * area * mask)) / total_obs
        print(f"{name}\t{100.0 * share:.6f}", flush=True)
    luh_secondary = np.asarray(data["luh2_secondary_fraction"], dtype=np.float64).mean(axis=0)
    print(
        f"LUH2_SECONDARY_DIAGNOSTIC land_mean={float(luh_secondary[land].mean()):.6f} "
        f"land_fraction_ge_0.5={float(np.mean(luh_secondary[land] >= 0.5)):.6f} "
        "excluded_from_remainder_partition=1",
        flush=True,
    )

    complement = land & ~primary_union
    regimes = remainder_partition(states, complement)
    partition_obs_share = sum(float(np.sum(obs_annual * area * mask)) for mask in regimes.values()) / total_obs
    print(
        f"PARTITION complement_obs_pct={100.0 * partition_obs_share:.6f} "
        f"partition_obs_pct={100.0 * partition_obs_share:.6f} "
        f"covered_plus_partition_obs_pct={100.0 * (primary_share + partition_obs_share):.6f} "
        f"partition_cells={sum(int(np.count_nonzero(mask)) for mask in regimes.values())}",
        flush=True,
    )

    canonical = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    print_model_table("canonical", canonical, observation, area, regimes)
    del canonical
    gc.collect()

    clean = climate.predict(
        data,
        "seasonal_percolation",
        annual_scale=1.65,
        safeguard_name="fuel_selective",
    )
    print_model_table("clean_climate_fuel_selective", clean, observation, area, regimes)
    del clean
    gc.collect()
    print(
        "DEFINITIONS abs_resid_pct is each regime's share of global area-weighted "
        "absolute annual-fraction error; log_resid_pct weights absolute annual "
        "log-ratio error by observed fire; seasonal_l1 is total-variation distance "
        "between aggregate normalized monthly cycles.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
