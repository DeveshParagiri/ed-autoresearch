"""Scratch-only stage split for the canonical phenology component.

The declared ``phenology`` component gates two distinct causal mechanisms at
different positions in the prediction stack.  ``_state_dependent_fire_season``
acts before rare ignition, crop management, dead-fuel timing, and conditional
allocation.  ``_live_fuel_greenup_brake`` acts after those mechanisms but
before the pathway opportunity banks.  This exact 3-by-3 physical bracket
varies their existing strengths independently so their marginal effects and
interaction can be identified without changing canonical component semantics.

Targets and ecological masks are used only after prediction.  Nothing from
GFED5, a region, coordinates, or a completed-record state enters the model.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.ecological_geography_audit import (  # noqa: E402
    MONTHS,
    cycle_and_annual,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


EXPECTED_MODEL_BLOB = "7a8511b761e83788a5af3a824389099761e06432"
METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)


def causal_mean_states(
    data: Mapping[str, np.ndarray], model
) -> dict[str, np.ndarray]:
    """Build only post-prediction regime states from causal monthly histories."""
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    monthly_rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    causal_rain = 12.0 * model._antecedent(monthly_rain, alpha_12)
    rain_mean = causal_rain.mean(axis=0)
    temperature_mean = model._antecedent(temperature, alpha_12).mean(axis=0)
    dry_fraction = (dryness / (dryness + 500.0)).mean(axis=0)
    del causal_rain, monthly_rain, temperature, dryness

    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)

    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    pasture = mean("luh2_pasture_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    urban = mean("luh2_urban_fraction")
    classified = np.clip(
        primary + crop + pasture + rangeland + urban, 0.0, 1.0
    )
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    return {
        "rain": rain_mean,
        "temperature": temperature_mean,
        "dry_fraction": dry_fraction,
        "primary": primary,
        "crop": crop,
        "rangeland": rangeland,
        "open": np.clip(pasture + rangeland + residual, 0.0, 1.0),
    }


def target_masks(
    data: Mapping[str, np.ndarray],
    model,
    land: np.ndarray,
) -> dict[str, np.ndarray]:
    """Define the high-weight timing target and two underburn guardrails."""
    states = causal_mean_states(data, model)
    rain = states["rain"]
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
    established = {
        "intact": (
            (temperature >= 20.0)
            & (rain >= 1200.0)
            & (canopy >= 20.0)
            & (lai >= 3.0)
            & (natural >= 0.7)
            & (primary >= 0.5)
        ),
        "temperate": (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (canopy >= 15.0)
            & (lai >= 2.5)
            & (natural >= 0.6)
        ),
        "boreal": (
            (temperature < 5.0)
            & (canopy >= 10.0)
            & (natural >= 0.6)
        ),
        "tropical_open": (
            (temperature >= 20.0)
            & (rain >= 500.0)
            & (rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        ),
        "rangeland": (
            (rangeland >= 0.4)
            & (rain >= 250.0)
            & (rain < 1500.0)
            & (biomass >= 0.2)
        ),
        "crop": crop >= 0.5,
        "arid": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }
    established_union = np.any(
        np.stack(tuple(established.values()), axis=0), axis=0
    )
    warm_open = (
        land
        & ~established_union
        & (temperature >= 18.0)
        & (rain >= 400.0)
        & (rain < 1600.0)
        & (states["dry_fraction"] >= 0.18)
        & (states["open"] >= 0.20)
    )
    return {
        "warm_seasonal_open": warm_open,
        "intact_tropical_closed": land & established["intact"],
        "cropland": land & established["crop"],
    }


def regime_statistics(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    pred_cycle, pred_annual = cycle_and_annual(prediction)
    obs_cycle, obs_annual = cycle_and_annual(observation)
    weight = area * mask
    pred_monthly = np.sum(pred_cycle * weight[None, ...], axis=(1, 2))
    obs_monthly = np.sum(obs_cycle * weight[None, ...], axis=(1, 2))
    pred_total = float(pred_monthly.sum())
    obs_total = float(obs_monthly.sum())
    pred_norm = pred_monthly / max(pred_total, 1e-12)
    obs_norm = obs_monthly / max(obs_total, 1e-12)
    pred_peak = int(np.argmax(pred_monthly))
    obs_peak = int(np.argmax(obs_monthly))
    phase = min(abs(pred_peak - obs_peak), 12 - abs(pred_peak - obs_peak))
    return {
        "cells": int(np.count_nonzero(mask)),
        "ratio": pred_total / max(obs_total, 1e-12),
        "seasonal_l1": 0.5 * float(np.sum(np.abs(pred_norm - obs_norm))),
        "model_peak": MONTHS[pred_peak],
        "obs_peak": MONTHS[obs_peak],
        "phase": phase,
        "annual_abs_error": float(
            np.sum(np.abs(pred_annual - obs_annual) * weight)
        ),
    }


def score_text(score: Mapping[str, float]) -> str:
    return " ".join(f"{name}={score[key]:.9f}" for name, key in METRICS)


def main() -> int:
    model_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if model_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(
            f"refusing moving-model audit: expected {EXPECTED_MODEL_BLOB}, "
            f"got {model_blob}"
        )

    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = target_masks(data, model, land)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference

    base_fire = float(model.PARAMS["fire_season_w"])
    base_green = float(model.PARAMS["greenup_brake"])
    fire_strengths = (0.0, 0.5 * base_fire, base_fire)
    green_strengths = (0.0, 0.5 * base_green, base_green)
    results: dict[
        tuple[float, float],
        tuple[dict[str, float], dict[str, dict[str, float | int | str]]],
    ] = {}
    base_key = (base_fire, base_green)

    # Run the incumbent first so a moving or malformed baseline is visible
    # before any counterfactual is interpreted.
    order = (base_key,) + tuple(
        (fire, green)
        for fire in fire_strengths
        for green in green_strengths
        if (fire, green) != base_key
    )
    print(
        f"MODEL_BLOB={model_blob} base_fire={base_fire:g} "
        f"base_green={base_green:g}",
        flush=True,
    )
    for fire_strength, green_strength in order:
        params = dict(model.PARAMS)
        params["fire_season_w"] = fire_strength
        params["greenup_brake"] = green_strength
        prediction = validate_prediction(model.predict(data, params, None))
        global_score = dict(evaluator.score(prediction)["global"])
        regime_scores = {
            name: regime_statistics(prediction, observation, area, mask)
            for name, mask in masks.items()
        }
        key = (fire_strength, green_strength)
        results[key] = (global_score, regime_scores)
        print(
            f"CONFIG fire={fire_strength:g} green={green_strength:g} "
            + score_text(global_score),
            flush=True,
        )
        for name, values in regime_scores.items():
            print(
                f"REGIME {name} fire={fire_strength:g} green={green_strength:g} "
                f"cells={values['cells']} ratio={float(values['ratio']):.9f} "
                f"seasonal_l1={float(values['seasonal_l1']):.9f} "
                f"peak={values['model_peak']}/{values['obs_peak']} "
                f"phase={values['phase']} "
                f"annual_abs_error={float(values['annual_abs_error']):.9e}",
                flush=True,
            )
        del prediction
        gc.collect()

    base_global, base_regimes = results[base_key]
    print("GLOBAL_DELTAS", flush=True)
    for key, (score, _) in results.items():
        print(
            f"fire={key[0]:g} green={key[1]:g} "
            + " ".join(
                f"d_{name}={score[metric] - base_global[metric]:+.9f}"
                for name, metric in METRICS
            ),
            flush=True,
        )

    print("REGIME_DELTAS", flush=True)
    for key, (_, regimes) in results.items():
        for name, values in regimes.items():
            base = base_regimes[name]
            print(
                f"{name} fire={key[0]:g} green={key[1]:g} "
                f"d_ratio={float(values['ratio']) - float(base['ratio']):+.9f} "
                f"d_seasonal_l1={float(values['seasonal_l1']) - float(base['seasonal_l1']):+.9f} "
                f"d_annual_abs_error={float(values['annual_abs_error']) - float(base['annual_abs_error']):+.9e}",
                flush=True,
            )

    # Exact second differences separate independent stage contributions from
    # their downstream interaction.  Positive means the joint weakening beats
    # the sum of the two isolated weakenings for that metric.
    print("INTERACTIONS", flush=True)
    for fire_strength in fire_strengths[:-1]:
        for green_strength in green_strengths[:-1]:
            joint = results[(fire_strength, green_strength)][0]
            fire_only = results[(fire_strength, base_green)][0]
            green_only = results[(base_fire, green_strength)][0]
            print(
                f"fire={fire_strength:g} green={green_strength:g} "
                + " ".join(
                    f"i_{name}="
                    f"{joint[metric] - fire_only[metric] - green_only[metric] + base_global[metric]:+.9f}"
                    for name, metric in METRICS
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
