"""Scratch-only climate reconstruction of vegetation structure for ED fire.

This falsification test deliberately excludes every field from ``ed.nc`` and
all modern-only fire-weather inputs.  It reconstructs the vegetation state
needed by the committed mechanistic fire equations from coupled meteorology,
LUH2 land state, and fixed lightning climatology.  No reference burned area,
coordinates, region labels, calendar indices, neighbours, or completed-record
statistics enter the prediction path.

The family asks whether three well established ecosystem constraints can
recover the spatial discriminator lost when the frozen ED state is removed:

* potential productivity rises with water and growing-season temperature;
* woody equilibrium rises with rainfall but is opposed by rainfall
  seasonality and accumulated water deficit;
* connected grass fuel requires enough production, then declines smoothly as
  woody closure shades and fragments the surface layer.

All states are initialized from month zero alone and advance locally, so the
construction is prefix causal.  The resulting GPP, biomass, LAI, canopy height,
and soil-carbon fields are mechanistic climate states, not fitted surrogates.
They are passed through the current globally shared fire machinery only in
this scratch diagnostic.  Nothing here is an official candidate.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch import model as incumbent  # noqa: E402
from autoresearch.scratchpad.clean_exogenous_rebuild_b867ed7 import (  # noqa: E402
    exogenous_ecology_ratios,
    metric_line,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


INPUTS = (
    "dryness",
    "monthly_precipitation",
    "air_temperature",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_primary_fraction",
    "luh2_secondary_fraction",
    "luh2_urban_fraction",
    "lightning_flash_rate",
)


STRUCTURES: dict[str, dict[str, float]] = {
    # A rainfall-driven equilibrium.  This is the least structured control:
    # precipitation establishes woody vegetation, but intra-annual rainfall
    # variability does not alter that equilibrium.
    "rain_equilibrium": {
        "productivity_scale": 2.4,
        "productivity_rain_half": 320.0,
        "woody_rain_center": 850.0,
        "woody_rain_width": 260.0,
        "seasonality_strength": 0.0,
        "deficit_strength": 0.0,
        "grass_rain_center": 290.0,
        "grass_rain_width": 110.0,
        "grass_shade": 2.2,
        "grass_establishment_power": 1.0,
        "humid_woody_floor": 0.0,
        "canopy_height": 27.0,
        "secondary_height_fraction": 0.68,
        "structure_tau": 48.0,
    },
    # Seasonal savanna competition: woody closure is harder to sustain where
    # rainfall is strongly seasonal, while grass continuity peaks after enough
    # rainfall to make fuel but before a closed canopy develops.
    "seasonal_competition": {
        "productivity_scale": 2.8,
        "productivity_rain_half": 300.0,
        "woody_rain_center": 900.0,
        "woody_rain_width": 240.0,
        "seasonality_strength": 1.4,
        "deficit_strength": 0.65,
        "grass_rain_center": 320.0,
        "grass_rain_width": 105.0,
        "grass_shade": 2.8,
        "grass_establishment_power": 1.0,
        "humid_woody_floor": 0.0,
        "canopy_height": 29.0,
        "secondary_height_fraction": 0.62,
        "structure_tau": 60.0,
    },
    # Stronger water-balance competition.  This puts the woody/open transition
    # nearer the wet end of the observed savanna range and makes dry-season
    # soil-water drawdown a stronger constraint on tree closure.
    "deficit_partition": {
        "productivity_scale": 3.2,
        "productivity_rain_half": 360.0,
        "woody_rain_center": 1050.0,
        "woody_rain_width": 230.0,
        "seasonality_strength": 1.9,
        "deficit_strength": 1.1,
        "grass_rain_center": 380.0,
        "grass_rain_width": 120.0,
        "grass_shade": 3.3,
        "grass_establishment_power": 1.0,
        "humid_woody_floor": 0.0,
        "canopy_height": 31.0,
        "secondary_height_fraction": 0.58,
        "structure_tau": 72.0,
    },
    # Refinement after the first exact round.  Squaring the establishment gate
    # represents loss of a percolating grass layer below the productivity
    # threshold.  A smooth humid-climate union restores woody closure where
    # the first seasonality response incorrectly held wet primary land open.
    "seasonal_percolation": {
        "productivity_scale": 3.0,
        "productivity_rain_half": 320.0,
        "woody_rain_center": 860.0,
        "woody_rain_width": 240.0,
        "seasonality_strength": 1.15,
        "deficit_strength": 0.60,
        "grass_rain_center": 360.0,
        "grass_rain_width": 90.0,
        "grass_shade": 3.0,
        "grass_establishment_power": 2.0,
        "humid_woody_floor": 0.72,
        "canopy_height": 31.0,
        "secondary_height_fraction": 0.62,
        "structure_tau": 60.0,
    },
}


SAFEGUARDS: dict[str, dict[str, float]] = {
    "moderate": {
        "arid_strength": 0.8,
        "arid_fuel_selectivity": 0.0,
        "forest_strength": 1.0,
        "temperate_strength": 0.0,
        "cold_strength": 0.45,
        "crop_strength": 0.8,
    },
    "strong": {
        "arid_strength": 1.6,
        "arid_fuel_selectivity": 0.0,
        "forest_strength": 1.8,
        "temperate_strength": 0.0,
        "cold_strength": 0.9,
        "crop_strength": 1.5,
    },
    "fuel_selective": {
        "arid_strength": 1.8,
        "arid_fuel_selectivity": 1.0,
        "forest_strength": 0.0,
        "temperate_strength": 1.0,
        "cold_strength": 0.0,
        "crop_strength": 0.8,
    },
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def causal_annual_rain(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Annualize the available trailing one-to-twelve monthly rain history.

    The installed ``annual_precipitation`` repeats each current-year total in
    every month and therefore contains future rainfall.  This function never
    reads it.  January at the beginning of a supplied prefix uses January
    alone, annualized; after twelve months the state is an exact trailing-year
    precipitation sum.
    """
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float32), 0.0, None)
    output = np.empty_like(rain, dtype=np.float32)
    accumulator = np.zeros_like(rain[0], dtype=np.float32)
    for time in range(rain.shape[0]):
        accumulator += rain[time]
        if time >= 12:
            accumulator -= rain[time - 12]
        available = min(time + 1, 12)
        output[time] = accumulator * np.float32(12.0 / available)
    return output


def _land_fractions(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Translate LUH2 shares into non-overlapping natural/secondary cover."""
    primary = np.clip(np.asarray(data["luh2_primary_fraction"], dtype=np.float32), 0.0, 1.0)
    secondary = np.clip(np.asarray(data["luh2_secondary_fraction"], dtype=np.float32), 0.0, 1.0)
    crop = np.clip(np.asarray(data["luh2_cropland_fraction"], dtype=np.float32), 0.0, 1.0)
    pasture = np.clip(np.asarray(data["luh2_pasture_fraction"], dtype=np.float32), 0.0, 1.0)
    rangeland = np.clip(np.asarray(data["luh2_rangeland_fraction"], dtype=np.float32), 0.0, 1.0)
    urban = np.clip(np.asarray(data["luh2_urban_fraction"], dtype=np.float32), 0.0, 1.0)
    residual = np.clip(1.0 - primary - secondary - crop - pasture - rangeland - urban, 0.0, 1.0)
    natural = np.clip(primary + residual, 0.0, 1.0)
    secondary = np.minimum(secondary, np.clip(1.0 - natural, 0.0, 1.0))
    return natural.astype(np.float32), secondary.astype(np.float32)


def reconstruct(
    data: Mapping[str, np.ndarray],
    structure: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Build climate-derived ED-like state using current and trailing inputs."""
    shape = np.asarray(data["dryness"]).shape
    ntime = shape[0]
    spatial = shape[1:]
    gpp = np.empty(shape, dtype=np.float32)
    biomass = np.empty(shape, dtype=np.float32)
    soil_carbon = np.empty(shape, dtype=np.float32)
    lai = np.empty(shape, dtype=np.float32)
    natural_height = np.empty(shape, dtype=np.float32)
    secondary_height = np.empty(shape, dtype=np.float32)
    annual_rain = causal_annual_rain(data)

    natural, secondary = _land_fractions(data)
    vegetated = np.clip(
        natural
        + secondary
        + np.asarray(data["luh2_pasture_fraction"], dtype=np.float32)
        + np.asarray(data["luh2_rangeland_fraction"], dtype=np.float32)
        + np.asarray(data["luh2_cropland_fraction"], dtype=np.float32),
        0.0,
        1.0,
    )

    rain0 = np.clip(np.asarray(data["monthly_precipitation"][0], dtype=np.float32), 0.0, None)
    temp0 = np.asarray(data["air_temperature"][0], dtype=np.float32)
    annual0 = annual_rain[0]
    deficit0 = np.clip(np.asarray(data["dryness"][0], dtype=np.float32), 0.0, None)

    # Each state is initialized solely from the first supplied month.  The
    # running deviation begins at zero rather than importing a climatological
    # seasonal cycle from the completed record.
    rain_mean = rain0.copy()
    rain_deviation = np.zeros(spatial, dtype=np.float32)
    temperature_mean = temp0.copy()
    root_water = np.clip(rain0 / (rain0 + 70.0) * np.exp(-deficit0 / 1800.0), 0.0, 1.0).astype(np.float32)

    warm0 = sigmoid((temperature_mean - 2.0) / 5.0)
    wet0 = sigmoid((annual0 - structure["woody_rain_center"]) / structure["woody_rain_width"])
    woody_state = np.clip(warm0 * wet0, 0.0, 1.0).astype(np.float32)
    productivity0 = (
        structure["productivity_scale"]
        * warm0
        * annual0 / (annual0 + structure["productivity_rain_half"] + 1e-6)
        * np.exp(-annual0 / 6500.0)
    )
    biomass_state = np.clip(17.0 * woody_state**1.35 * productivity0 / (productivity0 + 0.8), 0.0, 22.0).astype(np.float32)

    alpha3 = np.float32(1.0 - np.exp(-1.0 / 3.0))
    alpha12 = np.float32(1.0 - np.exp(-1.0 / 12.0))
    alpha_structure = np.float32(1.0 - np.exp(-1.0 / structure["structure_tau"]))
    alpha_biomass = np.float32(1.0 - np.exp(-1.0 / 96.0))

    for time in range(ntime):
        rain = np.clip(np.asarray(data["monthly_precipitation"][time], dtype=np.float32), 0.0, None)
        annual = annual_rain[time]
        temperature = np.asarray(data["air_temperature"][time], dtype=np.float32)
        deficit = np.clip(np.asarray(data["dryness"][time], dtype=np.float32), 0.0, None)

        old_rain_mean = rain_mean.copy()
        rain_mean += alpha12 * (rain - rain_mean)
        rain_deviation += alpha12 * (np.abs(rain - old_rain_mean) - rain_deviation)
        temperature_mean += alpha12 * (temperature - temperature_mean)

        # A bounded root-zone water bucket supplies causal phenology.  Rain
        # recharges it; warmth and the meteorological water deficit draw it
        # down.  It is state, not a transformed target or future climatology.
        recharge = np.clip(rain / 180.0, 0.0, 0.70)
        demand = (
            0.035
            + 0.075 * sigmoid((temperature - 12.0) / 5.0)
            + 0.18 * deficit / (deficit + 650.0)
        )
        root_water = np.clip(0.94 * root_water + recharge - demand, 0.0, 1.0).astype(np.float32)

        warm = sigmoid((temperature_mean - 2.0) / 5.0)
        water_productivity = annual / (annual + structure["productivity_rain_half"] + 1e-6)
        potential_productivity = (
            structure["productivity_scale"]
            * warm
            * water_productivity
            * np.exp(-annual / 6500.0)
        )

        # Normalized causal rainfall variability and accumulated water deficit
        # both penalize closed-forest equilibrium.  The deficit response is
        # broadest around the 400--800 mm transition and saturates smoothly.
        seasonality = np.clip(rain_deviation / (rain_mean + 25.0), 0.0, 2.0)
        water_deficit = deficit / (deficit + 650.0)
        woody_target = (
            warm
            * sigmoid((annual - structure["woody_rain_center"]) / structure["woody_rain_width"])
            * np.exp(-structure["seasonality_strength"] * seasonality)
            * np.exp(-structure["deficit_strength"] * water_deficit)
        )
        humid_floor = (
            structure["humid_woody_floor"]
            * warm
            * sigmoid((annual - 1450.0) / 260.0)
        )
        # Probabilistic union is a differentiable maximum-like operation.  It
        # preserves seasonal savanna competition while permitting humid forest
        # closure without a categorical biome switch.
        woody_target = 1.0 - (1.0 - np.clip(woody_target, 0.0, 1.0)) * (
            1.0 - np.clip(humid_floor, 0.0, 1.0)
        )
        woody_state += alpha_structure * (np.clip(woody_target, 0.0, 1.0) - woody_state)
        woody_state = np.clip(woody_state, 0.0, 1.0).astype(np.float32)

        # Grass requires enough water to establish a connected fuel bed, then
        # loses continuity as woody closure shades the surface.  Combined with
        # the rain-driven woody transition this creates a smooth intermediate-
        # cover fire window rather than a forest/nonforest switch.
        establishment = sigmoid((annual - structure["grass_rain_center"]) / structure["grass_rain_width"])
        open_surface = np.exp(-structure["grass_shade"] * woody_state)
        grass_continuity = np.clip(
            establishment ** structure["grass_establishment_power"] * open_surface,
            0.0,
            1.0,
        )
        green_phenology = np.sqrt(root_water)
        herbaceous_gpp = potential_productivity * grass_continuity * (0.08 + 0.92 * green_phenology)
        woody_gpp = 0.42 * potential_productivity * woody_state * (0.30 + 0.70 * green_phenology)
        gpp[time] = np.clip((herbaceous_gpp + woody_gpp) * vegetated[time], 0.0, 5.5)

        biomass_target = (
            17.0 * woody_state**1.35 * potential_productivity / (potential_productivity + 0.8)
            + 0.55 * grass_continuity
        )
        biomass_state += alpha_biomass * (np.clip(biomass_target, 0.0, 22.0) - biomass_state)
        biomass[time] = np.clip(biomass_state * vegetated[time], 0.0, 22.0)

        natural_height[time] = np.clip(
            structure["canopy_height"] * woody_state**0.75,
            0.0,
            35.0,
        )
        secondary_height[time] = np.clip(
            structure["secondary_height_fraction"] * natural_height[time],
            0.0,
            30.0,
        )
        lai[time] = np.clip(
            (
                5.2 * woody_state * (0.35 + 0.65 * green_phenology)
                + 1.35 * grass_continuity * green_phenology
            )
            * vegetated[time],
            0.0,
            7.5,
        )

        # This climate-derived slow carbon store is used only by the incumbent
        # cold-thaw carrier.  Cold and moist productive systems accumulate it;
        # warm, dry systems turn it over rapidly.
        cold_retention = sigmoid((7.0 - temperature_mean) / 5.0)
        soil_carbon[time] = np.clip(
            (0.7 + 10.0 * water_productivity * (0.25 + 0.75 * cold_retention))
            * vegetated[time],
            0.0,
            18.0,
        )

    reconstructed = dict(data)
    reconstructed.update(
        {
            "annual_precipitation": annual_rain,
            "gpp": gpp,
            "aboveground_biomass": biomass,
            "soil_carbon": soil_carbon,
            "leaf_area_index": lai,
            "natural_canopy_height": natural_height,
            "secondary_canopy_height": secondary_height,
            "natural_vegetation_fraction": natural,
            "secondary_vegetation_fraction": secondary,
        }
    )
    return reconstructed


def ecological_safeguard(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    reconstructed: Mapping[str, np.ndarray],
    strengths: Mapping[str, float],
) -> np.ndarray:
    """Apply smooth local fuel, shielding, snow, and fragmentation limits."""
    annual = np.asarray(reconstructed["annual_precipitation"], dtype=np.float32)
    canopy = np.asarray(reconstructed["natural_canopy_height"], dtype=np.float32)
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float32), 0.0, None)
    temperature = np.asarray(data["air_temperature"], dtype=np.float32)
    primary = np.clip(np.asarray(data["luh2_primary_fraction"], dtype=np.float32), 0.0, 1.0)
    crop = np.clip(np.asarray(data["luh2_cropland_fraction"], dtype=np.float32), 0.0, 1.0)
    urban = np.clip(np.asarray(data["luh2_urban_fraction"], dtype=np.float32), 0.0, 1.0)

    output = np.empty_like(prediction, dtype=np.float32)
    temperature_mean = temperature[0].copy()
    gpp = np.clip(np.asarray(reconstructed["gpp"], dtype=np.float32), 0.0, None)
    gpp_mean = gpp[0].copy()
    snowfall0 = rain[0] * sigmoid((1.0 - temperature[0]) / 2.0)
    snow = np.asarray(snowfall0, dtype=np.float32)
    alpha12 = np.float32(1.0 - np.exp(-1.0 / 12.0))

    for time in range(prediction.shape[0]):
        temperature_mean += alpha12 * (temperature[time] - temperature_mean)
        gpp_mean += alpha12 * (gpp[time] - gpp_mean)
        snowfall = rain[time] * sigmoid((1.0 - temperature[time]) / 2.0)
        melt_fraction = 1.0 - np.exp(-np.clip(temperature[time], 0.0, None) / 4.0)
        snow = np.clip((snow + snowfall) * (1.0 - melt_fraction), 0.0, 500.0).astype(np.float32)
        snow_cover = snow / (snow + 18.0)

        # Below roughly 250--350 mm yr-1, insufficient production prevents a
        # continuous fine-fuel bed even when the atmosphere is dry.
        low_rain = sigmoid((320.0 - annual[time]) / 85.0)
        absent_fuel = sigmoid((0.018 - gpp_mean) / 0.007)
        fuel_selectivity = strengths["arid_fuel_selectivity"]
        arid_fuel_failure = low_rain * (
            1.0 - fuel_selectivity + fuel_selectivity * absent_fuel
        )
        # Wet primary vegetation closes and shades the grassy surface.  Canopy
        # height is itself the causal climate-derived woody equilibrium above.
        closed_primary = (
            primary[time]
            * sigmoid((annual[time] - 1050.0) / 240.0)
            * sigmoid((canopy[time] - 13.0) / 3.0)
        )
        temperate_closed_primary = (
            primary[time]
            * sigmoid((temperature_mean - 5.0) / 3.0)
            * sigmoid((18.0 - temperature_mean) / 3.0)
            * sigmoid((annual[time] - 500.0) / 180.0)
            * sigmoid((canopy[time] - 10.0) / 3.0)
        )
        # Cold monthly combustion is suppressed both by a causal snow store
        # and by persistently cold primary vegetation.  Warm boreal fire-season
        # months remain available as both smooth factors approach zero.
        cold_primary = primary[time] * sigmoid((8.0 - temperature_mean) / 4.0) * (
            snow_cover + (1.0 - snow_cover) * sigmoid((3.0 - temperature[time]) / 3.0)
        )
        # Harvest, field boundaries, and urban infrastructure interrupt spread
        # continuously with managed cover; no population or regional label is
        # needed.
        crop_fragmentation = np.clip(crop[time] ** 1.5 + 4.0 * urban[time], 0.0, 1.0)

        log_suppression = (
            strengths["arid_strength"] * arid_fuel_failure
            + strengths["forest_strength"] * closed_primary
            + strengths["temperate_strength"] * temperate_closed_primary
            + strengths["cold_strength"] * cold_primary
            + strengths["crop_strength"] * crop_fragmentation
        )
        hazard = -np.log1p(-np.clip(prediction[time], 0.0, 1.0 - 1e-7))
        output[time] = 1.0 - np.exp(-hazard * np.exp(-log_suppression))
    return np.asarray(np.clip(output, 0.0, 1.0), dtype=np.float32)


def predict(
    data: Mapping[str, np.ndarray],
    structure_name: str,
    *,
    annual_scale: float = 1.0,
    safeguard_name: str | None = None,
) -> np.ndarray:
    reconstructed = reconstruct(data, STRUCTURES[structure_name])
    params = dict(incumbent.PARAMS)
    params["annual_scale"] *= annual_scale
    prediction = incumbent.predict(reconstructed, params=params)
    if safeguard_name is not None:
        prediction = ecological_safeguard(
            prediction,
            data,
            reconstructed,
            SAFEGUARDS[safeguard_name],
        )
    return validate_prediction(prediction)


def main() -> int:
    data = load_inputs(INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    results: list[tuple[float, str, float, str | None, dict[str, float]]] = []
    best_prediction: np.ndarray | None = None
    best_key: tuple[str, float, str | None] | None = None

    print("certified_inputs=" + ",".join(INPUTS), flush=True)
    requested_round = sys.argv[1] if len(sys.argv) > 1 else "refine"
    base_round = requested_round == "base"
    # The base round is retained for reproducibility.  The default refinement
    # is a three-member structural comparison, not a combinatorial grid.
    experiments = (
        tuple(
            (name, scale, None)
            for name in ("rain_equilibrium", "seasonal_competition", "deficit_partition")
            for scale in (0.80, 1.00, 1.25)
        )
        if base_round
        else (("seasonal_percolation", 1.65, "fuel_selective"),)
        if requested_round == "selective"
        else (
            ("seasonal_percolation", 1.40, None),
            ("seasonal_percolation", 1.55, "moderate"),
            ("seasonal_percolation", 1.75, "strong"),
        )
    )
    for structure_name, annual_scale, safeguard_name in experiments:
        prediction = predict(
            data,
            structure_name,
            annual_scale=annual_scale,
            safeguard_name=safeguard_name,
        )
        score = dict(evaluator.score(prediction)["global"])
        overall = float(score["overall_score"])
        guard_label = safeguard_name or "none"
        label = f"{structure_name}:annual_scale={annual_scale:g}:guard={guard_label}"
        print(metric_line(label, score), flush=True)
        audit_data = dict(data)
        audit_data["annual_precipitation"] = causal_annual_rain(data)
        ecology = exogenous_ecology_ratios(prediction, audit_data, evaluator)
        print(
            "ECOLOGY_CANDIDATE "
            + label
            + " "
            + " ".join(f"{name}={value:.6f}" for name, value in ecology.items()),
            flush=True,
        )
        results.append((overall, structure_name, annual_scale, safeguard_name, score))
        if best_prediction is None or overall > max(row[0] for row in results[:-1]):
            best_prediction = prediction.copy()
            best_key = (structure_name, annual_scale, safeguard_name)
        del prediction
        gc.collect()

    assert best_prediction is not None and best_key is not None
    best_overall, best_structure, best_scale, best_guard, best_score = max(
        results, key=lambda row: row[0]
    )
    if best_key != (best_structure, best_scale, best_guard):
        del best_prediction
        gc.collect()
        best_prediction = predict(
            data,
            best_structure,
            annual_scale=best_scale,
            safeguard_name=best_guard,
        )

    scores = evaluator.score(best_prediction)
    # Full-record means are allowed here only to label diagnostic ecology
    # regimes after prediction.  The prediction path above receives the exact
    # causal annual-rain history instead of the leaky installed annual field.
    audit_data = dict(data)
    audit_data["annual_precipitation"] = causal_annual_rain(data)
    ecology = exogenous_ecology_ratios(best_prediction, audit_data, evaluator)
    print(
        "BEST "
        + metric_line(
            f"{best_structure}:annual_scale={best_scale:g}:guard={best_guard or 'none'}",
            best_score,
        ),
        flush=True,
    )
    print(
        "REGIONAL "
        + " ".join(
            f"{name}={values['overall_score']:.6f}"
            for name, values in sorted(scores.items())
            if name != "global"
        ),
        flush=True,
    )
    print(
        "ECOLOGY " + " ".join(f"{name}={value:.6f}" for name, value in ecology.items()),
        flush=True,
    )

    prefix_months = 73
    # The public model interface has a fixed 192-month duration vector, so
    # prefix causality is tested by changing every future input while retaining
    # the full interface shape.  Earlier predictions must remain bit-identical.
    perturbed_data = {name: values.copy() for name, values in data.items()}
    for values in perturbed_data.values():
        values[prefix_months:] *= 0.5
    prefix_prediction = predict(
        perturbed_data,
        best_structure,
        annual_scale=best_scale,
        safeguard_name=best_guard,
    )
    prefix_difference = float(
        np.max(np.abs(prefix_prediction[:prefix_months] - best_prediction[:prefix_months]))
    )
    print(
        f"PREFIX months={prefix_months} max_abs_difference={prefix_difference:.12g}",
        flush=True,
    )
    print(
        "EQUATIONS P[t]=annualize(sum(monthly_rain[max(0,t-11):t+1])); "
        "potential_productivity=s*Tgate*P/(P+Phalf)*exp(-P/6500); "
        "woody_target=Tgate*logistic(P)*exp(-ks*rain_seasonality-kd*deficit); "
        "W[t]=W[t-1]+alpha*(woody_target-W[t-1]); "
        "grass_continuity=logistic(P)*exp(-kshade*W); "
        "root_water[t]=clip(.94*root_water+rain/180-demand,0,1); "
        "GPP=potential*(grass*(.08+.92*sqrt(water))+.42*W*(.30+.70*sqrt(water)))",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
