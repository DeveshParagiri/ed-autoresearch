"""Scratch-only exogenous self-organising vegetation and fire family.

This experiment uses only coupled meteorology, LUH2 land use, and fixed
lightning climatology.  It deliberately excludes every frozen ``ed.nc`` state,
modern-only weather field, geography, neighbours, and the benchmark at runtime.

Woody cover is initialized from a local climate equilibrium rather than from
prescribed vegetation structure.  Herbaceous live and dead fuel then compete
with woody shade, while fire consumes finite fuel and can top-kill woody
vegetation.  All updates are globally shared, pointwise, prefix-causal, and the
monthly burned fraction is a bounded Poisson footprint.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


INPUTS = (
    "dryness",
    "monthly_precipitation",
    "air_temperature",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_urban_fraction",
    "lightning_flash_rate",
)

MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float32),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def rising(values: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth increasing response with a physically legible center and width."""
    return 1.0 / (
        1.0
        + np.exp(
            np.clip(-(values - center) / max(float(width), 1e-6), -30.0, 30.0)
        )
    )


def climate_state(
    data: Mapping[str, np.ndarray],
    time: int,
    slow_rain: np.ndarray,
    slow_temperature: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive local climate carrying capacities without vegetation observations."""
    rain = np.clip(
        np.asarray(data["monthly_precipitation"][time], dtype=np.float32),
        0.0,
        None,
    )
    temperature = np.asarray(data["air_temperature"][time], dtype=np.float32)
    deficit = np.clip(
        np.asarray(data["dryness"][time], dtype=np.float32), 0.0, None
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"][time], dtype=np.float32),
        0.0,
        1.0,
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"][time], dtype=np.float32),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"][time], dtype=np.float32),
        0.0,
        1.0,
    )
    urban = np.clip(
        np.asarray(data["luh2_urban_fraction"][time], dtype=np.float32),
        0.0,
        1.0,
    )
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"][time], dtype=np.float32),
        0.0,
        None,
    )

    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    nonurban = np.clip(1.0 - urban, 0.0, 1.0)
    uncultivated = np.clip(1.0 - crop - urban, 0.0, 1.0)

    # The forcing's annual-precipitation field is intentionally absent: it is
    # a completed current-year total repeated into earlier months.  Twelve
    # times a trailing monthly rain state is instead a causal local water
    # supply estimate available at every step, including initialization.
    rain_supply = np.clip(12.0 * slow_rain, 0.0, 6000.0)

    # Trees establish only where both water and warmth support them.  Grazing,
    # cultivation, and urbanization reduce establishment smoothly, rather than
    # prescribing forest from a vegetation map.
    woody_moisture = rising(rain_supply, 720.0, 260.0)
    woody_thermal = rising(slow_temperature, 1.0, 5.5)
    woody_land_access = uncultivated * np.exp(
        -1.8 * managed_open - 4.0 * crop - 10.0 * urban
    )
    woody_capacity = np.clip(
        woody_land_access * woody_moisture * woody_thermal, 0.0, 1.0
    )

    # Grass production rises from desert with water, but its open-canopy
    # advantage declines in persistently wet climates.  Actual woody shade is
    # applied later, so wet sites can self-organize into forest or savanna.
    herb_water = (1.0 - np.exp(-rain_supply / 310.0)) * np.exp(
        -rain_supply / 5200.0
    )
    herb_thermal = rising(slow_temperature, -1.0, 5.0)
    herb_land = np.clip(
        nonurban * (1.0 - 0.70 * crop) * np.exp(-0.75 * managed_open * herb_water),
        0.0,
        1.0,
    )
    herb_capacity = np.clip(herb_land * herb_water * herb_thermal, 0.0, 1.0)

    rainfall_growth = rain / (rain + 42.0)
    thermal_growth = rising(temperature, 0.0, 5.0)
    combustion = (
        deficit / (deficit + 125.0)
        * 1.0 / (1.0 + rain / 55.0)
        * rising(temperature, 5.0, 4.5)
    )

    lightning_access = lightning / (lightning + 0.014)
    managed_access = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    managed_ignition = managed_access / (managed_access + 0.18)
    ignition = 1.0 - (1.0 - lightning_access) * (
        1.0 - 0.70 * managed_ignition
    )
    ignition = np.clip(0.025 + 0.975 * ignition, 0.0, 1.0)

    return {
        "rain_supply": rain_supply,
        "rain": rain,
        "temperature": temperature,
        "crop": crop,
        "managed_open": managed_open,
        "urban": urban,
        "woody_capacity": woody_capacity,
        "herb_capacity": herb_capacity,
        "rainfall_growth": rainfall_growth,
        "thermal_growth": thermal_growth,
        "combustion": np.clip(combustion, 0.0, 1.0),
        "ignition": ignition,
    }


STRUCTURES: dict[str, dict[str, float]] = {
    # Direct climate succession is the control: fire consumes fuel but has
    # only weak demographic feedback on mature woody cover.
    "climate_succession": {
        "shade": 2.2,
        "seedling_competition": 0.8,
        "woody_recruitment": 0.010,
        "woody_turnover": 0.0015,
        "fire_woody_mortality": 0.65,
        "water_bucket": 0.0,
        "two_stage": 0.0,
    },
    # Savanna fire maintains grass by top-killing woody vegetation, while
    # established canopy shades grass and grass competes with tree seedlings.
    "fire_grass_feedback": {
        "shade": 3.2,
        "seedling_competition": 2.0,
        "woody_recruitment": 0.009,
        "woody_turnover": 0.0015,
        "fire_woody_mortality": 2.6,
        "water_bucket": 0.0,
        "two_stage": 0.0,
    },
    # A shallow water bucket constrains grass more abruptly during drought;
    # rooted woody vegetation retains partial access to annual moisture.
    "root_depth_partition": {
        "shade": 2.8,
        "seedling_competition": 1.5,
        "woody_recruitment": 0.009,
        "woody_turnover": 0.0015,
        "fire_woody_mortality": 2.0,
        "water_bucket": 1.0,
        "two_stage": 0.0,
    },
    # Juveniles mature slowly and are more fire-sensitive than adult canopy;
    # this is a demographic alternative to instantaneous woody top-kill.
    "juvenile_bottleneck": {
        "shade": 3.0,
        "seedling_competition": 1.8,
        "woody_recruitment": 0.015,
        "woody_turnover": 0.0012,
        "fire_woody_mortality": 1.0,
        "water_bucket": 0.0,
        "two_stage": 1.0,
    },
}


def simulate(
    data: Mapping[str, np.ndarray],
    structure: str,
    event_scale: float,
    *,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evolve local vegetation and finite fuel through the supplied prefix."""
    p = STRUCTURES[structure]
    shape = np.asarray(data["dryness"]).shape
    prediction = np.empty(shape, dtype=np.float32)
    slow_temperature = np.asarray(
        data["air_temperature"][0], dtype=np.float32
    ).copy()
    slow_rain = np.clip(
        np.asarray(data["monthly_precipitation"][0], dtype=np.float32),
        0.0,
        None,
    ).copy()
    first = climate_state(data, 0, slow_rain, slow_temperature)

    # The initial state is an analytic climate equilibrium available at t=0.
    # No completed-record climatology or future month is consulted.
    woody = np.asarray(first["woody_capacity"], dtype=np.float32).copy()
    juvenile = (0.12 * woody).astype(np.float32)
    if p["two_stage"] > 0.5:
        woody *= np.float32(0.88)
    herb_capacity = first["herb_capacity"] * np.exp(-p["shade"] * woody)
    green = (0.55 * herb_capacity).astype(np.float32)
    dead = np.clip(
        0.45 * herb_capacity + 0.08 * woody, 0.0, 1.5
    ).astype(np.float32)
    shallow_water = np.clip(
        first["rain_supply"] / (first["rain_supply"] + 700.0), 0.0, 1.0
    ).astype(np.float32)

    alpha_temperature = np.float32(1.0 - np.exp(-1.0 / 24.0))
    alpha_rain = np.float32(1.0 - np.exp(-1.0 / 12.0))
    last_state: dict[str, np.ndarray] = {}
    for time in range(shape[0]):
        temperature = np.asarray(data["air_temperature"][time], dtype=np.float32)
        rain = np.clip(
            np.asarray(data["monthly_precipitation"][time], dtype=np.float32),
            0.0,
            None,
        )
        slow_rain += alpha_rain * (rain - slow_rain)
        slow_temperature += alpha_temperature * (temperature - slow_temperature)
        f = climate_state(data, time, slow_rain, slow_temperature)

        if p["water_bucket"] > 0.5:
            recharge = np.clip(f["rain"] / 115.0, 0.0, 0.8)
            evaporative_loss = (
                0.045
                + 0.13 * f["thermal_growth"]
                + 0.16 * f["combustion"]
            )
            shallow_water = np.clip(
                shallow_water + recharge - evaporative_loss, 0.0, 1.0
            ).astype(np.float32)
            grass_water = np.sqrt(
                np.clip(
                    f["rainfall_growth"] * (0.10 + 0.90 * shallow_water),
                    0.0,
                    1.0,
                )
            )
        else:
            grass_water = f["rainfall_growth"]

        total_woody = np.clip(woody + juvenile, 0.0, 1.0)
        herb_capacity = f["herb_capacity"] * np.exp(
            -p["shade"] * total_woody
        )
        herb_fill = np.clip(green / (herb_capacity + 1e-5), 0.0, 1.5)
        production = (
            0.22
            * herb_capacity
            * f["thermal_growth"]
            * (0.15 + 0.85 * grass_water)
            * np.clip(1.0 - herb_fill, 0.0, 1.0)
        )
        green = np.clip(green + production, 0.0, 1.25)

        curing_fraction = np.clip(
            0.025 + 0.30 * f["combustion"], 0.0, 0.50
        )
        cured = curing_fraction * green
        green -= cured
        woody_litter = 0.0030 * total_woody
        wet_decay = 0.025 + 0.10 * f["rain"] / (f["rain"] + 55.0)
        dead = np.clip(
            dead * np.exp(-wet_decay) + cured + woody_litter, 0.0, 1.75
        )

        # Dense wet canopy limits the continuity and drying of surface litter.
        surface_access = np.exp(
            -2.4
            * total_woody
            * rising(f["rain_supply"], 1150.0, 300.0)
        )
        crop_fragmentation = np.exp(-1.8 * f["crop"] - 8.0 * f["urban"])
        carrier_cover = np.clip(
            herb_capacity + 0.22 * total_woody + 0.18 * f["crop"],
            0.0,
            1.0,
        )
        connected_fuel = (1.0 - np.exp(-dead / 0.16)) * carrier_cover
        hazard = (
            event_scale
            * connected_fuel
            * surface_access
            * crop_fragmentation
            * f["combustion"]
            * f["ignition"]
            * MONTH_DAYS[time]
            / 30.4375
        )
        burn = 1.0 - np.exp(-np.clip(hazard, 0.0, 20.0))
        prediction[time] = burn

        # The same finite event removes herbaceous fuel.  Woody demographic
        # feedback is proportional to burned area and combustion severity.
        dead *= np.exp(-5.0 * burn)
        green *= np.exp(-1.8 * burn)
        fire_pressure = burn * (0.35 + 0.65 * f["combustion"])

        grass_competition = np.exp(-p["seedling_competition"] * herb_fill)
        woody_gap = np.clip(f["woody_capacity"] - total_woody, 0.0, 1.0)
        recruitment = (
            p["woody_recruitment"]
            * woody_gap
            * grass_competition
            * f["thermal_growth"]
        )
        if p["two_stage"] > 0.5:
            juvenile = np.clip(
                juvenile
                + recruitment
                - 0.020 * juvenile
                - 5.0 * fire_pressure * juvenile,
                0.0,
                1.0,
            )
            maturation = 0.012 * juvenile
            juvenile -= maturation
            woody = np.clip(
                woody
                + maturation
                - p["woody_turnover"] * woody
                - p["fire_woody_mortality"] * fire_pressure * woody,
                0.0,
                1.0,
            )
        else:
            woody = np.clip(
                woody
                + recruitment
                - p["woody_turnover"] * woody
                - p["fire_woody_mortality"] * fire_pressure * woody,
                0.0,
                1.0,
            )

        if return_state and time == shape[0] - 1:
            last_state = {
                "woody": np.clip(woody + juvenile, 0.0, 1.0).copy(),
                "green": green.copy(),
                "dead": dead.copy(),
                "woody_capacity": f["woody_capacity"].copy(),
                "herb_capacity": herb_capacity.copy(),
            }

    prediction = np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
    if return_state:
        return prediction, last_state
    return prediction


def score_line(label: str, score: Mapping[str, float]) -> str:
    return (
        f"{label:38s} overall={score['overall_score']:.9f} "
        f"bias={score['bias_score']:.6f} rmse={score['rmse_score']:.6f} "
        f"season={score['seasonal_cycle_score']:.6f} "
        f"spatial={score['spatial_distribution_score']:.6f} "
        f"annual_pct={12.0 * score['model_period_mean_percent']:.6f}"
    )


def ecological_ratios(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    evaluator: GFED5Evaluator,
) -> dict[str, tuple[int, float]]:
    """Audit climate and land-use regimes defined without frozen ED state."""
    pred_annual = prediction.reshape(-1, 12, 180, 360).mean(axis=0).sum(axis=0)
    obs_cycle = (
        np.asarray(evaluator.reference_cycle)
        .reshape(12, 180, 2, 360, 2)
        .mean(axis=(2, 4))
        / 100.0
    )
    obs_annual = obs_cycle.sum(axis=0)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))

    def period_mean(name: str) -> np.ndarray:
        return np.asarray(data[name]).reshape(-1, 12, 180, 360).mean(axis=(0, 1))

    annual_rain = 12.0 * period_mean("monthly_precipitation")
    temperature = period_mean("air_temperature")
    crop = period_mean("luh2_cropland_fraction")
    pasture = period_mean("luh2_pasture_fraction")
    rangeland = period_mean("luh2_rangeland_fraction")
    urban = period_mean("luh2_urban_fraction")
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    woody_potential = (
        np.clip(1.0 - crop - urban, 0.0, 1.0)
        * rising(annual_rain, 720.0, 260.0)
        * rising(temperature, 1.0, 5.5)
    )
    regimes = {
        "wet_warm_woody_potential": (
            (temperature >= 20.0)
            & (annual_rain >= 1500.0)
            & (woody_potential >= 0.55)
        ),
        "seasonal_warm_open": (
            (temperature >= 20.0)
            & (annual_rain >= 500.0)
            & (annual_rain < 1500.0)
            & (managed_open >= 0.25)
        ),
        "temperate_woody_potential": (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (annual_rain >= 600.0)
            & (woody_potential >= 0.45)
        ),
        "boreal_woody_potential": (
            (temperature < 5.0)
            & (annual_rain >= 300.0)
            & (woody_potential >= 0.25)
        ),
        "productive_rangeland": (
            (rangeland >= 0.40)
            & (annual_rain >= 250.0)
            & (annual_rain < 1500.0)
        ),
        "cropland_dominant": crop >= 0.50,
        "arid_low_rain": annual_rain < 250.0,
    }
    ratios: dict[str, tuple[int, float]] = {}
    for name, mask in regimes.items():
        weight = area * mask
        denominator = float(np.sum(obs_annual * weight))
        numerator = float(np.sum(pred_annual * weight))
        ratio = numerator / denominator if denominator > 0.0 else float("nan")
        ratios[name] = (int(np.sum(mask)), ratio)
    return ratios


def synthetic_smoke() -> int:
    """Exercise every state equation on a tiny deterministic input cube."""
    shape = (7, 4, 5)
    data: dict[str, np.ndarray] = {}
    data["dryness"] = np.linspace(0.0, 800.0, np.prod(shape), dtype=np.float32).reshape(shape)
    data["monthly_precipitation"] = np.linspace(5.0, 120.0, np.prod(shape), dtype=np.float32).reshape(shape)
    data["air_temperature"] = np.full(shape, 22.0, dtype=np.float32)
    for name in (
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
    ):
        data[name] = np.full(shape, 0.08, dtype=np.float32)
    data["lightning_flash_rate"] = np.full(shape, 0.02, dtype=np.float32)
    for structure in STRUCTURES:
        prediction = simulate(data, structure, 0.06)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == shape
        assert np.isfinite(prediction).all()
        assert float(prediction.min()) >= 0.0
        assert float(prediction.max()) <= 1.0
        prefix = 4
        shorter = {name: values[:prefix] for name, values in data.items()}
        prefix_prediction = simulate(shorter, structure, 0.06)
        assert isinstance(prefix_prediction, np.ndarray)
        prefix_delta = float(
            np.max(np.abs(prefix_prediction - prediction[:prefix]))
        )
        assert prefix_delta == 0.0
        perturbed = {name: values.copy() for name, values in data.items()}
        for values in perturbed.values():
            values[prefix:] *= np.float32(0.5)
        future_prediction = simulate(perturbed, structure, 0.06)
        assert isinstance(future_prediction, np.ndarray)
        future_delta = float(
            np.max(np.abs(future_prediction[:prefix] - prediction[:prefix]))
        )
        assert future_delta == 0.0
        print(
            f"SMOKE {structure} min={prediction.min():.8f} "
            f"max={prediction.max():.8f} prefix={prefix_delta:.1f} "
            f"future={future_delta:.1f}",
            flush=True,
        )
    return 0


def main() -> int:
    if "--smoke" in sys.argv:
        return synthetic_smoke()

    data = load_inputs(INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    records: list[tuple[float, str, float, dict[str, float]]] = []
    print("RUNTIME_INPUTS=" + ",".join(INPUTS), flush=True)

    # Event-scale brackets span small disconnected footprints through a broad
    # connected surface-fire month.  Structure, not fine coefficient tuning,
    # is the primary comparison.
    for structure in STRUCTURES:
        for event_scale in (0.018, 0.036, 0.072, 0.120):
            prediction = validate_prediction(
                simulate(data, structure, event_scale)
            )
            score = evaluator.score(prediction)["global"]
            print(
                score_line(f"{structure}/scale={event_scale:g}", score),
                flush=True,
            )
            records.append(
                (
                    float(score["overall_score"]),
                    structure,
                    event_scale,
                    dict(score),
                )
            )
            del prediction
            gc.collect()

    records.sort(reverse=True, key=lambda row: row[0])
    print("TOP", flush=True)
    for _, structure, event_scale, score in records[:8]:
        print(
            score_line(f"{structure}/scale={event_scale:g}", score),
            flush=True,
        )

    _, best_structure, best_scale, best_score = records[0]
    result = simulate(data, best_structure, best_scale, return_state=True)
    assert isinstance(result, tuple)
    best, state = result
    best = validate_prediction(best)
    scores = evaluator.score(best)
    print(
        "BEST "
        + score_line(f"{best_structure}/scale={best_scale:g}", best_score),
        flush=True,
    )
    print("REGIONS", flush=True)
    for region, score in scores.items():
        print(f"{region:7s} overall={score['overall_score']:.6f}", flush=True)
    print("ECOLOGY", flush=True)
    for regime, (cells, ratio) in ecological_ratios(best, data, evaluator).items():
        print(f"{regime:31s} cells={cells:5d} ratio={ratio:.6f}", flush=True)
    print(
        "FINAL_STATE "
        + " ".join(
            f"{name}_mean={float(np.mean(values)):.7f}"
            for name, values in state.items()
        ),
        flush=True,
    )

    # A shorter record must reproduce the complete shared prefix exactly.
    prefix = 73
    prefix_data = {name: values[:prefix] for name, values in data.items()}
    prefix_prediction = simulate(prefix_data, best_structure, best_scale)
    assert isinstance(prefix_prediction, np.ndarray)
    prefix_delta = float(np.max(np.abs(prefix_prediction - best[:prefix])))
    print(
        f"PREFIX months={prefix} max_abs_difference={prefix_delta:.12g}",
        flush=True,
    )

    # Stronger causality audit: alter every permitted forcing and land-use
    # value after month 96 while preserving the first half exactly.  Earlier
    # predictions must remain bit-identical despite that large future swing.
    future_start = 96
    perturbed = {name: np.asarray(values).copy() for name, values in data.items()}
    for values in perturbed.values():
        values[future_start:] *= np.float32(0.5)
    future_prediction = simulate(perturbed, best_structure, best_scale)
    assert isinstance(future_prediction, np.ndarray)
    future_delta = float(
        np.max(np.abs(future_prediction[:future_start] - best[:future_start]))
    )
    print(
        f"FUTURE_PERTURBATION start={future_start} factor=0.5 "
        f"prefix_max_abs_difference={future_delta:.12g}",
        flush=True,
    )
    print(
        "EQUATION woody_capacity=climate*land_access; "
        "herb_capacity=climate*land*exp(-shade*woody); "
        "green+=logistic_production-curing; dead+=curing+woody_litter-decay; "
        "burn=1-exp(-scale*connected_dead*access*combustion*ignition*duration); "
        "fire_consumes(green,dead) and topkills woody; woody recruits into "
        "climate_capacity under grass competition",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
