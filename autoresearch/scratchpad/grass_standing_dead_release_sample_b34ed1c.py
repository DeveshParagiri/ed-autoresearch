"""Sampled falsification of grass production-to-standing-dead release clocks.

The diagnostic tests whether the dominant warm seasonal open-fire error is a
missing ecological delay between wet-season grass growth and dry-season fuel
availability.  The candidate equations use only monthly rain, temperature,
dryness, and LUH2 land fractions.  Coordinates select score-bearing cells and
define held spatial blocks only; observations and the diagnostic carrier mask
never enter a prediction equation.

This family is not a duplicate of the committed temporal mechanisms.  The
seasonal-rain capacity is a rain-variance magnitude multiplier with no stock.
The dead-fuel response infers curing from ED GPP/LAI moving averages and then
renormalizes a multiplier.  The surface bank stores incumbent hazard and uses
instantaneous ED-GPP curing.  The multi-pathway bank explicitly delegates
natural surface fire to that earlier bank, and the pathway recovery reservoir
is a post-fire regeneration limit.  The older scratch dry-age clock also ages
incumbent hazard directly and uses ED GPP/structure for eligibility.  Here,
rain first creates a live grass stock, senescence or cohort transit transfers
that stock to standing dead fuel, and that finite stock controls the release
of a separately conserved open-surface hazard bank.

Three globally shared pointwise structures are tested with one fixed physical
bracket apiece.  ``drawdown_transfer`` transfers live grass during root-zone
water drawdown.  ``erlang_maturation`` routes production through two live
cohort stages before drought-dependent senescence.  ``dry_degree_curing``
integrates bounded atmospheric water stress into gradual curing while rain
causes litter loss.  Every hazard transform satisfies input hazard = released
hazard + terminal bank to numerical precision.  This script edits no tracked
file and never invokes the official evaluator command or experiment ledger.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    CYCLE_DAYS,
    MONTH_DAYS,
    antecedent,
    format_metrics,
    load_observed,
    load_selected,
    logistic,
    metrics,
    select_cells,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


@dataclass(frozen=True)
class GrassState:
    """Causal open-fuel eligibility and standing-dead readiness."""

    pathway_share: np.ndarray
    readiness: np.ndarray
    production: np.ndarray
    live_stock: np.ndarray
    dead_stock: np.ndarray


def field(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(data[name][:, 0, :], dtype=np.float64)


def common_drivers(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build causal climate productivity and open-continuity drivers."""
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    primary = np.clip(field(data, "luh2_primary_fraction"), 0.0, 1.0)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field(data, "luh2_urban_fraction"), 0.0, 1.0)

    rain12 = antecedent(rain, 12.0)
    rain24 = antecedent(rain, 24.0)
    annualized_rain = 12.0 * rain12
    temperature24 = antecedent(temperature, 24.0)
    thermal_growth = logistic((temperature - 10.0) / 4.0)
    warm_climate = logistic((temperature24 - 15.0) / 4.0)
    combustion = dryness / (dryness + 500.0)

    # A finite root-zone bucket: rainfall fills it and temperature/dryness set
    # monthly atmospheric demand.  It is initialized from month zero only.
    capacity_mm = 180.0
    demand = 45.0 * (0.2 + 0.8 * thermal_growth) * (0.25 + 0.75 * combustion)
    water = np.minimum(capacity_mm, rain[0]).copy()
    water_fraction = np.empty_like(rain)
    drawdown = np.empty_like(rain)
    previous_fraction = water / capacity_mm
    for time in range(rain.shape[0]):
        if time > 0:
            water = np.clip(water + rain[time] - demand[time], 0.0, capacity_mm)
        current_fraction = water / capacity_mm
        water_fraction[time] = current_fraction
        drawdown[time] = np.maximum(previous_fraction - current_fraction, 0.0)
        previous_fraction = current_fraction

    # Fuel establishment is low in deserts, peaks in seasonal mesic systems,
    # and declines smoothly where wet climate favours woody closure.
    dry_limit = annualized_rain / (annualized_rain + 280.0)
    wet_shading = np.exp(-annualized_rain / 4200.0)
    fuel_climate = np.clip(dry_limit * wet_shading / 0.68, 0.0, 1.0)
    woody_closure = primary * logistic((annualized_rain - 1050.0) / 260.0)
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual_open = np.clip(1.0 - classified, 0.0, 1.0)
    open_cover = np.clip(
        pasture + rangeland + residual_open + primary * (1.0 - woody_closure),
        0.0,
        1.0,
    )
    open_cover *= 1.0 - crop
    pathway_share = np.clip(open_cover * warm_climate * fuel_climate, 0.0, 1.0)
    production = np.clip(
        0.32 * pathway_share * thermal_growth * water_fraction,
        0.0,
        None,
    )
    return {
        "rain": rain,
        "rain12": rain12,
        "rain24": rain24,
        "annualized_rain": annualized_rain,
        "temperature": temperature,
        "temperature24": temperature24,
        "dryness": dryness,
        "combustion": combustion,
        "water_fraction": water_fraction,
        "drawdown": drawdown,
        "pathway_share": pathway_share,
        "production": production,
    }


def drawdown_transfer(drivers: Mapping[str, np.ndarray]) -> GrassState:
    """Transfer a live grass pool to standing dead fuel during drawdown."""
    production = drivers["production"]
    water = drivers["water_fraction"]
    drawdown = drivers["drawdown"]
    combustion = drivers["combustion"]
    rain = drivers["rain"]
    live = production[0].copy()
    dead = np.zeros_like(live)
    live_history = np.empty_like(production)
    dead_history = np.empty_like(production)
    readiness = np.empty_like(production)
    for time in range(production.shape[0]):
        live = 0.96 * live + production[time]
        water_stress = (1.0 - water[time]) * (0.25 + 0.75 * combustion[time])
        senescence_rate = 1.0 - np.exp(
            -(1.5 * drawdown[time] + 0.22 * water_stress)
        )
        transferred = live * senescence_rate
        live -= transferred
        wet_decomposition = 0.03 + 0.20 * rain[time] / (rain[time] + 60.0)
        dead = dead * np.exp(-wet_decomposition) + transferred
        dry_gate = water_stress * combustion[time]
        readiness[time] = dead / (dead + 0.30) * dry_gate
        live_history[time] = live
        dead_history[time] = dead
    return GrassState(
        drivers["pathway_share"], readiness, production, live_history, dead_history
    )


def erlang_maturation(drivers: Mapping[str, np.ndarray]) -> GrassState:
    """Use a two-stage live cohort before drought-dependent senescence."""
    production = drivers["production"]
    water = drivers["water_fraction"]
    combustion = drivers["combustion"]
    rain = drivers["rain"]
    juvenile = production[0].copy()
    mature = np.zeros_like(juvenile)
    dead = np.zeros_like(juvenile)
    live_history = np.empty_like(production)
    dead_history = np.empty_like(production)
    readiness = np.empty_like(production)
    for time in range(production.shape[0]):
        juvenile += production[time]
        matured = 0.42 * juvenile
        juvenile -= matured
        mature = 0.94 * mature + matured
        water_stress = (1.0 - water[time]) * (0.25 + 0.75 * combustion[time])
        senescence = mature * (1.0 - np.exp(-0.38 * water_stress))
        mature -= senescence
        litter_decay = 0.025 + 0.18 * rain[time] / (rain[time] + 50.0)
        dead = dead * np.exp(-litter_decay) + senescence
        readiness[time] = (
            dead / (dead + 0.28)
            * water_stress
            * combustion[time]
        )
        live_history[time] = juvenile + mature
        dead_history[time] = dead
    return GrassState(
        drivers["pathway_share"], readiness, production, live_history, dead_history
    )


def dry_degree_curing(drivers: Mapping[str, np.ndarray]) -> GrassState:
    """Accumulate dry-degree exposure and cure live grass gradually."""
    production = drivers["production"]
    water = drivers["water_fraction"]
    combustion = drivers["combustion"]
    rain = drivers["rain"]
    temperature = drivers["temperature"]
    live = production[0].copy()
    dead = np.zeros_like(live)
    dose = np.zeros_like(live)
    live_history = np.empty_like(production)
    dead_history = np.empty_like(production)
    readiness = np.empty_like(production)
    for time in range(production.shape[0]):
        live = 0.97 * live + production[time]
        thermal_dryness = (
            logistic((temperature[time] - 12.0) / 4.0)
            * (1.0 - water[time])
            * combustion[time]
        )
        # Rain relaxes the curing clock; dry degree-months integrate it.
        dose = dose * np.exp(-rain[time] / 80.0) + thermal_dryness
        curing_rate = 1.0 - np.exp(-0.16 * dose)
        transferred = live * curing_rate
        live -= transferred
        dead = dead * np.exp(-0.025 - rain[time] / 550.0) + transferred
        readiness[time] = (
            dead / (dead + 0.34)
            * combustion[time]
            * (1.0 - water[time])
        )
        live_history[time] = live
        dead_history[time] = dead
    return GrassState(
        drivers["pathway_share"], readiness, production, live_history, dead_history
    )


STATE_BUILDERS: dict[str, Callable[[Mapping[str, np.ndarray]], GrassState]] = {
    "drawdown_transfer": drawdown_transfer,
    "erlang_maturation": erlang_maturation,
    "dry_degree_curing": dry_degree_curing,
}


def finite_release_bank(
    prediction: np.ndarray,
    state: GrassState,
    store_fraction: float = 0.32,
    release_gain: float = 10.0,
) -> tuple[np.ndarray, float, float]:
    """Redistribute open-surface hazard and return exact stock diagnostics."""
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    output = np.empty_like(hazard)
    bank = np.zeros_like(hazard[0])
    total_input = float(hazard.sum(dtype=np.float64))
    for time in range(hazard.shape[0]):
        readiness = np.clip(state.readiness[time], 0.0, 1.0)
        stored = (
            store_fraction
            * state.pathway_share[time]
            * (1.0 - readiness)
            * hazard[time]
        )
        bank += stored
        release_fraction = 1.0 - np.exp(-(1.0 / 36.0 + release_gain * readiness))
        released = release_fraction * bank
        bank -= released
        output[time] = hazard[time] - stored + released
    closure = abs(float(output.sum(dtype=np.float64) + bank.sum() - total_input)) / (
        total_input + 1e-30
    )
    terminal_fraction = float(bank.sum() / (total_input + 1e-30))
    candidate = 1.0 - np.exp(-np.clip(output, 0.0, 50.0))
    return candidate, closure, terminal_fraction


def carrier_mask(data: Mapping[str, np.ndarray], drivers: Mapping[str, np.ndarray]) -> np.ndarray:
    """Reproduce the post-prediction warm-seasonal-open diagnostic subset."""
    annual_rain = drivers["annualized_rain"].mean(axis=0)
    temperature = drivers["temperature24"].mean(axis=0)
    dryness = drivers["combustion"].mean(axis=0)

    def mean(name: str) -> np.ndarray:
        return field(data, name).mean(axis=0)

    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    pasture = mean("luh2_pasture_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    urban = mean("luh2_urban_fraction")
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    open_cover = np.clip(pasture + rangeland + residual, 0.0, 1.0)
    canopy = mean("natural_canopy_height")
    lai = mean("leaf_area_index")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    established = (
        (
            (temperature >= 20.0)
            & (annual_rain >= 1200.0)
            & (canopy >= 20.0)
            & (lai >= 3.0)
            & (natural >= 0.7)
            & (primary >= 0.5)
        )
        | (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (canopy >= 15.0)
            & (lai >= 2.5)
            & (natural >= 0.6)
        )
        | ((temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6))
        | (
            (temperature >= 20.0)
            & (annual_rain >= 500.0)
            & (annual_rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        )
        | (
            (rangeland >= 0.4)
            & (annual_rain >= 250.0)
            & (annual_rain < 1500.0)
            & (biomass >= 0.2)
        )
        | (crop >= 0.5)
        | ((annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0))
    )
    return (
        ~established
        & (temperature >= 18.0)
        & (annual_rain >= 400.0)
        & (annual_rain < 1600.0)
        & (dryness >= 0.18)
        & (open_cover >= 0.20)
    )


def print_delta(
    label: str,
    candidate: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    weight: np.ndarray,
    folds: np.ndarray,
    mask: np.ndarray,
) -> None:
    current, current_folds = metrics(
        candidate[:, mask], observed[:, mask], area[mask], weight[mask], folds[mask]
    )
    reference, reference_folds = metrics(
        baseline[:, mask], observed[:, mask], area[mask], weight[mask], folds[mask]
    )
    names = ("alloc_rmse", "annual_log_rmse", "raw_cycle_rmse", "phase", "area_ratio")
    print(f"{label} " + format_metrics(current), flush=True)
    print(
        f"{label}_DELTA "
        + " ".join(
            f"{name}={current[index] - reference[index]:+.8f}"
            for index, name in enumerate(names)
        ),
        flush=True,
    )
    for fold in range(4):
        print(
            f"{label}_FOLD fold={fold} "
            + " ".join(
                f"{name}={current_folds[fold][index] - reference_folds[fold][index]:+.8f}"
                for index, name in enumerate(names[:4])
            ),
            flush=True,
        )


def prefix_audit(
    builder: Callable[[Mapping[str, np.ndarray]], GrassState],
    drivers: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    split: int = 96,
) -> float:
    perturbed = {name: np.asarray(values).copy() for name, values in drivers.items()}
    rng = np.random.default_rng(240831)
    for name in (
        "rain",
        "rain12",
        "rain24",
        "annualized_rain",
        "temperature",
        "temperature24",
        "dryness",
        "combustion",
        "water_fraction",
        "drawdown",
        "pathway_share",
        "production",
    ):
        values = perturbed[name]
        scale = np.maximum(np.abs(values[split:]), 1.0)
        values[split:] += rng.normal(0.0, 3.0, size=values[split:].shape) * scale
        if name not in ("temperature", "temperature24"):
            np.maximum(values, 0.0, out=values)
    original_state = builder(drivers)
    perturbed_state = builder(perturbed)
    original, _, _ = finite_release_bank(prediction, original_state)
    changed_prediction = prediction.copy()
    changed_prediction[split:] = np.clip(1.0 - prediction[split:], 0.0, 1.0 - 1e-7)
    changed, _, _ = finite_release_bank(changed_prediction, perturbed_state)
    return float(np.max(np.abs(original[:split] - changed[:split])))


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator, count=1536)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()

    baseline = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
    drivers = common_drivers(data)
    states = {name: builder(drivers) for name, builder in STATE_BUILDERS.items()}
    carrier = carrier_mask(data, drivers)
    noncarrier = ~carrier
    print(
        f"CARRIER cells={int(carrier.sum())} sampled_observed_weight="
        f"{reference_weight[carrier].sum() / reference_weight.sum():.8f}",
        flush=True,
    )
    all_mask = np.ones(carrier.shape, dtype=bool)
    baseline_all, _ = metrics(baseline, observed, area, reference_weight, folds)
    print("BASE_ALL " + format_metrics(baseline_all), flush=True)
    print_delta(
        "BASE_CARRIER_IDENTITY", baseline, baseline, observed, area,
        reference_weight, folds, carrier,
    )

    # The no-surface run empirically measures what the incumbent natural
    # surface bank contributes.  Each candidate then replaces that stage,
    # instead of stacking a second bank on top of it.
    original_surface = model._surface_fire_opportunity_bank
    model._surface_fire_opportunity_bank = lambda prediction, data, p, enabled: prediction
    try:
        no_surface = np.asarray(
            model.predict(data, dict(model.PARAMS), None), dtype=np.float64
        )[:, 0, :]
    finally:
        model._surface_fire_opportunity_bank = original_surface
    print_delta(
        "ABLATE_EXISTING_SURFACE_ALL", no_surface, baseline, observed, area,
        reference_weight, folds, all_mask,
    )
    print_delta(
        "ABLATE_EXISTING_SURFACE_CARRIER", no_surface, baseline, observed, area,
        reference_weight, folds, carrier,
    )

    for name, state in states.items():
        bank_audit: dict[str, float] = {}

        def replacement(prediction, candidate_data, p, enabled, *, state=state):
            candidate, closure, terminal = finite_release_bank(
                np.asarray(prediction, dtype=np.float64)[:, 0, :], state
            )
            bank_audit["closure"] = closure
            bank_audit["terminal"] = terminal
            return np.asarray(candidate[:, None, :], dtype=np.float32)

        model._surface_fire_opportunity_bank = replacement
        try:
            candidate = np.asarray(
                model.predict(data, dict(model.PARAMS), None), dtype=np.float64
            )[:, 0, :]
        finally:
            model._surface_fire_opportunity_bank = original_surface
        print(
            f"STOCK {name} closure={bank_audit['closure']:.12e} "
            f"terminal_fraction={bank_audit['terminal']:.8f} "
            f"prefix_max={prefix_audit(STATE_BUILDERS[name], drivers, baseline):.12e}",
            flush=True,
        )
        print_delta(
            f"REPLACE_{name}_ALL", candidate, baseline, observed, area,
            reference_weight, folds, all_mask,
        )
        print_delta(
            f"REPLACE_{name}_CARRIER", candidate, baseline, observed, area,
            reference_weight, folds, carrier,
        )
        print_delta(
            f"REPLACE_{name}_NONCARRIER", candidate, baseline, observed, area,
            reference_weight, folds, noncarrier,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
