"""Held-cell falsification of a cross-stratum propagation gate.

The incumbent partitions surface and woody capacity but never represents the
physical transition between them.  This experiment treats their smooth local
overlap as a vertically connected fuel share.  That share carries less hazard
outside a compound dry, warm combustion window and more hazard when the window
is active, allowing one pointwise state to change both annual allocation and
seasonal timing without storing fuel, moving hazard between cells, or fitting a
response surface.

Coordinates define four held audit folds only.  They do not enter the proposed
equation.  The script is diagnostic and never edits ``model.py`` or invokes the
official evaluator.
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

from autoresearch.scratchpad.causal_same_month_normals_bf42d58 import (  # noqa: E402
    ema,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    annual_loss,
    area_ratio,
    cycle_loss,
    ecological_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    select_high_weight,
)
from scripts.runtime import load_land_mask  # noqa: E402


EXPECTED_COMMIT = "33ac854"
EXPECTED_MODEL_BLOB = "3f63c96b9317d852e7b2973980ce77cc1bfc1b1f"
STRENGTHS = (0.125, 0.25, 0.5, 0.75, 1.0)
CANOPY_MOISTURE_MONTHS = (0.0, 2.0, 3.0, 6.0)


def pinned_model():
    source = subprocess.run(
        ("git", "show", f"{EXPECTED_COMMIT}:autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    blob = subprocess.run(
        ("git", "hash-object", "--stdin"),
        cwd=ROOT,
        input=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"unexpected pinned model blob {blob}")
    module = types.ModuleType("ed_fire_pinned_33ac854_stratum")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def fields(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values[:, 0, :], dtype=np.float32)
        for name, values in data.items()
    }


def rising(values: np.ndarray, slope: float, center: float) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(np.clip(-slope * (values - center), -30.0, 30.0))
    )


def cross_stratum_state(
    data: dict[str, np.ndarray],
    canopy_moisture_months: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return vertical fuel overlap and compound propagation readiness."""
    field = fields(data)
    gpp = np.clip(field["gpp"], 0.0, None)
    fine_fuel = ema(gpp, 12.0) / (ema(gpp, 12.0) + 0.35)
    natural = np.clip(field["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(field["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(field["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(field["secondary_canopy_height"], 0.0, None)
    biomass = np.clip(field["aboveground_biomass"], 0.0, None)
    crop = np.clip(field["luh2_cropland_fraction"], 0.0, 1.0)
    pasture = np.clip(field["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(field["luh2_rangeland_fraction"], 0.0, 1.0)
    urban = np.clip(field["luh2_urban_fraction"], 0.0, 1.0)

    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    open_cover = np.clip(
        natural_open + secondary_open + pasture + rangeland, 0.0, 2.0
    )
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface_capacity = (
        (1.0 - crop) * fine_fuel * open_cover * continuity
    )
    woody_capacity = biomass / (biomass + 1.0) * (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    )
    # Twice the harmonic mean is a bounded amount of fuel that exists in both
    # strata; unlike a ratio alone, trace fuel in either layer cannot qualify.
    vertical_overlap = np.clip(
        2.0
        * surface_capacity
        * woody_capacity
        / (surface_capacity + woody_capacity + 0.05),
        0.0,
        1.0,
    )

    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    rain_12 = ema(rain, 12.0)
    drought_maturation = np.maximum(
        (rain_12 - rain) / (rain_12 + rain + 10.0), 0.0
    )
    dryness = np.clip(field["dryness"], 0.0, None)
    temperature = field["air_temperature"]
    combustion = (
        dryness / (dryness + 250.0)
        * 1.0 / (1.0 + rain / 35.0)
        * rising(temperature, 1.0 / 3.0, 5.0)
    )
    # Canopy and coarse ladder fuels equilibrate more slowly than exposed fine
    # surface fuel.  Zero retains the instantaneous falsification; positive
    # values test a globally shared canopy-moisture response time.
    if canopy_moisture_months > 0.0:
        combustion = ema(combustion, canopy_moisture_months)
    # Crown transition needs both a matured rain deficit and active surface
    # combustion.  A geometric mean prevents one saturated limb from hiding a
    # weak one, while the smooth response avoids a categorical event trigger.
    compound_window = np.sqrt(
        np.clip(drought_maturation * combustion, 0.0, 1.0)
    )
    transition = rising(compound_window, 12.0, 0.28)
    return {
        "vertical_overlap": vertical_overlap,
        "compound_window": compound_window,
        "transition": transition,
        "signal": vertical_overlap * (0.5 + 1.5 * transition),
    }


def apply_gate(
    prediction: np.ndarray,
    state: dict[str, np.ndarray],
    strength: float,
) -> np.ndarray:
    """Blend the vertically connected share toward transition-controlled spread."""
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    overlap = state["vertical_overlap"]
    transition_multiplier = 0.5 + 1.5 * state["transition"]
    factor = 1.0 + float(strength) * overlap * (transition_multiplier - 1.0)
    adjusted = hazard * np.clip(factor, 0.25, 2.5)
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
    )


def weighted_quintiles(
    state: np.ndarray,
    baseline: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
) -> None:
    annual_state = state.mean(axis=0)
    order = np.argsort(annual_state)
    weight = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0) * area
    cumulative = np.cumsum(weight[order])
    total = float(cumulative[-1])
    edges = np.searchsorted(cumulative, total * np.arange(1, 5) / 5.0)
    groups = np.split(order, edges)
    for index, group in enumerate(groups):
        print(
            f"QUINTILE index={index} cells={group.size} "
            f"state={float(np.average(annual_state[group], weights=weight[group])):.9f} "
            f"area_ratio={area_ratio(baseline, observation, area, np.isin(np.arange(area.size), group)):.9f}",
            flush=True,
        )


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    observation_grid = load_observation()
    area_grid = one_degree_area()
    rows, columns, _, retained = select_high_weight(observation_grid, area_grid)
    land = load_land_mask()
    keep = land[rows, columns]
    rows, columns = rows[keep], columns[keep]
    data = {name: selected_input(name, rows, columns) for name in model.INPUTS}
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    del observation_grid, area_grid, land
    gc.collect()

    states = {
        months: cross_stratum_state(data, months)
        for months in CANOPY_MOISTURE_MONTHS
    }
    state = states[0.0]
    print(
        f"DESIGN cells={rows.size} retained_fire_weight={retained:.9f} "
        f"overlap_mean={float(state['vertical_overlap'].mean()):.9f} "
        f"overlap_p95={float(np.quantile(state['vertical_overlap'], 0.95)):.9f}",
        flush=True,
    )
    weighted_quintiles(
        state["signal"], baseline, observation, area
    )
    reference_annual = annual_loss(baseline, observation, area, folds)
    reference_cycle = cycle_loss(baseline, observation, area, folds)
    mean_fields = {
        name: values[:, 0, :].mean(axis=0) for name, values in data.items()
    }
    regimes = ecological_masks(mean_fields)
    reference_ecology = {
        name: area_ratio(baseline, observation, area, mask)
        for name, mask in regimes.items()
    }

    survivors: list[tuple[float, float, float]] = []
    for canopy_months, variant_state in states.items():
        print(
            f"STATE canopy_months={canopy_months:g} "
            f"transition_mean={float(variant_state['transition'].mean()):.9f}",
            flush=True,
        )
        for strength in STRENGTHS:
            candidate = apply_gate(baseline, variant_state, strength)
            candidate_annual = annual_loss(candidate, observation, area, folds)
            candidate_cycle = cycle_loss(candidate, observation, area, folds)
            annual_folds = sum(
                new < old
                for new, old in zip(candidate_annual[1], reference_annual[1])
            )
            cycle_folds = sum(
                new < old
                for new, old in zip(candidate_cycle[1], reference_cycle[1])
            )
            ecology_delta = {
                name: area_ratio(candidate, observation, area, mask) - reference_ecology[name]
                for name, mask in regimes.items()
            }
            objective = (
                candidate_annual[0]
                - reference_annual[0]
                + 12.0 * (candidate_cycle[0] - reference_cycle[0])
            )
            print(
                f"VARIANT canopy_months={canopy_months:g} strength={strength:.3f} "
                f"annual_delta={candidate_annual[0]-reference_annual[0]:+.9f} "
                f"annual_folds={annual_folds}/4 "
                f"cycle_delta={candidate_cycle[0]-reference_cycle[0]:+.9f} "
                f"cycle_folds={cycle_folds}/4 objective_delta={objective:+.9f} "
                f"eco_max_abs={max(abs(value) for value in ecology_delta.values()):.9f}",
                flush=True,
            )
            if (
                candidate_annual[0] < reference_annual[0]
                and candidate_cycle[0] < reference_cycle[0]
                and annual_folds >= 3
                and cycle_folds >= 3
            ):
                survivors.append((objective, strength, canopy_months))
    survivors.sort()

    prefix_data = {name: values.copy() for name, values in data.items()}
    prefix_states = {
        months: cross_stratum_state(prefix_data, months)
        for months in CANOPY_MOISTURE_MONTHS
    }
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_states = {
        months: cross_stratum_state(perturbed, months)
        for months in CANOPY_MOISTURE_MONTHS
    }
    for months in CANOPY_MOISTURE_MONTHS:
        print(
            f"PREFIX canopy_months={months:g} state_max_abs="
            f"{float(np.max(np.abs(prefix_states[months]['signal'][:96]-perturbed_states[months]['signal'][:96]))):.12g}",
            flush=True,
        )
    if survivors:
        print(
            f"DECISION exact=1 strength={survivors[0][1]:.3f} "
            f"canopy_months={survivors[0][2]:g} "
            f"objective_delta={survivors[0][0]:+.9f}",
            flush=True,
        )
    else:
        print(
            "DECISION exact=0 reject=no_strength_improves_annual_and_cycle_in_three_folds",
            flush=True,
        )
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
