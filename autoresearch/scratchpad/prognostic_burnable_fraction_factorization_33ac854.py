"""Held-cell screen of a prognostic burnable-fraction factorization.

The current model mixes annual fire propensity and monthly timing throughout
its operator stack.  This experiment adds one pointwise prognostic state after
that stack and explicitly factorizes incumbent hazard into a causal annual
propensity and a monthly allocation.  The annual term responds to the fraction
of locally supported fuel remaining after recovery and realized fire.  The
monthly term can additionally respond to combustion or ignition-combustion
readiness relative to its own trailing causal state.

Four fixed first-principles formulations are compared on held whole cells.
They use the existing 6-, 24-, and 120-month surface, residue, and woody
recovery times.  No target, coordinate, region, neighbour, future value, or
learned coefficient enters a candidate equation.  This is scratch triage only.
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
EPS = np.float32(1e-6)
FORMULATIONS = {
    # Annual fuel-stock control only.
    "stock_only": (0.5, 0.0, "combustion"),
    # The same annual state plus a weak physical combustion allocator.
    "stock_combustion": (0.5, 0.25, "combustion"),
    # Combustion must coincide with the union of natural and managed ignition.
    "stock_ignition_combustion": (0.5, 0.25, "ignition_combustion"),
    # A stronger structural bracket, not a fitted parameter search.
    "strong_ignition_combustion": (1.0, 0.5, "ignition_combustion"),
}


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
    module = types.ModuleType("ed_fire_pinned_33ac854_burnable_fraction")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def ema(model, values: np.ndarray, months: float) -> np.ndarray:
    return np.asarray(
        model._antecedent(
            np.asarray(values, dtype=np.float64),
            1.0 - np.exp(-1.0 / float(months)),
        ),
        dtype=np.float64,
    )


def prognostic_drivers(model, data: dict[str, np.ndarray]):
    field = {
        name: np.asarray(values[:, 0, :], dtype=np.float64)
        for name, values in data.items()
    }
    gpp = np.clip(field["gpp"], 0.0, None)
    fine_fuel = ema(model, gpp, 12.0) / (ema(model, gpp, 12.0) + 0.35)
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
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface = (
        np.clip(natural_open + secondary_open + pasture + rangeland, 0.0, 2.0)
        * fine_fuel
        * continuity
    )
    residue = crop * fine_fuel
    woody = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    total = 0.05 + surface + residue + woody
    surface_share = surface / total
    residue_share = residue / total
    woody_share = woody / total
    supported_fraction = np.clip((surface + residue + woody) / total, 0.0, 1.0)
    recovery = (
        surface_share * (1.0 - np.exp(-1.0 / 6.0))
        + residue_share * (1.0 - np.exp(-1.0 / 24.0))
        + woody_share * (1.0 - np.exp(-1.0 / 120.0))
        + (0.05 / total) * (1.0 - np.exp(-1.0 / 12.0))
    )

    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    dryness = np.clip(field["dryness"], 0.0, None)
    temperature = field["air_temperature"]
    combustion = (
        dryness / (dryness + 500.0)
        * 1.0 / (1.0 + rain / 35.0)
        * 1.0
        / (1.0 + np.exp(np.clip(-(temperature - 5.0) / 3.0, -30.0, 30.0)))
    )
    lightning = np.clip(field["lightning_flash_rate"], 0.0, None)
    natural_ignition = lightning / (lightning + 0.02)
    managed = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    managed_ignition = managed / (managed + 0.1)
    ignition = 1.0 - (1.0 - natural_ignition) * (1.0 - managed_ignition)
    return supported_fraction, recovery, combustion, ignition


def factorized_prediction(
    model,
    prediction: np.ndarray,
    data: dict[str, np.ndarray],
    formulation: tuple[float, float, str],
) -> np.ndarray:
    """Evolve one burnable fraction and factor annual versus monthly hazard."""
    annual_exponent, allocation_exponent, readiness_name = formulation
    capacity, recovery, combustion, ignition = prognostic_drivers(model, data)
    base_hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    state = np.asarray(capacity[0], dtype=np.float64).copy()
    annual_state = np.asarray(base_hazard[0], dtype=np.float64).copy()
    if readiness_name == "combustion":
        readiness = combustion
    elif readiness_name == "ignition_combustion":
        readiness = ignition * combustion
    else:
        raise ValueError(readiness_name)
    readiness_state = np.asarray(readiness[0], dtype=np.float64).copy()
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    output = np.empty_like(base_hazard, dtype=np.float64)
    for time_index in range(base_hazard.shape[0]):
        state += recovery[time_index] * (capacity[time_index] - state)
        np.clip(state, 0.0, 1.0, out=state)
        annual_state += alpha_12 * (base_hazard[time_index] - annual_state)
        readiness_state += alpha_12 * (
            readiness[time_index] - readiness_state
        )

        stock_ratio = np.clip(
            (state + 0.05) / (capacity[time_index] + 0.05), 0.5, 2.0
        )
        annual_factor = np.power(stock_ratio, annual_exponent)
        readiness_ratio = np.clip(
            (readiness[time_index] + 0.02) / (readiness_state + 0.02),
            0.5,
            2.0,
        )
        allocation_factor = np.power(readiness_ratio, allocation_exponent)

        annual_propensity = annual_state * annual_factor
        incumbent_allocation = base_hazard[time_index] / (
            annual_state + 1e-12
        )
        monthly_allocation = incumbent_allocation * allocation_factor
        output[time_index] = annual_propensity * monthly_allocation

        realized = 1.0 - np.exp(-np.clip(output[time_index], 0.0, 50.0))
        state -= realized
        np.clip(state, 0.0, 1.0, out=state)
    return np.asarray(
        1.0 - np.exp(-np.clip(output, 0.0, 50.0)), dtype=np.float32
    )


def main() -> int:
    started = time.perf_counter()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"moving canonical model {current_blob}")
    model = pinned_model()
    observation_grid = load_observation()
    area_grid = one_degree_area()
    rows, columns, _, retained = select_high_weight(observation_grid, area_grid)
    land = load_land_mask()
    keep = land[rows, columns]
    rows, columns = rows[keep], columns[keep]
    data = {name: selected_input(name, rows, columns) for name in model.INPUTS}
    incumbent = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    del observation_grid, area_grid, land
    gc.collect()
    print(
        f"DESIGN cells={rows.size} retained_fire_weight={retained:.9f} "
        "held_whole_cells=1 runtime_geography=0 future=0 neighbours=0",
        flush=True,
    )

    reference_annual = annual_loss(incumbent, observation, area, folds)
    reference_cycle = cycle_loss(incumbent, observation, area, folds)
    mean = {
        name: np.asarray(values[:, 0, :], dtype=np.float32).mean(axis=0)
        for name, values in data.items()
    }
    ecology = ecological_masks(mean)
    survivors: list[tuple[str, float]] = []
    variants: dict[str, np.ndarray] = {"incumbent": incumbent}
    for name, formulation in FORMULATIONS.items():
        candidate = factorized_prediction(model, incumbent, data, formulation)
        variants[name] = candidate
        annual = annual_loss(candidate, observation, area, folds)
        cycle = cycle_loss(candidate, observation, area, folds)
        annual_folds = sum(
            new < old for new, old in zip(annual[1], reference_annual[1])
        )
        cycle_folds = sum(
            new < old for new, old in zip(cycle[1], reference_cycle[1])
        )
        objective = (
            annual[0] - reference_annual[0]
            + 12.0 * (cycle[0] - reference_cycle[0])
        )
        print(
            f"VARIANT name={name} annual_delta={annual[0]-reference_annual[0]:+.9f} "
            f"cycle_delta={cycle[0]-reference_cycle[0]:+.9f} "
            f"annual_folds={annual_folds}/4 cycle_folds={cycle_folds}/4 "
            f"objective_delta={objective:+.9f}",
            flush=True,
        )
        if annual[0] < reference_annual[0] and cycle[0] < reference_cycle[0] and annual_folds >= 3 and cycle_folds >= 3:
            survivors.append((name, objective))
    survivors.sort(key=lambda item: item[1])

    audit_names = ["incumbent"] + [name for name, _ in survivors[:2]]
    if len(audit_names) == 1:
        audit_names += sorted(
            FORMULATIONS,
            key=lambda name: annual_loss(variants[name], observation, area, folds)[0]
            + 12.0 * cycle_loss(variants[name], observation, area, folds)[0],
        )[:1]
    for name in audit_names:
        for regime, mask in ecology.items():
            print(
                f"ECOLOGY name={name} regime={regime} cells={int(mask.sum())} "
                f"ratio={area_ratio(variants[name], observation, area, mask):.9f}",
                flush=True,
            )

    prefix_data = {name: values[:, :, :96].copy() for name, values in data.items()}
    prefix_incumbent = np.asarray(
        model.predict(prefix_data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_incumbent = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    for name, formulation in FORMULATIONS.items():
        original = factorized_prediction(
            model, prefix_incumbent, prefix_data, formulation
        )
        changed = factorized_prediction(
            model, perturbed_incumbent, perturbed, formulation
        )
        print(
            f"PREFIX name={name} max_abs={float(np.max(np.abs(original[:96]-changed[:96]))):.12g}",
            flush=True,
        )
    if survivors:
        print(
            f"DECISION exact=1 name={survivors[0][0]} "
            f"held_objective_delta={survivors[0][1]:+.9f}",
            flush=True,
        )
    else:
        print("DECISION exact=0 reject=no_joint_held_survivor", flush=True)
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
