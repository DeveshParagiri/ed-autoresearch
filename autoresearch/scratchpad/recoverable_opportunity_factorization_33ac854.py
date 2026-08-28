"""Held screen for recoverable annual opportunity with finite allocation.

Entry 191 showed that remaining post-fire stock improves monthly timing but is
the wrong annual-capacity brake.  Here annual propensity depends only on causal
fuel renewal and the trailing occupancy of physically coincident ignition and
combustion.  The prognostic stock is restricted to allocating that annual
source through a finite hazard budget; it never scales annual propensity.

Four fixed forms bracket absolute versus locally relative opportunity and the
presence of a rain-supported renewal gate.  All are globally shared,
pointwise, prefix causal, and use only the current coupled-valid inputs.  The
target and coordinates are used only after prediction for held-cell scoring.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from autoresearch.scratchpad.prognostic_burnable_fraction_factorization_33ac854 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    ema,
    pinned_model,
    prognostic_drivers,
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


FORMULATIONS = (
    "fuel_absolute",
    "rain_supported_absolute",
    "fuel_relative",
    "rain_supported_relative",
)


def recoverable_opportunity(model, data: dict[str, np.ndarray]):
    field = {
        name: np.asarray(values[:, 0, :], dtype=np.float64)
        for name, values in data.items()
    }
    gpp = np.clip(field["gpp"], 0.0, None)
    gpp_12 = ema(model, gpp, 12.0)
    fine_fuel_renewal = gpp_12 / (gpp_12 + 0.35)
    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    rain_12 = ema(model, rain, 12.0)
    annual_rain = 12.0 * rain_12
    rain_supported_renewal = (
        fine_fuel_renewal
        * annual_rain
        / (annual_rain + 500.0)
        * np.exp(-annual_rain / 3000.0)
        / (1.0 + np.exp(np.clip(-(annual_rain - 250.0) / 100.0, -30.0, 30.0)))
    )
    capacity, recovery, combustion, ignition = prognostic_drivers(model, data)
    available = ignition * combustion
    fuel_opportunity = ema(model, fine_fuel_renewal * available, 12.0)
    rain_opportunity = ema(model, rain_supported_renewal * available, 12.0)
    return {
        "fuel_absolute": fuel_opportunity,
        "rain_supported_absolute": rain_opportunity,
        "fuel_relative": fuel_opportunity,
        "rain_supported_relative": rain_opportunity,
    }, capacity, recovery, available


def annual_factor(model, opportunity: np.ndarray, formulation: str) -> np.ndarray:
    if formulation.endswith("_absolute"):
        # The 0.05 half-saturation is the incumbent unresolved background share.
        return np.clip(2.0 * opportunity / (opportunity + 0.05), 0.5, 1.5)
    if formulation.endswith("_relative"):
        slow_reference = ema(model, opportunity, 36.0)
        return np.clip(
            np.sqrt((opportunity + 0.02) / (slow_reference + 0.02)),
            0.75,
            1.25,
        )
    raise ValueError(formulation)


def allocate_finite_opportunity(
    model,
    prediction: np.ndarray,
    data: dict[str, np.ndarray],
    formulation: str,
) -> np.ndarray:
    """Separate annual source from a stock-controlled finite monthly budget."""
    opportunities, capacity, recovery, available = recoverable_opportunity(
        model, data
    )
    factor = annual_factor(model, opportunities[formulation], formulation)
    base_hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    annual_state = np.asarray(base_hazard[0], dtype=np.float64).copy()
    state = np.asarray(capacity[0], dtype=np.float64).copy()
    readiness_state = np.asarray(state * available[0], dtype=np.float64)
    budget = np.zeros_like(state)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    output = np.empty_like(base_hazard, dtype=np.float64)
    for time_index in range(base_hazard.shape[0]):
        state += recovery[time_index] * (capacity[time_index] - state)
        np.clip(state, 0.0, 1.0, out=state)
        readiness = state * available[time_index]
        readiness_state += alpha_12 * (readiness - readiness_state)
        annual_state += alpha_12 * (base_hazard[time_index] - annual_state)

        annual_propensity = annual_state * factor[time_index]
        incumbent_allocation = base_hazard[time_index] / (
            annual_state + 1e-12
        )
        monthly_source = annual_propensity * incumbent_allocation
        budget += monthly_source
        relative_readiness = np.clip(
            (readiness + 0.02) / (readiness_state + 0.02), 0.5, 2.0
        )
        desired = monthly_source * np.power(relative_readiness, 0.25)
        released = np.minimum(desired, budget)
        budget -= released
        output[time_index] = released

        realized = 1.0 - np.exp(-np.clip(released, 0.0, 50.0))
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
        "held_whole_cells=1 finite_budget=1 runtime_geography=0",
        flush=True,
    )

    opportunities, _, _, _ = recoverable_opportunity(model, data)
    for name in FORMULATIONS:
        factor = annual_factor(model, opportunities[name], name)
        print(
            f"FACTOR name={name} p05={np.quantile(factor,0.05):.6f} "
            f"median={np.median(factor):.6f} p95={np.quantile(factor,0.95):.6f}",
            flush=True,
        )
    reference_annual = annual_loss(incumbent, observation, area, folds)
    reference_cycle = cycle_loss(incumbent, observation, area, folds)
    variants: dict[str, np.ndarray] = {"incumbent": incumbent}
    survivors: list[tuple[str, float]] = []
    for name in FORMULATIONS:
        candidate = allocate_finite_opportunity(model, incumbent, data, name)
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

    mean = {
        name: np.asarray(values[:, 0, :], dtype=np.float32).mean(axis=0)
        for name, values in data.items()
    }
    ecology = ecological_masks(mean)
    ranked = [name for name, _ in survivors[:2]]
    if not ranked:
        ranked = sorted(
            FORMULATIONS,
            key=lambda name: annual_loss(variants[name], observation, area, folds)[0]
            + 12.0 * cycle_loss(variants[name], observation, area, folds)[0],
        )[:1]
    for name in ("incumbent", *ranked):
        for regime, mask in ecology.items():
            print(
                f"ECOLOGY name={name} regime={regime} cells={int(mask.sum())} "
                f"ratio={area_ratio(variants[name],observation,area,mask):.9f}",
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
    for name in FORMULATIONS:
        original = allocate_finite_opportunity(
            model, prefix_incumbent, prefix_data, name
        )
        changed = allocate_finite_opportunity(
            model, perturbed_incumbent, perturbed, name
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
