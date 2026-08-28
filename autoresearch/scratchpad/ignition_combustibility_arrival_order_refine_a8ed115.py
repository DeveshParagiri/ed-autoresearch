"""Held-cell refinement of the ignition/combustibility arrival state.

The parent experiment found a small annual-map gain from a -0.25 holdover
redistribution of existing natural surface and woody hazard, but the exact
candidate damaged several tropical-American regions.  This one-step refinement
keeps that sign and strength fixed.  It only tests whether the already-used
causal persistent-desiccation or coherent-surface states attenuate the
redistribution where dry-season combustion is intermittent or incoherent.

All equations are global, pointwise, prefix causal, and target independent.
Coordinates define held audit folds only.  No candidate enters ``model.py`` or
the official evaluator from this screening script.
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

from autoresearch.scratchpad.ignition_combustibility_arrival_order_a8ed115 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    arrival_order_states,
    fields,
    pinned_model,
    redistribute_hazard,
)
from autoresearch.scratchpad.causal_same_month_normals_bf42d58 import (  # noqa: E402
    ema,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    annual_loss,
    cycle_loss,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    select_high_weight,
)
from scripts.runtime import load_land_mask  # noqa: E402


STRENGTH = -0.25


def trailing_std(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for time_index in range(values.shape[0]):
        start = max(0, time_index - 11)
        output[time_index] = values[start : time_index + 1].std(axis=0)
    return output


def attenuation_gates(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reproduce incumbent dryness persistence and surface coherence states."""
    field = fields(data)
    dryness = np.clip(field["dryness"], 0.0, None)
    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    temperature = field["air_temperature"]
    primary = np.clip(field["luh2_primary_fraction"], 0.0, 1.0)
    dryness_12 = ema(dryness, 12.0)
    persistent = dryness_12 / (dryness_12 + 500.0)

    dryness_second_moment = ema(np.square(dryness), 12.0)
    dryness_variability = np.sqrt(
        np.maximum(dryness_second_moment - np.square(dryness_12), 0.0)
    ) / (dryness_12 + 1.0)
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    combustion_12 = ema(combustion, 12.0)
    temperature_12 = ema(temperature, 12.0)
    temperature_spread = trailing_std(temperature)
    combustion_temperature = ema(combustion * temperature, 12.0)
    alignment = (
        combustion_temperature / (combustion_12 + 1e-6) - temperature_12
    ) / (temperature_spread + 1.0)
    thermal_overlap = 2.0 / (
        1.0 + np.exp(np.clip(-alignment, -30.0, 30.0))
    )
    reliable_dryness = np.clip(
        2.0 / (1.0 + dryness_variability), 0.5, 1.5
    )
    coherent = np.clip(
        reliable_dryness
        * thermal_overlap
        / np.sqrt(1.0 + primary * np.square(dryness_variability)),
        0.5,
        1.5,
    )
    return {
        "ungated": np.ones_like(persistent),
        "persistent": persistent,
        "coherent": coherent,
        "persistent_coherent": persistent * coherent,
    }


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
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    del observation_grid, area_grid, land
    gc.collect()

    state = arrival_order_states(data)
    gates = attenuation_gates(data)
    mean_temperature = state["temperature"].mean(axis=0)
    mean_open = state["open_cover"].mean(axis=0)
    mean_rain = state["annual_rain"].mean(axis=0)
    warm_open = (
        (mean_temperature >= 15.0)
        & (mean_open >= 0.4)
        & (mean_rain >= 250.0)
        & (mean_rain < 1800.0)
    )
    print(
        f"DESIGN cells={rows.size} warm_open={int(warm_open.sum())} "
        f"retained_fire_weight={retained:.9f} strength={STRENGTH:+g}",
        flush=True,
    )
    for name, gate in gates.items():
        print(
            f"GATE name={name} min={float(gate.min()):.9f} "
            f"mean={float(gate.mean()):.9f} max={float(gate.max()):.9f}",
            flush=True,
        )

    reference_annual_all = annual_loss(baseline, observation, area, folds)
    reference_cycle_all = cycle_loss(baseline, observation, area, folds)
    reference_annual_warm = annual_loss(
        baseline[:, warm_open], observation[:, warm_open], area[warm_open], folds[warm_open]
    )
    reference_cycle_warm = cycle_loss(
        baseline[:, warm_open], observation[:, warm_open], area[warm_open], folds[warm_open]
    )
    survivors: list[tuple[str, float]] = []
    for name, gate in gates.items():
        candidate = redistribute_hazard(
            baseline, state["natural_signal"] * gate, STRENGTH
        )
        annual_all = annual_loss(candidate, observation, area, folds)
        cycle_all = cycle_loss(candidate, observation, area, folds)
        annual_warm = annual_loss(
            candidate[:, warm_open],
            observation[:, warm_open],
            area[warm_open],
            folds[warm_open],
        )
        cycle_warm = cycle_loss(
            candidate[:, warm_open],
            observation[:, warm_open],
            area[warm_open],
            folds[warm_open],
        )
        fold_counts = (
            sum(new < old for new, old in zip(annual_all[1], reference_annual_all[1])),
            sum(new < old for new, old in zip(cycle_all[1], reference_cycle_all[1])),
            sum(new < old for new, old in zip(annual_warm[1], reference_annual_warm[1])),
            sum(new < old for new, old in zip(cycle_warm[1], reference_cycle_warm[1])),
        )
        objective = (
            annual_all[0]
            - reference_annual_all[0]
            + annual_warm[0]
            - reference_annual_warm[0]
            + 12.0
            * (
                cycle_all[0]
                - reference_cycle_all[0]
                + cycle_warm[0]
                - reference_cycle_warm[0]
            )
        )
        print(
            f"VARIANT gate={name} "
            f"annual_all_delta={annual_all[0]-reference_annual_all[0]:+.9f} "
            f"cycle_all_delta={cycle_all[0]-reference_cycle_all[0]:+.9f} "
            f"annual_warm_delta={annual_warm[0]-reference_annual_warm[0]:+.9f} "
            f"cycle_warm_delta={cycle_warm[0]-reference_cycle_warm[0]:+.9f} "
            f"annual_all_folds={fold_counts[0]}/4 cycle_all_folds={fold_counts[1]}/4 "
            f"annual_warm_folds={fold_counts[2]}/4 cycle_warm_folds={fold_counts[3]}/4 "
            f"objective_delta={objective:+.9f}",
            flush=True,
        )
        if name != "ungated" and min(fold_counts) >= 3:
            survivors.append((name, objective))
    survivors.sort(key=lambda item: item[1])

    prefix_data = {name: values[:, :, :96].copy() for name, values in data.items()}
    prefix_state = arrival_order_states(prefix_data)
    prefix_gates = attenuation_gates(prefix_data)
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_state = arrival_order_states(perturbed)
    perturbed_gates = attenuation_gates(perturbed)
    for name in gates:
        original = prefix_state["natural_signal"] * prefix_gates[name]
        changed = perturbed_state["natural_signal"] * perturbed_gates[name]
        print(
            f"PREFIX gate={name} max_abs={float(np.max(np.abs(original[:96]-changed[:96]))):.12g}",
            flush=True,
        )
    if survivors:
        print(
            f"DECISION exact=1 gate={survivors[0][0]} "
            f"held_objective_delta={survivors[0][1]:+.9f}",
            flush=True,
        )
    else:
        print("DECISION exact=0 reject=no_gate_stable_in_all_four_audits", flush=True)
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
