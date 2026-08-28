"""Held-cell test of ignition-combustibility arrival order.

This diagnostic tests whether an ignition pulse arriving after fuels have
become combustible carries more fire than combustibility arriving after the
ignition pulse has passed.  For every site it constructs the causal state

    X_t = [EMA12(I_t C_{t-1}) - EMA12(I_{t-1} C_t)]
          / [EMA12(I_t C_{t-1}) + EMA12(I_{t-1} C_t) + 0.02]

where ``I`` is bounded lightning ignition access and ``C`` is bounded dry,
rain-free, thermally viable combustion.  The current month enters only the
current state and the first lag is initialized locally, so no future month or
neighbour enters the equation.  A correction may only redistribute incumbent
hazard through existing surface or woody pathway shares; it cannot add an
ignition source.  Coordinates define held audit folds only and never enter a
candidate equation.
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
    cycle_loss,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    select_high_weight,
)
from scripts.runtime import load_land_mask  # noqa: E402


EXPECTED_COMMIT = "a8ed115"
EXPECTED_MODEL_BLOB = "731e1ee048fd1099dffe75d11a738fd9125f8064"
EPS = np.float32(1e-6)
STRENGTHS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)


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
    module = types.ModuleType("ed_fire_pinned_a8ed115_arrival")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def fields(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values[:, 0, :], dtype=np.float32)
        for name, values in data.items()
    }


def arrival_order_states(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return bounded causal arrival state and incumbent pathway shares."""
    field = fields(data)
    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    dryness = np.clip(field["dryness"], 0.0, None)
    temperature = field["air_temperature"]
    lightning = np.clip(field["lightning_flash_rate"], 0.0, None)
    gpp = np.clip(field["gpp"], 0.0, None)

    ignition = lightning / (lightning + 0.02)
    thermal = 1.0 / (1.0 + np.exp(np.clip(-(temperature - 5.0) / 3.0, -30.0, 30.0)))
    combustion = (
        dryness / (dryness + 250.0)
        * 1.0 / (1.0 + rain / 35.0)
        * thermal
    )
    previous_ignition = np.empty_like(ignition)
    previous_combustion = np.empty_like(combustion)
    previous_ignition[0] = ignition[0]
    previous_combustion[0] = combustion[0]
    previous_ignition[1:] = ignition[:-1]
    previous_combustion[1:] = combustion[:-1]
    ignition_after_combustion = ema(ignition * previous_combustion, 12.0)
    combustion_after_ignition = ema(previous_ignition * combustion, 12.0)
    arrival_order = np.clip(
        (ignition_after_combustion - combustion_after_ignition)
        / (ignition_after_combustion + combustion_after_ignition + 0.02),
        -1.0,
        1.0,
    )

    natural = np.clip(field["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(field["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(field["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(field["secondary_canopy_height"], 0.0, None)
    biomass = np.clip(field["aboveground_biomass"], 0.0, None)
    crop = np.clip(field["luh2_cropland_fraction"], 0.0, 1.0)
    pasture = np.clip(field["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(field["luh2_rangeland_fraction"], 0.0, 1.0)
    urban = np.clip(field["luh2_urban_fraction"], 0.0, 1.0)
    fine_fuel = ema(gpp, 12.0) / (ema(gpp, 12.0) + 0.35)
    open_cover = np.clip(
        natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0)
        + pasture
        + rangeland,
        0.0,
        2.0,
    )
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover * continuity
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
        + secondary
        * secondary_canopy
        / (secondary_canopy + 8.0)
        * biomass
        / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel
    total_capacity = 0.05 + surface_capacity + woody_capacity + crop_capacity
    surface_share = surface_capacity / total_capacity
    woody_share = woody_capacity / total_capacity
    return {
        "arrival_order": arrival_order,
        "surface_signal": arrival_order * surface_share,
        "woody_signal": arrival_order * woody_share,
        "natural_signal": arrival_order * (surface_share + woody_share),
        "open_cover": open_cover,
        "temperature": temperature,
        "annual_rain": 12.0 * ema(rain, 12.0),
    }


def redistribute_hazard(
    prediction: np.ndarray,
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Causally redistribute incumbent hazard with a bounded moving reference."""
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    raw_factor = np.exp(np.clip(float(strength) * signal, -0.5, 0.5))
    reference = ema(raw_factor, 12.0)
    factor = np.clip(raw_factor / np.maximum(reference, 1e-6), 0.5, 2.0)
    adjusted = hazard * factor
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
    )


def weighted_association(
    signal: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    weight_sum = max(float(np.sum(weight)), 1e-12)
    x_mean = float(np.sum(signal * weight) / weight_sum)
    y_mean = float(np.sum(residual * weight) / weight_sum)
    dx = signal - x_mean
    dy = residual - y_mean
    covariance = float(np.sum(weight * dx * dy) / weight_sum)
    variance = float(np.sum(weight * np.square(dx)) / weight_sum)
    slope = covariance / max(variance, 1e-12)
    correlation = covariance / max(
        np.sqrt(
            variance
            * float(np.sum(weight * np.square(dy)) / weight_sum)
        ),
        1e-12,
    )
    return slope, correlation


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
    print(
        f"DESIGN cells={rows.size} retained_fire_weight={retained:.9f} "
        "held_whole_cells=1 runtime_geography=0 future=0 neighbours=0",
        flush=True,
    )

    state = arrival_order_states(data)
    mean_temperature = state["temperature"].mean(axis=0)
    mean_open = state["open_cover"].mean(axis=0)
    mean_rain = state["annual_rain"].mean(axis=0)
    warm_open = (
        (mean_temperature >= 15.0)
        & (mean_open >= 0.4)
        & (mean_rain >= 250.0)
        & (mean_rain < 1800.0)
    )
    base_cycle = baseline.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation.reshape(16, 12, -1).mean(axis=0)
    arrival_cycle = state["arrival_order"].reshape(16, 12, -1).mean(axis=0)
    target = np.log((obs_cycle + EPS) / (base_cycle + EPS))
    floor = float(np.sum(obs_cycle * area[None, :]) / np.sum(area)) * 0.02
    weights = area[None, :] * (obs_cycle + floor)
    sign_positive = 0
    for fold in range(4):
        chosen = warm_open & (folds == fold)
        slope, correlation = weighted_association(
            arrival_cycle[:, chosen].reshape(-1),
            target[:, chosen].reshape(-1),
            weights[:, chosen].reshape(-1),
        )
        sign_positive += int(slope > 0.0)
        print(
            f"SIGN fold={fold} cells={int(chosen.sum())} slope={slope:+.9f} "
            f"correlation={correlation:+.9f}",
            flush=True,
        )
    print(
        f"SIGN_STABILITY positive={sign_positive}/4 negative={4-sign_positive}/4 "
        "physical_negative=holdover_ignition_precedes_peak_combustibility",
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
    viable: list[tuple[str, float, float]] = []
    for pathway in ("surface_signal", "woody_signal", "natural_signal"):
        for strength in STRENGTHS:
            candidate = redistribute_hazard(baseline, state[pathway], strength)
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
            warm_cycle_folds = sum(
                new < old
                for new, old in zip(cycle_warm[1], reference_cycle_warm[1])
            )
            all_cycle_folds = sum(
                new < old
                for new, old in zip(cycle_all[1], reference_cycle_all[1])
            )
            warm_annual_folds = sum(
                new < old
                for new, old in zip(annual_warm[1], reference_annual_warm[1])
            )
            all_annual_folds = sum(
                new < old
                for new, old in zip(annual_all[1], reference_annual_all[1])
            )
            print(
                f"VARIANT pathway={pathway} strength={strength:+g} "
                f"annual_all_delta={annual_all[0]-reference_annual_all[0]:+.9f} "
                f"cycle_all_delta={cycle_all[0]-reference_cycle_all[0]:+.9f} "
                f"annual_warm_delta={annual_warm[0]-reference_annual_warm[0]:+.9f} "
                f"cycle_warm_delta={cycle_warm[0]-reference_cycle_warm[0]:+.9f} "
                f"annual_all_folds={all_annual_folds}/4 "
                f"annual_warm_folds={warm_annual_folds}/4 "
                f"cycle_all_folds={all_cycle_folds}/4 "
                f"cycle_warm_folds={warm_cycle_folds}/4",
                flush=True,
            )
            stable_direction = (
                (sign_positive >= 3 and strength > 0.0)
                or ((4 - sign_positive) >= 3 and strength < 0.0)
            )
            if (
                stable_direction
                and warm_cycle_folds >= 3
                and all_cycle_folds >= 3
                and warm_annual_folds >= 3
                and all_annual_folds >= 3
            ):
                viable.append(
                    (
                        pathway,
                        strength,
                        cycle_all[0] - reference_cycle_all[0]
                        + cycle_warm[0] - reference_cycle_warm[0],
                    )
                )
            del candidate
    viable.sort(key=lambda item: item[2])

    prefix_data = {
        name: values[:, :, :96].copy() for name, values in data.items()
    }
    prefix_state = arrival_order_states(prefix_data)
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_state = arrival_order_states(perturbed)
    prefix_delta = float(
        np.max(
            np.abs(
                prefix_state["arrival_order"][:96]
                - perturbed_state["arrival_order"][:96]
            )
        )
    )
    print(f"PREFIX signal_max_abs={prefix_delta:.12g}", flush=True)
    if max(sign_positive, 4 - sign_positive) < 3:
        print("DECISION reject_reason=unstable_sign exact=0", flush=True)
    elif not viable:
        print("DECISION reject_reason=no_stable_positive_redistribution exact=0", flush=True)
    else:
        best = viable[0]
        print(
            f"DECISION exact=1 pathway={best[0]} strength={best[1]:g} "
            f"held_objective_delta={best[2]:+.9f}",
            flush=True,
        )
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
