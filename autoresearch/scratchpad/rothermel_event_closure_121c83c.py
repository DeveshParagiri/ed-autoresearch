"""Held-block test of a pointwise Rothermel-inspired event-size closure.

This is a dimensionless structural proxy, not a claim to reproduce an
operational Rothermel fuel model.  It retains the current incumbent and lets a
bounded surface-hazard share respond to a separately factorized occurrence
rate and conditional event footprint.  The spread proxy follows the Rothermel
heat-source / heat-sink structure, while the footprint is finite and overlaps
as a Poisson process::

    I_R = w_n eta_M
    R = I_R / (rho_b Q_ig)
    A = 1 - exp(-a R^2 tau)
    lambda = L_12 / .02 * natural_open + managed / .15
    B_event = 1 - exp(-lambda A)

Here GPP and biomass proxy net fuel load and reaction intensity, LAI and canopy
height proxy fuel-bed bulk density, rain and dryness proxy fuel moisture and
moisture of extinction, and air temperature enters ignition energy.  A causal
12-month reference isolates event-size allocation from the incumbent mean
map.  GFED and coordinates are used only after prediction for held losses.

No canonical model, official result, progress artifact, or audit ledger is
written.  A full-grid candidate score is forbidden unless one fixed bracket
improves annual-log, normalized-allocation, and raw-cycle error in every held
whole-cell block.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.additive_pathway_replacement_a8ed115 import (  # noqa: E402
    ecological_ratios_selected,
)
from autoresearch.scratchpad.rain_fuel_pathway_probe import ecological_ratios  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "121c83c"
EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
EXPECTED_INCUMBENT = 0.719892388
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ("git", "show", f"{PINNED}:autoresearch/model.py"),
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
    module = types.ModuleType(f"model_{PINNED}_rothermel")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def rothermel_terms(
    data: dict[str, np.ndarray],
    moisture_of_extinction: float,
    area_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return event burn, its causal relative factor, surface share, and ROS."""
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None)
    dryness = np.clip(np.asarray(data["dryness"], dtype=np.float64), 0.0, None)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    biomass = np.clip(np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None)
    lai = np.clip(np.asarray(data["leaf_area_index"], dtype=np.float64), 0.0, None)
    lightning = np.clip(np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None)
    natural = np.clip(np.asarray(data["natural_vegetation_fraction"], dtype=np.float64), 0.0, 1.0)
    secondary = np.clip(np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64), 0.0, 1.0)
    natural_height = np.clip(np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None)
    secondary_height = np.clip(np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None)
    crop = np.clip(np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0)
    range_ = np.clip(np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0)
    pasture = np.clip(np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0)
    urban = np.clip(np.asarray(data["luh2_urban_fraction"], dtype=np.float64), 0.0, 1.0)

    gpp_12 = antecedent(gpp, 12.0)
    lightning_12 = antecedent(lightning, 12.0)
    fine_load = gpp_12 / (gpp_12 + 0.35)
    woody_load = biomass / (biomass + 1.0)
    net_fuel_load = 0.65 * fine_load + 0.35 * woody_load

    combustion = dryness / (dryness + 350.0) / (1.0 + rain / 35.0)
    fuel_moisture = np.clip(1.0 - combustion, 0.0, 1.0)
    moisture_ratio = np.clip(fuel_moisture / moisture_of_extinction, 0.0, 1.0)
    moisture_damping = np.clip(
        1.0
        - 2.59 * moisture_ratio
        + 5.11 * np.square(moisture_ratio)
        - 3.52 * np.power(moisture_ratio, 3.0),
        0.0,
        1.0,
    )
    reaction_intensity = net_fuel_load * moisture_damping

    cover_sum = natural + secondary + 1e-8
    bed_height = (
        natural * natural_height + secondary * secondary_height
    ) / cover_sum
    depth = bed_height / (bed_height + 8.0)
    packing = lai / (lai + 2.0)
    bulk_density = (0.10 + packing) / (0.25 + depth)

    # Rothermel's Q_ig=250+1116M is normalized by 250.  The first term adds
    # the sensible-heating distance from ambient air to a fixed 320 C ignition
    # temperature, normalized at 20 C, so warmer air weakly lowers the sink.
    ignition_energy = np.clip(
        (320.0 - temperature) / 300.0 + (1116.0 / 250.0) * fuel_moisture,
        0.25,
        8.0,
    )
    ros = reaction_intensity / (bulk_density * ignition_energy + 1e-12)

    # Moist fuel and rain terminate a spreading event within the monthly
    # forcing interval.  This is a finite residence time, not the earlier
    # accumulated consecutive-combustible-month state.
    extinction_duration = 1.0 / (
        1.0 + fuel_moisture / moisture_of_extinction + rain / 70.0
    )
    event_footprint = -np.expm1(
        -np.clip(area_scale * np.square(ros) * extinction_duration, 0.0, 50.0)
    )

    natural_open = np.clip(
        natural * 8.0 / (natural_height + 8.0)
        + secondary * 8.0 / (secondary_height + 8.0),
        0.0,
        1.0,
    )
    managed = np.clip(range_ + pasture + crop, 0.0, 1.0)
    event_rate = natural_open * lightning_12 / 0.02 + managed / 0.15
    event_burn = -np.expm1(-np.clip(event_rate * event_footprint, 0.0, 50.0))
    event_reference = antecedent(event_burn, 12.0)
    relative = np.clip((event_burn + 0.02) / (event_reference + 0.02), 0.5, 2.0)

    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    open_cover = np.clip(range_ + pasture + natural_open, 0.0, 1.0)
    surface_capacity = (1.0 - crop) * fine_load * open_cover * continuity
    woody_capacity = (
        natural * natural_height / (natural_height + 8.0)
        + secondary * secondary_height / (secondary_height + 8.0)
    ) * woody_load
    residue_capacity = crop * fine_load
    surface_share = surface_capacity / (
        0.05 + surface_capacity + woody_capacity + residue_capacity
    )
    return event_burn, relative, surface_share, ros


def candidate(
    incumbent: np.ndarray,
    relative: np.ndarray,
    surface_share: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    multiplier = np.clip(
        1.0 + strength * surface_share * (relative - 1.0), 0.5, 1.5
    )
    return np.asarray(-np.expm1(-np.clip(hazard * multiplier, 0.0, 50.0)), dtype=np.float32)


def held_losses(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    obs_annual: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    weight = area * obs_annual
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_allocation = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-12)
    pred_allocation = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-12)
    annual = []
    allocation = []
    raw_cycle = []
    for fold in range(4):
        held = folds == fold
        held_weight = weight[held]
        denominator = np.sum(held_weight) + 1e-15
        annual.append(
            np.sqrt(
                np.sum(
                    held_weight
                    * np.square(
                        np.log(obs_annual[held] + 1e-5)
                        - np.log(pred_annual[held] + 1e-5)
                    )
                )
                / denominator
            )
        )
        allocation.append(
            np.sqrt(
                np.sum(
                    held_weight[None, :]
                    * np.square(obs_allocation[:, held] - pred_allocation[:, held])
                )
                / (12.0 * denominator)
            )
        )
        raw_cycle.append(
            np.sqrt(
                np.sum(
                    held_weight[None, :]
                    * np.square(obs_cycle[:, held] - pred_cycle[:, held])
                )
                / (12.0 * denominator)
            )
        )
    return np.asarray(annual), np.asarray(allocation), np.asarray(raw_cycle)


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual_grid = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_annual_grid = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area_grid * obs_annual_grid
    excess_weight = area_grid * np.maximum(pred_annual_grid - obs_annual_grid, 0.0)

    def top(weight: np.ndarray) -> np.ndarray:
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, 0.90) + 1)]

    cells = np.union1d(top(observed_weight), top(excess_weight))
    rows, columns = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    selected_data = {
        name: np.asarray(values[:, rows, columns], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float64)
    selected_observed = np.asarray(observed[:, rows, columns], dtype=np.float64)
    selected_area = area_grid[rows, columns]
    selected_obs_annual = obs_annual_grid[rows, columns]
    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        selected_obs_annual,
        folds,
    )
    incumbent_score = evaluator.score(incumbent)["global"]
    if abs(incumbent_score["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(
            f"incumbent drift {incumbent_score['overall_score']:.9f}"
        )
    print(
        f"BASE pinned={PINNED} overall={incumbent_score['overall_score']:.9f} "
        f"cells={len(cells)} observed_coverage="
        f"{observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}",
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )

    survivors: list[tuple[float, float, float, float]] = []
    best: tuple[float, float, float, float, np.ndarray] | None = None
    for moisture_of_extinction in (0.20, 0.30, 0.40):
        for area_scale in (0.5, 1.0, 2.0):
            event_burn, relative, surface_share, ros = rothermel_terms(
                selected_data, moisture_of_extinction, area_scale
            )
            print(
                f"STATE mx={moisture_of_extinction:.2f} area_scale={area_scale:g} "
                f"event_mean={event_burn.mean():.9f} event_p95={np.quantile(event_burn, .95):.9f} "
                f"ros_mean={ros.mean():.9f} ros_p95={np.quantile(ros, .95):.9f}",
                flush=True,
            )
            for strength in (0.05, 0.10, 0.20):
                trial = candidate(
                    selected_incumbent, relative[:, 0, :], surface_share[:, 0, :], strength
                )
                losses = held_losses(
                    trial,
                    selected_observed,
                    selected_area,
                    selected_obs_annual,
                    folds,
                )
                gains = tuple(base_losses[index] - losses[index] for index in range(3))
                held = bool(all(np.all(gain > 0.0) for gain in gains))
                aggregate = float(sum(gain.sum() for gain in gains))
                print(
                    f"BRACKET mx={moisture_of_extinction:.2f} area_scale={area_scale:g} "
                    f"strength={strength:.2f} held={int(held)} annual_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[0])
                    + " allocation_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[1])
                    + " raw_cycle_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[2]),
                    flush=True,
                )
                if best is None or aggregate > best[0]:
                    best = (
                        aggregate,
                        moisture_of_extinction,
                        area_scale,
                        strength,
                        trial,
                    )
                if held:
                    survivors.append(
                        (aggregate, moisture_of_extinction, area_scale, strength)
                    )

    assert best is not None
    best_ecology = ecological_ratios_selected(
        best[4],
        selected_observed,
        selected_data,
        selected_area,
    )
    base_ecology = ecological_ratios_selected(
        selected_incumbent,
        selected_observed,
        selected_data,
        selected_area,
    )
    print(
        f"HELD_ECOLOGY best=mx{best[1]:.2f}:a{best[2]:g}:s{best[3]:.2f} "
        + ",".join(
            f"{name}:{base_ecology[name]:.5f}->{best_ecology[name]:.5f}"
            for name in base_ecology
        ),
        flush=True,
    )

    probe = np.linspace(0, len(cells) - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    _, relative_before, share_before, _ = rothermel_terms(
        prefix_data, best[1], best[2]
    )
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    before = candidate(prefix_incumbent, relative_before, share_before, best[3])
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    _, relative_after, share_after, _ = rothermel_terms(changed, best[1], best[2])
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)
    after = candidate(changed_incumbent, relative_after, share_after, best[3])
    print(
        f"PREFIX best=mx{best[1]:.2f}:a{best[2]:g}:s{best[3]:.2f} "
        f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}",
        flush=True,
    )

    if not survivors:
        print("DECISION exact=0 reject=no_all_block_all_metric_survivor", flush=True)
        return 0

    survivors.sort(reverse=True)
    _, moisture_of_extinction, area_scale, strength = survivors[0]
    _, relative, surface_share, _ = rothermel_terms(
        data, moisture_of_extinction, area_scale
    )
    trial = validate_prediction(candidate(incumbent, relative, surface_share, strength))
    trial_scores = evaluator.score(trial)
    global_score = trial_scores["global"]
    print(
        f"DECISION exact=1 mx={moisture_of_extinction:.2f} area_scale={area_scale:g} "
        f"strength={strength:.2f}",
        flush=True,
    )
    print(
        f"EXACT overall={global_score['overall_score']:.9f} "
        f"delta={global_score['overall_score']-incumbent_score['overall_score']:+.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f}",
        flush=True,
    )
    land = load_land_mask()
    base_ecology = ecological_ratios(incumbent, data, observed, area_grid, land)
    trial_ecology = ecological_ratios(trial, data, observed, area_grid, land)
    print(
        "ECOLOGY "
        + ",".join(
            f"{name}:{base_ecology[name]:.5f}->{trial_ecology[name]:.5f}"
            for name in base_ecology
        ),
        flush=True,
    )
    print(
        "REGIONS "
        + ",".join(
            f"{name}:{trial_scores[name]['overall_score']-evaluator.score(incumbent)[name]['overall_score']:+.6f}"
            for name in sorted(key for key in trial_scores if key != "global")
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
