"""Falsify an annual footprint role for open secondary regrowth.

The phase-space audit finds stable annual underburn in secondary-open fuel,
while the active secondary litter banks only redistribute incumbent hazard in
time and the local event footprint omits secondary vegetation. This script
tests a slow, pointwise capacity multiplier with fixed strengths. A full-grid
exact score is computed only if a formulation improves annual loss in all four
held spatial blocks and its aggregate cycle cost is below five percent of the
annual gain. This allows a bounded annual-map trade rather than demanding an
unrelated cycle Pareto improvement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from residual_phase_space_33ac854 import antecedent, load_pinned  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def rising(values: np.ndarray, scale: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def secondary_regrowth_states(data) -> dict[str, np.ndarray]:
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None)
    annual_rain = 12.0 * antecedent(rain, 12.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_12 = antecedent(temperature, 12.0)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = antecedent(gpp, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)

    secondary = np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0)
    secondary_canopy = np.clip(data["secondary_canopy_height"], 0.0, None)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    secondary_capacity = secondary_open * fine_fuel * continuity
    secondary_structure = secondary_open * continuity

    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    biomass = np.clip(data["aboveground_biomass"], 0.0, None)
    surface_capacity = (
        (1.0 - crop)
        * fine_fuel
        * np.clip(rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0)
        * continuity
    )
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel
    share = secondary_capacity / (
        0.05 + secondary_capacity + surface_capacity + woody_capacity + crop_capacity
    )
    surface_structure = (
        (1.0 - crop)
        * np.clip(rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0)
        * continuity
    )
    woody_structure = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    structural_share = secondary_structure / (
        0.05 + secondary_structure + surface_structure + woody_structure + crop
    )
    rain_support = (
        rising(annual_rain, 100.0, 300.0)
        * annual_rain / (annual_rain + 500.0)
        * np.exp(-annual_rain / 3000.0)
    )
    warm_support = rising(temperature_12, 3.0, 10.0)
    # This is a slowly varying event-size capacity. Existing combustion and
    # ignition equations retain responsibility for monthly timing.
    return {
        "structural": np.clip(structural_share, 0.0, 1.0),
        "supported": np.clip(share * rain_support * warm_support, 0.0, 1.0),
    }


def candidate(incumbent: np.ndarray, state: np.ndarray, strength: float) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * (1.0 + strength * state), 0.0, 50.0)),
        dtype=np.float32,
    )


def losses(prediction, observed, area, cells, folds):
    rows, cols = cells // 360, cells % 360
    pred = np.asarray(prediction[:, rows, cols], dtype=np.float64)
    obs = np.asarray(observed[:, rows, cols], dtype=np.float64)
    obs_ann = np.average(obs, axis=0, weights=MONTH_DAYS)
    pred_ann = np.average(pred, axis=0, weights=MONTH_DAYS)
    weight = area[rows, cols] * obs_ann
    obs_cycle = obs.reshape(16, 12, -1).mean(axis=0)
    pred_cycle = pred.reshape(16, 12, -1).mean(axis=0)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-12)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-12)
    annual = []
    cycle = []
    for fold in range(4):
        held = folds == fold
        annual.append(
            np.sqrt(
                np.sum(
                    weight[held]
                    * np.square(
                        np.log(obs_ann[held] + 1e-5) - np.log(pred_ann[held] + 1e-5)
                    )
                )
                / (np.sum(weight[held]) + 1e-15)
            )
        )
        cycle.append(
            np.sum(
                weight[held][None, :]
                * np.abs(obs_alloc[:, held] - pred_alloc[:, held])
            )
            / (np.sum(weight[held]) + 1e-15)
        )
    return np.asarray(annual), np.asarray(cycle)


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    states = secondary_regrowth_states(data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine_observed = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine_observed.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    weight = area * obs_annual
    ranking = np.argsort(weight.ravel())[::-1]
    coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
    count = int(np.searchsorted(coverage, 0.90) + 1)
    cells = ranking[:count]
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4

    base_annual, base_cycle = losses(incumbent, observed, area, cells, folds)
    print(
        "BASE annual=" + ",".join(f"{value:.6f}" for value in base_annual)
        + " cycle=" + ",".join(f"{value:.6f}" for value in base_cycle)
    )
    selected_obs = observed[:, rows, cols]
    selected_pred = incumbent[:, rows, cols]
    monthly_weight = np.broadcast_to(weight[rows, cols], selected_obs.shape)
    survivor = None
    for name, state in states.items():
        selected_state = state[:, rows, cols]
        quantiles = np.quantile(selected_state, np.linspace(0.0, 1.0, 6))
        print(f"STATE_QUINTILES {name} model_over_observed")
        for index in range(5):
            if index == 4:
                mask = (selected_state >= quantiles[index]) & (selected_state <= quantiles[index + 1])
            else:
                mask = (selected_state >= quantiles[index]) & (selected_state < quantiles[index + 1])
            ratio = np.sum(monthly_weight * selected_pred * mask) / (
                np.sum(monthly_weight * selected_obs * mask) + 1e-15
            )
            print(
                f"q{index + 1} state={quantiles[index]:.5f}:{quantiles[index + 1]:.5f} "
                f"ratio={ratio:.5f}"
            )

        for strength in (0.5, 1.0, 2.0, 4.0):
            trial = candidate(incumbent, state, strength)
            annual, cycle = losses(trial, observed, area, cells, folds)
            annual_gain = base_annual - annual
            cycle_gain = base_cycle - cycle
            held = bool(
                np.all(annual_gain > 0.0)
                and -cycle_gain.sum() <= 0.05 * annual_gain.sum()
            )
            print(
                f"variant={name} strength={strength:g} held={held} annual_gain="
                + ",".join(f"{value:+.6f}" for value in annual_gain)
                + " cycle_gain="
                + ",".join(f"{value:+.6f}" for value in cycle_gain)
            )
            if held and survivor is None:
                survivor = (name, strength, trial, annual_gain.sum() + cycle_gain.sum())
            elif held and survivor is not None:
                objective = annual_gain.sum() + cycle_gain.sum()
                if objective > survivor[3]:
                    survivor = (name, strength, trial, objective)

    if survivor is None:
        print("EXACT skipped: no stable held survivor")
        return
    name, strength, trial, _ = survivor
    score = evaluator.score(validate_prediction(trial))
    global_score = score["global"]
    print(
        f"EXACT variant={name} strength={strength:g} overall={global_score['overall_score']:.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f}"
    )


if __name__ == "__main__":
    main()
