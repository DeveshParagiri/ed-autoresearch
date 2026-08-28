"""Falsify a low-woody rangeland management-opportunity interaction.

The current model represents generic managed access and separately brakes
productive rangeland, but it has no interaction in which management access is
largest in low-woody, rain-supported swards. Fixed strengths are screened in
held spatial blocks; only the weakest stable survivor is replayed exactly.
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

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from secondary_regrowth_footprint_33ac854 import MONTH_DAYS, losses  # noqa: E402
from temperature_pathway_blend import ecological_ratios  # noqa: E402
from unrepresented_state_audit_9f957d7 import antecedent, load_pinned  # noqa: E402


def rising(values: np.ndarray, scale: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def falling(values: np.ndarray, scale: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip((values - center) / scale, -50.0, 50.0)))


def management_state(data) -> np.ndarray:
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None)
    annual_rain = 12.0 * antecedent(rain, 12.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_12 = antecedent(temperature, 12.0)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = antecedent(gpp, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    biomass = np.clip(np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None)
    low_woody = 0.5 / (biomass + 0.5)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    rain_support = (
        rising(annual_rain, 100.0, 250.0)
        * falling(annual_rain, 250.0, 1500.0)
    )
    warm = rising(temperature_12, 3.0, 15.0)
    return np.clip(
        rangeland * low_woody * fine_fuel * continuity * rain_support * warm,
        0.0,
        1.0,
    )


def candidate(incumbent, state, strength):
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * (1.0 + strength * state), 0.0, 50.0)),
        dtype=np.float32,
    )


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    state = management_state(data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    weight = area * obs_annual
    ranking = np.argsort(weight.ravel())[::-1]
    coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
    cells = ranking[: int(np.searchsorted(coverage, 0.90) + 1)]
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    base_annual, base_cycle = losses(incumbent, observed, area, cells, folds)
    base_score = evaluator.score(incumbent)
    print(f"BASE overall={base_score['global']['overall_score']:.9f}")

    survivor = None
    for strength in (0.25, 0.5, 1.0, 2.0):
        trial = candidate(incumbent, state, strength)
        annual, cycle = losses(trial, observed, area, cells, folds)
        annual_gain = base_annual - annual
        cycle_gain = base_cycle - cycle
        held = bool(
            np.all(annual_gain > 0.0)
            and -cycle_gain.sum() <= 0.05 * annual_gain.sum()
        )
        print(
            f"strength={strength:g} held={held} annual_gain="
            + ",".join(f"{value:+.6f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.6f}" for value in cycle_gain)
        )
        if held and survivor is None:
            survivor = (strength, trial)
    if survivor is None:
        print("EXACT skipped: no stable held survivor")
        return

    strength, trial = survivor
    score = evaluator.score(validate_prediction(trial))
    global_score = score["global"]
    deltas = {
        region: score[region]["overall_score"] - base_score[region]["overall_score"]
        for region in score
        if region != "global"
    }
    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 12.0
    )
    ecology_base = ecological_ratios(
        incumbent, prepared, observed, area, load_land_mask()
    )
    ecology = ecological_ratios(trial, prepared, observed, area, load_land_mask())

    prefix_cells = ranking[:64]
    prefix_rows, prefix_cols = prefix_cells // 360, prefix_cells % 360
    prefix_data = {
        key: np.asarray(values[:, prefix_rows, prefix_cols])[:, None, :]
        for key, values in data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    prefix_trial = candidate(prefix_incumbent, management_state(prefix_data), strength)
    perturbed = {key: values.copy() for key, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    perturbed_incumbent = model.predict(perturbed, dict(model.PARAMS), None)
    perturbed_trial = candidate(
        perturbed_incumbent, management_state(perturbed), strength
    )
    print(
        f"EXACT strength={strength:g} overall={global_score['overall_score']:.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f} "
        f"regions={sum(value > 0.0 for value in deltas.values())}/{len(deltas)} "
        f"prefix={np.max(np.abs(prefix_trial[:96]-perturbed_trial[:96])):.12g}"
    )
    print(
        "REGIONAL "
        + ",".join(f"{region}:{value:+.6f}" for region, value in sorted(deltas.items()))
    )
    print(
        "ECOLOGY "
        + ",".join(
            f"{name}:{ecology_base[name]:.5f}->{ecology[name]:.5f}"
            for name in ecology_base
        )
    )


if __name__ == "__main__":
    main()
