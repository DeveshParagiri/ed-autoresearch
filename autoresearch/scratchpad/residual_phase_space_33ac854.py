"""Weighted ecological phase-space audit of the 33ac854 residual.

Diagnostic only. Coordinates assign held spatial blocks but never enter a
candidate state or equation. All dynamic summaries are current or prefix
causal. The script does not edit or officially evaluate the canonical model.
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

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "33ac854"
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def signed_ratio(observed, predicted, weight, mask) -> float:
    selected = weight * mask
    numerator = float(np.sum(selected * (observed - predicted)))
    denominator = float(np.sum(selected * 0.5 * (observed + predicted)))
    return numerator / (denominator + 1e-15)


def fold_values(observed, predicted, weight, mask, folds) -> np.ndarray:
    return np.asarray(
        [
            signed_ratio(observed, predicted, weight, mask * (folds == fold))
            for fold in range(4)
        ]
    )


def sign_text(values: np.ndarray) -> str:
    signs = np.sign(values)
    stable = bool(np.all(signs > 0.0) or np.all(signs < 0.0))
    return ("stable" if stable else "mixed") + ":" + ",".join(
        f"{value:+.3f}" for value in values
    )


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    base = evaluator.score(prediction)["global"]
    with Dataset(GFED5_PATH) as dataset:
        fine_observed = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine_observed.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))

    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    observed_weight = area * obs_annual
    ranking = np.argsort(observed_weight.ravel())[::-1]
    cumulative = np.cumsum(observed_weight.ravel()[ranking]) / observed_weight.sum()
    count = int(np.searchsorted(cumulative, 0.90) + 1)
    cells = ranking[:count]
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"base={base['overall_score']:.9f} selected_cells={count} "
        f"observed_fire_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f}"
    )

    selected = {
        name: np.asarray(data[name][:, rows, cols], dtype=np.float64)
        for name in model.INPUTS
    }
    pred = np.asarray(prediction[:, rows, cols], dtype=np.float64)
    obs = np.asarray(observed[:, rows, cols], dtype=np.float64)
    obs_ann = obs_annual[rows, cols]
    pred_ann = pred_annual[rows, cols]
    area_cell = area[rows, cols]
    del data, prediction, observed, fine_observed

    rain = np.clip(selected["monthly_precipitation"], 0.0, None)
    dry = np.clip(selected["dryness"], 0.0, None)
    temp = selected["air_temperature"]
    gpp = np.clip(selected["gpp"], 0.0, None)
    lightning = np.clip(selected["lightning_flash_rate"], 0.0, None)
    rain3, rain12 = antecedent(rain, 3.0), antecedent(rain, 12.0)
    gpp3, gpp12 = antecedent(gpp, 3.0), antecedent(gpp, 12.0)
    lightning3 = antecedent(lightning, 3.0)

    crop = np.clip(selected["luh2_cropland_fraction"], 0.0, 1.0)
    range_ = np.clip(selected["luh2_rangeland_fraction"], 0.0, 1.0)
    pasture = np.clip(selected["luh2_pasture_fraction"], 0.0, 1.0)
    natural = np.clip(selected["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(selected["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(selected["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(selected["secondary_canopy_height"], 0.0, None)
    biomass = np.clip(selected["aboveground_biomass"], 0.0, None)
    urban = np.clip(selected["luh2_urban_fraction"], 0.0, 1.0)
    # The coupled-valid model reconstructs annual moisture causally because no
    # precomputed full-year precipitation field is an active input.
    annual_rain = 12.0 * rain12

    fine_fuel = gpp12 / (gpp12 + 0.35)
    open_cover = np.clip(
        range_
        + pasture
        + natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0),
        0.0,
        2.0,
    )
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover * continuity
    woody_capacity = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    capacity_total = 0.05 + surface_capacity + woody_capacity + crop_capacity
    surface_share = surface_capacity / capacity_total
    woody_share = woody_capacity / capacity_total
    crop_share = crop_capacity / capacity_total

    mean_rain = annual_rain.mean(axis=0)
    mean_temp = temp.mean(axis=0)
    mean_surface = surface_share.mean(axis=0)
    mean_woody = woody_share.mean(axis=0)
    mean_crop = crop_share.mean(axis=0)
    mean_fuel = fine_fuel.mean(axis=0)
    mean_continuity = continuity.mean(axis=0)
    primary = np.clip(selected["luh2_primary_fraction"], 0.0, 1.0)
    mean_primary = primary.mean(axis=0)
    mean_secondary = secondary.mean(axis=0)
    mean_secondary_open = (
        secondary * 8.0 / (secondary_canopy + 8.0)
    ).mean(axis=0)
    mean_secondary_woody = (
        secondary * secondary_canopy / (secondary_canopy + 8.0)
    ).mean(axis=0)
    mean_pasture = pasture.mean(axis=0)
    mean_range = range_.mean(axis=0)
    mean_canopy = canopy.mean(axis=0)
    mean_biomass = biomass.mean(axis=0)
    mean_lai = np.clip(selected["leaf_area_index"], 0.0, None).mean(axis=0)
    mean_soil_c = np.clip(selected["soil_carbon"], 0.0, None).mean(axis=0)
    mean_lightning = lightning.mean(axis=0)
    rain_variability = np.sqrt(
        np.maximum(antecedent(np.square(rain), 12.0) - np.square(rain12), 0.0)
    ).mean(axis=0)
    temperature_variability = np.sqrt(
        np.maximum(
            antecedent(np.square(temp), 12.0) - np.square(antecedent(temp, 12.0)),
            0.0,
        )
    ).mean(axis=0)
    fire_weight = area_cell * 0.5 * (obs_ann + pred_ann)

    hydro = np.where(mean_rain < 500.0, "dry", np.where(mean_rain < 1200.0, "seasonal", "humid"))
    thermal = np.where(mean_temp < 8.0, "cold", np.where(mean_temp < 20.0, "mild", "warm"))
    structure = np.full(count, "mixed", dtype="U8")
    structure[mean_surface >= np.maximum(0.50, np.maximum(mean_woody, mean_crop))] = "surface"
    structure[mean_woody >= np.maximum(0.45, np.maximum(mean_surface, mean_crop))] = "woody"
    structure[mean_crop >= np.maximum(0.35, np.maximum(mean_surface, mean_woody))] = "crop"

    states = np.asarray(
        [f"{h}/{t}/{s}" for h, t, s in zip(hydro, thermal, structure)], dtype="U32"
    )
    state_rows = []
    total_observed_fire = float(np.sum(area_cell * obs_ann))
    for state in np.unique(states):
        mask = states == state
        share = float(np.sum(area_cell[mask] * obs_ann[mask]) / total_observed_fire)
        residual = signed_ratio(obs_ann, pred_ann, area_cell, mask)
        held = fold_values(obs_ann, pred_ann, area_cell, mask, folds)
        state_rows.append((share, state, residual, held, int(mask.sum())))
    state_rows.sort(reverse=True, key=lambda row: row[0])
    print("ANNUAL_PHASE_SPACE share residual held_blocks cells")
    for share, state, residual, held, ncell in state_rows[:20]:
        print(
            f"{state:24s} share={share:.4f} residual={residual:+.4f} "
            f"held={sign_text(held)} cells={ncell}"
        )

    axes = {
        "surface_high": mean_surface >= 0.55,
        "woody_high": mean_woody >= 0.45,
        "crop_high": mean_crop >= 0.35,
        "fuel_low": mean_fuel < 0.55,
        "fuel_high": mean_fuel >= 0.75,
        "fragmented": mean_continuity < 0.75,
        "primary_high": mean_primary >= 0.55,
        "secondary_high": mean_secondary >= 0.35,
        "secondary_open": mean_secondary_open >= 0.25,
        "secondary_woody": mean_secondary_woody >= 0.25,
        "pasture_high": mean_pasture >= 0.20,
        "rangeland_high": mean_range >= 0.20,
        "natural_open": (mean_surface >= 0.45) & (mean_primary >= 0.35),
        "tall_canopy": mean_canopy >= 15.0,
        "high_biomass": mean_biomass >= 2.0,
        "low_lai": mean_lai < 1.5,
        "high_soil_c": mean_soil_c >= 5.0,
        "rare_lightning": mean_lightning < 0.01,
        "frequent_lightning": mean_lightning >= 0.05,
        "rain_aseasonal": rain_variability < 35.0,
        "rain_seasonal": rain_variability >= 70.0,
        "temperature_seasonal": temperature_variability >= 7.0,
        "dry_climate": mean_rain < 500.0,
        "seasonal_climate": (mean_rain >= 500.0) & (mean_rain < 1200.0),
        "humid_climate": mean_rain >= 1200.0,
        "cold": mean_temp < 8.0,
        "warm": mean_temp >= 20.0,
    }
    print("ANNUAL_AXES share residual held_blocks")
    for name, mask in axes.items():
        share = float(np.sum(area_cell[mask] * obs_ann[mask]) / total_observed_fire)
        residual = signed_ratio(obs_ann, pred_ann, area_cell, mask)
        held = fold_values(obs_ann, pred_ann, area_cell, mask, folds)
        print(f"{name:18s} share={share:.4f} residual={residual:+.4f} held={sign_text(held)}")

    # Monthly allocation residual, classified by causal hydrologic/phenology phase.
    obs_cycle = obs.reshape(16, 12, count).mean(axis=0)
    pred_cycle = pred.reshape(16, 12, count).mean(axis=0)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-12)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-12)
    rain_deficit = np.maximum((rain3 - rain) / (rain3 + rain + 10.0), 0.0)
    curing = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.2), 0.0)
    combustion = dry / (dry + 250.0) / (1.0 + rain / 35.0)
    greenup = np.maximum((gpp - gpp3) / (gpp3 + gpp + 0.2), 0.0)
    lightning_pulse = np.maximum(
        (lightning - lightning3) / (lightning + lightning3 + 0.002), 0.0
    )
    def annual_cycle(values):
        return values.reshape(16, 12, count).mean(axis=0)
    deficit_cycle = annual_cycle(rain_deficit)
    curing_cycle = annual_cycle(curing)
    combustion_cycle = annual_cycle(combustion)
    greenup_cycle = annual_cycle(greenup)
    lightning_cycle = annual_cycle(lightning_pulse)
    phases = {
        "drying_curing": (deficit_cycle > 0.10) & (curing_cycle > 0.04),
        "peak_combustible": combustion_cycle > 0.45,
        "wet_greenup": greenup_cycle > 0.04,
        "lightning_arrival": lightning_cycle > 0.12,
    }
    cycle_weight = np.broadcast_to(fire_weight, (12, count))
    cycle_folds = np.broadcast_to(folds, (12, count))
    print("CYCLE_PHASE_SPACE state/phase share residual held_blocks")
    phase_rows = []
    for state in np.unique(states):
        cell_mask = states == state
        for phase, phase_mask in phases.items():
            mask = phase_mask & cell_mask[None, :]
            support = float(np.sum(cycle_weight * mask * 0.5 * (obs_alloc + pred_alloc)))
            total = float(np.sum(cycle_weight * 0.5 * (obs_alloc + pred_alloc)))
            share = support / (total + 1e-15)
            if share < 0.003:
                continue
            residual = signed_ratio(obs_alloc, pred_alloc, cycle_weight, mask)
            held = fold_values(obs_alloc, pred_alloc, cycle_weight, mask, cycle_folds)
            phase_rows.append((share, state, phase, residual, held))
    phase_rows.sort(reverse=True, key=lambda row: row[0])
    for share, state, phase, residual, held in phase_rows[:30]:
        print(
            f"{state}/{phase:18s} share={share:.4f} residual={residual:+.4f} "
            f"held={sign_text(held)}"
        )


if __name__ == "__main__":
    main()
