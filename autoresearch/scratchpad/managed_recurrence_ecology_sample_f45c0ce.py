"""Stratified sampled ecology audit of managed recurrence equilibrium.

The sample retains the most observation-weighted cells from each established
ecological guardrail plus the global high-fire carrier.  Masks are diagnostics
only and never enter the candidate equation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    load_observed,
    load_selected,
)
from autoresearch.scratchpad.warm_seasonal_open_held_block_f45c0ce import (  # noqa: E402
    MONTH_DAYS,
    input_index,
    stream_max,
    stream_mean,
)
from autoresearch.scratchpad.warm_open_transition_capacity_probe_f45c0ce import (  # noqa: E402
    antecedent,
    annual_capacity_candidate,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def ecology_masks():
    index = input_index()
    rain = 12.0 * stream_mean("monthly_precipitation", index)
    temperature = stream_mean("air_temperature", index)
    canopy = stream_mean("natural_canopy_height", index)
    lai = stream_mean("leaf_area_index", index)
    biomass = stream_mean("aboveground_biomass", index)
    natural = stream_mean("natural_vegetation_fraction", index)
    primary = stream_mean("luh2_primary_fraction", index)
    crop = stream_mean("luh2_cropland_fraction", index)
    rangeland = stream_mean("luh2_rangeland_fraction", index)
    land = (
        (stream_max("annual_precipitation", index) > 0.0)
        | (stream_max("air_temperature", index, absolute=True) > 1e-6)
        | (stream_max("natural_vegetation_fraction", index) > 0.0)
        | (stream_max("secondary_vegetation_fraction", index) > 0.0)
    )
    masks = {
        "intact_tropical_closed": (temperature >= 20) & (rain >= 1200) & (canopy >= 20) & (lai >= 3) & (natural >= .7) & (primary >= .5),
        "temperate_closed": (temperature >= 5) & (temperature < 20) & (canopy >= 15) & (lai >= 2.5) & (natural >= .6),
        "boreal": (temperature < 5) & (canopy >= 10) & (natural >= .6),
        "tropical_open": (temperature >= 20) & (rain >= 500) & (rain < 1500) & (canopy >= 5) & (canopy < 20) & (natural >= .5),
        "productive_rangeland": (rangeland >= .4) & (rain >= 250) & (rain < 1500) & (biomass >= .2),
        "cropland": crop >= .5,
        "arid_low_fuel": (rain < 250) & (biomass < .3) & (lai < 1),
    }
    return {name: mask & land for name, mask in masks.items()}, land


def select_stratified(evaluator, masks, land, per_mask=256, global_count=768):
    mean = np.asarray(evaluator.reference_mean).reshape(180, 2, 360, 2).mean(axis=(1, 3)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    weight = mean * area
    selected = set()
    for mask in list(masks.values()) + [land]:
        candidates = np.flatnonzero(mask.ravel())
        count = global_count if mask is land else per_mask
        order = candidates[np.argsort(weight.ravel()[candidates])[::-1]]
        selected.update(map(int, order[: min(count, order.size)]))
    cells = np.asarray(sorted(selected), dtype=np.int64)
    return cells // 360, cells % 360, area, weight


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    masks, land = ecology_masks()
    rows, cols, full_area, full_weight = select_stratified(evaluator, masks, land)
    print(f"DESIGN selected={rows.size}", flush=True)
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    baseline = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]

    def field(name):
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    rain12 = antecedent(rain, 12.0)
    temperature12 = antecedent(field("air_temperature"), 12.0)
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    hazard12 = antecedent(hazard, 12.0)
    managed = np.clip(
        field("luh2_rangeland_fraction") + field("luh2_pasture_fraction"),
        0.0,
        1.0,
    )
    managed_access = managed / (managed + 0.15)
    gap = 0.01 / (hazard12 + 0.01)
    recurrence = hazard12 / (hazard12 + 0.01)
    annual_rain = 12.0 * rain12
    fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(-annual_rain / 3000.0)
    warm = 1.0 / (1.0 + np.exp(np.clip(-(temperature12 - 18.0) / 4.0, -30.0, 30.0)))
    signal = managed_access * (gap - recurrence) * fuel * warm
    candidate = annual_capacity_candidate(hazard, signal, 0.5)

    base_annual = np.average(baseline, axis=0, weights=MONTH_DAYS)
    candidate_annual = np.average(candidate, axis=0, weights=MONTH_DAYS)
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    for name, mask in masks.items():
        selected = mask[rows, cols]
        area = full_area[rows[selected], cols[selected]]
        denominator = float(observed_annual[selected] @ area)
        old = float(base_annual[selected] @ area) / (denominator + 1e-12)
        new = float(candidate_annual[selected] @ area) / (denominator + 1e-12)
        retained = float(full_weight[rows[selected], cols[selected]].sum() / (full_weight[mask].sum() + 1e-12))
        print(
            f"ECOLOGY {name} cells={int(selected.sum())} retained_obs_weight={retained:.6f} "
            f"ratio={old:.6f}->{new:.6f} delta={new-old:+.6f} relative_mass={(new/old-1.0):+.6f}",
            flush=True,
        )
    prefix = 96
    perturbed = {name: values.copy() for name, values in data.items()}
    for values in perturbed.values():
        values[prefix:] *= np.float32(0.5)
    altered_base = np.asarray(model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
    altered_rain = np.clip(np.asarray(perturbed["monthly_precipitation"][:, 0, :], dtype=np.float64), 0.0, None)
    altered_rain12 = antecedent(altered_rain, 12.0)
    altered_t12 = antecedent(np.asarray(perturbed["air_temperature"][:, 0, :], dtype=np.float64), 12.0)
    altered_hazard = -np.log1p(-np.clip(altered_base, 0.0, 1.0 - 1e-7))
    altered_h12 = antecedent(altered_hazard, 12.0)
    altered_managed = np.clip(
        np.asarray(perturbed["luh2_rangeland_fraction"][:, 0, :], dtype=np.float64)
        + np.asarray(perturbed["luh2_pasture_fraction"][:, 0, :], dtype=np.float64), 0.0, 1.0
    )
    altered_access = altered_managed / (altered_managed + .15)
    altered_gap = .01 / (altered_h12 + .01)
    altered_rec = altered_h12 / (altered_h12 + .01)
    altered_a = 12 * altered_rain12
    altered_fuel = np.square(altered_a / (altered_a + 250)) * np.exp(-altered_a / 3000)
    altered_warm = 1 / (1 + np.exp(np.clip(-(altered_t12 - 18) / 4, -30, 30)))
    altered_candidate = annual_capacity_candidate(
        altered_hazard,
        altered_access * (altered_gap - altered_rec) * altered_fuel * altered_warm,
        .5,
    )
    print(
        f"PREFIX max_abs={float(np.max(np.abs(altered_candidate[:prefix] - candidate[:prefix]))):.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
