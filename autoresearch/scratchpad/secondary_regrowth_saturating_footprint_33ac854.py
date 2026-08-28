"""Held and exact audit of a saturating secondary-regrowth footprint.

The linear secondary-open footprint ``1 + 0.5 q2`` improves the global map but
keeps expanding event size as secondary structural share approaches one.  A
finite patch mosaic instead has diminishing returns once burnable regrowth
patches overlap.  This experiment preserves the linear law's initial slope but
caps its additional event footprint at 25 percent:

    footprint = 1 + 0.25 * (1 - exp(-2 q2)).

The constants are linked rather than fitted: ``0.25 * 2 = 0.5``, so rare
secondary cover has the incumbent response while dominant secondary cover
saturates.  Region labels are used only after prediction for diagnosis.  The
candidate equation is globally shared, pointwise, prefix causal, and target
independent.  No canonical or official artifact is written.
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
from secondary_regrowth_footprint_33ac854 import (  # noqa: E402
    MONTH_DAYS,
    candidate as linear_candidate,
    losses,
    secondary_regrowth_states,
)
from scripts.fast_ilamb import GFED5Evaluator, GFED_REGIONS  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402


CACHE = SCRATCH / "canonical_3f63c96b_chunked.npy"
EXPECTED_INCUMBENT = 0.719107756
FOOTPRINT_CAP = 0.25
INITIAL_SLOPE = 0.5


def saturating_candidate(
    incumbent: np.ndarray,
    structural_share: np.ndarray,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    rate = INITIAL_SLOPE / FOOTPRINT_CAP
    added_footprint = FOOTPRINT_CAP * (
        1.0 - np.exp(-rate * np.clip(structural_share, 0.0, 1.0))
    )
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * (1.0 + added_footprint), 0.0, 50.0)),
        dtype=np.float32,
    )


def coarse_region_mask(
    bounds: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if bounds is None:
        return np.ones((180, 360), dtype=bool)
    south, north, west, east = bounds
    latitude = -89.5 + np.arange(180, dtype=np.float64)
    longitude = -179.5 + np.arange(360, dtype=np.float64)
    return (
        (latitude[:, None] > south)
        & (latitude[:, None] <= north)
        & (longitude[None, :] > west)
        & (longitude[None, :] <= east)
    )


def main() -> None:
    model = load_pinned()
    if not CACHE.exists():
        raise RuntimeError(f"missing pinned prediction cache {CACHE}")
    incumbent = np.asarray(np.load(CACHE, mmap_mode="r"), dtype=np.float32)
    data = load_inputs(model.INPUTS)
    state = secondary_regrowth_states(data)["structural"]
    linear = linear_candidate(incumbent, state, 0.5)
    trial = saturating_candidate(incumbent, state)

    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_score = evaluator.score(incumbent)
    incumbent_global = incumbent_score["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError("failed pinned incumbent reproduction")
    with Dataset(GFED5_PATH) as dataset:
        fine_observed = np.asarray(dataset.variables["burntArea"][:192])
    observed = (
        fine_observed.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    weight = area * obs_annual
    ranking = np.argsort(weight.ravel())[::-1]
    coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
    count = int(np.searchsorted(coverage, 0.90) + 1)
    cells = ranking[:count]
    rows, columns = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (columns // 15)) % 4

    base_annual, base_cycle = losses(
        incumbent, observed, area, cells, folds
    )
    linear_annual, linear_cycle = losses(
        linear, observed, area, cells, folds
    )
    trial_annual, trial_cycle = losses(
        trial, observed, area, cells, folds
    )
    annual_gain = base_annual - trial_annual
    cycle_gain = base_cycle - trial_cycle
    held = bool(
        np.all(annual_gain > 0.0)
        and -cycle_gain.sum() <= 0.05 * annual_gain.sum()
    )
    print(
        "HELD linear_annual_gain="
        + ",".join(f"{value:+.6f}" for value in base_annual - linear_annual)
        + " linear_cycle_gain="
        + ",".join(f"{value:+.6f}" for value in base_cycle - linear_cycle)
    )
    print(
        f"HELD saturated={held} annual_gain="
        + ",".join(f"{value:+.6f}" for value in annual_gain)
        + " cycle_gain="
        + ",".join(f"{value:+.6f}" for value in cycle_gain)
    )

    state_mean = state.mean(axis=0)
    incumbent_annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    trial_annual_map = np.average(trial, axis=0, weights=MONTH_DAYS)
    print("REGION_DIAG name area_ratio_before area_ratio_after fire_weighted_q2")
    for name, bounds in GFED_REGIONS.items():
        if name == "global":
            continue
        mask = coarse_region_mask(bounds) & load_land_mask()
        observed_total = float(np.sum(obs_annual[mask] * area[mask]))
        before = float(np.sum(incumbent_annual[mask] * area[mask])) / max(
            observed_total, 1e-15
        )
        after = float(np.sum(trial_annual_map[mask] * area[mask])) / max(
            observed_total, 1e-15
        )
        q2 = float(
            np.sum(state_mean[mask] * obs_annual[mask] * area[mask])
            / max(observed_total, 1e-15)
        )
        print(
            f"REGION_DIAG name={name} area_ratio_before={before:.6f} "
            f"area_ratio_after={after:.6f} fire_weighted_q2={q2:.6f}"
        )

    if not held:
        print("EXACT skipped: saturating law failed held gate")
        return
    trial_score = evaluator.score(validate_prediction(trial))
    trial_global = trial_score["global"]
    print(
        f"EXACT overall={trial_global['overall_score']:.9f} "
        f"delta={trial_global['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"bias={trial_global['bias_score']:.9f} "
        f"rmse={trial_global['rmse_score']:.9f} "
        f"seasonal={trial_global['seasonal_cycle_score']:.9f} "
        f"spatial={trial_global['spatial_distribution_score']:.9f}"
    )
    deltas = {
        name: trial_score[name]["overall_score"]
        - incumbent_score[name]["overall_score"]
        for name in trial_score
        if name != "global"
    }
    print(
        f"REGIONS improved={sum(value > 0.0 for value in deltas.values())}/14 "
        + ",".join(f"{name}:{value:+.6f}" for name, value in sorted(deltas.items()))
    )

    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 12.0
    )
    land = load_land_mask()
    incumbent_ecology = ecological_ratios(
        incumbent, prepared, observed, area, land
    )
    trial_ecology = ecological_ratios(trial, prepared, observed, area, land)
    print(
        "ECOLOGY "
        + ",".join(
            f"{name}:{incumbent_ecology[name]:.5f}->{trial_ecology[name]:.5f}"
            for name in incumbent_ecology
        )
    )

    prefix_cells = ranking[:64]
    prefix_rows, prefix_columns = prefix_cells // 360, prefix_cells % 360
    prefix_data = {
        name: np.asarray(values[:, prefix_rows, prefix_columns])[:, None, :]
        for name, values in data.items()
    }
    prefix_incumbent = model.predict(
        prefix_data, dict(model.PARAMS), None
    )
    prefix_state = secondary_regrowth_states(prefix_data)["structural"]
    prefix_trial = saturating_candidate(prefix_incumbent, prefix_state)
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    perturbed_incumbent = model.predict(
        perturbed, dict(model.PARAMS), None
    )
    perturbed_state = secondary_regrowth_states(perturbed)["structural"]
    perturbed_trial = saturating_candidate(
        perturbed_incumbent, perturbed_state
    )
    print(
        "PREFIX max_abs="
        f"{np.max(np.abs(prefix_trial[:96]-perturbed_trial[:96])):.12g}"
    )


if __name__ == "__main__":
    main()
