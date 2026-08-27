"""Provenance and sampled falsification of LUH2 net-change fire mechanisms.

The prepared LUH2 fields are annual state snapshots repeated for all twelve
months.  Their differences therefore identify only net annual state changes,
not the gross transition matrix supplied by native LUH2.  This diagnostic first
quantifies those changes and then tests three fixed, global, pointwise equations
that cannot spend an annual event twelve times.  Clearing and conversion enter
a finite residue stock at the January state update and are released only when
fuel, moisture, temperature, and ignition allow combustion.  Fragmentation is
represented as a finite memory of newly converted land that removes no more
than its own connected surface share.

Observations and coordinates select diagnostic cells and spatial folds only.
They never enter a candidate equation.  The script does not run the official
evaluator, edit the canonical model, fit a coefficient, or write a tracked
artifact.

Semantics references are the LUH2 v2f variable README at
https://luh.umd.edu/LUH2/LUH2_v2f_README_v6.pdf and Hurtt et al. (2020),
https://doi.org/10.5194/gmd-13-5425-2020.  Those sources distinguish state
fractions from explicit annual ``state1_to_state2`` gross transitions.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    MONTH_DAYS,
    antecedent,
    format_metrics,
    input_index,
    load_observed,
    load_selected,
    metrics,
    weighted_corr,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, INPUTS_DIR, load_model  # noqa: E402


LUH2_PATH = INPUTS_DIR / "luh2.nc"
LUH_NAMES = (
    "luh2_primary_fraction",
    "luh2_secondary_fraction",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_urban_fraction",
)
COMPOSITIONAL_LAND_NAMES = tuple(
    name for name in LUH_NAMES if name != "luh2_secondary_fraction"
)
MONTH_WEIGHTS = MONTH_DAYS / MONTH_DAYS.sum()


def rising(values: np.ndarray, center: float, width: float) -> np.ndarray:
    z = np.clip((values - center) / width, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def field(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(data[name][:, 0, :], dtype=np.float64)


def union_fraction(*values: np.ndarray) -> np.ndarray:
    complement = np.ones_like(values[0], dtype=np.float64)
    for value in values:
        complement *= 1.0 - np.clip(value, 0.0, 1.0)
    return np.clip(1.0 - complement, 0.0, 1.0)


def smooth_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Smooth bounded analogue of min(left, right), zero if either is zero."""
    return 2.0 * left * right / (left + right + 1e-12)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def coarse_area_and_reference(evaluator: GFED5Evaluator):
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    reference = (
        np.asarray(evaluator.reference_mean, dtype=np.float64)
        .reshape(180, 2, 360, 2)
        .mean(axis=(1, 3))
        / 100.0
    )
    return area, reference, area * reference


def read_annual_states() -> tuple[dict[str, np.ndarray], float]:
    annual: dict[str, np.ndarray] = {}
    repeat_error = 0.0
    with Dataset(LUH2_PATH) as dataset:
        print(
            "PROVENANCE "
            f"source={getattr(dataset, 'source', '')!r} "
            f"processing={getattr(dataset, 'processing', '')!r}",
            flush=True,
        )
        for name in LUH_NAMES:
            values = np.asarray(dataset.variables[name][:], dtype=np.float64)
            annual[name] = values[::12]
            repeated = np.repeat(annual[name], 12, axis=0)
            repeat_error = max(repeat_error, float(np.max(np.abs(values - repeated))))
    return annual, repeat_error


def provenance_audit(
    annual: Mapping[str, np.ndarray], area: np.ndarray, repeat_error: float
) -> np.ndarray:
    land = np.max(
        np.maximum.reduce([annual[name] for name in COMPOSITIONAL_LAND_NAMES]), axis=0
    ) > 1e-8
    land_area = area[land]
    tiled_area = np.broadcast_to(land_area, (15, land_area.size))
    print(
        f"REPETITION max_within_year_difference={repeat_error:.12g} "
        "annual_differences_land_only_in_january=true",
        flush=True,
    )
    for name in LUH_NAMES:
        values = annual[name][:, land]
        change = np.diff(values, axis=0)
        gain = np.clip(change, 0.0, None)
        loss = np.clip(-change, 0.0, None)
        endpoint = weighted_mean(values[-1] - values[0], land_area)
        mean_state = weighted_mean(values.mean(axis=0), land_area)
        mean_abs = weighted_mean(np.abs(change), tiled_area)
        mean_gain = weighted_mean(gain, tiled_area)
        mean_loss = weighted_mean(loss, tiled_area)
        active = " ".join(
            f"area_year_gt_{threshold:g}="
            f"{weighted_mean(np.abs(change) > threshold, tiled_area):.8f}"
            for threshold in (1e-6, 1e-3, 1e-2, 5e-2)
        )
        extreme = int(np.count_nonzero(np.abs(change) > 0.1))
        print(
            f"STATE {name} mean={mean_state:.9f} endpoint={endpoint:+.9f} "
            f"annual_abs={mean_abs:.9f} gain={mean_gain:.9f} "
            f"loss={mean_loss:.9f} max_abs={np.max(np.abs(change)):.9f} "
            f"extreme_cell_years_gt_0.1={extreme} {active}",
            flush=True,
        )

    secondary = annual["luh2_secondary_fraction"][:, land]
    secondary_change = np.diff(secondary, axis=0)
    print(
        "SECONDARY_DIAGNOSTIC numeric_range="
        f"{secondary.min():.9f}..{secondary.max():.9f} "
        f"median={np.median(secondary):.9f} share_gt_0.9={np.mean(secondary > 0.9):.9f} "
        f"cell_year_change_gt_1e-6={np.mean(np.abs(secondary_change) > 1e-6):.9f}; "
        "finite numeric field, but not used as a transition proxy because it is "
        "nearly saturated on the non-secondary-defined land mask and changes rarely",
        flush=True,
    )

    primary_loss = np.clip(-np.diff(annual["luh2_primary_fraction"], axis=0), 0.0, None)
    gains = {
        key: np.clip(np.diff(annual[key], axis=0), 0.0, None)
        for key in (
            "luh2_cropland_fraction",
            "luh2_pasture_fraction",
            "luh2_rangeland_fraction",
            "luh2_urban_fraction",
        )
    }
    managed_gain = union_fraction(*gains.values())
    managed_loss = union_fraction(
        *(
            np.clip(-np.diff(annual[key], axis=0), 0.0, None)
            for key in (
                "luh2_cropland_fraction",
                "luh2_pasture_fraction",
                "luh2_rangeland_fraction",
            )
        )
    )
    pl = primary_loss[:, land]
    mg = managed_gain[:, land]
    ml = managed_loss[:, land]
    matched = smooth_overlap(pl, mg)
    primary_weight = float(np.sum(pl * tiled_area))
    print(
        "OVERLAP primary_loss_managed_gain_corr="
        f"{weighted_corr(pl, mg, tiled_area):.8f} "
        f"smoothly_matched_primary_loss_share="
        f"{float(np.sum(matched * tiled_area) / (primary_weight + 1e-30)):.8f} "
        f"managed_contraction_mean={weighted_mean(ml, tiled_area):.9f}; "
        "managed contraction is a net-abandonment proxy only, not a gross transition",
        flush=True,
    )
    for name, gain in gains.items():
        values = gain[:, land]
        print(
            f"DESTINATION {name} primary_loss_gain_corr="
            f"{weighted_corr(pl, values, tiled_area):.8f} "
            f"smoothly_matched_primary_loss_share="
            f"{float(np.sum(smooth_overlap(pl, values) * tiled_area) / (primary_weight + 1e-30)):.8f}",
            flush=True,
        )
    return land


def select_cells(
    reference_weight: np.ndarray,
    annual: Mapping[str, np.ndarray],
    observed_count: int = 768,
    transition_count: int = 512,
):
    primary_loss = np.clip(
        -np.diff(annual["luh2_primary_fraction"], axis=0), 0.0, None
    )
    activity = primary_loss.copy()
    for name in (
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
    ):
        activity += np.abs(np.diff(annual[name], axis=0))
    activity = activity.max(axis=0)
    observed_cells = np.argsort(reference_weight.ravel())[::-1][:observed_count]
    transition_cells = np.argsort(activity.ravel())[::-1][:transition_count]
    cells = np.unique(np.concatenate((observed_cells, transition_cells)))
    rows, cols = cells // 360, cells % 360
    observed_members = np.isin(cells, observed_cells)
    transition_members = activity.ravel()[cells] > 1e-3
    retained = float(reference_weight.ravel()[cells].sum() / reference_weight.sum())
    return rows, cols, observed_members, transition_members, retained


def annual_change(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.zeros_like(selected)
    change[1:] = selected[1:] - selected[:-1]
    return np.clip(change, 0.0, None), np.clip(-change, 0.0, None)


def finite_residue_release(
    source: np.ndarray, readiness: np.ndarray, residence_months: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    decay = np.exp(-1.0 / residence_months)
    stock = np.zeros(source.shape[1], dtype=np.float64)
    release = np.zeros_like(source)
    source_total = float(source.sum())
    decayed_total = 0.0
    released_total = 0.0
    for time in range(source.shape[0]):
        retained = stock * decay
        decayed_total += float(np.sum(stock - retained))
        stock = retained + source[time]
        release[time] = stock * np.clip(readiness[time], 0.0, 1.0)
        stock -= release[time]
        released_total += float(np.sum(release[time]))
    terminal = float(stock.sum())
    closure = source_total - released_total - decayed_total - terminal
    return release, (source_total, released_total, decayed_total, closure)


def annual_propensity_correlation(
    state: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    name: str,
) -> None:
    state_year = state.reshape(16, 12, -1).sum(axis=1)
    observed_year = np.average(
        observed.reshape(16, 12, -1), axis=1, weights=MONTH_DAYS[:12]
    )
    cell_state = state_year.mean(axis=0)
    cell_observed = observed_year.mean(axis=0)
    correlations = [weighted_corr(cell_state, cell_observed, weights)]
    correlations.extend(
        weighted_corr(cell_state[folds == fold], cell_observed[folds == fold], weights[folds == fold])
        for fold in range(4)
    )
    print(
        f"PROPENSITY {name} corr_all_and_folds="
        + ",".join(f"{value:+.8f}" for value in correlations),
        flush=True,
    )


def subset_summary(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    reference_weight: np.ndarray,
    mask: np.ndarray,
) -> str:
    pred_annual = np.average(prediction[:, mask], axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed[:, mask], axis=0, weights=MONTH_DAYS)
    log_rmse = np.sqrt(
        np.average(
            np.square(np.log((pred_annual + 1e-5) / (obs_annual + 1e-5))),
            weights=reference_weight[mask],
        )
    )
    ratio = np.sum(pred_annual * area[mask]) / np.sum(obs_annual * area[mask])
    pred_cycle = prediction[:, mask].reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed[:, mask].reshape(16, 12, -1).mean(axis=0)
    pred_anomaly = pred_cycle - pred_cycle.mean(axis=0, keepdims=True)
    obs_anomaly = obs_cycle - obs_cycle.mean(axis=0, keepdims=True)
    seasonal = np.sqrt(
        np.average(
            np.average(
                np.square(pred_anomaly - obs_anomaly),
                axis=0,
                weights=MONTH_DAYS[:12],
            ),
            weights=reference_weight[mask],
        )
    )
    return f"annual_log_rmse={log_rmse:.8f} raw_cycle_rmse={seasonal:.8f} area_ratio={ratio:.8f}"


def audit_candidate(
    name: str,
    candidate: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    reference_weight: np.ndarray,
    folds: np.ndarray,
    observed_members: np.ndarray,
    transition_members: np.ndarray,
) -> None:
    values, held = metrics(candidate, observed, area, reference_weight, folds)
    baseline_values, baseline_held = metrics(
        baseline, observed, area, reference_weight, folds
    )
    print(f"CANDIDATE {name} " + format_metrics(values), flush=True)
    print(
        f"DELTA {name} all="
        + ",".join(f"{value - base:+.8f}" for value, base in zip(values, baseline_values))
        + " held_annual_log_rmse="
        + ",".join(
            f"{value[1] - base[1]:+.8f}" for value, base in zip(held, baseline_held)
        ),
        flush=True,
    )
    print(
        f"SUBSETS {name} observed_top="
        f"[{subset_summary(candidate, observed, area, reference_weight, observed_members)}] "
        f"transition_active=[{subset_summary(candidate, observed, area, reference_weight, transition_members)}]",
        flush=True,
    )


def build_mechanisms(data: Mapping[str, np.ndarray]):
    primary = field(data, "luh2_primary_fraction")
    crop = field(data, "luh2_cropland_fraction")
    pasture = field(data, "luh2_pasture_fraction")
    rangeland = field(data, "luh2_rangeland_fraction")
    urban = field(data, "luh2_urban_fraction")
    crop_gain, _ = annual_change(crop)
    pasture_gain, _ = annual_change(pasture)
    range_gain, _ = annual_change(rangeland)
    urban_gain, _ = annual_change(urban)
    _, primary_loss = annual_change(primary)

    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    temperature = field(data, "air_temperature")
    lightning = np.clip(field(data, "lightning_flash_rate"), 0.0, None)
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)
    gpp12 = antecedent(gpp, 12.0)
    rain12 = antecedent(rain, 12.0)

    combustion = dryness / (dryness + 500.0)
    rain_exclusion = 1.0 / (1.0 + rain / 30.0)
    rain_deficit = np.clip((rain12 - rain) / (rain12 + rain + 10.0), 0.0, 1.0)
    thermal = rising(temperature, 12.0, 4.0)
    managed_access = union_fraction(crop, pasture, rangeland, urban)
    lightning_chance = lightning / (lightning + 0.02)
    ignition = union_fraction(lightning_chance, managed_access)
    readiness = combustion * rain_exclusion * (0.25 + 0.75 * rain_deficit) * thermal * ignition

    fine_fuel = gpp12 / (gpp12 + 0.35)
    woody_fuel = fine_fuel * canopy / (canopy + 10.0) * biomass / (biomass + 2.0)
    destination_gain = union_fraction(crop_gain, pasture_gain, range_gain, urban_gain)
    clearing_source = smooth_overlap(primary_loss, destination_gain) * woody_fuel
    clearing_release, clearing_budget = finite_residue_release(
        clearing_source, readiness, 18.0
    )

    forest_gate = primary * canopy / (canopy + 8.0)
    forest_range_gain = range_gain * forest_gate
    managed_source = union_fraction(crop_gain, pasture_gain, forest_range_gain)
    managed_source *= fine_fuel * (0.35 + 0.65 * woody_fuel)
    managed_release, managed_budget = finite_residue_release(
        managed_source, readiness, 12.0
    )

    expansion = union_fraction(crop_gain, urban_gain)
    fragmentation = np.zeros_like(expansion)
    stock = np.zeros(expansion.shape[1], dtype=np.float64)
    retention = np.exp(-1.0 / 60.0)
    for time in range(expansion.shape[0]):
        stock = np.clip(retention * stock + expansion[time], 0.0, 1.0)
        fragmentation[time] = stock
    surface_fuel = fine_fuel * union_fraction(pasture, rangeland, primary)
    fragmentation_brake = np.clip(fragmentation * surface_fuel, 0.0, 1.0)

    states = {
        "clearing_residue": clearing_release,
        "managed_conversion": managed_release,
        "fragmentation_change": fragmentation_brake,
    }
    budgets = {
        "clearing_residue": clearing_budget,
        "managed_conversion": managed_budget,
    }
    return states, budgets


def apply_mechanism(
    baseline: np.ndarray, name: str, state: np.ndarray
) -> np.ndarray:
    if name == "fragmentation_change":
        return np.clip(baseline * (1.0 - state), 0.0, 1.0)
    return np.clip(1.0 - (1.0 - baseline) * np.exp(-state), 0.0, 1.0)


def prefix_audit(data: Mapping[str, np.ndarray], states: Mapping[str, np.ndarray]) -> None:
    cutoff = 101
    perturbed = {name: np.asarray(values).copy() for name, values in data.items()}
    for name in LUH_NAMES:
        if name not in perturbed:
            continue
        perturbed[name][cutoff:] = np.clip(
            perturbed[name][cutoff:] * 0.31 + 0.17, 0.0, 1.0
        )
    altered, _ = build_mechanisms(perturbed)
    for name, state in states.items():
        error = float(np.max(np.abs(state[:cutoff] - altered[name][:cutoff])))
        print(f"PREFIX {name} max_pre_cutoff_difference={error:.12g}", flush=True)
        if error != 0.0:
            raise AssertionError(f"future perturbation leaked into {name}: {error}")


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    area_grid, _, reference_weight_grid = coarse_area_and_reference(evaluator)
    annual, repeat_error = read_annual_states()
    provenance_audit(annual, area_grid, repeat_error)
    rows, cols, observed_members, transition_members, retained = select_cells(
        reference_weight_grid, annual
    )
    area = area_grid[rows, cols]
    reference_weight = reference_weight_grid[rows, cols]
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"SAMPLE cells={rows.size} retained_reference_weight={retained:.8f} "
        f"observed_top={observed_members.sum()} transition_active={transition_members.sum()} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )

    model = load_model()
    names = tuple(dict.fromkeys((*model.INPUTS, *LUH_NAMES)))
    data = load_selected(names, rows, cols)
    prediction_data = {name: data[name] for name in model.INPUTS}
    baseline = np.asarray(
        model.predict(prediction_data, dict(model.PARAMS), None), dtype=np.float64
    )[:, 0, :]
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()
    baseline_values, _ = metrics(
        baseline, observed, area, reference_weight, folds
    )
    print("BASE " + format_metrics(baseline_values), flush=True)

    states, budgets = build_mechanisms(data)
    for name, budget in budgets.items():
        print(
            f"BUDGET {name} source={budget[0]:.9f} released={budget[1]:.9f} "
            f"decayed={budget[2]:.9f} closure={budget[3]:+.12g}",
            flush=True,
        )
    for name, state in states.items():
        annual_propensity_correlation(
            state, observed, reference_weight, folds, name
        )
        candidate = apply_mechanism(baseline, name, state)
        audit_candidate(
            name,
            candidate,
            baseline,
            observed,
            area,
            reference_weight,
            folds,
            observed_members,
            transition_members,
        )
    prefix_audit(data, states)
    print(
        "INTERPRETATION native transitions.nc is absent, so these equations use "
        "net state changes only. Direct timing from a raw monthly difference would "
        "be a January artifact; finite stock allows physically gated later release. "
        "A twelve-month difference would repeat and spend the same annual event twelve times.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
