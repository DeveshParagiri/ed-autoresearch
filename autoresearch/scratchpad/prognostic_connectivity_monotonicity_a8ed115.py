"""Held-cell falsifier for a prognostic burnable-patch connectivity state.

The target is used only after the state has been constructed.  ``rho`` is a
pointwise, prefix-causal fraction of connected surface fuel.  It recovers
toward a cover- and trailing-GPP-supported capacity and is depleted by a fixed
input-only proxy for realised fire pressure.  The test asks whether GFED5
burned area per crude ignition opportunity rises with effective burnable
connectivity in held spatial blocks.  It is a scratch diagnostic, not a fitted
runtime surface and not an official candidate.
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
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask  # noqa: E402

PINNED = "a8ed115"
MONTHS = 192


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


def antecedent(model, values, months):
    return np.asarray(
        model._antecedent(
            np.asarray(values, dtype=np.float32),
            1.0 - np.exp(-1.0 / float(months)),
        ),
        dtype=np.float32,
    )


def connectivity_drivers(model, data):
    """Return capacity, crude ignition, and current burnability.

    Capacity contains no fire-weather field: it is the connected surface-fuel
    fraction supported by current cover and trailing production.  Ignition is
    the union of a finite lightning-patch occupancy and managed access.
    Burnability is kept separate so the tested state means connected *burnable*
    patches rather than productive vegetation in general.
    """
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0).astype(np.float32)
    secondary = np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0).astype(np.float32)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0).astype(np.float32)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0).astype(np.float32)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0).astype(np.float32)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0).astype(np.float32)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None).astype(np.float32)
    secondary_canopy = np.clip(data["secondary_canopy_height"], 0.0, None).astype(np.float32)

    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    managed = np.clip(rangeland + pasture + crop, 0.0, 1.0)
    carrier = np.clip(
        natural_open + secondary_open + rangeland + pasture + 0.20 * crop,
        0.0,
        1.0,
    )
    fragmentation = np.exp(-1.5 * crop**1.5 - 5.0 * urban)
    gpp = np.clip(data["gpp"], 0.0, None).astype(np.float32)
    gpp_12 = antecedent(model, gpp, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    capacity = np.asarray(
        np.clip(carrier * fragmentation * np.sqrt(fine_fuel), 0.0, 1.0),
        dtype=np.float32,
    )

    lightning = np.clip(data["lightning_flash_rate"], 0.0, None).astype(np.float32)
    lightning_occupancy = 1.0 - np.exp(-60.0 * lightning)
    managed_access = managed / (managed + 0.10)
    ignition = np.asarray(
        1.0 - (1.0 - lightning_occupancy) * (1.0 - managed_access),
        dtype=np.float32,
    )

    rain = np.clip(data["monthly_precipitation"], 0.0, None).astype(np.float32)
    rain_6 = antecedent(model, rain, 6.0)
    rain_deficit = np.maximum(
        (rain_6 - rain) / (rain_6 + rain + 10.0), 0.0
    )
    dryness = np.clip(data["dryness"], 0.0, None).astype(np.float32)
    combustion = dryness / (dryness + 500.0)
    burnability = np.asarray(
        np.sqrt(np.clip(combustion * (0.20 + 0.80 * rain_deficit), 0.0, 1.0)),
        dtype=np.float32,
    )
    return capacity, ignition, burnability


def prognostic_rho(capacity, ignition, burnability, recovery_months, depletion):
    """Evolve input-only connected fuel and return its pre-fire burnable state."""
    recovery = 1.0 - np.exp(-1.0 / float(recovery_months))
    state = np.asarray(capacity[0], dtype=np.float64).copy()
    effective = np.empty_like(capacity, dtype=np.float32)
    for time in range(capacity.shape[0]):
        state += recovery * (capacity[time] - state)
        np.clip(state, 0.0, 1.0, out=state)
        effective[time] = state * burnability[time]
        # This fixed Poisson pressure is deliberately not calibrated to GFED5.
        # It is only the no-target falsifier for post-fire network breakage.
        disturbed = 1.0 - np.exp(
            -float(depletion) * ignition[time] * burnability[time]
        )
        state *= 1.0 - disturbed
    return effective


def weighted_slope_and_corr(x, y, weight):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0.0)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    weight = np.asarray(weight[valid], dtype=np.float64)
    if x.size < 3 or weight.sum() <= 0.0:
        return float("nan"), float("nan")
    xbar = np.average(x, weights=weight)
    ybar = np.average(y, weights=weight)
    dx = x - xbar
    dy = y - ybar
    covariance = np.sum(weight * dx * dy)
    variance_x = np.sum(weight * dx * dx)
    variance_y = np.sum(weight * dy * dy)
    slope = covariance / max(variance_x, 1e-18)
    corr = covariance / np.sqrt(max(variance_x * variance_y, 1e-36))
    return float(slope), float(corr)


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    weights = np.asarray(weights[order], dtype=np.float64)
    cumulative = np.cumsum(weights)
    if not values.size or cumulative[-1] <= 0.0:
        return np.full(len(probabilities), np.nan)
    positions = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(probabilities, positions, values)


def held_fold_report(rho, ignition, observation, area, land):
    rows, cols = np.where(land)
    cell_fold = ((rows // 15) + 3 * (cols // 15)) % 4
    ncell = len(rows)
    rho_cell = rho[:, rows, cols]
    ignition_cell = ignition[:, rows, cols]
    observation_cell = observation[:, rows, cols]
    area_cell = area[rows, cols]

    opportunity_sum = ignition_cell.sum(axis=0, dtype=np.float64)
    observed_sum = observation_cell.sum(axis=0, dtype=np.float64)
    rho_bar = np.sum(rho_cell * ignition_cell, axis=0, dtype=np.float64) / (
        opportunity_sum + 1e-12
    )
    event_proxy = observed_sum / (opportunity_sum + 1e-12)
    cell_weight = area_cell * opportunity_sum
    cell_valid = opportunity_sum > 0.12

    fold_rows = []
    for fold in range(4):
        train = cell_valid & (cell_fold != fold)
        held = cell_valid & (cell_fold == fold)
        edges = weighted_quantile(
            rho_bar[train], cell_weight[train], (0.2, 0.4, 0.6, 0.8)
        )
        bins = np.digitize(rho_bar[held], edges)
        bin_means = []
        for index in range(5):
            chosen = held.copy()
            chosen[held] = bins == index
            if not np.any(chosen):
                bin_means.append(float("nan"))
                continue
            bin_means.append(
                float(np.average(event_proxy[chosen], weights=cell_weight[chosen]))
            )
        annual_slope, annual_corr = weighted_slope_and_corr(
            rho_bar[held], event_proxy[held], cell_weight[held]
        )

        monthly_fold = np.broadcast_to(cell_fold[None, :], rho_cell.shape) == fold
        monthly_valid = monthly_fold & (ignition_cell > 1e-4)
        monthly_y = observation_cell / (ignition_cell + 1e-12)
        monthly_weight = area_cell[None, :] * ignition_cell
        monthly_slope, monthly_corr = weighted_slope_and_corr(
            rho_cell[monthly_valid],
            monthly_y[monthly_valid],
            monthly_weight[monthly_valid],
        )
        strict_monotone = bool(
            np.all(np.diff(np.asarray(bin_means, dtype=np.float64)) >= 0.0)
        )
        fold_rows.append((annual_slope, monthly_slope, strict_monotone))
        print(
            f"fold={fold} cells={int(held.sum())} "
            f"annual_slope={annual_slope:+.8g} annual_corr={annual_corr:+.5f} "
            f"monthly_slope={monthly_slope:+.8g} monthly_corr={monthly_corr:+.5f} "
            f"strict_monotone={int(strict_monotone)} "
            + "bins="
            + ",".join(f"{value:.7g}" for value in bin_means),
            flush=True,
        )
    annual_positive = sum(row[0] > 0.0 for row in fold_rows)
    monthly_positive = sum(row[1] > 0.0 for row in fold_rows)
    monotone = sum(row[2] for row in fold_rows)
    print(
        f"signs annual={annual_positive}/4 monthly={monthly_positive}/4 "
        f"strict_monotone={monotone}/4 "
        f"pass={int(annual_positive >= 3 and monthly_positive >= 3 and monotone >= 3)}",
        flush=True,
    )
    return annual_positive, monthly_positive, monotone, rho_bar, event_proxy, cell_weight, cell_fold


def ecological_masks(model, data, land):
    def mean(name):
        return np.asarray(data[name], dtype=np.float64).reshape(16, 12, 180, 360).mean(axis=(0, 1))

    rain = 12.0 * antecedent(model, np.clip(data["monthly_precipitation"], 0.0, None), 12.0)
    annual_rain = rain.reshape(16, 12, 180, 360).mean(axis=(0, 1))
    temperature = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    return {
        "intact_tropical_closed": land & (temperature >= 20.0) & (annual_rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": land & (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": land & (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": land & (temperature >= 20.0) & (annual_rain >= 500.0) & (annual_rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": land & (rangeland >= 0.4) & (annual_rain >= 250.0) & (annual_rain < 1500.0) & (biomass >= 0.2),
        "cropland": land & (crop >= 0.5),
        "arid_low_fuel": land & (annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecology_report(masks, land, rho_bar, event_proxy, cell_weight):
    rows, cols = np.where(land)
    print("ECOLOGY primary_r12_d0.5", flush=True)
    for name, mask in masks.items():
        chosen = mask[rows, cols] & (cell_weight > 0.0)
        slope, corr = weighted_slope_and_corr(
            rho_bar[chosen], event_proxy[chosen], cell_weight[chosen]
        )
        print(
            f"{name} cells={int(chosen.sum())} slope={slope:+.8g} "
            f"corr={corr:+.5f} rho={np.average(rho_bar[chosen], weights=cell_weight[chosen]):.5f} "
            f"ba_per_opp={np.average(event_proxy[chosen], weights=cell_weight[chosen]):.7g}",
            flush=True,
        )


def main():
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    capacity, ignition, burnability = connectivity_drivers(model, data)
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    with Dataset(GFED5_PATH) as dataset:
        raw = np.asarray(dataset.variables["burntArea"][:MONTHS])
    observation = np.asarray(
        raw.reshape(MONTHS, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0,
        dtype=np.float32,
    )

    print(
        "capacity_quantiles="
        + ",".join(f"{value:.5f}" for value in np.quantile(capacity[:, land], (0.1, 0.5, 0.9, 0.99)))
        + " ignition_quantiles="
        + ",".join(f"{value:.5f}" for value in np.quantile(ignition[:, land], (0.1, 0.5, 0.9, 0.99))),
        flush=True,
    )
    primary = None
    summaries = []
    for recovery_months, depletion in (
        (6.0, 0.25),
        (12.0, 0.25),
        (12.0, 0.50),
        (12.0, 1.00),
        (24.0, 0.50),
    ):
        print(f"CONFIG recovery={recovery_months:g} depletion={depletion:g}", flush=True)
        rho = prognostic_rho(
            capacity, ignition, burnability, recovery_months, depletion
        )
        result = held_fold_report(rho, ignition, observation, area, land)
        summaries.append(
            (recovery_months, depletion, result[0], result[1], result[2])
        )
        if recovery_months == 12.0 and depletion == 0.50:
            primary = result[3:6]
    print(
        "SUMMARY "
        + ";".join(
            f"r{recovery:g}_d{depletion:g}:annual{annual}/4_monthly{monthly}/4_monotone{monotone}/4"
            for recovery, depletion, annual, monthly, monotone in summaries
        ),
        flush=True,
    )
    assert primary is not None
    ecology_report(
        ecological_masks(model, data, land),
        land,
        primary[0],
        primary[1],
        primary[2],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
