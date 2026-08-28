"""Falsify a pointwise subgrid patch-cohort replacement for neighbour spread.

Historical high-score models benefited from cross-cell neighbour averaging and
fitted annual/seasonal surfaces. Neither is admissible in a coupled ED site.
This scratch experiment transfers only the physical spread hypothesis: event
extent grows when a local mosaic contains mature, connected surface fuel.

Three within-site surface-fuel cohorts recover and age with fixed global time
constants. Current combustion and ignition determine their burn opportunity;
incoming model hazard depletes mature cohorts back into the young cohort. The
resulting event-capacity factor is divided by its causal local twelve-month
reference, so it redistributes an existing surface hazard rather than adding a
target-calibrated annual source. No fitted coefficient, target value, future
input, coordinate, region, neighbour, or disallowed forcing enters the law.
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


PINNED = "121c83c"
MONTH_DAYS = np.tile(
    np.asarray(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
        dtype=np.float64,
    ),
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


def rising(values: np.ndarray, width: float, center: float) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(np.clip(-(values - center) / width, -50.0, 50.0))
    )


def patch_factor(data, incumbent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return causal mosaic factor and the affected surface-hazard share.

    For young, recovering, and mature patch fractions ``x=(x0,x1,x2)``, the
    monthly pre-fire update is

        x0 <- x0 + (1-exp(-1/6)) x0_capacity - transition_01
        x1 <- x1 + transition_01 - (1-exp(-1/18)) x1
        x2 <- x2 + transition_12.

    Capacity changes first scale or seed the local mosaic. Cohort maturity is
    ``M=(.05*x0+.40*x1+x2)/sum(x)``. Combustion ``C`` and ignition occupancy
    ``I`` give ``q=.2+C*I*M`` and the transfer factor

        F_t = q_t / EMA12(q)_t.

    Incoming surface hazard burns cohort ``a`` with probability
    ``1-exp(-12*h*s*C*m_a)`` and returns that patch to the young cohort. The
    candidate later applies ``h' = h*(1+w*s*(F-1))`` for a fixed small tile
    fraction ``w``.
    """
    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    dryness = np.clip(np.asarray(data["dryness"], dtype=np.float64), 0.0, None)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    range_ = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    urban = np.clip(
        np.asarray(data["luh2_urban_fraction"], dtype=np.float64), 0.0, 1.0
    )
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    secondary = np.clip(
        np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    secondary_canopy = np.clip(
        np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )

    gpp_3 = antecedent(gpp, 3.0)
    gpp_12 = antecedent(gpp, 12.0)
    lightning_12 = antecedent(lightning, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)
    curing = curing / (curing + 0.05)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    open_cover = np.clip(
        range_
        + pasture
        + natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0),
        0.0,
        2.0,
    )
    surface_capacity = np.clip(
        (1.0 - crop) * fine_fuel * open_cover * continuity,
        0.0,
        1.0,
    )
    woody_capacity = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    surface_share = surface_capacity / (
        0.05 + surface_capacity + woody_capacity + crop_capacity
    )

    combustion = (
        dryness
        / (dryness + 250.0)
        / (1.0 + rain / 35.0)
        * rising(temperature, 3.0, 5.0)
        * (0.4 + 0.6 * curing)
    )
    managed = np.clip(range_ + pasture + crop, 0.0, 1.0)
    ignition = 1.0 - np.exp(
        -np.clip(lightning_12 / 0.02 + managed / 0.15, 0.0, 50.0)
    )

    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    maturity_weights = np.asarray((0.05, 0.40, 1.00), dtype=np.float64)
    transition_01_rate = 1.0 - np.exp(-1.0 / 6.0)
    transition_12_rate = 1.0 - np.exp(-1.0 / 18.0)
    cohorts = np.zeros((3,) + incumbent.shape[1:], dtype=np.float64)
    cohorts[2] = surface_capacity[0]
    reference = np.zeros_like(incumbent[0], dtype=np.float64)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    factor = np.empty_like(incumbent, dtype=np.float64)

    for time in range(incumbent.shape[0]):
        target = surface_capacity[time]
        total = cohorts.sum(axis=0)
        contraction = np.minimum(1.0, target / (total + 1e-12))
        cohorts *= contraction[None, ...]
        cohorts[0] += np.maximum(target - cohorts.sum(axis=0), 0.0)

        transition_01 = transition_01_rate * cohorts[0]
        transition_12 = transition_12_rate * cohorts[1]
        cohorts[0] -= transition_01
        cohorts[1] += transition_01 - transition_12
        cohorts[2] += transition_12

        maturity = np.sum(maturity_weights[:, None, None] * cohorts, axis=0) / (
            cohorts.sum(axis=0) + 1e-12
        )
        opportunity = 0.2 + combustion[time] * ignition[time] * maturity
        if time == 0:
            reference[...] = opportunity
        else:
            reference += alpha_12 * (opportunity - reference)
        factor[time] = np.clip(opportunity / (reference + 1e-12), 0.5, 2.0)

        burn_probability = 1.0 - np.exp(
            -12.0
            * hazard[time][None, ...]
            * surface_share[time][None, ...]
            * combustion[time][None, ...]
            * maturity_weights[:, None, None]
        )
        burned = cohorts * burn_probability
        cohorts -= burned
        cohorts[0] += burned.sum(axis=0)

    return factor, surface_share


def candidate(
    incumbent: np.ndarray,
    factor: np.ndarray,
    surface_share: np.ndarray,
    tile_fraction: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    multiplier = 1.0 + tile_fraction * surface_share * (factor - 1.0)
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * multiplier, 0.0, 50.0)),
        dtype=np.float32,
    )


def held_losses(prediction, observed, area_cell, obs_annual, folds):
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    weight = area_cell * obs_annual
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_allocation = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-12)
    pred_allocation = pred_cycle / (
        pred_cycle.sum(axis=0, keepdims=True) + 1e-12
    )
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
            np.sum(
                held_weight[None, :]
                * np.abs(
                    obs_allocation[:, held] - pred_allocation[:, held]
                )
            )
            / denominator
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


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual_grid = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_annual_grid = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area * obs_annual_grid
    excess_weight = area * np.maximum(pred_annual_grid - obs_annual_grid, 0.0)

    def top(weight):
        ranking = np.argsort(weight.ravel())[::-1]
        coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
        return ranking[: int(np.searchsorted(coverage, 0.90) + 1)]

    cells = np.union1d(top(observed_weight), top(excess_weight))
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    selected_data = {
        name: np.asarray(values[:, rows, cols], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    selected_incumbent = np.asarray(incumbent[:, rows, cols], dtype=np.float64)[
        :, None, :
    ]
    selected_observed = np.asarray(observed[:, rows, cols], dtype=np.float64)
    area_cell = area[rows, cols]
    obs_annual = obs_annual_grid[rows, cols]
    factor, surface_share = patch_factor(selected_data, selected_incumbent)
    factor = factor[:, 0, :]
    surface_share = surface_share[:, 0, :]
    selected_incumbent = selected_incumbent[:, 0, :]

    base = held_losses(
        selected_incumbent,
        selected_observed,
        area_cell,
        obs_annual,
        folds,
    )
    base_global = evaluator.score(incumbent)["global"]
    print(
        f"BASE overall={base_global['overall_score']:.9f} cells={len(cells)} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}"
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.6f}" for value in base[0])
        + " allocation=" + ",".join(f"{value:.6f}" for value in base[1])
        + " raw_cycle=" + ",".join(f"{value:.6f}" for value in base[2])
    )

    any_survivor = False
    for tile_fraction in (0.05, 0.10, 0.20, 0.35):
        trial = candidate(
            selected_incumbent,
            factor,
            surface_share,
            tile_fraction,
        )
        losses = held_losses(
            trial,
            selected_observed,
            area_cell,
            obs_annual,
            folds,
        )
        gains = tuple(base[index] - losses[index] for index in range(3))
        survivor = bool(
            np.all(gains[2] > 0.0)
            and np.count_nonzero(gains[0] > 0.0) >= 3
            and np.count_nonzero(gains[1] > 0.0) >= 3
            and gains[0].sum() > 0.0
            and gains[1].sum() > 0.0
        )
        any_survivor |= survivor
        print(
            f"tile_fraction={tile_fraction:.2f} held={survivor} annual_gain="
            + ",".join(f"{value:+.6f}" for value in gains[0])
            + " allocation_gain="
            + ",".join(f"{value:+.6f}" for value in gains[1])
            + " raw_cycle_gain="
            + ",".join(f"{value:+.6f}" for value in gains[2])
        )

    prefix_cells = cells[:64]
    prefix_rows, prefix_cols = prefix_cells // 360, prefix_cells % 360
    prefix_data = {
        name: np.asarray(values[:, prefix_rows, prefix_cols])[:, None, :]
        for name, values in data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    prefix_factor, prefix_share = patch_factor(prefix_data, prefix_incumbent)
    prefix_trial = candidate(prefix_incumbent, prefix_factor, prefix_share, 0.20)
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    perturbed_incumbent = model.predict(perturbed, dict(model.PARAMS), None)
    perturbed_factor, perturbed_share = patch_factor(
        perturbed, perturbed_incumbent
    )
    perturbed_trial = candidate(
        perturbed_incumbent,
        perturbed_factor,
        perturbed_share,
        0.20,
    )
    print(
        "PREFIX max_abs="
        f"{np.max(np.abs(prefix_trial[:96]-perturbed_trial[:96])):.12g}"
    )
    if not any_survivor:
        print("EXACT skipped: no stable held survivor")


if __name__ == "__main__":
    main()
