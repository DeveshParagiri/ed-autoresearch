"""Held-block falsification of compound fire-weather opportunity duration.

The incumbent represents instantaneous ignition and combustion, their signed
arrival order, and separate marginal variability terms. It does not represent
the effective number of distinct compound ignition-ready months in a causal
annual window. This diagnostic asks whether a concentrated compound window can
support a larger natural-fuel event footprint than the same summed opportunity
spread diffusely across the year.

The candidate state is globally shared, point-local, prefix-causal, and uses
only current valid inputs. GFED5 and coordinates are used after prediction only
to compute four held-block losses. No observation-derived field or coordinate
enters the state or candidate equation.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402
from secondary_regrowth_footprint_33ac854 import MONTH_DAYS, losses  # noqa: E402


PINNED = "121c83c"


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


def trailing_sum(values: np.ndarray) -> np.ndarray:
    accumulator = np.zeros_like(values[0], dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        accumulator += values[time]
        if time >= 12:
            accumulator -= values[time - 12]
        output[time] = accumulator
    return output


def rising(values: np.ndarray, scale: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def compound_duration_state(data: dict[str, np.ndarray]) -> np.ndarray:
    rain = np.clip(np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None)
    dryness = np.clip(np.asarray(data["dryness"], dtype=np.float64), 0.0, None)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    lightning = np.clip(np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)

    ignition = lightning / (lightning + 0.02)
    combustion = (
        dryness / (dryness + 250.0)
        * 1.0 / (1.0 + rain / 35.0)
        * rising(temperature, 3.0, 5.0)
    )
    opportunity = np.clip(ignition * combustion, 0.0, 1.0)
    total = trailing_sum(opportunity)
    square_total = trailing_sum(np.square(opportunity))
    effective_months = np.clip(np.square(total) / (square_total + 1e-8), 0.0, 12.0)
    coverage = 1.0 - np.exp(-total)
    concentration = 1.0 - effective_months / 12.0

    gpp_12 = antecedent(gpp, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(data["luh2_urban_fraction"], 0.0, 1.0)
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(data["secondary_vegetation_fraction"], 0.0, 1.0)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(data["secondary_canopy_height"], 0.0, None)
    biomass = np.clip(data["aboveground_biomass"], 0.0, None)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    open_natural = np.clip(
        rangeland
        + pasture
        + natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0),
        0.0,
        2.0,
    )
    surface = (1.0 - crop) * fine_fuel * open_natural * continuity
    woody = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    natural_share = (surface + woody) / (0.05 + surface + woody + crop_capacity)
    return np.asarray(
        np.clip(natural_share * coverage * concentration, 0.0, 1.0),
        dtype=np.float32,
    )


def candidate(incumbent: np.ndarray, state: np.ndarray, strength: float) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    factor = np.exp(np.clip(strength * state, -0.5, 0.5))
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * factor, 0.0, 50.0)), dtype=np.float32
    )


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    state = compound_duration_state(data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_score = evaluator.score(incumbent)["global"]

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_ann = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_ann = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area * obs_ann
    excess_weight = area * np.maximum(pred_ann - obs_ann, 0.0)

    def top(weight: np.ndarray) -> np.ndarray:
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, 0.90) + 1)]

    cells = np.union1d(top(observed_weight), top(excess_weight))
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    base_annual, base_cycle = losses(incumbent, observed, area, cells, folds)
    print(
        f"BASE pinned={PINNED} overall={base_score['overall_score']:.9f} "
        f"cells={len(cells)} observed_coverage="
        f"{observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}",
        flush=True,
    )
    print(
        f"STATE weighted_mean={np.average(state[:, rows, cols], weights=np.broadcast_to(observed_weight[rows, cols], (192, len(cells)))):.9f} "
        f"p95={np.quantile(state[:, rows, cols], 0.95):.9f}",
        flush=True,
    )

    selected = {name: values[:, rows, cols].copy() for name, values in data.items()}
    changed = {name: values.copy() for name, values in selected.items()}
    for values in changed.values():
        values[96:] *= 0.5
    prefix_delta = float(
        np.max(
            np.abs(
                compound_duration_state(selected)[:96]
                - compound_duration_state(changed)[:96]
            )
        )
    )
    print(f"PREFIX future_half_after=96 state_max_abs={prefix_delta:.12g}", flush=True)

    survivors: list[tuple[float, np.ndarray]] = []
    for strength in (0.05, 0.10, 0.20, 0.40):
        trial = candidate(incumbent, state, strength)
        annual, cycle = losses(trial, observed, area, cells, folds)
        annual_gain = base_annual - annual
        cycle_gain = base_cycle - cycle
        held = bool(
            np.all(annual_gain > 0.0)
            and cycle_gain.sum() >= -0.05 * annual_gain.sum()
        )
        print(
            f"BRACKET strength={strength:g} held={int(held)} annual_gain="
            + ",".join(f"{value:+.9f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.9f}" for value in cycle_gain),
            flush=True,
        )
        if held:
            survivors.append((strength, trial))

    if not survivors:
        print("DECISION reject=no_fixed_bracket_clears_four_block_gate", flush=True)
        return
    for strength, trial in survivors:
        score = evaluator.score(validate_prediction(trial))["global"]
        print(
            f"EXACT strength={strength:g} overall={score['overall_score']:.9f} "
            f"delta={score['overall_score']-base_score['overall_score']:+.9f} "
            f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
            f"seasonal={score['seasonal_cycle_score']:.9f} "
            f"spatial={score['spatial_distribution_score']:.9f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
