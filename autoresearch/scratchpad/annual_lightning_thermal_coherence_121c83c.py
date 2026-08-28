"""Held test of a lightning-by-thermal-coherence annual capacity.

The deeper reverse-ML audit at ``121c83c`` found the same signed annual
interaction in all four whole-cell folds: trailing lightning is associated
with underburn only when trailing temperature variability is low, while high
temperature variability with little lightning is associated with overburn.
This script translates that shape once, without copying a learned surface::

    I = EMA12(lightning) / (EMA12(lightning) + 0.02)
    V = sigma12(temperature) / (sigma12(temperature) + 4)
    M = (1 + k S I (1 - V)) exp(-k S V (1 - I))

``S`` is natural/open fine-fuel support.  The first term lets lightning expand
event footprint only through a coherent thermal season; the second suppresses
continental event footprint when natural ignition is insufficient.  Every
constant is a fixed physical saturation already used in the incumbent family,
and ``k`` is selected only from a small declared bracket.

The equation is globally shared, point-local, and prefix causal.  Coordinates
assign held blocks only.  GFED enters losses only after prediction.  Exact
full-grid scoring is forbidden unless annual-log, normalized-allocation, and
raw-cycle losses all improve in every held block.
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
from autoresearch.scratchpad.rain_fuel_pathway_probe import (  # noqa: E402
    ecological_ratios,
)
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
STRENGTHS = (0.05, 0.10, 0.20, 0.40)
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
    module = types.ModuleType(f"model_{PINNED}_lightning_thermal_coherence")
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


def capacity_terms(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_12 = antecedent(lightning, 12.0)
    ignition = lightning_12 / (lightning_12 + 0.02)

    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_12 = antecedent(temperature, 12.0)
    temperature_second_12 = antecedent(np.square(temperature), 12.0)
    temperature_sigma_12 = np.sqrt(
        np.maximum(temperature_second_12 - np.square(temperature_12), 0.0)
    )
    variability = temperature_sigma_12 / (temperature_sigma_12 + 4.0)

    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = antecedent(gpp, 12.0)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    secondary = np.clip(
        np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    natural_height = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    secondary_height = np.clip(
        np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    urban = np.clip(
        np.asarray(data["luh2_urban_fraction"], dtype=np.float64), 0.0, 1.0
    )
    natural_open = np.clip(
        natural * 8.0 / (natural_height + 8.0)
        + secondary * 8.0 / (secondary_height + 8.0),
        0.0,
        1.0,
    )
    continuity = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    support = np.clip(natural_open * fine_fuel * continuity, 0.0, 1.0)

    coherent_ignition = support * ignition * (1.0 - variability)
    continental_without_ignition = support * variability * (1.0 - ignition)
    return (
        coherent_ignition,
        continental_without_ignition,
        support,
        ignition,
        variability,
    )


def candidate(
    incumbent: np.ndarray,
    coherent_ignition: np.ndarray,
    continental_without_ignition: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    multiplier = (
        1.0 + strength * coherent_ignition
    ) * np.exp(-strength * continental_without_ignition)
    adjusted = hazard * np.clip(multiplier, 0.5, 1.5)
    return np.asarray(-np.expm1(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def held_losses(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    observed_annual: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    weight = area * observed_annual
    observed_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    predicted_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    observed_allocation = observed_cycle / (
        observed_cycle.sum(axis=0, keepdims=True) + 1e-12
    )
    predicted_allocation = predicted_cycle / (
        predicted_cycle.sum(axis=0, keepdims=True) + 1e-12
    )
    annual: list[float] = []
    allocation: list[float] = []
    raw_cycle: list[float] = []
    for fold in range(4):
        held = folds == fold
        held_weight = weight[held]
        denominator = np.sum(held_weight) + 1e-15
        annual.append(
            np.sqrt(
                np.sum(
                    held_weight
                    * np.square(
                        np.log(observed_annual[held] + 1e-5)
                        - np.log(predicted_annual[held] + 1e-5)
                    )
                )
                / denominator
            )
        )
        allocation.append(
            np.sqrt(
                np.sum(
                    held_weight[None, :]
                    * np.square(
                        observed_allocation[:, held] - predicted_allocation[:, held]
                    )
                )
                / (12.0 * denominator)
            )
        )
        raw_cycle.append(
            np.sqrt(
                np.sum(
                    held_weight[None, :]
                    * np.square(observed_cycle[:, held] - predicted_cycle[:, held])
                )
                / (12.0 * denominator)
            )
        )
    return np.asarray(annual), np.asarray(allocation), np.asarray(raw_cycle)


def main() -> int:
    model = load_pinned()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(
            f"current model blob {current_blob} differs from pinned {EXPECTED_MODEL_BLOB}"
        )

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_scores = evaluator.score(incumbent)
    incumbent_global = incumbent_scores["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(
            f"incumbent drift {incumbent_global['overall_score']:.9f}"
        )

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observed_annual_grid = np.average(observed, axis=0, weights=MONTH_DAYS)
    predicted_annual_grid = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area_grid * observed_annual_grid
    excess_weight = area_grid * np.maximum(
        predicted_annual_grid - observed_annual_grid, 0.0
    )

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
    selected_observed_annual = observed_annual_grid[rows, columns]
    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        selected_observed_annual,
        folds,
    )
    terms = capacity_terms(selected_data)
    coherent_ignition = terms[0][:, 0, :]
    continental_without_ignition = terms[1][:, 0, :]
    support = terms[2][:, 0, :]
    ignition = terms[3][:, 0, :]
    variability = terms[4][:, 0, :]

    print(
        f"BASE pinned={PINNED} model_blob={current_blob} "
        f"overall={incumbent_global['overall_score']:.9f} cells={cells.size} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}",
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )
    print(
        "STATE "
        f"support_mean={support.mean():.9f} support_p95={np.quantile(support, .95):.9f} "
        f"ignition_mean={ignition.mean():.9f} ignition_p95={np.quantile(ignition, .95):.9f} "
        f"variability_mean={variability.mean():.9f} variability_p95={np.quantile(variability, .95):.9f} "
        f"coherent_ignition_mean={coherent_ignition.mean():.9f} "
        f"continental_brake_mean={continental_without_ignition.mean():.9f}",
        flush=True,
    )

    survivors: list[tuple[float, float, np.ndarray]] = []
    best: tuple[float, float, np.ndarray] | None = None
    for strength in STRENGTHS:
        trial = candidate(
            selected_incumbent,
            coherent_ignition,
            continental_without_ignition,
            strength,
        )
        losses = held_losses(
            trial,
            selected_observed,
            selected_area,
            selected_observed_annual,
            folds,
        )
        gains = tuple(base_losses[index] - losses[index] for index in range(3))
        stable = bool(all(np.all(gain > 0.0) for gain in gains))
        aggregate = float(sum(gain.sum() for gain in gains))
        print(
            f"BRACKET strength={strength:.2f} stable={int(stable)} annual_gain="
            + ",".join(f"{value:+.9f}" for value in gains[0])
            + " allocation_gain="
            + ",".join(f"{value:+.9f}" for value in gains[1])
            + " raw_cycle_gain="
            + ",".join(f"{value:+.9f}" for value in gains[2]),
            flush=True,
        )
        if best is None or aggregate > best[0]:
            best = (aggregate, strength, trial)
        if stable:
            survivors.append((aggregate, strength, trial))

    assert best is not None
    base_ecology = ecological_ratios_selected(
        selected_incumbent,
        selected_observed,
        selected_data,
        selected_area,
    )
    best_ecology = ecological_ratios_selected(
        best[2],
        selected_observed,
        selected_data,
        selected_area,
    )
    print(
        f"HELD_ECOLOGY best_strength={best[1]:.2f} "
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
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    prefix_terms = capacity_terms(prefix_data)
    before = candidate(prefix_incumbent, prefix_terms[0], prefix_terms[1], best[1])
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)
    changed_terms = capacity_terms(changed)
    after = candidate(
        changed_incumbent, changed_terms[0], changed_terms[1], best[1]
    )
    print(
        f"PREFIX best_strength={best[1]:.2f} "
        f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}",
        flush=True,
    )

    if not survivors:
        print(
            "DECISION exact=0 reject=no_all_block_all_metric_survivor",
            flush=True,
        )
        return 0

    survivors.sort(reverse=True)
    _, strength, _ = survivors[0]
    full_terms = capacity_terms(data)
    trial = validate_prediction(
        candidate(incumbent, full_terms[0], full_terms[1], strength)
    )
    trial_scores = evaluator.score(trial)
    trial_global = trial_scores["global"]
    print(f"DECISION exact=1 strength={strength:.2f}", flush=True)
    print(
        f"EXACT overall={trial_global['overall_score']:.9f} "
        f"delta={trial_global['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"bias={trial_global['bias_score']:.9f} rmse={trial_global['rmse_score']:.9f} "
        f"seasonal={trial_global['seasonal_cycle_score']:.9f} "
        f"spatial={trial_global['spatial_distribution_score']:.9f}",
        flush=True,
    )
    land = load_land_mask()
    base_full_ecology = ecological_ratios(
        incumbent, data, observed, area_grid, land
    )
    trial_full_ecology = ecological_ratios(trial, data, observed, area_grid, land)
    print(
        "ECOLOGY "
        + ",".join(
            f"{name}:{base_full_ecology[name]:.5f}->{trial_full_ecology[name]:.5f}"
            for name in base_full_ecology
        ),
        flush=True,
    )
    print(
        "REGIONS "
        + ",".join(
            f"{name}:{trial_scores[name]['overall_score']-incumbent_scores[name]['overall_score']:+.6f}"
            for name in sorted(key for key in trial_scores if key != "global")
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
