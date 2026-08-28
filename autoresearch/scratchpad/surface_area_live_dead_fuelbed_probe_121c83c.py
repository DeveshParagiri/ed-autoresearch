"""Held-fold probe of a surface-area-weighted live/dead fine-fuel bed.

This scratch mechanism uses LAI in its native units of one-sided leaf area per
ground area.  It never infers fuel mass from GPP or biomass.  A globally shared
leaf turnover law transfers exposed live area into a dead-surface pool, and
live and dead moisture responses are combined by their fractional exposed
surface areas.  This is deliberately distinct from the mass-weighted
Rothermel and live/dead litter probes.

Coordinates define held spatial folds only and GFED enters losses only.  No
geographic quantity, region label, observation, fitted threshold, or future
summary enters the prediction equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.additive_pathway_replacement_a8ed115 import (  # noqa: E402
    ecological_ratios_selected,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    MONTH_DAYS,
    held_losses,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


STRENGTHS = (0.05, 0.10, 0.20)
FORMULATIONS = ("signed_surface_contrast", "relative_reactive_area")


@dataclass
class SurfaceBed:
    live_area: np.ndarray
    dead_area: np.ndarray
    dead_ready_area: np.ndarray
    live_green_area: np.ndarray
    reactive_area: np.ndarray
    support: np.ndarray
    closure: float


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -50.0, 50.0)))


def field(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    values = np.asarray(data[name], dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        return values[:, 0, :]
    return values


def surface_bed(data: dict[str, np.ndarray]) -> SurfaceBed:
    """Diagnose live area and integrate a dead fine-fuel surface pool.

    LAI is the live one-sided surface area.  Every month, a fixed six-month
    turnover fraction plus a bounded phenological decline enters the dead
    pool.  Dead surface has a nine-month reference half-life, shortened by wet
    warmth.  Those constants are shared globally and never fitted by region.
    """
    lai = np.clip(field(data, "leaf_area_index"), 0.0, None)
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    natural = np.clip(field(data, "natural_vegetation_fraction"), 0.0, 1.0)
    secondary = np.clip(field(data, "secondary_vegetation_fraction"), 0.0, 1.0)
    natural_height = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    secondary_height = np.clip(field(data, "secondary_canopy_height"), 0.0, None)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field(data, "luh2_urban_fraction"), 0.0, 1.0)

    open_natural = (
        natural * 8.0 / (natural_height + 8.0)
        + secondary * 8.0 / (secondary_height + 8.0)
    )
    support = np.clip(
        (1.0 - urban) * (open_natural + rangeland + pasture + crop), 0.0, 1.0
    )
    live_area = lai * support
    live_area_3 = antecedent(live_area, 3.0)
    decline = np.maximum(live_area_3 - live_area, 0.0)
    base_turnover = 1.0 - np.exp(-1.0 / 6.0)
    deposition = np.minimum(
        live_area,
        base_turnover * live_area + 0.75 * decline,
    )

    wet = rain / (rain + 30.0)
    warm = logistic((temperature - 10.0) / 4.0)
    # ln(2)/9 is a fixed reference surface-area half-life of nine months.
    decay_rate = (np.log(2.0) / 9.0) * (0.25 + 1.75 * wet * warm)
    dead = np.zeros_like(live_area[0])
    dead_history = np.empty_like(live_area)
    total_deposition = 0.0
    total_decay = 0.0
    for time in range(live_area.shape[0]):
        surviving = dead * np.exp(-decay_rate[time])
        total_decay += float(np.sum(dead - surviving))
        dead = surviving + deposition[time]
        total_deposition += float(np.sum(deposition[time]))
        dead_history[time] = dead

    closure = abs(total_deposition - total_decay - float(np.sum(dead))) / (
        total_deposition + 1e-30
    )
    dry_combustion = (
        dryness / (dryness + 350.0)
        / (1.0 + rain / 35.0)
        * logistic((temperature - 5.0) / 3.0)
    )
    curing = np.clip(
        decline / (live_area_3 + live_area + 0.25), 0.0, 1.0
    )
    dead_moisture_response = np.sqrt(np.clip(dry_combustion, 0.0, 1.0))
    live_moisture_response = np.clip(
        dry_combustion * (0.10 + 0.90 * curing), 0.0, 1.0
    )
    background_area = 0.25 * support
    total_area = background_area + live_area + dead_history + 1e-12
    dead_ready_area = dead_history * dead_moisture_response / total_area
    live_green_area = live_area * (1.0 - live_moisture_response) / total_area
    reactive_area = (
        dead_history * dead_moisture_response
        + live_area * live_moisture_response
    ) / total_area
    return SurfaceBed(
        live_area=live_area,
        dead_area=dead_history,
        dead_ready_area=np.clip(dead_ready_area, 0.0, 1.0),
        live_green_area=np.clip(live_green_area, 0.0, 1.0),
        reactive_area=np.clip(reactive_area, 0.0, 1.0),
        support=support,
        closure=float(closure),
    )


def candidate(
    incumbent: np.ndarray,
    bed: SurfaceBed,
    formulation: str,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    if formulation == "signed_surface_contrast":
        signal = np.clip(bed.dead_ready_area - bed.live_green_area, -1.0, 1.0)
    elif formulation == "relative_reactive_area":
        reference = antecedent(bed.reactive_area, 12.0)
        relative = (bed.reactive_area + 0.05) / (reference + 0.05)
        signal = np.clip(relative - 1.0, -1.0, 1.0)
    else:
        raise ValueError(formulation)
    adjusted = hazard * np.exp(np.clip(strength * signal, -0.25, 0.25))
    return np.asarray(-np.expm1(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_global = evaluator.score(incumbent)["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {incumbent_global['overall_score']:.9f}")

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
    bed = surface_bed(selected_data)
    print(
        f"BASE overall={incumbent_global['overall_score']:.9f} cells={cells.size} "
        f"fold_cells={','.join(str(int(np.sum(folds == f))) for f in range(4))} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}",
        flush=True,
    )
    print(
        f"STATE closure={bed.closure:.12g} support_mean={bed.support.mean():.9f} "
        f"live_area_mean={bed.live_area.mean():.9f} dead_area_mean={bed.dead_area.mean():.9f} "
        f"dead_ready_mean={bed.dead_ready_area.mean():.9f} "
        f"live_green_mean={bed.live_green_area.mean():.9f} "
        f"reactive_mean={bed.reactive_area.mean():.9f}",
        flush=True,
    )

    best = None
    for formulation in FORMULATIONS:
        for strength in STRENGTHS:
            trial = candidate(selected_incumbent, bed, formulation, strength)
            losses = held_losses(
                trial,
                selected_observed,
                selected_area,
                selected_observed_annual,
                folds,
            )
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            all_gates = bool(all(np.all(gain > 0.0) for gain in gains))
            aggregate = float(
                sum(np.sum(gains[index] / base_losses[index]) for index in range(3))
            )
            print(
                f"HELD formulation={formulation} strength={strength:.2f} "
                f"all_gates={int(all_gates)} aggregate={aggregate:+.9f} annual_gain="
                + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain="
                + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain="
                + ",".join(f"{value:+.9f}" for value in gains[2]),
                flush=True,
            )
            record = (aggregate, formulation, strength, trial, gains, all_gates)
            if best is None or aggregate > best[0]:
                best = record

    assert best is not None
    base_ecology = ecological_ratios_selected(
        selected_incumbent, selected_observed, selected_data, selected_area
    )
    best_ecology = ecological_ratios_selected(
        best[3], selected_observed, selected_data, selected_area
    )
    print(
        f"HELD_ECOLOGY best={best[1]}:{best[2]:.2f} "
        + ",".join(
            f"{name}:{base_ecology[name]:.6f}->{best_ecology[name]:.6f}"
            for name in base_ecology
        ),
        flush=True,
    )

    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    before_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)[:, 0, :]
    before_bed = surface_bed(prefix_data)
    before = candidate(before_incumbent, before_bed, best[1], best[2])
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    after_incumbent = model.predict(changed, dict(model.PARAMS), None)[:, 0, :]
    after_bed = surface_bed(changed)
    after = candidate(after_incumbent, after_bed, best[1], best[2])
    prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
    print(
        f"PREFIX best={best[1]}:{best[2]:.2f} max_abs={prefix_max:.12g}",
        flush=True,
    )
    print(
        f"DECISION held_survivor={int(best[5])} best={best[1]}:{best[2]:.2f} "
        f"rule=all_four_folds_all_three_losses",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
