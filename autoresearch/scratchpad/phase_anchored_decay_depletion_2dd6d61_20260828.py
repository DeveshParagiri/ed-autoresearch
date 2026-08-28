"""Held-cell probe of phase-anchored post-peak fine-fuel exhaustion.

This scratch experiment is pinned to canonical ``2dd6d61``.  It asks whether
the centered seasonal waveform retains too much fire on the decay limb after
the local litter-supported peak.  It does not add another opportunity bank or
fuel reservoir: no hazard is stored and nothing is released later.  Instead,
the already-computed local hazard is attenuated only during a contiguous
decline in both hazard and the causal live-to-dead litter support.

For incumbent hazard ``h_t``, causal dead-litter support ``L_t``, incumbent
combustion pressure ``b_t``, and fine-fuel share ``f_t``, a litter peak starts
the decay limb when ``L[t-1] >= L[t-2] > L[t]`` and ``h[t] < h[t-1]``.  While
both series continue declining,

    E_t = E_(t-1) + b_(t-1) f_(t-1) L_(t-1)
    h'_t = h_t exp(-kappa E_t).

Any rise resets ``E`` to zero.  Therefore the anchor month and every rising or
local-maximum month are unchanged exactly; only the following decay limb can
be suppressed.  The fixed physical bracket ``kappa=(0.05, 0.10, 0.20)`` means
five, ten, or twenty percent additional exhaustion per unit cumulative
combustible fine-litter exposure.  It is declared here before scoring.

All prediction features are current coupled-valid inputs or prefix-causal
local model state.  Coefficients are globally shared.  Coordinates are used
only to construct disjoint whole-cell held folds, and GFED enters only losses
and post-prediction ecological audits.  This script never records an official
evaluation or edits canonical artifacts.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from autoresearch.scratchpad.live_dead_litter_mass_balance_121c83c import (  # noqa: E402
    LitterState,
    litter_state,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    held_losses,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "2dd6d61"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
EXPECTED_OVERALL = 0.720105466
KAPPAS = (0.05, 0.10, 0.20)
CYCLE_DAYS = np.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    dtype=np.float64,
)


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
    module = types.ModuleType(f"model_{PINNED}_phase_anchored_depletion")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def field(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    values = np.asarray(data[name], dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        return values[:, 0, :]
    return values


def combustion_pressure(
    data: dict[str, np.ndarray], incumbent: np.ndarray
) -> np.ndarray:
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    temperature = field(data, "air_temperature")
    warm = 1.0 / (
        1.0 + np.exp(np.clip(-(temperature - 5.0) / 3.0, -50.0, 50.0))
    )
    combustion = dryness / (dryness + 500.0) / (1.0 + rain / 35.0) * warm
    hazard = -np.log1p(
        -np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
    )
    return 1.0 - np.exp(
        -2.0 * hazard / (hazard + 0.04) * combustion
    )


def phase_anchored_depletion(
    incumbent: np.ndarray,
    data: dict[str, np.ndarray],
    state: LitterState,
    kappa: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Suppress only a litter-anchored, monotonically falling hazard limb."""
    hazard = -np.log1p(
        -np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
    )
    litter = np.asarray(state.litter_load, dtype=np.float64)
    fine_share = np.asarray(state.fine_share, dtype=np.float64)
    burn = combustion_pressure(data, incumbent)
    output = hazard.copy()
    exhaustion = np.zeros_like(hazard[0])
    active = np.zeros_like(hazard[0], dtype=bool)
    active_count = 0

    for time in range(1, hazard.shape[0]):
        hazard_falling = hazard[time] < hazard[time - 1]
        litter_falling = litter[time] < litter[time - 1]
        if time == 1:
            litter_peak = np.ones_like(active)
        else:
            litter_peak = litter[time - 1] >= litter[time - 2]
        starts = hazard_falling & litter_falling & litter_peak
        continues = active & hazard_falling & litter_falling
        active = starts | continues
        exposure = (
            burn[time - 1]
            * fine_share[time - 1]
            * litter[time - 1]
        )
        exhaustion = np.where(active, exhaustion + exposure, 0.0)
        output[time] = hazard[time] * np.exp(-kappa * exhaustion)
        active_count += int(np.count_nonzero(active))

    candidate = np.asarray(
        -np.expm1(-np.clip(output, 0.0, 50.0)), dtype=np.float32
    )
    nondeclining = np.zeros_like(hazard, dtype=bool)
    nondeclining[0] = True
    nondeclining[1:] = hazard[1:] >= hazard[:-1]
    peak_max_abs = float(
        np.max(np.abs(candidate[nondeclining] - np.asarray(incumbent)[nondeclining]))
    )
    diagnostics = {
        "active_fraction": active_count / float(hazard.size),
        "mean_suppression": float(np.mean(np.asarray(incumbent) - candidate)),
        "maximum_suppression": float(np.max(np.asarray(incumbent) - candidate)),
        "nondeclining_max_abs": peak_max_abs,
    }
    return candidate, diagnostics


def centered_cycle_losses(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    observed_annual: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    pred_cycle = np.asarray(prediction, dtype=np.float64).reshape(
        16, 12, -1
    ).mean(axis=0)
    obs_cycle = np.asarray(observed, dtype=np.float64).reshape(
        16, 12, -1
    ).mean(axis=0)
    pred_anomaly = pred_cycle - pred_cycle.mean(axis=0, keepdims=True)
    obs_anomaly = obs_cycle - obs_cycle.mean(axis=0, keepdims=True)
    difference = np.square(pred_anomaly - obs_anomaly)
    cell_weight = np.asarray(area) * np.asarray(observed_annual)
    losses = []
    for fold in range(4):
        held = folds == fold
        numerator = np.sum(
            difference[:, held]
            * CYCLE_DAYS[:, None]
            * cell_weight[None, held]
        )
        denominator = CYCLE_DAYS.sum() * cell_weight[held].sum() + 1e-15
        losses.append(np.sqrt(numerator / denominator))
    return np.asarray(losses)


def held_peak_months(
    prediction: np.ndarray, area: np.ndarray, folds: np.ndarray
) -> tuple[int, ...]:
    cycle = np.asarray(prediction, dtype=np.float64).reshape(
        16, 12, -1
    ).mean(axis=0)
    peaks = []
    for fold in range(4):
        held = folds == fold
        monthly = np.sum(cycle[:, held] * area[None, held], axis=1)
        peaks.append(int(np.argmax(monthly)))
    return tuple(peaks)


def cell_peak_changes(baseline: np.ndarray, trial: np.ndarray) -> int:
    baseline_cycle = np.asarray(baseline).reshape(16, 12, -1).mean(axis=0)
    trial_cycle = np.asarray(trial).reshape(16, 12, -1).mean(axis=0)
    return int(
        np.count_nonzero(
            np.argmax(baseline_cycle, axis=0)
            != np.argmax(trial_cycle, axis=0)
        )
    )


def full_grid_candidate(
    incumbent: np.ndarray,
    data: dict[str, np.ndarray],
    kappa: float,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, dict[str, float]]:
    rows = np.repeat(np.arange(180), 360)
    columns = np.tile(np.arange(360), 180)
    candidate = np.asarray(incumbent, dtype=np.float32).copy()
    weighted = {
        "active_fraction": 0.0,
        "mean_suppression": 0.0,
        "maximum_suppression": 0.0,
        "nondeclining_max_abs": 0.0,
    }
    total_cells = rows.size
    for start in range(0, total_cells, chunk_size):
        stop = min(start + chunk_size, total_cells)
        chunk_rows = rows[start:stop]
        chunk_columns = columns[start:stop]
        chunk_data = {
            name: np.asarray(values[:, chunk_rows, chunk_columns])[:, None, :]
            for name, values in data.items()
        }
        chunk_incumbent = np.asarray(
            incumbent[:, chunk_rows, chunk_columns], dtype=np.float64
        )
        state = litter_state(chunk_data, chunk_incumbent)
        chunk_candidate, diagnostics = phase_anchored_depletion(
            chunk_incumbent, chunk_data, state, kappa
        )
        candidate[:, chunk_rows, chunk_columns] = chunk_candidate
        fraction = (stop - start) / total_cells
        weighted["active_fraction"] += diagnostics["active_fraction"] * fraction
        weighted["mean_suppression"] += diagnostics["mean_suppression"] * fraction
        weighted["maximum_suppression"] = max(
            weighted["maximum_suppression"], diagnostics["maximum_suppression"]
        )
        weighted["nondeclining_max_abs"] = max(
            weighted["nondeclining_max_abs"], diagnostics["nondeclining_max_abs"]
        )
    return candidate, weighted


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
        raise RuntimeError(f"current canonical model drifted to {current_blob}")

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline_score = evaluator.score(incumbent)["global"]
    if abs(baseline_score["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(
            f"incumbent score drift {baseline_score['overall_score']:.9f}"
        )

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observed_annual_grid = np.average(observation, axis=0, weights=MONTH_DAYS)
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
    selected_observation = np.asarray(
        observation[:, rows, columns], dtype=np.float64
    )
    selected_area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    selected_observed_annual = np.asarray(
        observed_annual_grid[rows, columns], dtype=np.float64
    )
    baseline_held = held_losses(
        selected_incumbent,
        selected_observation,
        selected_area,
        selected_observed_annual,
        folds,
    )
    baseline_centered = centered_cycle_losses(
        selected_incumbent,
        selected_observation,
        selected_area,
        selected_observed_annual,
        folds,
    )
    baseline_peaks = held_peak_months(selected_incumbent, selected_area, folds)
    state = litter_state(selected_data, selected_incumbent)

    print(
        f"BASE pinned={PINNED} blob={current_blob} "
        f"overall={baseline_score['overall_score']:.9f} "
        f"rmse={baseline_score['rmse_score']:.9f} cells={cells.size} "
        "folds=" + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in baseline_held[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in baseline_held[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in baseline_held[2])
        + " centered_cycle=" + ",".join(f"{value:.9f}" for value in baseline_centered)
        + " peaks=" + ",".join(str(value) for value in baseline_peaks),
        flush=True,
    )
    print(
        "PREDECLARED kappa=0.05,0.10,0.20 gate=centered_positive_all_folds,"
        "fold_peak_unchanged,coupled_annual_allocation_raw_reported,"
        "exact_seasonal_nonnegative,no_severe_ecology",
        flush=True,
    )

    records = []
    for kappa in KAPPAS:
        trial, diagnostics = phase_anchored_depletion(
            selected_incumbent, selected_data, state, kappa
        )
        trial_held = held_losses(
            trial,
            selected_observation,
            selected_area,
            selected_observed_annual,
            folds,
        )
        trial_centered = centered_cycle_losses(
            trial,
            selected_observation,
            selected_area,
            selected_observed_annual,
            folds,
        )
        gains = tuple(
            baseline_held[index] - trial_held[index] for index in range(3)
        )
        centered_gain = baseline_centered - trial_centered
        peaks = held_peak_months(trial, selected_area, folds)
        changed = cell_peak_changes(selected_incumbent, trial)
        centered_stable = bool(np.all(centered_gain > 0.0))
        peak_stable = peaks == baseline_peaks
        aggregate = float(np.sum(centered_gain / baseline_centered))
        records.append(
            (aggregate, kappa, centered_stable, peak_stable, trial, diagnostics)
        )
        print(
            f"HELD kappa={kappa:.2f} centered_stable={int(centered_stable)} "
            f"peak_stable={int(peak_stable)} cell_peak_changes={changed} "
            f"active_fraction={diagnostics['active_fraction']:.9f} "
            f"nondeclining_max_abs={diagnostics['nondeclining_max_abs']:.12g} "
            "annual_gain=" + ",".join(f"{value:+.9f}" for value in gains[0])
            + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
            + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2])
            + " centered_gain=" + ",".join(f"{value:+.9f}" for value in centered_gain)
            + " peaks=" + ",".join(str(value) for value in peaks),
            flush=True,
        )

    best = max(records, key=lambda record: record[0])
    _, best_kappa, _, _, _, _ = best
    del state
    gc.collect()

    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    before_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    changed_data = {name: values.copy() for name, values in before_data.items()}
    for values in changed_data.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    before_incumbent = np.asarray(
        model.predict(before_data, dict(model.PARAMS), None)[:, 0, :],
        dtype=np.float64,
    )
    changed_incumbent = np.asarray(
        model.predict(changed_data, dict(model.PARAMS), None)[:, 0, :],
        dtype=np.float64,
    )
    before_state = litter_state(before_data, before_incumbent)
    changed_state = litter_state(changed_data, changed_incumbent)
    prefix_max = 0.0
    for kappa in KAPPAS:
        before_candidate = phase_anchored_depletion(
            before_incumbent, before_data, before_state, kappa
        )[0]
        changed_candidate = phase_anchored_depletion(
            changed_incumbent, changed_data, changed_state, kappa
        )[0]
        local_prefix = float(
            np.max(np.abs(before_candidate[:96] - changed_candidate[:96]))
        )
        prefix_max = max(prefix_max, local_prefix)
        print(
            f"PREFIX kappa={kappa:.2f} cutoff=96 cells={probe.size} "
            f"max_abs={local_prefix:.12g}",
            flush=True,
        )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix causality failed: {prefix_max}")
    del before_state, changed_state, before_candidate, changed_candidate
    gc.collect()

    land = load_land_mask()
    masks = regime_masks(data)
    baseline_ecology = ecological_statistics(
        incumbent, masks, observation, area_grid, land
    )
    exact_records = []
    for _, kappa, centered_stable, peak_stable, _, _ in records:
        full_candidate, full_diagnostics = full_grid_candidate(
            incumbent, data, kappa
        )
        full_candidate = validate_prediction(full_candidate)
        trial_score = evaluator.score(full_candidate)["global"]
        trial_ecology = ecological_statistics(
            full_candidate, masks, observation, area_grid, land
        )
        severe = []
        for name in baseline_ecology:
            old = float(baseline_ecology[name]["ratio"])
            new = float(trial_ecology[name]["ratio"])
            relative = new / old if old > 0.0 else float("inf")
            if (
                not np.isfinite(new)
                or new < 0.25
                or new > 4.0
                or relative < 0.75
                or relative > 1.25
            ):
                severe.append(name)
        accepted = bool(
            centered_stable
            and peak_stable
            and trial_score["rmse_score"] > baseline_score["rmse_score"]
            and trial_score["seasonal_cycle_score"]
            >= baseline_score["seasonal_cycle_score"]
            and not severe
        )
        exact_records.append(
            (accepted, trial_score["overall_score"], kappa, severe)
        )
        print(
            f"EXACT_PROXY kappa={kappa:.2f} "
            f"overall={trial_score['overall_score']:.9f} "
            f"delta_overall={trial_score['overall_score']-baseline_score['overall_score']:+.9f} "
            f"rmse={trial_score['rmse_score']:.9f} "
            f"delta_rmse={trial_score['rmse_score']-baseline_score['rmse_score']:+.9f} "
            f"seasonal={trial_score['seasonal_cycle_score']:.9f} "
            f"delta_seasonal={trial_score['seasonal_cycle_score']-baseline_score['seasonal_cycle_score']:+.9f} "
            f"bias_delta={trial_score['bias_score']-baseline_score['bias_score']:+.9f} "
            f"spatial_delta={trial_score['spatial_distribution_score']-baseline_score['spatial_distribution_score']:+.9f} "
            f"active_fraction={full_diagnostics['active_fraction']:.9f} "
            f"nondeclining_max_abs={full_diagnostics['nondeclining_max_abs']:.12g}",
            flush=True,
        )
        print(
            f"ECOLOGY kappa={kappa:.2f} "
            + ",".join(
                f"{name}:{float(baseline_ecology[name]['ratio']):.9f}"
                f"->{float(trial_ecology[name]['ratio']):.9f}"
                for name in baseline_ecology
            )
            + " severe=" + (",".join(severe) if severe else "none"),
            flush=True,
        )
        del full_candidate, trial_ecology
        gc.collect()

    survivors = [record for record in exact_records if record[0]]
    accepted = bool(survivors)
    if survivors:
        _, _, decision_kappa, decision_severe = max(
            survivors, key=lambda record: record[1]
        )
    else:
        decision_kappa = best_kappa
        decision_severe = []
    print(
        f"DECISION accept={int(accepted)} kappa={decision_kappa:.2f} "
        f"prefix={prefix_max:.12g} severe_ecology={len(decision_severe)} "
        f"reason={'survives_all_predeclared_gates' if accepted else 'no_exact_phase_safe_survivor'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
