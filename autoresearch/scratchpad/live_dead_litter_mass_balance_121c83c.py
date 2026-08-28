"""Held-block probe of a two-timescale live-to-dead litter mass balance.

The incumbent repeatedly uses ``EMA12(GPP)/(EMA12(GPP)+0.35)`` as a fine-fuel
proxy.  This scratch experiment asks whether that proxy is missing explicit
production, turnover, decomposition, combustion readiness, and fire
consumption.  It builds globally shared fast herbaceous and slow woody litter
pools using only current coupled-valid GPP, LAI, rain, temperature, dryness,
biomass/canopy, and local incumbent hazard.

Three genuinely distinct uses of the same state are declared before scoring.
``load_replacement`` substitutes litter load for the incumbent fine-fuel load;
``relative_allocation`` changes timing while retaining a causal local running
reference; ``finite_release`` stores the unavailable fine-hazard share and
releases it when litter becomes combustible.  Fixed blend fractions are not a
parameter fit.  Coordinates assign whole-cell folds only, and GFED enters only
the held losses.  Full-grid scoring is forbidden unless annual-log,
normalized-allocation, and raw-cycle loss improve in all four folds.
"""

from __future__ import annotations

import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.additive_pathway_replacement_a8ed115 import (  # noqa: E402
    ecological_ratios_selected,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    held_losses,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "121c83c"
EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
EXPECTED_INCUMBENT = 0.719892388
BLENDS = (0.25, 0.50, 1.00)


@dataclass
class LitterState:
    old_fine: np.ndarray
    litter_load: np.ndarray
    readiness: np.ndarray
    fine_share: np.ndarray
    live_fast: np.ndarray
    live_slow: np.ndarray
    dead_fast: np.ndarray
    dead_slow: np.ndarray
    closure: float


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
    module = types.ModuleType(f"model_{PINNED}_live_dead_litter")
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


def rising(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def field(data, name: str) -> np.ndarray:
    values = np.asarray(data[name], dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        return values[:, 0, :]
    return values


def litter_state(
    data: dict[str, np.ndarray],
    incumbent: np.ndarray,
) -> LitterState:
    """Integrate an explicit fast/slow live-to-dead local mass balance.

    The fast live pool turns over on three months and its dead litter decays on
    six months.  The woody live pool turns over on eighteen months and its
    litter decays on thirty-six months.  Wet warmth accelerates decomposition;
    incumbent combustion opportunity consumes only dead mass.  All transfers
    are accounted for explicitly in the closure diagnostic.
    """
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    lai = np.clip(field(data, "leaf_area_index"), 0.0, None)
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)
    natural_canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    secondary_canopy = np.clip(field(data, "secondary_canopy_height"), 0.0, None)
    canopy = np.maximum(natural_canopy, secondary_canopy)
    hazard = -np.log1p(-np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7))

    productivity = gpp / (gpp + 0.35)
    leaf_support = lai / (lai + 2.0)
    woody_structure = canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    open_structure = 8.0 / (canopy + 8.0)
    fast_target = productivity * leaf_support * open_structure
    slow_target = productivity * leaf_support * woody_structure
    fine_share = open_structure * leaf_support / (
        0.05 + open_structure * leaf_support + woody_structure
    )

    gpp12 = antecedent(gpp, 12.0)
    old_fine = gpp12 / (gpp12 + 0.35)
    lai3 = antecedent(lai, 3.0)
    leaf_fall = np.maximum((lai3 - lai) / (lai3 + lai + 0.5), 0.0)
    warm = rising(temperature, 10.0, 4.0)
    wet = rain / (rain + 30.0)
    combustion = (
        dryness / (dryness + 500.0)
        / (1.0 + rain / 35.0)
        * rising(temperature, 5.0, 3.0)
    )

    fast_turn_base = 1.0 - np.exp(-1.0 / 3.0)
    slow_turn_base = 1.0 - np.exp(-1.0 / 18.0)
    live_fast = np.zeros_like(hazard[0])
    live_slow = np.zeros_like(hazard[0])
    dead_fast = np.zeros_like(hazard[0])
    dead_slow = np.zeros_like(hazard[0])
    histories = [np.empty_like(hazard) for _ in range(4)]
    litter_load = np.empty_like(hazard)
    readiness = np.empty_like(hazard)
    total_input = 0.0
    total_decay = 0.0
    total_consumed = 0.0

    for time in range(hazard.shape[0]):
        fast_input = fast_turn_base * fast_target[time]
        slow_input = slow_turn_base * slow_target[time]
        total_input += float(np.sum(fast_input) + np.sum(slow_input))
        live_fast += fast_input
        live_slow += slow_input

        fast_turn = 1.0 - np.exp(-(1.0 / 3.0 + 1.5 * leaf_fall[time]))
        slow_turn = 1.0 - np.exp(-(1.0 / 18.0 + 0.5 * leaf_fall[time]))
        fast_transfer = live_fast * fast_turn
        slow_transfer = live_slow * slow_turn
        live_fast -= fast_transfer
        live_slow -= slow_transfer

        environment = 0.25 + 1.5 * warm[time] * wet[time]
        fast_decay = 1.0 - np.exp(-environment / 6.0)
        slow_decay = 1.0 - np.exp(-environment / 36.0)
        fast_loss = dead_fast * fast_decay
        slow_loss = dead_slow * slow_decay
        dead_fast -= fast_loss
        dead_slow -= slow_loss
        total_decay += float(np.sum(fast_loss) + np.sum(slow_loss))
        dead_fast += fast_transfer
        dead_slow += slow_transfer

        fast_ready = dead_fast / (dead_fast + live_fast + 0.05)
        slow_ready = dead_slow / (dead_slow + live_slow + 0.10)
        litter_load[time] = 0.75 * fast_ready + 0.25 * slow_ready
        readiness[time] = combustion[time] * litter_load[time]

        burn_pressure = 1.0 - np.exp(
            -2.0 * hazard[time] / (hazard[time] + 0.04) * combustion[time]
        )
        fast_consumed = dead_fast * burn_pressure
        slow_consumed = dead_slow * (1.0 - np.exp(-0.35 * burn_pressure))
        dead_fast -= fast_consumed
        dead_slow -= slow_consumed
        total_consumed += float(np.sum(fast_consumed) + np.sum(slow_consumed))

        histories[0][time] = live_fast
        histories[1][time] = live_slow
        histories[2][time] = dead_fast
        histories[3][time] = dead_slow

    final_mass = float(np.sum(live_fast + live_slow + dead_fast + dead_slow))
    closure = abs(total_input - total_decay - total_consumed - final_mass) / (
        total_input + 1e-30
    )
    return LitterState(
        old_fine=np.asarray(old_fine, dtype=np.float32),
        litter_load=np.asarray(np.clip(litter_load, 0.0, 1.0), dtype=np.float32),
        readiness=np.asarray(np.clip(readiness, 0.0, 1.0), dtype=np.float32),
        fine_share=np.asarray(np.clip(fine_share, 0.0, 1.0), dtype=np.float32),
        live_fast=np.asarray(histories[0], dtype=np.float32),
        live_slow=np.asarray(histories[1], dtype=np.float32),
        dead_fast=np.asarray(histories[2], dtype=np.float32),
        dead_slow=np.asarray(histories[3], dtype=np.float32),
        closure=float(closure),
    )


def load_replacement(
    incumbent: np.ndarray, state: LitterState, blend: float
) -> np.ndarray:
    """Replace the inferred fine-fuel share with explicit dead litter load."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    replacement = (state.litter_load + 0.05) / (state.old_fine + 0.05)
    factor = 1.0 + blend * state.fine_share * (replacement - 1.0)
    return np.asarray(
        -np.expm1(-np.clip(hazard * np.clip(factor, 0.25, 4.0), 0.0, 50.0)),
        dtype=np.float32,
    )


def relative_allocation(
    incumbent: np.ndarray, state: LitterState, blend: float
) -> np.ndarray:
    """Use litter readiness as a mean-neutral local seasonal allocator."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    raw = (state.readiness + 0.05) / (state.old_fine + 0.05)
    reference = antecedent(raw, 12.0)
    relative = np.clip(raw / (reference + 1e-8), 0.25, 4.0)
    factor = 1.0 + blend * state.fine_share * (relative - 1.0)
    return np.asarray(
        -np.expm1(-np.clip(hazard * np.clip(factor, 0.25, 4.0), 0.0, 50.0)),
        dtype=np.float32,
    )


def finite_release(
    incumbent: np.ndarray, state: LitterState, blend: float
) -> tuple[np.ndarray, float]:
    """Store unavailable fine-path hazard and release it on litter readiness."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    output = np.empty_like(hazard)
    bank = np.zeros_like(hazard[0])
    total_input = float(np.sum(hazard, dtype=np.float64))
    for time in range(hazard.shape[0]):
        stored = (
            blend
            * state.fine_share[time]
            * (1.0 - state.readiness[time])
            * hazard[time]
        )
        bank += stored
        release = 1.0 - np.exp(-(1.0 / 24.0 + 8.0 * state.readiness[time]))
        released = release * bank
        bank -= released
        output[time] = hazard[time] - stored + released
    terminal = float(np.sum(bank) / (total_input + 1e-30))
    return (
        np.asarray(-np.expm1(-np.clip(output, 0.0, 50.0)), dtype=np.float32),
        terminal,
    )


def apply_formulation(name, incumbent, state, blend):
    if name == "load_replacement":
        return load_replacement(incumbent, state, blend), 0.0
    if name == "relative_allocation":
        return relative_allocation(incumbent, state, blend), 0.0
    if name == "finite_release":
        return finite_release(incumbent, state, blend)
    raise ValueError(name)


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

    def top(weight):
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
    state = litter_state(selected_data, selected_incumbent)
    print(
        f"BASE pinned={PINNED} model_blob={current_blob} overall={incumbent_global['overall_score']:.9f} "
        f"cells={cells.size} observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}"
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2])
    )
    print(
        f"STATE closure={state.closure:.12g} old_fine_mean={state.old_fine.mean():.9f} "
        f"litter_mean={state.litter_load.mean():.9f} readiness_mean={state.readiness.mean():.9f} "
        f"fast_dead_mean={state.dead_fast.mean():.9f} slow_dead_mean={state.dead_slow.mean():.9f} "
        f"fine_share_mean={state.fine_share.mean():.9f}"
    )

    survivors = []
    best = None
    formulations = ("load_replacement", "relative_allocation", "finite_release")
    for formulation in formulations:
        for blend in BLENDS:
            trial, terminal = apply_formulation(
                formulation, selected_incumbent, state, blend
            )
            trial_losses = held_losses(
                trial,
                selected_observed,
                selected_area,
                selected_observed_annual,
                folds,
            )
            gains = tuple(base_losses[index] - trial_losses[index] for index in range(3))
            held = bool(all(np.all(gain > 0.0) for gain in gains))
            aggregate = float(sum(gain.sum() for gain in gains))
            print(
                f"BRACKET formulation={formulation} blend={blend:.2f} held={int(held)} "
                f"terminal_bank={terminal:.9f} annual_gain="
                + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain="
                + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain="
                + ",".join(f"{value:+.9f}" for value in gains[2])
            )
            record = (aggregate, formulation, blend, trial, terminal, gains)
            if best is None or aggregate > best[0]:
                best = record
            if held:
                survivors.append(record)

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
            f"{name}:{base_ecology[name]:.5f}->{best_ecology[name]:.5f}"
            for name in base_ecology
        )
    )

    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)[:, 0, :]
    prefix_state = litter_state(prefix_data, prefix_incumbent)
    before, _ = apply_formulation(best[1], prefix_incumbent, prefix_state, best[2])
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)[:, 0, :]
    changed_state = litter_state(changed, changed_incumbent)
    after, _ = apply_formulation(best[1], changed_incumbent, changed_state, best[2])
    print(
        f"PREFIX best={best[1]}:{best[2]:.2f} "
        f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}"
    )

    if not survivors:
        print("DECISION exact=0 reject=no_all_block_all_metric_survivor")
        return 0

    survivors.sort(key=lambda record: record[0], reverse=True)
    _, formulation, blend, _, _, _ = survivors[0]
    full_state = litter_state(data, incumbent)
    trial, terminal = apply_formulation(formulation, incumbent, full_state, blend)
    score = evaluator.score(validate_prediction(trial))["global"]
    print(
        f"DECISION exact=1 formulation={formulation} blend={blend:.2f} "
        f"terminal_bank={terminal:.9f}"
    )
    print(
        f"EXACT overall={score['overall_score']:.9f} "
        f"delta={score['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
        f"seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
