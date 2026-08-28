"""Test a causal slow-capacity / fast-timing factorization of litter physics.

The live-to-dead litter experiment found a repeatable annual-map signal but its
monthly correction sometimes damaged the raw seasonal cycle.  This diagnostic
splits exactly the same target-blind physical correction in log-hazard space:

    log F_t = C_t + W_t
    C_t = EMA_tau(log F_t)
    W_t = log F_t - C_t

``C`` is a slowly evolving fuel-capacity state and ``W`` is the current timing
anomaly relative to that state.  Both are pointwise, globally parameterized,
and prefix causal.  Fixed time constants and blends are declared before any
score is read.  GFED is used only for held losses and exact post-hoc audit.
This script never invokes the recording evaluator or edits the canonical model.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.land_cover_interface_exact_121c83c import (  # noqa: E402
    METRICS,
    REGIONS,
    audit_masks,
    ecology,
    global_area_ratio,
    severe_pathology,
)
from autoresearch.scratchpad.live_dead_litter_exact_tradeoff_121c83c import (  # noqa: E402
    CompactState,
    full_litter_state_chunked,
)
from autoresearch.scratchpad.live_dead_litter_mass_balance_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    EXPECTED_MODEL_BLOB,
    PINNED,
    litter_state,
    load_pinned,
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


@dataclass(frozen=True)
class Config:
    tau_months: float
    blend: float
    timing_share: float

    @property
    def label(self) -> str:
        return (
            f"tau{self.tau_months:g}_b{self.blend:.2f}"
            f"_w{self.timing_share:.2f}"
        )


# Slow states spanning seasonal carry-over through a two-year litter memory.
# The weakest exact litter winner used blend=.10; .25 tests whether separating
# timing permits a materially stronger expression of the same mechanism.
CONFIGS = tuple(
    Config(tau, blend, timing_share)
    for tau in (6.0, 12.0, 24.0)
    for blend in (0.10, 0.25)
    for timing_share in (0.0, 0.25)
)


def causal_ema(values: np.ndarray, tau_months: float) -> np.ndarray:
    """Return a current-inclusive pointwise EMA with a global time constant."""
    alpha = 1.0 - np.exp(-1.0 / tau_months)
    values64 = np.asarray(values, dtype=np.float64)
    state = values64[0].copy()
    output = np.empty_like(values64)
    for time in range(values64.shape[0]):
        state += alpha * (values64[time] - state)
        output[time] = state
    return output


def correction_log(state: CompactState, blend: float) -> np.ndarray:
    """Log of the bounded fine-fuel replacement used by the prior experiment."""
    replacement = (state.litter_load + 0.05) / (state.old_fine + 0.05)
    factor = 1.0 + blend * state.fine_share * (replacement - 1.0)
    return np.log(np.clip(factor, 0.25, 4.0))


def candidate(
    incumbent: np.ndarray,
    state: CompactState,
    config: Config,
) -> np.ndarray:
    """Apply slow capacity plus a bounded share of the fast timing anomaly."""
    hazard = -np.log1p(
        -np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
    )
    raw = correction_log(state, config.blend)
    capacity = causal_ema(raw, config.tau_months)
    timing = raw - capacity
    separated = capacity + config.timing_share * timing
    adjusted = hazard * np.exp(np.clip(separated, -1.5, 1.5))
    return np.asarray(
        -np.expm1(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
    )


def selected_cells(
    observed: np.ndarray,
    incumbent: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    incumbent_annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area * observed_annual
    excess_weight = area * np.maximum(incumbent_annual - observed_annual, 0.0)

    def cover(weight: np.ndarray) -> np.ndarray:
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, 0.90) + 1)]

    cells = np.union1d(cover(observed_weight), cover(excess_weight))
    rows, columns = cells // incumbent.shape[2], cells % incumbent.shape[2]
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    return cells, rows, columns, folds


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
    base_scores = evaluator.score(incumbent)
    base_global = base_scores["global"]
    if abs(base_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {base_global['overall_score']:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    cells, rows, columns, folds = selected_cells(observed, incumbent, area)
    selected_data = {
        name: np.asarray(values[:, rows, columns], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float64)
    selected_observed = np.asarray(observed[:, rows, columns], dtype=np.float64)
    selected_area = area[rows, columns]
    selected_observed_annual = np.average(
        selected_observed, axis=0, weights=MONTH_DAYS
    )
    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        selected_observed_annual,
        folds,
    )
    selected_state = litter_state(selected_data, selected_incumbent)
    compact_selected = CompactState(
        old_fine=selected_state.old_fine,
        litter_load=selected_state.litter_load,
        readiness=selected_state.readiness,
        fine_share=selected_state.fine_share,
    )
    print(
        f"BASE pinned={PINNED} model_blob={current_blob} "
        f"overall={base_global['overall_score']:.9f} cells={cells.size} "
        f"mass_closure={selected_state.closure:.12g}",
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )

    survivors: list[tuple[float, Config]] = []
    for config in CONFIGS:
        trial = candidate(selected_incumbent, compact_selected, config)
        trial_losses = held_losses(
            trial,
            selected_observed,
            selected_area,
            selected_observed_annual,
            folds,
        )
        gains = tuple(base_losses[index] - trial_losses[index] for index in range(3))
        annual_stable = bool(np.all(gains[0] > 0.0))
        aggregate = float(sum(gain.sum() for gain in gains))
        print(
            f"HELD label={config.label} annual_stable={int(annual_stable)} "
            f"aggregate_gain={aggregate:+.9f} annual_gain="
            + ",".join(f"{value:+.9f}" for value in gains[0])
            + " allocation_gain="
            + ",".join(f"{value:+.9f}" for value in gains[1])
            + " raw_cycle_gain="
            + ",".join(f"{value:+.9f}" for value in gains[2]),
            flush=True,
        )
        if annual_stable and aggregate > 0.0:
            survivors.append((aggregate, config))

    # Prefix audit all declared candidates on a distributed cell sample.
    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    probe_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    probe_incumbent = model.predict(probe_data, dict(model.PARAMS), None)[:, 0, :]
    probe_state_raw = litter_state(probe_data, probe_incumbent)
    probe_state = CompactState(
        old_fine=probe_state_raw.old_fine,
        litter_load=probe_state_raw.litter_load,
        readiness=probe_state_raw.readiness,
        fine_share=probe_state_raw.fine_share,
    )
    changed = {name: values.copy() for name, values in probe_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)[:, 0, :]
    changed_state_raw = litter_state(changed, changed_incumbent)
    changed_state = CompactState(
        old_fine=changed_state_raw.old_fine,
        litter_load=changed_state_raw.litter_load,
        readiness=changed_state_raw.readiness,
        fine_share=changed_state_raw.fine_share,
    )
    for config in CONFIGS:
        before = candidate(probe_incumbent, probe_state, config)
        after = candidate(changed_incumbent, changed_state, config)
        prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
        print(
            f"PREFIX label={config.label} max_abs={prefix_max:.12g}", flush=True
        )
        if prefix_max != 0.0:
            raise RuntimeError(f"prefix causality failed for {config.label}")

    if not survivors:
        print("DECISION exact=0 reason=no_annual_stable_positive_aggregate_candidate")
        return 0

    survivors.sort(key=lambda item: item[0], reverse=True)
    exact_configs = [config for _aggregate, config in survivors]
    full_state, full_closure = full_litter_state_chunked(data, incumbent)
    land = load_land_mask()
    masks, fractional_countries = audit_masks(data)
    base_ecology = ecology(
        incumbent, masks, fractional_countries, observed, area, land
    )
    base_area_ratio = global_area_ratio(incumbent, observed, area, land)
    winners: list[tuple[float, Config]] = []
    for config in exact_configs:
        trial = validate_prediction(candidate(incumbent, full_state, config))
        scores = evaluator.score(trial)
        global_scores = scores["global"]
        trial_ecology = ecology(
            trial, masks, fractional_countries, observed, area, land
        )
        pathologies = severe_pathology(base_ecology, trial_ecology)
        ratio = global_area_ratio(trial, observed, area, land)
        delta = global_scores["overall_score"] - base_global["overall_score"]
        print(
            f"EXACT label={config.label} "
            + " ".join(
                f"{metric}={global_scores[key]:.9f}" for metric, key in METRICS
            )
            + " deltas="
            + ",".join(
                f"{metric}:{global_scores[key]-base_global[key]:+.9f}"
                for metric, key in METRICS
            )
            + f" area_ratio={ratio:.9f} area_delta={ratio-base_area_ratio:+.9f} "
            + f"mass_closure={full_closure:.12g}",
            flush=True,
        )
        print(
            f"REGIONS label={config.label} "
            + ",".join(
                f"{region}:{scores[region]['overall_score']-base_scores[region]['overall_score']:+.9f}"
                for region in REGIONS
            ),
            flush=True,
        )
        print(
            f"ECOLOGY label={config.label} "
            + ",".join(
                f"{name}:{float(base_ecology[name]['ratio']):.9f}"
                f"->{float(values['ratio']):.9f}"
                for name, values in trial_ecology.items()
            )
            + " severe=" + (",".join(pathologies) if pathologies else "none"),
            flush=True,
        )
        if delta > 0.0 and not pathologies:
            winners.append((delta, config))
        del trial, scores, trial_ecology
        gc.collect()

    if not winners:
        print("DECISION accept=0 reason=no_positive_exact_ecologically_safe_candidate")
        return 0
    winners.sort(key=lambda item: item[0], reverse=True)
    delta, config = winners[-1]
    # ``winners`` is ascending after the key sort; the final element is best.
    print(
        f"DECISION accept=1 label={config.label} overall_delta={delta:+.9f} "
        f"rule=overall_first_prefix_exact_no_severe_ecology",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
