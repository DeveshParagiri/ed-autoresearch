"""Prefix-causal capacity/timing factorization at canonical ``121c83c``.

This is a read-only held-cell diagnostic.  It factors the canonical monthly
hazard ``h_t = -log(1-p_t)`` into a slow capacity and a timing channel,

    C_t = C_(t-1) + alpha * (h_t - C_(t-1))
    T_t = h_t / C_t
    h_t = C_t * T_t,       alpha = 1 - exp(-1/tau).

For a positive ecological multiplier ``q_t``, its prefix-causal slow and
timing parts are

    log Q_t = log Q_(t-1) + alpha * (log q_t - log Q_(t-1))
    U_t = q_t / Q_t
    q_t = Q_t * U_t.

Thus the full mechanism is ``C Q T U``, the capacity-only diagnostic is
``C Q T`` (incumbent timing retained exactly), and the timing-only diagnostic
is ``C T U`` (incumbent capacity retained).  Initialization uses only month
zero, and every update is one-sided.  There is no calendar-year, completed-
year, future, target, coordinate, region, residual, or fitted runtime feature.

Coordinates assign disjoint whole-cell folds only.  GFED enters only held
losses after prediction.  The tested laws are the previously declared direct
live-dead litter replacement and Simpson-weighted land-cover access.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.land_cover_interface_mechanism_121c83c import (  # noqa: E402
    interface_signals,
)
from autoresearch.scratchpad.live_dead_litter_mass_balance_121c83c import (  # noqa: E402
    litter_state,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    PINNED,
    held_losses,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask  # noqa: E402


EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
TAUS = (12.0, 24.0)
MECHANISMS = (
    ("litter", 0.05),
    ("litter", 0.10),
    ("litter", 0.20),
    ("simpson", 0.02),
    ("simpson", 0.05),
    ("simpson", 0.10),
)


@dataclass
class CompactLitter:
    old_fine: np.ndarray
    litter_load: np.ndarray
    fine_share: np.ndarray
    max_closure: float


def causal_mean(values: np.ndarray, months: float) -> np.ndarray:
    """One-sided exponential mean initialized from the first available month."""
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def compact_litter_state(
    data: dict[str, np.ndarray], incumbent: np.ndarray, chunk_size: int = 4096
) -> CompactLitter:
    """Evaluate the exact declared litter state in independent cell chunks."""
    time, cells = incumbent.shape
    output = {
        name: np.empty((time, cells), dtype=np.float32)
        for name in ("old_fine", "litter_load", "fine_share")
    }
    max_closure = 0.0
    for start in range(0, cells, chunk_size):
        stop = min(start + chunk_size, cells)
        chunk = {
            name: np.asarray(values[:, start:stop])[:, None, :]
            for name, values in data.items()
        }
        state = litter_state(chunk, incumbent[:, start:stop])
        for name in output:
            output[name][:, start:stop] = getattr(state, name)
        max_closure = max(max_closure, state.closure)
    return CompactLitter(**output, max_closure=float(max_closure))


def raw_factor(
    name: str,
    strength: float,
    data: dict[str, np.ndarray],
    litter: CompactLitter,
) -> np.ndarray:
    """Return the exact positive multiplier of a previously declared law."""
    if name == "litter":
        replacement = (litter.litter_load + 0.05) / (litter.old_fine + 0.05)
        return np.clip(
            1.0 + strength * litter.fine_share * (replacement - 1.0),
            0.25,
            4.0,
        )
    if name == "simpson":
        access, _, _ = interface_signals(data, "simpson")
        return np.exp(np.clip(strength * access, -0.5, 0.5))
    raise ValueError(name)


def factorized_predictions(
    incumbent: np.ndarray, factor: np.ndarray, months: float
) -> tuple[dict[str, np.ndarray], float, float]:
    """Return full/capacity/timing predictions and reconstruction residuals."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    capacity = causal_mean(hazard, months)
    timing = np.divide(
        hazard,
        capacity,
        out=np.zeros_like(hazard),
        where=capacity > 0.0,
    )
    log_factor = np.log(np.clip(factor, 1e-8, 1e8))
    slow_factor = np.exp(causal_mean(log_factor, months))
    residual_factor = factor / slow_factor
    hazards = {
        "full": capacity * slow_factor * timing * residual_factor,
        "capacity": capacity * slow_factor * timing,
        "timing": capacity * timing * residual_factor,
    }
    predictions = {
        name: np.asarray(-np.expm1(-np.clip(value, 0.0, 50.0)), dtype=np.float32)
        for name, value in hazards.items()
    }
    base_reconstruction = float(np.max(np.abs(capacity * timing - hazard)))
    factor_reconstruction = float(
        np.max(np.abs(slow_factor * residual_factor - factor))
    )
    return predictions, base_reconstruction, factor_reconstruction


def load_observation(evaluator: GFED5Evaluator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    return observed, area, annual


def describe_gains(base: tuple[np.ndarray, ...], trial: tuple[np.ndarray, ...]) -> tuple:
    gains = tuple(base[index] - trial[index] for index in range(3))
    stable = tuple(bool(np.all(gain > 0.0)) for gain in gains)
    aggregate = float(
        sum(np.sum(gains[index] / base[index]) for index in range(3))
    )
    return gains, stable, aggregate


def prefix_test(model) -> None:
    data = load_inputs(model.INPUTS)
    land = load_land_mask()
    rows, columns = np.where(land)
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    before_data_3d = {
        name: np.asarray(values[:, rows[probe], columns[probe]])[:, None, :]
        for name, values in data.items()
    }
    after_data_3d = {name: values.copy() for name, values in before_data_3d.items()}
    for values in after_data_3d.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    before_incumbent = np.asarray(
        model.predict(before_data_3d, dict(model.PARAMS), None)[:, 0, :],
        dtype=np.float64,
    )
    after_incumbent = np.asarray(
        model.predict(after_data_3d, dict(model.PARAMS), None)[:, 0, :],
        dtype=np.float64,
    )
    before_data = {name: values[:, 0, :] for name, values in before_data_3d.items()}
    after_data = {name: values[:, 0, :] for name, values in after_data_3d.items()}
    before_litter = compact_litter_state(before_data, before_incumbent, 64)
    after_litter = compact_litter_state(after_data, after_incumbent, 64)
    maximum = float(np.max(np.abs(before_incumbent[:96] - after_incumbent[:96])))
    for mechanism, strength in (("litter", 0.10), ("simpson", 0.05)):
        before_factor = raw_factor(mechanism, strength, before_data, before_litter)
        after_factor = raw_factor(mechanism, strength, after_data, after_litter)
        maximum = max(
            maximum,
            float(np.max(np.abs(before_factor[:96] - after_factor[:96]))),
        )
        for tau in TAUS:
            before, _, _ = factorized_predictions(before_incumbent, before_factor, tau)
            after, _, _ = factorized_predictions(after_incumbent, after_factor, tau)
            for channel in before:
                maximum = max(
                    maximum,
                    float(np.max(np.abs(before[channel][:96] - after[channel][:96]))),
                )
    print(
        f"PREFIX cutoff=96 cells={probe.size} mechanisms=2 taus=2 channels=3 "
        f"max_abs={maximum:.12g}",
        flush=True,
    )
    if maximum != 0.0:
        raise RuntimeError(f"prefix causality failed: {maximum}")


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
        raise RuntimeError(f"current model drift: {current_blob}")

    data = load_inputs(model.INPUTS)
    incumbent_grid = np.asarray(model.predict(data, dict(model.PARAMS), None))
    land = load_land_mask()
    rows, columns = np.where(land)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    incumbent = np.asarray(incumbent_grid[:, rows, columns], dtype=np.float64)
    selected_data = {
        name: np.asarray(values[:, rows, columns]) for name, values in data.items()
    }

    evaluator = GFED5Evaluator(GFED5_PATH)
    observed_grid, area_grid, observed_annual_grid = load_observation(evaluator)
    observed = np.asarray(observed_grid[:, rows, columns], dtype=np.float64)
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    observed_annual = np.asarray(observed_annual_grid[rows, columns], dtype=np.float64)
    base = held_losses(incumbent, observed, area, observed_annual, folds)
    litter = compact_litter_state(selected_data, incumbent)

    print(
        f"BASE pinned={PINNED} blob={current_blob} cells={rows.size} folds="
        + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base[2]),
        flush=True,
    )
    print(
        f"STATE litter_max_closure={litter.max_closure:.12g} "
        f"old_fine_mean={litter.old_fine.mean():.9f} "
        f"litter_mean={litter.litter_load.mean():.9f} "
        f"fine_share_mean={litter.fine_share.mean():.9f}",
        flush=True,
    )
    print(
        "EQUATION h=-log(1-p); C=EMA_tau(h); T=h/C; "
        "Q=exp(EMA_tau(log(q))); U=q/Q; "
        "full=C*Q*T*U capacity=C*Q*T timing=C*T*U",
        flush=True,
    )

    records = []
    factors = {}
    for mechanism, strength in MECHANISMS:
        key = (mechanism, strength)
        factors[key] = raw_factor(mechanism, strength, selected_data, litter)
        factor = factors[key]
        print(
            f"FACTOR mechanism={mechanism} strength={strength:.2f} "
            f"mean={factor.mean():.9f} p05={np.quantile(factor,.05):.9f} "
            f"p95={np.quantile(factor,.95):.9f}",
            flush=True,
        )
        for tau in TAUS:
            predictions, base_error, factor_error = factorized_predictions(
                incumbent, factor, tau
            )
            for channel, prediction in predictions.items():
                trial = held_losses(prediction, observed, area, observed_annual, folds)
                gains, stable, aggregate = describe_gains(base, trial)
                records.append(
                    (aggregate, mechanism, strength, tau, channel, gains, stable)
                )
                print(
                    f"HELD mechanism={mechanism} strength={strength:.2f} "
                    f"tau={tau:.0f} channel={channel} "
                    f"annual_stable={int(stable[0])} allocation_stable={int(stable[1])} "
                    f"raw_stable={int(stable[2])} aggregate={aggregate:+.9f} "
                    f"annual_gain=" + ",".join(f"{value:+.9f}" for value in gains[0])
                    + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
                    + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2])
                    + f" base_recon={base_error:.3g} factor_recon={factor_error:.3g}",
                    flush=True,
                )

    annual_stable = [record for record in records if record[6][0]]
    all_stable = [record for record in records if all(record[6])]
    capacity_stable = [
        record for record in annual_stable if record[4] == "capacity"
    ]
    best = max(records, key=lambda record: record[0])
    print(
        f"SUMMARY annual_stable={len(annual_stable)} "
        f"capacity_annual_stable={len(capacity_stable)} all_metric_stable={len(all_stable)} "
        f"best={best[1]}:{best[2]:.2f}:tau{best[3]:.0f}:{best[4]} "
        f"aggregate={best[0]:+.9f}",
        flush=True,
    )
    prefix_test(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
