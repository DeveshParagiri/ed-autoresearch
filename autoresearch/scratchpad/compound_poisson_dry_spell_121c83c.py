"""Held-block compound-Poisson submonthly weather probe on ``121c83c``.

Monthly precipitation ``P`` and a globally fixed mean storm depth ``d`` imply
an expected storm count ``lambda=P/d``.  Uniform independent storm arrivals
within a month of ``D`` days then give daily zero-rain probability
``exp(-lambda/D)``.  A finite-state Bernoulli recursion calculates the exact
probability of at least one seven-day event-free run.  This is submonthly event
topology inferred from coupled-valid rain, not another smooth monthly rain
response and not an installed wet-day or dry-spell input.

Two distinct physical roles are preregistered.  ``occurrence`` treats the dry
run as a latent monthly state in which incumbent ignition hazard may occur.
``conserved_release`` stores the incumbent hazard share blocked by absent dry
runs and releases that finite stock when a later run occurs.  Storm depths and
blend fractions are fixed brackets, never target-fitted.  Coordinates assign
whole-cell folds only, and GFED enters only the held losses.  Full-grid exact
scoring is forbidden unless annual-log, normalized-allocation, and raw-cycle
loss improve in every one of four folds.
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
STORM_DEPTHS_MM = (5.0, 10.0, 20.0)
RUN_DAYS = 7
BLENDS = (0.10, 0.25, 0.50)
BUCKET_CAPACITY_MM = 30.0
BUCKET_DRYDOWN_MONTHS = 2.0
TERMINAL_LEAK_MONTHS = 24.0


@dataclass
class WeatherState:
    zero_day_probability: np.ndarray
    dry_run_probability: np.ndarray
    bucket_deficit: np.ndarray
    combustion: np.ndarray
    opportunity: np.ndarray


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
    module = types.ModuleType(f"model_{PINNED}_compound_poisson")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def field(data, name: str) -> np.ndarray:
    values = np.asarray(data[name], dtype=np.float64)
    if values.ndim == 3 and values.shape[1] == 1:
        return values[:, 0, :]
    return values


def at_least_one_dry_run(
    zero_day_probability: np.ndarray,
    run_days: int,
) -> np.ndarray:
    """Return exact P(at least one run) for independent daily storm arrivals.

    State ``j`` holds the probability that no qualifying run has occurred and
    the current trailing dry run has length ``j``.  Wet days reset the state to
    zero and dry days advance it.  Probability leaving state ``run_days-1`` on
    a dry day is absorbed into the event "at least one run".
    """
    output = np.empty_like(zero_day_probability, dtype=np.float32)
    for time in range(zero_day_probability.shape[0]):
        dry = np.clip(zero_day_probability[time], 0.0, 1.0)
        states = [np.ones_like(dry)] + [np.zeros_like(dry) for _ in range(run_days - 1)]
        for _ in range(int(MONTH_DAYS[time])):
            surviving = np.sum(states, axis=0)
            states = [(1.0 - dry) * surviving] + [
                dry * states[length - 1] for length in range(1, run_days)
            ]
        output[time] = np.asarray(
            np.clip(1.0 - np.sum(states, axis=0), 0.0, 1.0),
            dtype=np.float32,
        )
    return output


def weather_state(
    data: dict[str, np.ndarray],
    storm_depth_mm: float,
) -> WeatherState:
    """Construct a globally shared causal event-free combustion opportunity."""
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    days = MONTH_DAYS.reshape((-1,) + (1,) * (rain.ndim - 1))
    expected_storms = rain / storm_depth_mm
    zero_day_probability = np.exp(-expected_storms / days)
    dry_run_probability = at_least_one_dry_run(
        zero_day_probability,
        RUN_DAYS,
    )

    combustion = dryness / (dryness + 500.0)
    bucket = np.minimum(rain[0], BUCKET_CAPACITY_MM).astype(np.float64)
    bucket_deficit = np.empty_like(rain, dtype=np.float32)
    recession = np.exp(
        -np.clip(combustion, 0.0, 1.0) / BUCKET_DRYDOWN_MONTHS
    )
    for time in range(rain.shape[0]):
        if time > 0:
            bucket = np.minimum(BUCKET_CAPACITY_MM, bucket + rain[time])
        bucket *= recession[time]
        bucket_deficit[time] = np.asarray(
            np.clip(1.0 - bucket / BUCKET_CAPACITY_MM, 0.0, 1.0),
            dtype=np.float32,
        )

    fuel_readiness = np.sqrt(
        np.clip(combustion * bucket_deficit, 0.0, 1.0)
    )
    opportunity = np.clip(
        dry_run_probability * fuel_readiness,
        0.0,
        1.0,
    )
    return WeatherState(
        zero_day_probability=np.asarray(zero_day_probability, dtype=np.float32),
        dry_run_probability=np.asarray(dry_run_probability, dtype=np.float32),
        bucket_deficit=np.asarray(bucket_deficit, dtype=np.float32),
        combustion=np.asarray(combustion, dtype=np.float32),
        opportunity=np.asarray(opportunity, dtype=np.float32),
    )


def occurrence(
    incumbent: np.ndarray,
    state: WeatherState,
    blend: float,
) -> tuple[np.ndarray, float]:
    """Mix in a latent-window occurrence probability, not a hazard multiplier.

    If a combustible dry run exists with probability ``O``, the conditional
    hazard is ``H/O``.  Marginal event probability is therefore
    ``O*(1-exp(-H/O))``.  For small hazard it approaches the incumbent event
    probability, but it caps large monthly occurrence when windows are rare.
    """
    incumbent = np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
    hazard = -np.log1p(-incumbent)
    opportunity = np.asarray(state.opportunity, dtype=np.float64)
    conditional = opportunity * -np.expm1(
        -hazard / np.maximum(opportunity, 1e-8)
    )
    conditional = np.where(opportunity > 0.0, conditional, 0.0)
    candidate = (1.0 - blend) * incumbent + blend * conditional
    return np.asarray(np.clip(candidate, 0.0, 1.0), dtype=np.float32), 0.0


def conserved_release(
    incumbent: np.ndarray,
    state: WeatherState,
    blend: float,
) -> tuple[np.ndarray, float]:
    """Store blocked hazard and release its finite stock in later dry runs."""
    hazard = -np.log1p(
        -np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
    )
    opportunity = np.asarray(state.opportunity, dtype=np.float64)
    output = np.empty_like(hazard)
    bank = np.zeros_like(hazard[0])
    total_input = float(np.sum(hazard, dtype=np.float64))
    leak_survival = np.exp(-1.0 / TERMINAL_LEAK_MONTHS)
    for time in range(hazard.shape[0]):
        stored = blend * (1.0 - opportunity[time]) * hazard[time]
        bank += stored
        release_fraction = 1.0 - (1.0 - opportunity[time]) * leak_survival
        released = release_fraction * bank
        bank -= released
        output[time] = hazard[time] - stored + released
    closure = abs(
        total_input - float(np.sum(output, dtype=np.float64)) - float(np.sum(bank))
    ) / (total_input + 1e-30)
    return (
        np.asarray(-np.expm1(-np.clip(output, 0.0, 50.0)), dtype=np.float32),
        float(closure),
    )


def apply_role(
    role: str,
    incumbent: np.ndarray,
    state: WeatherState,
    blend: float,
) -> tuple[np.ndarray, float]:
    if role == "occurrence":
        return occurrence(incumbent, state, blend)
    if role == "conserved_release":
        return conserved_release(incumbent, state, blend)
    raise ValueError(role)


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
        predicted_annual_grid - observed_annual_grid,
        0.0,
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
    print(
        f"BASE pinned={PINNED} model_blob={current_blob} "
        f"overall={incumbent_global['overall_score']:.9f} cells={cells.size} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}"
    )
    print(
        "BASE_HELD annual="
        + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation="
        + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle="
        + ",".join(f"{value:.9f}" for value in base_losses[2])
    )

    states = {}
    for depth in STORM_DEPTHS_MM:
        state = weather_state(selected_data, depth)
        states[depth] = state
        print(
            f"STATE depth_mm={depth:.1f} run_days={RUN_DAYS} "
            f"expected_storms_mean={field(selected_data, 'monthly_precipitation').mean()/depth:.6f} "
            f"zero_day_mean={state.zero_day_probability.mean():.6f} "
            f"run_probability_mean={state.dry_run_probability.mean():.6f} "
            f"bucket_deficit_mean={state.bucket_deficit.mean():.6f} "
            f"opportunity_mean={state.opportunity.mean():.6f}"
        )

    survivors = []
    best = None
    for role in ("occurrence", "conserved_release"):
        for depth in STORM_DEPTHS_MM:
            for blend in BLENDS:
                trial, closure = apply_role(
                    role,
                    selected_incumbent,
                    states[depth],
                    blend,
                )
                losses = held_losses(
                    trial,
                    selected_observed,
                    selected_area,
                    selected_observed_annual,
                    folds,
                )
                gains = tuple(
                    base_losses[index] - losses[index] for index in range(3)
                )
                held = bool(all(np.all(gain > 0.0) for gain in gains))
                aggregate = float(sum(gain.sum() for gain in gains))
                print(
                    f"BRACKET role={role} depth_mm={depth:.1f} "
                    f"run_days={RUN_DAYS} blend={blend:.2f} held={int(held)} "
                    f"closure={closure:.12g} annual_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[0])
                    + " allocation_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[1])
                    + " raw_cycle_gain="
                    + ",".join(f"{value:+.9f}" for value in gains[2])
                )
                record = (aggregate, role, depth, blend, trial, closure, gains)
                if best is None or aggregate > best[0]:
                    best = record
                if held:
                    survivors.append(record)

    assert best is not None
    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)[:, 0, :]
    prefix_state = weather_state(prefix_data, best[2])
    before, _ = apply_role(best[1], prefix_incumbent, prefix_state, best[3])
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)[:, 0, :]
    changed_state = weather_state(changed, best[2])
    after, _ = apply_role(best[1], changed_incumbent, changed_state, best[3])
    print(
        f"PREFIX best={best[1]}:depth{best[2]:.1f}:blend{best[3]:.2f} "
        f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}"
    )

    if not survivors:
        print("DECISION exact=0 reject=no_all_block_all_metric_survivor")
        return 0

    survivors.sort(key=lambda record: record[0], reverse=True)
    _, role, depth, blend, _, _, _ = survivors[0]
    full_state = weather_state(data, depth)
    trial, closure = apply_role(role, incumbent, full_state, blend)
    score = evaluator.score(validate_prediction(trial))["global"]
    print(
        f"DECISION exact=1 role={role} depth_mm={depth:.1f} "
        f"run_days={RUN_DAYS} blend={blend:.2f} closure={closure:.12g}"
    )
    print(
        f"EXACT overall={score['overall_score']:.9f} "
        f"delta={score['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
        f"seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}"
    )
    base_ecology = ecological_ratios_selected(
        incumbent[:, rows, columns],
        observed[:, rows, columns],
        selected_data,
        selected_area,
    )
    trial_ecology = ecological_ratios_selected(
        trial[:, rows, columns],
        observed[:, rows, columns],
        selected_data,
        selected_area,
    )
    print(
        "EXACT_ECOLOGY "
        + ",".join(
            f"{name}:{base_ecology[name]:.5f}->{trial_ecology[name]:.5f}"
            for name in base_ecology
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
