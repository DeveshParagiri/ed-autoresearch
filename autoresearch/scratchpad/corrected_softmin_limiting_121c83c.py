"""Held-block re-test of limiting-factor aggregation on canonical ``121c83c``.

The canonical base fire rate multiplies four favourability factors: dryness,
precipitation, fuel, and temperature.  Dormant soft-min code historically used
an unnormalised sum of exponentials, which subtracts ``log(n)/sharpness`` and
can collapse a perfectly equal factor vector toward zero.  This scratch probe
uses the corrected *mean* log-sum-exp soft minimum, compares fixed sharpnesses
and the hard-min limit, and also tests a homogeneous harmonic mean.

The mechanism is globally shared, pointwise, and uses only the incumbent valid
inputs and causal base-factor construction.  Coordinates define held spatial
folds only; GFED enters only the losses.  Full exact scoring and ecology are
forbidden unless annual-log, normalized-allocation, and raw-cycle loss improve
in every one of four disjoint whole-cell folds.
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
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    held_losses,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "121c83c"
EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
EXPECTED_INCUMBENT = 0.719892388
SHARPNESSES = (2.0, 8.0, 25.0, np.inf)
BLENDS = (0.10, 0.25, 0.50, 1.00)


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
    module = types.ModuleType(f"model_{PINNED}_corrected_softmin")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def factors_and_product(model, data, p, enabled):
    """Reproduce only the canonical four-factor base-rate construction."""
    factors = []
    product = np.ones_like(data["dryness"], dtype=np.float32)
    if "dryness" in enabled:
        term = model._rising(data["dryness"], p["k1"], p["D_low"]) * model._falling(
            data["dryness"], p["k2"], p["D_high"]
        )
        factors.append(term)
        product = product * term
    if "precipitation" in enabled:
        annual = data["annual_precipitation"]
        monthly = data["monthly_precipitation"]
        term = (annual / (annual + p["P_half"] + 1e-12)) * (
            1.0 / (1.0 + monthly / (p["pre_dampen_half"] + 1e-12))
        )
        factors.append(term)
        product = product * term
    if "fuel" in enabled:
        term = model._hump(p["gpp_af"] * data["gpp"], p["gpp_b"], p["gpp_d"])
        factors.append(term)
        product = product * term
    if "temperature" in enabled:
        term = model._managed_open_temperature_gate(data, p)
        factors.append(term)
        product = product * term
    if len(factors) != 4:
        raise RuntimeError(f"expected four base factors, got {len(factors)}")
    return np.stack(factors, axis=0), product


def mean_lse_softmin(stack: np.ndarray, sharpness: float) -> np.ndarray:
    """Return normalized log-mean-exp soft minimum with a stable shift."""
    values = np.clip(np.asarray(stack, dtype=np.float64), 1e-9, 1.0)
    if np.isinf(sharpness):
        return np.min(values, axis=0)
    minimum = np.min(values, axis=0)
    relative = np.exp(-sharpness * (values - minimum[None, ...]))
    return minimum - np.log(np.mean(relative, axis=0)) / sharpness


def harmonic_limit(stack: np.ndarray) -> np.ndarray:
    """Return the homogeneous generalized mean of order minus one."""
    values = np.clip(np.asarray(stack, dtype=np.float64), 1e-9, 1.0)
    return values.shape[0] / np.sum(1.0 / values, axis=0)


def fire_rate_factory(model, family: str, sharpness: float, blend: float):
    """Blend the product geometrically toward one limiting-factor law."""
    def fire_rate(data, p, enabled):
        stack, product = factors_and_product(model, data, p, enabled)
        if family == "softmin":
            limit = mean_lse_softmin(stack, sharpness)
        elif family == "harmonic":
            limit = harmonic_limit(stack)
        else:
            raise ValueError(family)
        product = np.clip(product, 1e-12, 1.0)
        limit = np.clip(limit, 1e-12, 1.0)
        rate = np.exp(
            (1.0 - blend) * np.log(product) + blend * np.log(limit)
        )
        rate = np.power(np.clip(rate, 0.0, None), p["fire_exp"])
        if "fuel" in enabled and "fuel_k" in p:
            capacity = data["gpp"].mean(axis=0, keepdims=True)
            capacity = capacity / (capacity + p["fuel_half"] + 1e-9)
            rate *= 1.0 + p["fuel_k"] * capacity
        elif "fire_amp" in p:
            rate *= p["fire_amp"]
        return rate

    return fire_rate


def candidate(model, data, family: str, sharpness: float, blend: float):
    original = model._fire_rate
    model._fire_rate = fire_rate_factory(model, family, sharpness, blend)
    try:
        prediction = np.asarray(
            model.predict(data, dict(model.PARAMS), None),
            dtype=np.float32,
        )
        if not np.isfinite(prediction).all():
            raise RuntimeError("candidate contains non-finite values")
        if float(prediction.min()) < 0.0 or float(prediction.max()) > 1.0:
            raise RuntimeError("candidate falls outside burned-fraction bounds")
        return prediction
    finally:
        model._fire_rate = original


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

    equal = np.asarray((0.05, 0.25, 0.50, 0.90), dtype=np.float64)
    equal_stack = np.repeat(equal[:, None, None], 4, axis=1).transpose(1, 0, 2)
    for sharpness in SHARPNESSES:
        check = mean_lse_softmin(equal_stack, sharpness)[:, 0]
        print(
            f"NORMALIZATION family=softmin sharpness={sharpness:g} "
            f"equal_max_abs={np.max(np.abs(check-equal)):.12g}"
        )
    harmonic_check = harmonic_limit(equal_stack)[:, 0]
    print(
        f"NORMALIZATION family=harmonic equal_max_abs="
        f"{np.max(np.abs(harmonic_check-equal)):.12g}"
    )

    configs = [
        ("softmin", sharpness, blend)
        for sharpness in SHARPNESSES
        for blend in BLENDS
    ] + [("harmonic", -1.0, blend) for blend in BLENDS]
    survivors = []
    best = None
    for family, sharpness, blend in configs:
        trial = candidate(model, selected_data, family, sharpness, blend)[:, 0, :]
        losses = held_losses(
            trial,
            selected_observed,
            selected_area,
            selected_observed_annual,
            folds,
        )
        gains = tuple(base_losses[index] - losses[index] for index in range(3))
        held = bool(all(np.all(gain > 0.0) for gain in gains))
        aggregate = float(sum(gain.sum() for gain in gains))
        sharp_label = "hard" if np.isinf(sharpness) else f"{sharpness:g}"
        print(
            f"BRACKET family={family} sharpness={sharp_label} blend={blend:.2f} "
            f"held={int(held)} annual_gain="
            + ",".join(f"{value:+.9f}" for value in gains[0])
            + " allocation_gain="
            + ",".join(f"{value:+.9f}" for value in gains[1])
            + " raw_cycle_gain="
            + ",".join(f"{value:+.9f}" for value in gains[2])
        )
        record = (aggregate, family, sharpness, blend, trial, gains)
        if best is None or aggregate > best[0]:
            best = record
        if held:
            survivors.append(record)

    assert best is not None
    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    before = candidate(model, prefix_data, best[1], best[2], best[3])[:, 0, :]
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    after = candidate(model, changed, best[1], best[2], best[3])[:, 0, :]
    best_sharp = "hard" if np.isinf(best[2]) else f"{best[2]:g}"
    print(
        f"PREFIX best={best[1]}:sharp{best_sharp}:blend{best[3]:.2f} "
        f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}"
    )

    if not survivors:
        print("DECISION exact=0 reject=no_all_block_all_metric_survivor")
        return 0

    survivors.sort(key=lambda record: record[0], reverse=True)
    _, family, sharpness, blend, _, _ = survivors[0]
    trial = validate_prediction(candidate(model, data, family, sharpness, blend))
    score = evaluator.score(trial)["global"]
    sharp_label = "hard" if np.isinf(sharpness) else f"{sharpness:g}"
    print(
        f"DECISION exact=1 family={family} sharpness={sharp_label} "
        f"blend={blend:.2f}"
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
