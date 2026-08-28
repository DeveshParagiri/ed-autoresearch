"""Held-block test of one reverse-ML-derived ecological mechanism family.

The deep cycle learner repeatedly finds negative main effects of current fire
opportunity and absolute temperature but a positive interaction: their joint
penalty is sub-additive.  This script translates that shape into a refractory
fuel-pressure law.  High current opportunity or sustained heat can each make a
fuel bed temporarily refractory, but their overlap cannot consume the same
fuel twice.  The inclusion-exclusion signal is globally shared and uses only
current incumbent opportunity plus current/past local temperature.

No learned surface, learned cut, coordinate, region, cell identity, benchmark
field, or target-derived value enters the candidate.  The four strengths are
fixed.  Exact scoring is permitted only if normalized-cycle loss improves in
every held spatial block and annual cost is at most five percent of aggregate
cycle gain.
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


def rising(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def saturation_signal(incumbent: np.ndarray, data) -> np.ndarray:
    """Return A + H - A*H for relative opportunity A and heat stress H."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    trailing_hazard = antecedent(hazard, 12.0)
    relative_opportunity = hazard / (hazard + trailing_hazard + 1e-8)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    # Twenty C is the incumbent global ignition center, and three C is its
    # existing broad ecological thermal scale; neither value comes from ML.
    heat_stress = rising(temperature, 20.0, 3.0)
    return np.asarray(
        np.clip(
            relative_opportunity
            + heat_stress
            - relative_opportunity * heat_stress,
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )


def candidate(incumbent: np.ndarray, signal: np.ndarray, strength: float) -> np.ndarray:
    """Redistribute hazard away from refractory months using a causal reference."""
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    raw_factor = np.exp(np.clip(-strength * signal, -0.5, 0.0))
    reference = antecedent(raw_factor, 12.0)
    factor = np.clip(raw_factor / np.maximum(reference, 1e-8), 0.5, 2.0)
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * factor, 0.0, 50.0)), dtype=np.float32
    )


def main() -> None:
    model = load_pinned()
    current_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pinned_blob = subprocess.run(
        ["git", "rev-parse", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != pinned_blob:
        raise RuntimeError(f"current model blob {current_blob} differs from {PINNED} blob {pinned_blob}")

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    signal = saturation_signal(incumbent, data)
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

    def top(weight):
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, 0.90) + 1)]

    cells = np.union1d(top(observed_weight), top(excess_weight))
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    base_annual, base_cycle = losses(incumbent, observed, area, cells, folds)
    print(
        f"BASE pinned={PINNED} model_blob={current_blob} overall={base_score['overall_score']:.9f} "
        f"cells={cells.size} observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}"
    )
    print(
        f"FORMULA A=h/(h+EMA12(h)); H=sigmoid((T-20)/3); S=A+H-AH; "
        f"F=exp(-kS)/EMA12(exp(-kS)) signal_mean={np.mean(signal[:, rows, cols]):.9f} "
        f"signal_p95={np.quantile(signal[:, rows, cols], 0.95):.9f}"
    )

    selected_data = {
        name: np.asarray(values[:, rows[:64], cols[:64]])[:, None, :]
        for name, values in data.items()
    }
    selected_incumbent = model.predict(selected_data, dict(model.PARAMS), None)
    selected_trial = candidate(
        selected_incumbent, saturation_signal(selected_incumbent, selected_data), 0.2
    )
    changed_data = {name: values.copy() for name, values in selected_data.items()}
    for values in changed_data.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed_data, dict(model.PARAMS), None)
    changed_trial = candidate(
        changed_incumbent, saturation_signal(changed_incumbent, changed_data), 0.2
    )
    print(
        f"PREFIX future_half_after=96 max_abs={np.max(np.abs(selected_trial[:96]-changed_trial[:96])):.12g}"
    )

    survivors = []
    for strength in (0.05, 0.10, 0.20, 0.40):
        trial = candidate(incumbent, signal, strength)
        annual, cycle = losses(trial, observed, area, cells, folds)
        annual_gain = base_annual - annual
        cycle_gain = base_cycle - cycle
        held = bool(
            np.all(cycle_gain > 0.0)
            and annual_gain.sum() >= -0.05 * cycle_gain.sum()
        )
        print(
            f"BRACKET strength={strength:g} held={int(held)} annual_gain="
            + ",".join(f"{value:+.9f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.9f}" for value in cycle_gain)
        )
        if held:
            survivors.append((strength, trial))

    if not survivors:
        print("EXACT skipped: no fixed bracket clears every held cycle gate")
        return
    for strength, trial in survivors:
        score = evaluator.score(validate_prediction(trial))["global"]
        print(
            f"EXACT strength={strength:g} overall={score['overall_score']:.9f} "
            f"delta={score['overall_score']-base_score['overall_score']:+.9f} "
            f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
            f"seasonal={score['seasonal_cycle_score']:.9f} "
            f"spatial={score['spatial_distribution_score']:.9f}"
        )


if __name__ == "__main__":
    main()
