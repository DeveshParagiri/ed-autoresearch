"""Factor the 121c83c/HEAD GFED5 score plateau into diagnostic oracles.

This file is host-side diagnosis only. GFED5 is read solely to construct and
score counterfactual oracles. Observations, observation-derived ranks, and
coordinates never enter ``model.predict`` or any proposed candidate equation.
The script does not edit the canonical model, official ledger, or progress
artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import (  # noqa: E402
    OVERALL_WEIGHTS,
    GFED5Evaluator,
    _inferred_time_weights,
)
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


PINNED = "121c83c"
METRICS = (
    ("bias_score", "bias"),
    ("rmse_score", "rmse"),
    ("seasonal_cycle_score", "seasonal"),
    ("spatial_distribution_score", "spatial"),
)


def score_line(evaluator: GFED5Evaluator, label: str, values: np.ndarray) -> dict[str, float]:
    score = evaluator.score(validate_prediction(values))["global"]
    print(
        f"SCORE {label:<30} overall={score['overall_score']:.9f} "
        f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
        f"seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}",
        flush=True,
    )
    return score


def factor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inferred-time-weighted mean map and unit-mean monthly shape."""
    weights = _inferred_time_weights(16)
    mean = np.average(values, axis=0, weights=weights)
    cycle = values.reshape(16, 12, 180, 360).mean(axis=0)
    month_weights = np.asarray((31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
    cycle_mean = np.average(cycle, axis=0, weights=month_weights)
    shape = cycle / (cycle_mean[None, ...] + 1e-12)
    return mean.astype(np.float32), shape.astype(np.float32)


def rebuild(mean: np.ndarray, shape: np.ndarray) -> np.ndarray:
    cycle = mean[None, ...] * shape
    return np.asarray(np.clip(np.tile(cycle, (16, 1, 1)), 0.0, 1.0), dtype=np.float32)


def reorder_values(values: np.ndarray, target_rank: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Preserve ``values``' marginal distribution but impose ``target_rank`` ordering."""
    output = np.asarray(values, dtype=np.float32).copy()
    cells = np.flatnonzero(mask.ravel())
    source_sorted = np.sort(values.ravel()[cells])
    target_order = cells[np.argsort(target_rank.ravel()[cells], kind="stable")]
    output.ravel()[target_order] = source_sorted
    return output


def normalize_shape(shape: np.ndarray) -> np.ndarray:
    month_weights = np.asarray((31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
    clipped = np.clip(shape, 0.0, None)
    mean = np.average(clipped, axis=0, weights=month_weights)
    return np.asarray(clipped / (mean[None, ...] + 1e-12), dtype=np.float32)


def phase_match(model_shape: np.ndarray, observed_shape: np.ndarray) -> np.ndarray:
    model_peak = np.argmax(model_shape, axis=0)
    observed_peak = np.argmax(observed_shape, axis=0)
    shift = observed_peak - model_peak
    source = (np.arange(12)[:, None, None] - shift[None, ...]) % 12
    return np.take_along_axis(model_shape, source, axis=0)


def amplitude_match(model_shape: np.ndarray, observed_shape: np.ndarray) -> np.ndarray:
    model_amplitude = np.sqrt(np.mean(np.square(model_shape - 1.0), axis=0))
    observed_amplitude = np.sqrt(np.mean(np.square(observed_shape - 1.0), axis=0))
    ratio = observed_amplitude / (model_amplitude + 1e-8)
    matched = 1.0 + ratio[None, ...] * (model_shape - 1.0)
    return normalize_shape(matched)


def main() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    pinned_blob = subprocess.run(
        ["git", "rev-parse", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head_blob != pinned_blob:
        raise RuntimeError(f"working model blob {head_blob} differs from {PINNED} blob {pinned_blob}")
    print(f"IDENTITY head={head} pinned={PINNED} model_blob={head_blob}", flush=True)

    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    del data

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192], dtype=np.float32)
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / np.float32(100.0)
    del fine

    evaluator = GFED5Evaluator(GFED5_PATH)
    model_mean, model_shape = factor(incumbent)
    observed_mean, observed_shape = factor(observed)
    land = load_land_mask()

    scores: dict[str, dict[str, float]] = {}
    scores["incumbent"] = score_line(evaluator, "incumbent", incumbent)
    scores["climatology"] = score_line(
        evaluator, "incumbent_climatology", rebuild(model_mean, model_shape)
    )
    scores["empirical_oracle"] = score_line(
        evaluator, "observed_climatology_oracle", rebuild(observed_mean, observed_shape)
    )

    print("COMPONENT_HEADROOM empirical oracle minus incumbent", flush=True)
    for key, label in METRICS:
        raw = scores["empirical_oracle"][key] - scores["incumbent"][key]
        weighted = raw * OVERALL_WEIGHTS[key] / sum(OVERALL_WEIGHTS.values())
        perfect = (1.0 - scores["incumbent"][key]) * OVERALL_WEIGHTS[key] / sum(
            OVERALL_WEIGHTS.values()
        )
        print(
            f"HEADROOM {label:<8} raw={raw:+.9f} weighted_overall={weighted:+.9f} "
            f"to_mathematical_one={perfect:+.9f}",
            flush=True,
        )

    print("ANNUAL_MAP_INTERVENTIONS incumbent monthly shape held fixed", flush=True)
    coarse_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    global_scale = float(np.sum(coarse_area * observed_mean) / np.sum(coarse_area * model_mean))
    score_line(evaluator, "global_magnitude_oracle", rebuild(model_mean * global_scale, model_shape))
    rank_oracle = reorder_values(model_mean, observed_mean, land)
    score_line(evaluator, "local_rank_oracle", rebuild(rank_oracle, model_shape))
    marginal_oracle = reorder_values(observed_mean, model_mean, land)
    score_line(evaluator, "marginal_magnitude_oracle", rebuild(marginal_oracle, model_shape))
    score_line(evaluator, "full_annual_map_oracle", rebuild(observed_mean, model_shape))

    print("SEASONAL_INTERVENTIONS incumbent annual map held fixed", flush=True)
    phase = normalize_shape(phase_match(model_shape, observed_shape))
    amplitude = amplitude_match(model_shape, observed_shape)
    phase_amplitude = amplitude_match(phase, observed_shape)
    score_line(evaluator, "phase_oracle", rebuild(model_mean, phase))
    score_line(evaluator, "amplitude_oracle", rebuild(model_mean, amplitude))
    score_line(evaluator, "phase_plus_amplitude", rebuild(model_mean, phase_amplitude))
    score_line(evaluator, "full_cycle_shape_oracle", rebuild(model_mean, observed_shape))


if __name__ == "__main__":
    main()
