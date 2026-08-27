"""Exact eligibility audit for pruning the stale arid fine-fuel capacity.

The prior official model is loaded directly from its committed source so the
regional and ecological deltas are exact.  The candidate is the committed
canonical model at 2d8d5c2.  This script does not run or record official ILAMB.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
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
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


BASE_COMMIT = "3efb3bf"


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def committed_model(commit: str):
    source = subprocess.check_output(
        ("git", "show", f"{commit}:autoresearch/model.py"),
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType(f"model_{commit}")
    module.__file__ = f"git:{commit}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def metric_line(label: str, scores: dict[str, dict[str, float]]) -> str:
    values = scores["global"]
    return (
        f"{label}\toverall={values['overall_score']:.9f}"
        f"\tbias={values['bias_score']:.9f}"
        f"\trmse={values['rmse_score']:.9f}"
        f"\tseasonal={values['seasonal_cycle_score']:.9f}"
        f"\tspatial={values['spatial_distribution_score']:.9f}"
        f"\tannual_pct={values['model_period_mean_percent']:.9f}"
    )


def main() -> int:
    started = time.perf_counter()
    candidate_model = load_model()
    baseline_model = committed_model(BASE_COMMIT)
    if tuple(candidate_model.INPUTS) != tuple(baseline_model.INPUTS):
        raise RuntimeError("candidate and baseline input contracts differ")
    data = load_inputs(candidate_model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    masks = regime_masks(data)
    land = load_land_mask()
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))

    baseline = validate_prediction(
        baseline_model.predict(data, dict(baseline_model.PARAMS), None)
    )
    baseline_scores = evaluator.score(baseline)
    baseline_ecology = ecological_statistics(baseline, masks, observation, area, land)
    print(metric_line("baseline", baseline_scores) + f"\trss_mb={rss_mb():.1f}", flush=True)
    del baseline, baseline_model
    gc.collect()

    candidate = validate_prediction(
        candidate_model.predict(data, dict(candidate_model.PARAMS), None)
    )
    candidate_scores = evaluator.score(candidate)
    candidate_ecology = ecological_statistics(candidate, masks, observation, area, land)
    print(
        metric_line("candidate", candidate_scores)
        + f"\tdelta={candidate_scores['global']['overall_score'] - baseline_scores['global']['overall_score']:+.9f}"
        + f"\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    positive = 0
    for region in sorted(key for key in candidate_scores if key != "global"):
        old = float(baseline_scores[region]["overall_score"])
        new = float(candidate_scores[region]["overall_score"])
        positive += int(new > old)
        print(
            f"REGION {region} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH positive={positive}/14", flush=True)
    for name in masks:
        old = float(baseline_ecology[name]["ratio"])
        new = float(candidate_ecology[name]["ratio"])
        print(
            f"ECOLOGY {name} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f} "
            f"phase={candidate_ecology[name]['phase_months']}",
            flush=True,
        )

    prefix = 96
    expected = candidate[:prefix].copy()
    del candidate, masks, observation, area, land, baseline_ecology, candidate_ecology
    gc.collect()
    for values in data.values():
        values[prefix:] *= np.float32(0.5)
    perturbed = validate_prediction(
        candidate_model.predict(data, dict(candidate_model.PARAMS), None)
    )
    difference = float(np.max(np.abs(perturbed[:prefix] - expected)))
    print(
        f"PREFIX future_half_after={prefix} max_abs_difference={difference:.12g}",
        flush=True,
    )
    print(
        f"DONE elapsed={time.perf_counter() - started:.3f}s peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
