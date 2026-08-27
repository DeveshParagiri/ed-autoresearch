"""Exact operating-point leave-one-out audit of annual-capacity subterms.

The canonical component audit can show that a grouped component is useful while
concealing a stale physical subterm inside it.  This scratch audit disables one
globally shared submechanism at a time without changing any equation, input, or
other parameter.  It is diagnostic only and never records an official result.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_model, validate_prediction  # noqa: E402


SUBTERMS = {
    "capacity:cold_forest": {"cold_forest_capacity": 0.0},
    "capacity:arid_fine_fuel": {"arid_fine_fuel_capacity": 0.0},
    "capacity:productive_range": {"productive_range_brake": 0.0},
    "closure:warm_open": {"persistent_warm_open_brake": 0.0},
    "closure:cold_thaw": {"cold_thaw_source": 0.0},
    "closure:cold_thaw_boost": {"cold_thaw_capacity_boost": 0.0},
}


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def metric_line(label: str, scores: dict[str, dict[str, float]]) -> str:
    values = scores["global"]
    return (
        f"{label}\toverall={values['overall_score']:.9f}"
        f"\tbias={values['bias_score']:.9f}"
        f"\trmse={values['rmse_score']:.9f}"
        f"\tseasonal={values['seasonal_cycle_score']:.9f}"
        f"\tspatial={values['spatial_distribution_score']:.9f}"
    )


def main() -> int:
    started = time.perf_counter()
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    baseline_scores = evaluator.score(baseline)
    baseline_overall = float(baseline_scores["global"]["overall_score"])
    print(metric_line("baseline", baseline_scores) + f"\trss_mb={rss_mb():.1f}", flush=True)
    del baseline
    gc.collect()

    for label, overrides in SUBTERMS.items():
        params = dict(model.PARAMS)
        params.update(overrides)
        candidate = validate_prediction(model.predict(data, params, None))
        scores = evaluator.score(candidate)
        delta = float(scores["global"]["overall_score"]) - baseline_overall
        positive = sum(
            float(scores[region]["overall_score"])
            > float(baseline_scores[region]["overall_score"])
            for region in scores
            if region != "global"
        )
        regional = sorted(
            (
                float(scores[region]["overall_score"])
                - float(baseline_scores[region]["overall_score"]),
                region,
            )
            for region in scores
            if region != "global"
        )
        print(
            metric_line(label, scores)
            + f"\tdelta={delta:+.9f}\tregions_positive={positive}/14"
            + f"\tworst={regional[0][1]}:{regional[0][0]:+.9f}"
            + f"\tbest={regional[-1][1]}:{regional[-1][0]:+.9f}"
            + f"\trss_mb={rss_mb():.1f}",
            flush=True,
        )
        del candidate, scores
        gc.collect()

    print(
        f"DONE elapsed={time.perf_counter() - started:.3f}s peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
