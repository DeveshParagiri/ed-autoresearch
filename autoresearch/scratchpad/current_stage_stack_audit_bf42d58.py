"""Trace exact score changes through the current mechanistic operator stack."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch import model  # noqa: E402
from autoresearch.scratchpad.surface_capacity_stage_placement_bf42d58 import (  # noqa: E402
    EXPECTED_INCUMBENT,
    EXPECTED_MODEL_BLOB,
    STAGES,
    metric_text,
    prepared_inputs,
    transformed_prediction,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, validate_prediction  # noqa: E402


def main() -> int:
    started = time.perf_counter()
    observed_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"unexpected model blob {observed_blob}")

    data = prepared_inputs()
    params = dict(model.PARAMS)
    enabled = set(model.COMPONENTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    prediction = transformed_prediction(data, params, enabled)
    previous = evaluator.score(validate_prediction(prediction))["global"]
    print(f"STAGE name=transform {metric_text(previous)}", flush=True)

    for label, function in STAGES:
        prediction = function(prediction, data, params, enabled)
        current = evaluator.score(validate_prediction(prediction))["global"]
        print(
            f"STAGE name={label} {metric_text(current)} "
            f"delta_overall={current['overall_score']-previous['overall_score']:+.9f} "
            f"delta_bias={current['bias_score']-previous['bias_score']:+.9f} "
            f"delta_rmse={current['rmse_score']-previous['rmse_score']:+.9f} "
            f"delta_seasonal={current['seasonal_cycle_score']-previous['seasonal_cycle_score']:+.9f} "
            f"delta_spatial={current['spatial_distribution_score']-previous['spatial_distribution_score']:+.9f}",
            flush=True,
        )
        previous = current

    prediction = model._surface_seasonality_capacity(
        prediction, data, params, enabled
    )
    current = evaluator.score(validate_prediction(prediction))["global"]
    print(
        f"STAGE name=surface_seasonality_capacity {metric_text(current)} "
        f"delta_overall={current['overall_score']-previous['overall_score']:+.9f} "
        f"delta_bias={current['bias_score']-previous['bias_score']:+.9f} "
        f"delta_rmse={current['rmse_score']-previous['rmse_score']:+.9f} "
        f"delta_seasonal={current['seasonal_cycle_score']-previous['seasonal_cycle_score']:+.9f} "
        f"delta_spatial={current['spatial_distribution_score']-previous['spatial_distribution_score']:+.9f}",
        flush=True,
    )
    if abs(current["overall_score"] - EXPECTED_INCUMBENT) > 5e-9:
        raise RuntimeError(
            f"failed incumbent reproduction {current['overall_score']:.9f}"
        )
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
