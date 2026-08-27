"""Bounded exact sweep of the corrected limiting-factor aggregation.

The earlier soft minimum omitted normalization by the number of physical
factors. That expression can become negative for ordinary positive factors and
was clipped to a near-zero constant. This experiment tests the mathematically
normalized log-mean-exp soft minimum and blends it with the incumbent product.
It is a globally shared mechanistic aggregation, not a fitted response surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch import model  # noqa: E402
from autoresearch.scratchpad.clean_exogenous_rebuild_b867ed7 import (  # noqa: E402
    exogenous_ecology_ratios,
    metric_line,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


def main() -> int:
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    best: tuple[float, float, float, np.ndarray] | None = None
    for sharpness in (1.0, 2.0, 4.0):
        for weight in (0.0, 0.03, 0.07, 0.12, 0.20):
            params = dict(model.PARAMS)
            params["soft_s"] = sharpness
            params["soft_w"] = weight
            prediction = validate_prediction(model.predict(data, params=params))
            score = dict(evaluator.score(prediction)["global"])
            print(
                metric_line(
                    f"normalized_softmin:sharp={sharpness:g}:weight={weight:g}",
                    score,
                ),
                flush=True,
            )
            candidate = (float(score["overall_score"]), sharpness, weight, prediction)
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None
    overall, sharpness, weight, prediction = best
    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * model._antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float32),
        1.0 - np.exp(-1.0 / 12.0),
    )
    ecology = exogenous_ecology_ratios(prediction, prepared, evaluator)
    print(
        f"BEST overall={overall:.9f} sharp={sharpness:g} weight={weight:g}",
        flush=True,
    )
    print(
        "ECOLOGY " + " ".join(f"{name}={value:.6f}" for name, value in ecology.items()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
