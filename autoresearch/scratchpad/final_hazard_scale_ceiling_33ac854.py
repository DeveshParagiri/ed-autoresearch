"""Diagnostic ceiling from a globally uniform final-hazard rescaling.

This is not a candidate model. It asks whether the current score gap is merely
an overall-area calibration error or instead requires changing the spatial and
seasonal allocation of fire. The current committed mechanistic prediction is
converted to hazard, multiplied by fixed factors, and scored without changing
``model.py`` or official artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_model,
    validate_model,
    validate_prediction,
)


EXPECTED_MODEL_BLOB = "3f63c96b9317d852e7b2973980ce77cc1bfc1b1f"
FACTORS = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20)


def main() -> int:
    blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"moving canonical model {blob}")

    model = load_model()
    inputs, _ = validate_model(model)
    data = load_inputs(inputs)
    prediction = validate_prediction(
        model.predict(data, dict(model.PARAMS), None)
    )
    evaluator = GFED5Evaluator(GFED5_PATH)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    for factor in FACTORS:
        candidate = np.asarray(
            1.0 - np.exp(-np.clip(factor * hazard, 0.0, 50.0)),
            dtype=np.float32,
        )
        score = evaluator.score(candidate)["global"]
        print(
            f"factor={factor:.2f} "
            f"overall={score['overall_score']:.9f} "
            f"bias={score['bias_score']:.9f} "
            f"rmse={score['rmse_score']:.9f} "
            f"seasonal={score['seasonal_cycle_score']:.9f} "
            f"spatial={score['spatial_distribution_score']:.9f} "
            f"annual_pct={12.0 * score['model_period_mean_percent']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
