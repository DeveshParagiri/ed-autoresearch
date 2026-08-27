"""Test a causal site-local fuel-recovery reservoir on the current prediction."""

from __future__ import annotations

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


def main() -> int:
    model = load_model()
    inputs, _ = validate_model(model)
    data = load_inputs(inputs)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)

    for recovery_months in (6.0, 12.0, 24.0, 48.0):
        decay = np.exp(-1.0 / recovery_months)
        for strength in (1.0, 2.0, 4.0, 8.0, 16.0):
            candidate = np.empty_like(baseline)
            scar = np.zeros_like(baseline[0], dtype=np.float64)
            for time in range(baseline.shape[0]):
                candidate[time] = baseline[time] * np.exp(-strength * scar)
                scar = decay * scar + candidate[time]
            score = evaluator.score(candidate)["global"]
            print(
                f"recovery_months={recovery_months:g} strength={strength:g} "
                f"overall={score['overall_score']:.4f} "
                f"bias={score['bias_score']:.4f} "
                f"rmse={score['rmse_score']:.4f} "
                f"seasonal={score['seasonal_cycle_score']:.4f} "
                f"spatial={score['spatial_distribution_score']:.4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
