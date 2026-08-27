"""Probe smooth causal rain-memory mechanisms before a model experiment.

Every field is current or an exponentially decayed prior state at the same
site. The probe contains no coordinates, regions, neighbours, or completed-
record climatology. It tests whether wet-season fuel construction followed by
drying can sharpen recurrent fire without introducing annual fire mass.
"""

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
    validate_prediction,
)


def running(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def trailing_annual(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        start = max(0, time - 11)
        output[time] = values[start : time + 1].sum(axis=0)
        output[time] *= 12.0 / (time - start + 1)
    return output


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> None:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} "
        f"bias={score['bias_score']:.4f} rmse={score['rmse_score']:.4f} "
        f"seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}",
        flush=True,
    )


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float64)
    rain_6m = running(rain, 6.0)
    rain_12m = running(rain, 12.0)
    drying = np.maximum((rain_6m - rain) / (rain_6m + rain + 10.0), 0.0)
    fuel_construction = (
        rain_12m / (rain_12m + 15.0)
        * 120.0 / (rain_12m + 120.0)
    )
    gpp_24m = running(np.asarray(data["gpp"], dtype=np.float64), 24.0)
    fine_fuel = gpp_24m / (gpp_24m + 0.35)
    canopy = np.asarray(data["natural_canopy_height"], dtype=np.float64)
    natural = np.asarray(data["natural_vegetation_fraction"], dtype=np.float64)
    open_natural = natural * 10.0 / (canopy + 10.0)
    trailing = trailing_annual(incumbent)
    recurrent = trailing / (trailing + 0.04)
    low_opportunity = 1.0 / (1.0 + trailing / 0.04)

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    families = {
        "recurrent-rain-curing": drying * fuel_construction * fine_fuel * recurrent,
        "open-rain-curing": drying * fuel_construction * fine_fuel * open_natural,
        "low-opportunity-rain-curing": (
            drying * fuel_construction * fine_fuel * open_natural * low_opportunity
        ),
    }
    for label, mechanism in families.items():
        for strength in (0.25, 0.5, 1.0, 2.0, 4.0):
            factor = np.exp(np.clip(strength * mechanism, -5.0, 5.0))
            normalizer = running(factor, 12.0)
            relative = factor / (normalizer + 1e-12)
            candidate = np.clip(incumbent * relative, 0.0, 1.0).astype(np.float32)
            report(evaluator, f"{label} strength={strength:g}", candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
