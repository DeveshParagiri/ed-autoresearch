"""Probe a smooth causal wet-growth to dry-curing fire pulse."""

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


def running_mean(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def normalize_causally(factor: np.ndarray, months: float = 12.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(factor[0], dtype=np.float64).copy()
    relative = np.empty_like(factor)
    for time in range(factor.shape[0]):
        state += alpha * (factor[time] - state)
        relative[time] = factor[time] / (state + 1e-12)
    return relative


def main() -> int:
    model = load_model()
    inputs, _ = validate_model(model)
    data = load_inputs(inputs)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)

    def report(label: str, candidate: np.ndarray) -> None:
        score = evaluator.score(np.clip(candidate, 0.0, 1.0))["global"]
        print(
            f"{label} overall={score['overall_score']:.4f} "
            f"bias={score['bias_score']:.4f} rmse={score['rmse_score']:.4f} "
            f"seasonal={score['seasonal_cycle_score']:.4f} "
            f"spatial={score['spatial_distribution_score']:.4f}",
            flush=True,
        )

    for scale in (1.0, 1.15, 1.30, 1.45, 1.60):
        report(f"scale={scale:.2f}", baseline * scale)

    rain = np.asarray(data["monthly_precipitation"], dtype=np.float64)
    for timescale in (6.0, 12.0):
        reservoir = running_mean(rain, timescale)
        deficit = np.maximum(reservoir - rain, 0.0)
        for half_saturation in (25.0, 50.0, 100.0):
            drying = deficit / (deficit + half_saturation)
            antecedent_fuel = reservoir / (reservoir + half_saturation)
            pulse = drying * antecedent_fuel
            for strength in (0.5, 1.0, 2.0, 4.0):
                factor = normalize_causally(np.exp(strength * pulse))
                report(
                    f"scale=1.30 tau={timescale:g} half={half_saturation:g} "
                    f"strength={strength:g}",
                    baseline * 1.30 * factor,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
