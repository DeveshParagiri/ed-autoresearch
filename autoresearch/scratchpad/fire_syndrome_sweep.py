"""Probe a smooth sub-grid fire-return mechanism derived from the ML teacher.

The learner repeatedly assigns missing fire to low-incumbent cells only when
local climate variability and burnable vegetation identify a fire-carrying
ecological syndrome.  This probe expresses that response as four continuous,
globally shared gates rather than a tree or a geographic dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_model, validate_prediction  # noqa: E402


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -40.0, 40.0)))


def causal_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.empty_like(values, dtype=np.float32)
    spread = np.empty_like(values, dtype=np.float32)
    state = np.zeros(values.shape[1:], dtype=np.float64)
    m2 = np.zeros(values.shape[1:], dtype=np.float64)
    for time in range(values.shape[0]):
        count = time + 1
        current = np.asarray(values[time], dtype=np.float64)
        delta = current - state
        state += delta / count
        m2 += delta * (current - state)
        mean[time] = state
        spread[time] = np.sqrt(m2 / count)
    return mean, spread


def running(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float32)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)

    rain_mean, rain_std = causal_mean_std(data["monthly_precipitation"])
    temp_mean, temp_std = causal_mean_std(data["air_temperature"])
    fuel_state = running(data["gpp"], 12.0)
    fuel = fuel_state / (fuel_state + 0.20)
    annual_rain = np.asarray(data["annual_precipitation"], dtype=np.float32)
    biomass = np.asarray(data["aboveground_biomass"], dtype=np.float32)
    canopy = np.asarray(data["natural_canopy_height"], dtype=np.float32)
    natural = np.asarray(data["natural_vegetation_fraction"], dtype=np.float32)
    crop = np.asarray(data["luh2_cropland_fraction"], dtype=np.float32)
    range_ = np.asarray(data["luh2_rangeland_fraction"], dtype=np.float32)

    rain_seasonality = rain_std / (rain_mean + 25.0)
    open_cover = np.clip(natural * 10.0 / (canopy + 10.0) + range_, 0.0, 1.0)
    seasonal_grass = (
        sigmoid((rain_seasonality - 0.35) / 0.15)
        * sigmoid((annual_rain - 250.0) / 150.0)
        * sigmoid((1500.0 - annual_rain) / 300.0)
        * open_cover
        * fuel
    )
    managed = (
        sigmoid((temp_std - 5.0) / 2.0)
        * np.clip(crop + range_, 0.0, 1.0)
        * fuel
    )
    cold_forest = (
        sigmoid((8.0 - temp_mean) / 3.0)
        * sigmoid((temp_std - 7.0) / 2.0)
        * natural
        * canopy / (canopy + 8.0)
        * biomass / (biomass + 2.0)
    )
    tropical_woodland = (
        sigmoid((temp_mean - 18.0) / 3.0)
        * sigmoid((rain_std - 35.0) / 15.0)
        * sigmoid((18.0 - canopy) / 3.0)
        * natural
        * fuel
    )
    gate = 1.0 - (
        (1.0 - seasonal_grass)
        * (1.0 - managed)
        * (1.0 - cold_forest)
        * (1.0 - tropical_woodland)
    )
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))

    base_score = evaluator.score(baseline)["global"]
    print(f"baseline={base_score['overall_score']:.4f}", flush=True)
    for reference in (0.001, 0.003, 0.01):
        opportunity = np.log1p(reference / (hazard + 1e-7))
        for strength in (0.15, 0.30, 0.50, 0.75):
            shaped = 1.0 - np.exp(
                -hazard * np.exp(np.clip(strength * gate * opportunity, 0.0, 8.0))
            )
            for scale in (0.65, 0.80, 0.95, 1.10):
                candidate = np.clip(shaped * scale, 0.0, 1.0)
                score = evaluator.score(candidate)["global"]
                print(
                    f"reference={reference:g} strength={strength:g} scale={scale:g} "
                    f"overall={score['overall_score']:.4f} bias={score['bias_score']:.4f} "
                    f"rmse={score['rmse_score']:.4f} seasonal={score['seasonal_cycle_score']:.4f} "
                    f"spatial={score['spatial_distribution_score']:.4f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
