"""Test a smooth ignition pulse for fire windows missed by the incumbent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_model, validate_prediction  # noqa: E402


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-value, -40.0, 40.0)))


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    annual = cycle.sum(axis=0)
    allocation = cycle / (annual[None, ...] + 1e-12)
    evaluator = GFED5Evaluator(GFED5_PATH)

    cycles = {
        name: np.asarray(values, dtype=np.float64)
        .reshape(16, 12, 180, 360)
        .mean(axis=0)
        for name, values in data.items()
    }

    def anomaly(name: str) -> np.ndarray:
        values = cycles[name]
        return np.clip(
            (values - values.mean(axis=0)[None, ...])
            / (values.std(axis=0)[None, ...] + 1e-6),
            -4.0,
            4.0,
        )

    temperature = anomaly("air_temperature")
    rain = anomaly("monthly_precipitation")
    previous_rain = np.roll(rain, 1, axis=0)
    gpp = anomaly("gpp")
    previous_gpp = np.roll(gpp, 1, axis=0)
    lightning = np.clip(cycles["lightning_flash_rate"], 0.0, None)
    lightning_gate = lightning / (lightning + 0.01)
    crop = cycles["luh2_cropland_fraction"].mean(axis=0)[None, ...]
    pasture = cycles["luh2_pasture_fraction"].mean(axis=0)[None, ...]
    rangeland = cycles["luh2_rangeland_fraction"].mean(axis=0)[None, ...]
    ignition = np.clip(0.15 + lightning_gate + 0.5 * crop + 0.3 * pasture, 0.0, 2.0)
    burnable_fuel = sigmoid((previous_gpp + 0.3) / 0.6) + 0.25 * rangeland

    warm_reopening = sigmoid((temperature - 0.8) / 0.35)
    antecedent_drying = sigmoid((-previous_rain - 0.5) / 0.35)
    current_drying = sigmoid((-rain - 0.4) / 0.35)
    missed_window = np.exp(-allocation / 0.018)
    pulses = {
        "warm": missed_window * warm_reopening * ignition * burnable_fuel,
        "antecedent_dry": missed_window
        * antecedent_drying
        * ignition
        * burnable_fuel,
        "warm_or_dry": missed_window
        * (warm_reopening + antecedent_drying)
        * ignition
        * burnable_fuel,
        "warm_dry_union": missed_window
        * (1.0 - (1.0 - warm_reopening) * (1.0 - antecedent_drying))
        * ignition
        * burnable_fuel,
        "current_or_antecedent_dry": missed_window
        * (1.0 - (1.0 - current_drying) * (1.0 - antecedent_drying))
        * ignition
        * burnable_fuel,
    }

    def report(label: str, prediction: np.ndarray) -> None:
        scores = evaluator.score(prediction)
        global_score = scores["global"]
        print(
            f"{label}\toverall={global_score['overall_score']:.5f}"
            f"\trmse={global_score['rmse_score']:.5f}"
            f"\tseasonal={global_score['seasonal_cycle_score']:.5f}"
            f"\tBONA={scores['bona']['overall_score']:.5f}"
            f"\tTENA={scores['tena']['overall_score']:.5f}"
            f"\tEURO={scores['euro']['overall_score']:.5f}"
            f"\tNHAF={scores['nhaf']['overall_score']:.5f}"
            f"\tSHAF={scores['shaf']['overall_score']:.5f}"
            f"\tAUST={scores['aust']['overall_score']:.5f}",
            flush=True,
        )

    report("incumbent", incumbent)
    for name, pulse in pulses.items():
        for strength in (0.02, 0.08, 0.20, 0.50, 1.0, 2.0):
            adjusted = allocation + strength * pulse
            adjusted /= adjusted.sum(axis=0, keepdims=True) + 1e-12
            prediction = np.tile(annual[None, ...] * adjusted, (16, 1, 1))
            report(f"{name} strength={strength}", prediction.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
