"""Probe smooth state-defined annual corrections before editing model.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_model, validate_prediction  # noqa: E402


def rising(value: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(value - center) / width, -40.0, 40.0)))


def falling(value: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 - rising(value, center, width)


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)

    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=(0, 1))

    temperature = mean("air_temperature")
    rain = mean("annual_precipitation")
    gpp = mean("gpp")
    crop = mean("luh2_cropland_fraction")
    pasture = mean("luh2_pasture_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    natural = mean("natural_vegetation_fraction")

    gates = {
        "cool_crop_18": crop * falling(temperature, 18.0, 3.0),
        "cool_crop_22": crop * falling(temperature, 22.0, 3.0),
        "cool_managed": (crop + 0.35 * pasture)
        * falling(temperature, 20.0, 3.0),
        "cool_productive_range": rangeland
        * falling(temperature, 20.0, 3.0)
        * rising(gpp, 0.4, 0.2)
        * rising(rain, 350.0, 120.0),
        "warm_seasonal_range": rangeland
        * rising(temperature, 20.0, 3.0)
        * rising(rain, 350.0, 120.0)
        * falling(rain, 1500.0, 250.0)
        * rising(gpp, 0.25, 0.15),
        "warm_dry_fuel": (0.5 * natural + 0.5 * rangeland)
        * rising(temperature, 20.0, 3.0)
        * falling(rain, 900.0, 180.0)
        * rising(gpp, 0.15, 0.10),
    }

    def report(label: str, prediction: np.ndarray) -> float:
        scores = evaluator.score(prediction)
        global_score = scores["global"]
        print(
            f"{label}\toverall={global_score['overall_score']:.5f}"
            f"\tbias={global_score['bias_score']:.5f}"
            f"\trmse={global_score['rmse_score']:.5f}"
            f"\tspatial={global_score['spatial_distribution_score']:.5f}"
            f"\tBONA={scores['bona']['overall_score']:.5f}"
            f"\tTENA={scores['tena']['overall_score']:.5f}"
            f"\tEURO={scores['euro']['overall_score']:.5f}"
            f"\tSHSA={scores['shsa']['overall_score']:.5f}"
            f"\tSHAF={scores['shaf']['overall_score']:.5f}"
            f"\tAUST={scores['aust']['overall_score']:.5f}",
            flush=True,
        )
        return float(global_score["overall_score"])

    report("incumbent", incumbent)
    for name in (
        "cool_crop_18",
        "cool_crop_22",
        "cool_managed",
        "cool_productive_range",
    ):
        for strength in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            candidate = incumbent * np.exp(-strength * gates[name])[None, ...]
            report(f"brake {name} strength={strength}", candidate.astype(np.float32))
    for name in ("warm_seasonal_range", "warm_dry_fuel"):
        for strength in (0.25, 0.5, 1.0, 1.5, 2.0):
            candidate = incumbent * np.exp(strength * gates[name])[None, ...]
            report(f"boost {name} strength={strength}", np.clip(candidate, 0.0, 1.0).astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
