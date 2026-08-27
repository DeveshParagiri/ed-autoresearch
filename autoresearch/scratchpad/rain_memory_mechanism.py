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
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
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
    if "--audit-bins" in sys.argv:
        with Dataset(GFED5_PATH) as dataset:
            reference = np.asarray(dataset.variables["burntArea"][:192])
        observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
        land = load_land_mask()[None, ...]
        area = np.cos(
            np.deg2rad(-89.5 + np.arange(180, dtype=np.float64))
        )[None, :, None]
        mass_weight = np.broadcast_to(area, incumbent.shape)

        def audit(name: str, values: np.ndarray) -> None:
            selected_values = values[land.repeat(values.shape[0], axis=0)]
            edges = np.unique(np.quantile(selected_values, np.linspace(0.0, 1.0, 7)))
            print(f"mass audit {name} edges={np.array2string(edges, precision=6)}")
            for index in range(edges.size - 1):
                selected = land & (values >= edges[index]) & (
                    values <= edges[index + 1]
                    if index == edges.size - 2
                    else values < edges[index + 1]
                )
                predicted_mass = float((incumbent * mass_weight * selected).sum())
                observed_mass = float((observed * mass_weight * selected).sum())
                global_observed = float((observed * mass_weight * land).sum())
                print(
                    f"bin={index} ratio={predicted_mass / (observed_mass + 1e-12):.3f} "
                    f"observed_share={observed_mass / global_observed:.3f}",
                    flush=True,
                )

        audit("gpp_memory_24m", gpp_24m)
        audit("dryness_memory_12m", running(np.asarray(data["dryness"]), 12.0))
        audit("rain_memory_12m", rain_12m)
        return 0
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
