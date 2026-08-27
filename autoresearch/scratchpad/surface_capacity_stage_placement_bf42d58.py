"""Test where coherent surface capacity belongs in the physical operator stack.

The current model applies the bounded surface-capacity response after every
fuel and recurrence state.  This exact-grid diagnostic moves the same equation
upstream, where added event capacity can enter or bypass the conserved banks.
No equation, coefficient, input, or target-dependent field changes between
variants; only the physical stage at which capacity enters changes.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch import model  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    validate_prediction,
)


EXPECTED_MODEL_BLOB = "a1966275d22874d1c71c45c7b8a8f5c8e473358d"
EXPECTED_INCUMBENT = 0.718995365


STAGES = (
    ("ecological_brakes", model._ecological_regime_brakes),
    ("pathway_event_scaling", model._pathway_event_scaling),
    ("regime_capacity", model._ecological_fire_capacity),
    ("seasonal_rain_capacity", model._seasonal_rainfall_capacity),
    ("state_fire_season", model._state_dependent_fire_season),
    ("rare_lightning", model._rare_lightning_ignition),
    ("crop_management", model._rain_conditioned_crop_management),
    ("dead_fuel_pool", model._dead_fuel_pool_response),
    ("conditional_allocation", model._conditional_fire_allocation),
    ("greenup_brake", model._live_fuel_greenup_brake),
    ("surface_bank", model._surface_fire_opportunity_bank),
    ("local_footprint", model._local_fire_footprint),
    ("annual_regime_closure", model._annual_regime_closure),
    ("multi_path_bank", model._multi_pathway_opportunity_bank),
    ("fuel_recovery", model._pathway_fuel_recovery_reservoir),
    ("secondary_litter", model._secondary_fuel_litter_banks),
    ("fragment_recurrence", model._fragmented_managed_recurrence_brake),
)

INSERTIONS = (
    "regime_capacity",
    "greenup_brake",
    "local_footprint",
    "annual_regime_closure",
    "fragment_recurrence",
)


def prepared_inputs() -> dict[str, np.ndarray]:
    data = dict(load_inputs(model.INPUTS))
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float32)
    data["annual_precipitation"] = 12.0 * model._antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    return data


def transformed_prediction(
    data: dict[str, np.ndarray],
    params: dict[str, float],
    enabled: set[str],
) -> np.ndarray:
    rate = model._fire_rate(data, params, enabled)
    if "cropland" in enabled and params.get("crop_k", 0.0) > 0.0:
        crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
        rate *= 1.0 / (
            1.0 + params["crop_k"] * np.power(crop, params["crop_n"])
        )
    if "curing" in enabled:
        rate *= model._curing(data, params)
    if "lag" in enabled:
        rate = model._lag(rate, params)
    return model._transform(rate, params)


def predict_at_stage(
    data: dict[str, np.ndarray],
    insertion: str,
) -> np.ndarray:
    params = dict(model.PARAMS)
    enabled = set(model.COMPONENTS)
    prediction = transformed_prediction(data, params, enabled)
    inserted = False
    for label, function in STAGES:
        prediction = function(prediction, data, params, enabled)
        if label == insertion:
            prediction = model._surface_seasonality_capacity(
                prediction, data, params, enabled
            )
            inserted = True
    if not inserted:
        raise ValueError(f"unknown insertion stage {insertion}")
    return validate_prediction(np.asarray(prediction, dtype=np.float32))


def metric_text(scores: dict[str, float]) -> str:
    keys = (
        "overall_score",
        "bias_score",
        "rmse_score",
        "seasonal_cycle_score",
        "spatial_distribution_score",
    )
    return " ".join(f"{key}={scores[key]:.9f}" for key in keys)


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
    evaluator = GFED5Evaluator(GFED5_PATH)
    all_scores: dict[str, dict[str, dict[str, float]]] = {}
    for insertion in INSERTIONS:
        prediction = predict_at_stage(data, insertion)
        scores = evaluator.score(prediction)
        all_scores[insertion] = scores
        global_scores = scores["global"]
        if insertion == "fragment_recurrence":
            if abs(global_scores["overall_score"] - EXPECTED_INCUMBENT) > 5e-9:
                raise RuntimeError(
                    f"failed incumbent reproduction {global_scores['overall_score']:.9f}"
                )
        print(
            f"VARIANT insertion={insertion} {metric_text(global_scores)}",
            flush=True,
        )
        del prediction
        gc.collect()

    incumbent = all_scores["fragment_recurrence"]
    for insertion, scores in all_scores.items():
        for name in sorted(key for key in scores if key != "global"):
            print(
                f"REGION insertion={insertion} name={name} "
                f"overall={scores[name]['overall_score']:.9f} "
                f"delta={scores[name]['overall_score']-incumbent[name]['overall_score']:+.9f}",
                flush=True,
            )
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
