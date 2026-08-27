"""Test where cold-forest and productive-range capacity enters the stack."""

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
from autoresearch.scratchpad.surface_capacity_stage_placement_bf42d58 import (  # noqa: E402
    metric_text,
    prepared_inputs,
    transformed_prediction,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, validate_prediction  # noqa: E402


EXPECTED_MODEL_BLOB = "731e1ee048fd1099dffe75d11a738fd9125f8064"
EXPECTED_INCUMBENT = 0.719021686

STAGES = (
    ("pathway_event_scaling", model._pathway_event_scaling),
    ("ecological_brakes", model._ecological_regime_brakes),
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
    "transform",
    "pathway_event_scaling",
    "ecological_brakes",
    "seasonal_rain_capacity",
    "local_footprint",
    "annual_regime_closure",
    "fragment_recurrence",
    "surface_seasonality_capacity",
)


def predict_at_stage(data: dict[str, np.ndarray], insertion: str) -> np.ndarray:
    params = dict(model.PARAMS)
    enabled = set(model.COMPONENTS)
    prediction = transformed_prediction(data, params, enabled)
    inserted = False
    if insertion == "transform":
        prediction = model._ecological_fire_capacity(
            prediction, data, params, enabled
        )
        inserted = True
    for label, function in STAGES:
        if label == "regime_capacity":
            continue
        prediction = function(prediction, data, params, enabled)
        if label == insertion:
            prediction = model._ecological_fire_capacity(
                prediction, data, params, enabled
            )
            inserted = True
    prediction = model._surface_seasonality_capacity(
        prediction, data, params, enabled
    )
    if insertion == "surface_seasonality_capacity":
        prediction = model._ecological_fire_capacity(
            prediction, data, params, enabled
        )
        inserted = True
    if not inserted:
        raise ValueError(f"unknown insertion stage {insertion}")
    return validate_prediction(np.asarray(prediction, dtype=np.float32))


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
        print(
            f"VARIANT insertion={insertion} {metric_text(scores['global'])}",
            flush=True,
        )
        del prediction
        gc.collect()

    incumbent = all_scores["ecological_brakes"]
    if abs(incumbent["global"]["overall_score"] - EXPECTED_INCUMBENT) > 5e-9:
        raise RuntimeError(
            f"failed incumbent reproduction {incumbent['global']['overall_score']:.9f}"
        )
    for insertion, scores in all_scores.items():
        print(
            f"DELTA insertion={insertion} "
            f"overall={scores['global']['overall_score']-EXPECTED_INCUMBENT:+.9f} "
            f"regions_positive={sum(scores[name]['overall_score'] > incumbent[name]['overall_score'] for name in scores if name != 'global')}/14",
            flush=True,
        )
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
