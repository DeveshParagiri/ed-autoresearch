"""Exact stage-placement audit of the supported secondary-open footprint.

The footprint equation and strength are pinned to the dead-fuel-pruned 7838128
model.  Only its insertion point changes.  All stages are pointwise and prefix
causal; the script does not edit the canonical model or invoke official
evaluation.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
import types
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
    validate_prediction,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402


EXPECTED_COMMIT = "7838128"
EXPECTED_MODEL_BLOB = "de74aa63e2d99b1f1416c4c0fc6f35255966bc33"
EXPECTED_INCUMBENT = 0.719748275
INSERTIONS = (
    "final",
    "pathway_event_scaling",
    "seasonal_rain_capacity",
    "local_footprint",
    "annual_regime_closure",
    "secondary_litter",
)


def pinned_model():
    source = subprocess.run(
        ("git", "show", f"{EXPECTED_COMMIT}:autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    blob = subprocess.run(
        ("git", "hash-object", "--stdin"),
        cwd=ROOT,
        input=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"unexpected pinned model blob {blob}")
    module = types.ModuleType("ed_fire_pinned_7838128_footprint_stage")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def metric_text(scores: dict[str, float]) -> str:
    return " ".join(
        f"{name}={scores[key]:.9f}"
        for name, key in (
            ("overall", "overall_score"),
            ("bias", "bias_score"),
            ("rmse", "rmse_score"),
            ("seasonal", "seasonal_cycle_score"),
            ("spatial", "spatial_distribution_score"),
        )
    )


def prepared_inputs(model) -> dict[str, np.ndarray]:
    data = dict(load_inputs(model.INPUTS))
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float32)
    data["annual_precipitation"] = 12.0 * model._antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    return data


def transformed_prediction(
    model,
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
    model,
    data: dict[str, np.ndarray],
    insertion: str,
) -> np.ndarray:
    params = dict(model.PARAMS)
    enabled = set(model.COMPONENTS)
    prediction = transformed_prediction(model, data, params, enabled)
    stages = (
        ("pathway_event_scaling", model._pathway_event_scaling),
        ("ecological_brakes", model._ecological_regime_brakes),
        ("regime_capacity", model._ecological_fire_capacity),
        ("seasonal_rain_capacity", model._seasonal_rainfall_capacity),
        ("state_fire_season", model._state_dependent_fire_season),
        ("rare_lightning", model._rare_lightning_ignition),
        ("crop_management", model._rain_conditioned_crop_management),
        ("conditional_allocation", model._conditional_fire_allocation),
        ("greenup_brake", model._live_fuel_greenup_brake),
        ("surface_bank", model._surface_fire_opportunity_bank),
        ("local_footprint", model._local_fire_footprint),
        ("annual_regime_closure", model._annual_regime_closure),
        ("multi_path_bank", model._multi_pathway_opportunity_bank),
        ("fuel_recovery", model._pathway_fuel_recovery_reservoir),
        ("secondary_litter", model._secondary_fuel_litter_banks),
        ("fragment_recurrence", model._fragmented_managed_recurrence_brake),
        ("surface_seasonality", model._surface_seasonality_capacity),
        ("arrival_order", model._ignition_combustibility_arrival_order),
    )
    inserted = False
    for label, function in stages:
        prediction = function(prediction, data, params, enabled)
        if label == insertion:
            prediction = model._secondary_open_footprint(
                prediction, data, params, enabled
            )
            inserted = True
    if insertion == "final":
        prediction = model._secondary_open_footprint(
            prediction, data, params, enabled
        )
        inserted = True
    if not inserted:
        raise ValueError(f"unknown insertion stage {insertion}")
    return validate_prediction(np.asarray(prediction, dtype=np.float32))


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    if "dead_fuel_pool" in model.COMPONENTS:
        raise RuntimeError("pinned stack has not pruned dead-fuel component")
    data = prepared_inputs(model)
    evaluator = GFED5Evaluator(GFED5_PATH)

    incumbent_prediction = predict_at_stage(model, data, "final")
    incumbent_scores = evaluator.score(incumbent_prediction)
    incumbent_global = incumbent_scores["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(
            "failed exact incumbent reproduction "
            f"{incumbent_global['overall_score']:.9f}"
        )
    print(f"VARIANT insertion=final {metric_text(incumbent_global)}", flush=True)

    with Dataset(GFED5_PATH) as dataset:
        fine_observed = np.asarray(dataset.variables["burntArea"][:192])
    observed = (
        fine_observed.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    incumbent_ecology = ecological_ratios(
        incumbent_prediction, data, observed, area, land
    )

    for insertion in INSERTIONS[1:]:
        prediction = predict_at_stage(model, data, insertion)
        scores = evaluator.score(prediction)
        global_scores = scores["global"]
        delta = global_scores["overall_score"] - EXPECTED_INCUMBENT
        positive = sum(
            scores[name]["overall_score"]
            > incumbent_scores[name]["overall_score"]
            for name in scores
            if name != "global"
        )
        print(
            f"VARIANT insertion={insertion} {metric_text(global_scores)} "
            f"delta={delta:+.9f} regions_positive={positive}/14",
            flush=True,
        )
        for name in sorted(key for key in scores if key != "global"):
            print(
                f"REGION insertion={insertion} name={name} "
                f"delta={scores[name]['overall_score']-incumbent_scores[name]['overall_score']:+.9f}",
                flush=True,
            )
        if delta > 0.0:
            ecology = ecological_ratios(
                prediction, data, observed, area, land
            )
            print(
                f"ECOLOGY insertion={insertion} "
                + ",".join(
                    f"{name}:{incumbent_ecology[name]:.6f}->{ecology[name]:.6f}"
                    for name in incumbent_ecology
                ),
                flush=True,
            )
        del prediction
        gc.collect()
    print(f"DONE wall_seconds={time.perf_counter()-started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
