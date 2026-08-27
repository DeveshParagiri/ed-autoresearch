"""Exact scratch audit of half-strength live-fuel green-up braking.

The canonical dry-phase allocator remains unchanged at 0.3.  Only the later
green-up brake is reduced from 2.0 to 1.0.  The script is read-only with
respect to the canonical model and official experiment ledger.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.ecological_geography_audit import (  # noqa: E402
    cycle_and_annual,
)
from autoresearch.scratchpad.phenology_stage_split_80368d8 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    causal_mean_states,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)


def ecology_masks(data, model, land: np.ndarray) -> dict[str, np.ndarray]:
    states = causal_mean_states(data, model)
    rain = states["rain"]
    temperature = states["temperature"]
    primary = states["primary"]
    crop = states["crop"]
    rangeland = states["rangeland"]

    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)

    canopy = mean("natural_canopy_height")
    lai = mean("leaf_area_index")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    return {
        "intact_tropical_closed": land
        & (temperature >= 20.0)
        & (rain >= 1200.0)
        & (canopy >= 20.0)
        & (lai >= 3.0)
        & (natural >= 0.7)
        & (primary >= 0.5),
        "temperate_closed": land
        & (temperature >= 5.0)
        & (temperature < 20.0)
        & (canopy >= 15.0)
        & (lai >= 2.5)
        & (natural >= 0.6),
        "boreal": land
        & (temperature < 5.0)
        & (canopy >= 10.0)
        & (natural >= 0.6),
        "tropical_open": land
        & (temperature >= 20.0)
        & (rain >= 500.0)
        & (rain < 1500.0)
        & (canopy >= 5.0)
        & (canopy < 20.0)
        & (natural >= 0.5),
        "productive_rangeland": land
        & (rangeland >= 0.4)
        & (rain >= 250.0)
        & (rain < 1500.0)
        & (biomass >= 0.2),
        "cropland": land & (crop >= 0.5),
        "arid_low_fuel": land
        & (rain < 250.0)
        & (biomass < 0.3)
        & (lai < 1.0),
    }


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    _, pred_annual = cycle_and_annual(prediction)
    _, obs_annual = cycle_and_annual(observation)
    result: dict[str, float] = {}
    for name, mask in masks.items():
        weight = area * mask
        result[name] = float(np.sum(pred_annual * weight)) / max(
            float(np.sum(obs_annual * weight)), 1e-12
        )
    return result


def score_text(score: dict[str, float]) -> str:
    return " ".join(f"{name}={score[key]:.9f}" for name, key in METRICS)


def main() -> int:
    started = time.perf_counter()
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(
            f"refusing moving-model audit: expected {EXPECTED_MODEL_BLOB}, got {blob}"
        )

    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    params = dict(model.PARAMS)
    if params["fire_season_w"] != 0.3 or params["greenup_brake"] != 2.0:
        raise RuntimeError("unexpected canonical phenology operating point")

    incumbent = validate_prediction(model.predict(data, params, None))
    incumbent_scores = evaluator.score(incumbent)
    candidate_params = dict(params)
    candidate_params["greenup_brake"] = 1.0
    candidate = validate_prediction(
        model.predict(data, candidate_params, None)
    )
    candidate_scores = evaluator.score(candidate)
    print(f"MODEL_BLOB={blob}", flush=True)
    print("INCUMBENT " + score_text(incumbent_scores["global"]), flush=True)
    print("CANDIDATE " + score_text(candidate_scores["global"]), flush=True)
    print(
        "GLOBAL_DELTA "
        + " ".join(
            f"d_{name}={candidate_scores['global'][key] - incumbent_scores['global'][key]:+.9f}"
            for name, key in METRICS
        ),
        flush=True,
    )

    for region in sorted(name for name in candidate_scores if name != "global"):
        old = incumbent_scores[region]["overall_score"]
        new = candidate_scores[region]["overall_score"]
        print(
            f"REGION {region} incumbent={old:.9f} candidate={new:.9f} "
            f"delta={new - old:+.9f}",
            flush=True,
        )

    land = load_land_mask()
    masks = ecology_masks(data, model, land)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    incumbent_ecology = ecology_ratios(incumbent, observation, area, masks)
    candidate_ecology = ecology_ratios(candidate, observation, area, masks)
    for name in incumbent_ecology:
        old = incumbent_ecology[name]
        new = candidate_ecology[name]
        print(
            f"ECOLOGY {name} incumbent={old:.9f} candidate={new:.9f} "
            f"delta={new - old:+.9f}",
            flush=True,
        )

    candidate_prefix = candidate[:96].copy()
    del incumbent, candidate, observation, masks, land
    gc.collect()
    for values in data.values():
        values[96:] *= np.float32(0.5)
    future = validate_prediction(
        model.predict(data, candidate_params, None)
    )
    prefix_delta = float(np.max(np.abs(candidate_prefix - future[:96])))
    print(
        f"PREFIX future_start=96 factor=0.5 max_abs_difference={prefix_delta:.12g}",
        flush=True,
    )
    elapsed = time.perf_counter() - started
    peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # This experiment runs on macOS, where ru_maxrss is bytes.
    peak_gib = peak_raw / (1024.0**3)
    print(
        f"RESOURCES wall_seconds={elapsed:.6f} peak_rss_raw={peak_raw} "
        f"peak_rss_gib={peak_gib:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
