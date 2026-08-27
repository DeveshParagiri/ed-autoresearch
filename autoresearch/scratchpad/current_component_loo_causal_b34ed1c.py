"""Exact operating-point pruning audit for the causal-rain canonical model.

This scratch diagnostic evaluates the full committed mechanistic model, each
single-component removal, and the pair formed by the two least useful removals.
It never edits the model or records an official evaluation.  Geographic and
ecological masks are used only after prediction as plausibility diagnostics.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.ecological_geography_audit import (  # noqa: E402
    area_statistics,
    cycle_and_annual,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_model,
    validate_prediction,
)


METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)
EXPECTED_HEAD = "95cae58"
EXPECTED_OVERALL = 0.717405482


def regime_masks(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return established ecological masks without future state in prediction."""
    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name]).reshape(16, 12, 180, 360).mean(axis=(0, 1))

    # For audit classification only, long-run rain is the period mean of the
    # primitive monthly forcing.  It never enters the prediction path.
    rain = 12.0 * mean("monthly_precipitation")
    temperature = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    return {
        "intact_tropical_closed": (
            (temperature >= 20.0)
            & (rain >= 1200.0)
            & (canopy >= 20.0)
            & (lai >= 3.0)
            & (natural >= 0.7)
            & (primary >= 0.5)
        ),
        "temperate_closed": (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (canopy >= 15.0)
            & (lai >= 2.5)
            & (natural >= 0.6)
        ),
        "boreal": (
            (temperature < 5.0)
            & (canopy >= 10.0)
            & (natural >= 0.6)
        ),
        "tropical_open": (
            (temperature >= 20.0)
            & (rain >= 500.0)
            & (rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        ),
        "productive_rangeland": (
            (rangeland >= 0.4)
            & (rain >= 250.0)
            & (rain < 1500.0)
            & (biomass >= 0.2)
        ),
        "cropland": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecological_statistics(
    prediction: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observation: np.ndarray,
    area: np.ndarray,
    land: np.ndarray,
) -> dict[str, dict[str, float | int | str]]:
    model_cycle, model_annual = cycle_and_annual(prediction)
    obs_cycle, obs_annual = cycle_and_annual(observation)
    return {
        name: area_statistics(
            mask & land,
            model_cycle,
            model_annual,
            obs_cycle,
            obs_annual,
            area,
        )
        for name, mask in masks.items()
    }


def format_deltas(
    without: Mapping[str, float], full: Mapping[str, float]
) -> str:
    return " ".join(
        f"d_{label}={without[key] - full[key]:+.9f}"
        for label, key in METRICS
    )


def main() -> int:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(
            f"refusing moving-worktree audit: expected {EXPECTED_HEAD}, got {head}"
        )
    model = load_model()
    inputs, components = validate_model(model, require_components=True)
    data = load_inputs(inputs)
    evaluator = GFED5Evaluator(GFED5_PATH)
    params = dict(model.PARAMS)

    full_prediction = validate_prediction(model.predict(data, params, None))
    full_scores = evaluator.score(full_prediction)
    full_global = full_scores["global"]
    if abs(float(full_global["overall_score"]) - EXPECTED_OVERALL) > 5e-10:
        raise RuntimeError(
            "refusing mismatched baseline: "
            f"expected {EXPECTED_OVERALL:.9f}, "
            f"got {float(full_global['overall_score']):.9f}"
        )
    print(
        f"FULL head={head} "
        + " ".join(f"{label}={full_global[key]:.9f}" for label, key in METRICS),
        flush=True,
    )
    print(
        "component\toverall_without\td_overall\td_bias\td_rmse\t"
        "d_seasonal\td_spatial",
        flush=True,
    )

    rows: list[tuple[str, dict[str, float]]] = []
    for component in components:
        enabled = tuple(name for name in components if name != component)
        prediction = validate_prediction(model.predict(data, params, enabled))
        without = dict(evaluator.score(prediction)["global"])
        rows.append((component, without))
        deltas = [without[key] - full_global[key] for _, key in METRICS]
        print(
            component
            + "\t"
            + f"{without['overall_score']:.9f}\t"
            + "\t".join(f"{delta:+.9f}" for delta in deltas),
            flush=True,
        )
        del prediction
        gc.collect()

    ranked = sorted(
        rows,
        key=lambda row: row[1]["overall_score"] - full_global["overall_score"],
        reverse=True,
    )
    print("RANKED_REMOVAL_DELTA", flush=True)
    for component, without in ranked:
        print(
            f"{component} " + format_deltas(without, full_global),
            flush=True,
        )

    suspicious = tuple(component for component, _ in ranked[:2])
    pair_enabled = tuple(
        component for component in components if component not in suspicious
    )
    pair_prediction = validate_prediction(
        model.predict(data, params, pair_enabled)
    )
    pair_scores = evaluator.score(pair_prediction)
    print(
        "PAIR_REMOVE components="
        + ",".join(suspicious)
        + " "
        + format_deltas(pair_scores["global"], full_global),
        flush=True,
    )

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    full_ecology = ecological_statistics(
        full_prediction, masks, observation, area, land
    )

    print("SUSPICIOUS_ECOLOGY", flush=True)
    for component in suspicious:
        enabled = tuple(name for name in components if name != component)
        prediction = validate_prediction(model.predict(data, params, enabled))
        ecology = ecological_statistics(prediction, masks, observation, area, land)
        print(f"REMOVAL component={component}", flush=True)
        for regime in masks:
            old = full_ecology[regime]
            new = ecology[regime]
            print(
                f"{regime} cells={old['cells']} "
                f"full_ratio={float(old['ratio']):.9f} "
                f"without_ratio={float(new['ratio']):.9f} "
                f"d_ratio={float(new['ratio']) - float(old['ratio']):+.9f} "
                f"full_phase={old['phase_months']} "
                f"without_phase={new['phase_months']}",
                flush=True,
            )
        del prediction
        gc.collect()

    pair_ecology = ecological_statistics(
        pair_prediction, masks, observation, area, land
    )
    print("PAIR_ECOLOGY components=" + ",".join(suspicious), flush=True)
    for regime in masks:
        old = full_ecology[regime]
        new = pair_ecology[regime]
        print(
            f"{regime} full_ratio={float(old['ratio']):.9f} "
            f"without_ratio={float(new['ratio']):.9f} "
            f"d_ratio={float(new['ratio']) - float(old['ratio']):+.9f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
