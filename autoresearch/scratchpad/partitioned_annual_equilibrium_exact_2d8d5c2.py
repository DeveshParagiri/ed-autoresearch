"""One exact proxy for a partitioned annual-regime closure at strength 0.25.

The incumbent cold-thaw source is preserved exactly.  Only the warm persistent-
fire correction is partitioned continuously between natural-open and managed-
open pathways, avoiding the destructive stacked recurrence multiplier.  This is
a scratch proxy audit and never records official ILAMB.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


STRENGTH = 0.25


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def make_partitioned_closure(model, original):
    def closure(prediction, data, p, enabled):
        if "annual_regime_closure" not in enabled:
            return prediction
        base = np.asarray(prediction, dtype=np.float64)
        warm_parameters = dict(p)
        warm_parameters["cold_thaw_source"] = 0.0
        cold_parameters = dict(p)
        cold_parameters["persistent_warm_open_brake"] = 0.0
        warm_result = original(base, data, warm_parameters, enabled)
        cold_result = original(base, data, cold_parameters, enabled)

        base_hazard = -np.log1p(-np.clip(base, 0.0, 1.0 - 1e-7))
        warm_hazard = -np.log1p(
            -np.clip(warm_result, 0.0, 1.0 - 1e-7)
        )
        cold_hazard = -np.log1p(
            -np.clip(cold_result, 0.0, 1.0 - 1e-7)
        )
        warm_delta = np.zeros_like(base_hazard)
        active = base_hazard > 1e-15
        warm_delta[active] = np.log(
            np.clip(warm_hazard[active] / base_hazard[active], 1e-12, 1e12)
        )
        cold_source = np.maximum(cold_hazard - base_hazard, 0.0)

        alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
        recurrence_hazard = model._antecedent(base_hazard, alpha_12)
        recurrence = recurrence_hazard / (recurrence_hazard + 0.01)
        rangeland = np.clip(
            np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        pasture = np.clip(
            np.asarray(data["luh2_pasture_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        managed_open = np.clip(rangeland + pasture, 0.0, 1.0)
        managed_access = managed_open / (managed_open + 0.15)
        natural = np.clip(
            np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        canopy = np.clip(
            np.asarray(data["natural_canopy_height"], dtype=np.float64),
            0.0,
            None,
        )
        natural_open = natural * 8.0 / (canopy + 8.0)
        managed_share = managed_open / (
            managed_open + natural_open + 0.10
        )

        rain = np.clip(
            np.asarray(data["monthly_precipitation"], dtype=np.float64),
            0.0,
            None,
        )
        annual_rain = 12.0 * model._antecedent(rain, alpha_12)
        fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
            -annual_rain / 3000.0
        )
        temperature = np.asarray(data["air_temperature"], dtype=np.float64)
        temperature_12 = model._antecedent(temperature, alpha_12)
        warm = model._rising(temperature_12, 0.25, 18.0)
        support = managed_access * fuel * warm

        gap = 1.0 - recurrence
        managed_delta = (
            recurrence * warm_delta + STRENGTH * gap * support
        )
        log_adjustment = (
            (1.0 - managed_share) * warm_delta
            + managed_share * managed_delta
        )
        adjusted_hazard = (
            base_hazard * np.exp(np.clip(log_adjustment, -5.0, 5.0))
            + cold_source
        )
        return np.asarray(
            1.0 - np.exp(-np.clip(adjusted_hazard, 0.0, 50.0)),
            dtype=np.float32,
        )

    return closure


def metric_line(label: str, scores: dict[str, dict[str, float]]) -> str:
    values = scores["global"]
    return (
        f"{label}\toverall={values['overall_score']:.9f}"
        f"\tbias={values['bias_score']:.9f}"
        f"\trmse={values['rmse_score']:.9f}"
        f"\tseasonal={values['seasonal_cycle_score']:.9f}"
        f"\tspatial={values['spatial_distribution_score']:.9f}"
    )


def main() -> int:
    started = time.perf_counter()
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    baseline_scores = evaluator.score(baseline)
    print(metric_line("baseline", baseline_scores) + f"\trss_mb={rss_mb():.1f}", flush=True)

    original = model._annual_regime_closure
    model._annual_regime_closure = make_partitioned_closure(model, original)
    try:
        candidate = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    finally:
        model._annual_regime_closure = original
    candidate_scores = evaluator.score(candidate)
    print(
        metric_line("partitioned:k=.25", candidate_scores)
        + f"\tdelta={candidate_scores['global']['overall_score'] - baseline_scores['global']['overall_score']:+.9f}"
        + f"\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    positive = 0
    for region in sorted(key for key in candidate_scores if key != "global"):
        old = float(baseline_scores[region]["overall_score"])
        new = float(candidate_scores[region]["overall_score"])
        positive += int(new > old)
        print(
            f"REGION {region} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH positive={positive}/14", flush=True)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    baseline_ecology = ecological_statistics(baseline, masks, observation, area, land)
    candidate_ecology = ecological_statistics(candidate, masks, observation, area, land)
    for name in masks:
        old = float(baseline_ecology[name]["ratio"])
        new = float(candidate_ecology[name]["ratio"])
        print(
            f"ECOLOGY {name} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f} "
            f"phase={candidate_ecology[name]['phase_months']}",
            flush=True,
        )

    prefix = 96
    expected = candidate[:prefix].copy()
    del baseline, candidate, observation, area, land, masks, baseline_ecology, candidate_ecology
    gc.collect()
    for values in data.values():
        values[prefix:] *= np.float32(0.5)
    model._annual_regime_closure = make_partitioned_closure(model, original)
    try:
        perturbed = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    finally:
        model._annual_regime_closure = original
    difference = float(np.max(np.abs(perturbed[:prefix] - expected)))
    print(f"PREFIX future_half_after={prefix} max_abs_difference={difference:.12g}", flush=True)
    print(
        f"DONE elapsed={time.perf_counter() - started:.3f}s peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
