"""Single exact proxy of managed-open recurrence equilibrium at k=0.5.

This scratch-only candidate applies a globally shared smooth negative feedback
to final hazard.  It boosts low-recurrence managed-open fire capacity and
brakes high-recurrence capacity using causal rain-built fuel and temperature
memory.  No coordinate, region, benchmark field, or future state enters the
prediction equation.
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

from autoresearch.scratchpad.clean_exogenous_rebuild_b867ed7 import metric_line  # noqa: E402
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


STRENGTH = 0.5


def rss_mb() -> float:
    # macOS reports ru_maxrss in bytes.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def candidate(model, prediction, data):
    hazard = -np.log1p(
        -np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0 - 1e-7)
    )
    alpha12 = 1.0 - np.exp(-1.0 / 12.0)
    trailing_hazard = model._antecedent(hazard, alpha12)
    recurrence = trailing_hazard / (trailing_hazard + 0.01)

    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    managed_open = np.clip(rangeland + pasture, 0.0, 1.0)
    managed_access = managed_open / (managed_open + 0.15)

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    annual_rain = 12.0 * model._antecedent(rain, alpha12)
    fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
        -annual_rain / 3000.0
    )
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature12 = model._antecedent(temperature, alpha12)
    warm = model._rising(temperature12, 0.25, 18.0)

    equilibrium = managed_access * (1.0 - 2.0 * recurrence) * fuel * warm
    adjusted_hazard = hazard * np.exp(STRENGTH * equilibrium)
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted_hazard, 0.0, 50.0)),
        dtype=np.float32,
    )


def timed_prediction(model, data):
    started = time.perf_counter()
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    return prediction, time.perf_counter() - started


def main() -> int:
    total_started = time.perf_counter()
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    print(f"LOADED elapsed={time.perf_counter() - total_started:.3f}s rss_mb={rss_mb():.1f}", flush=True)

    baseline, baseline_runtime = timed_prediction(model, data)
    baseline_scores = evaluator.score(baseline)
    print(
        metric_line("baseline", baseline_scores["global"])
        + f"\truntime={baseline_runtime:.3f}s\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    started = time.perf_counter()
    proposed = validate_prediction(candidate(model, baseline, data))
    candidate_runtime = time.perf_counter() - started
    proposed_scores = evaluator.score(proposed)
    print(
        metric_line("managed_recurrence:k=.5", proposed_scores["global"])
        + f"\tdelta={proposed_scores['global']['overall_score'] - baseline_scores['global']['overall_score']:+.9f}"
        + f"\truntime={candidate_runtime:.3f}s\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    positive = 0
    for name in sorted(key for key in proposed_scores if key != "global"):
        old = float(baseline_scores[name]["overall_score"])
        new = float(proposed_scores[name]["overall_score"])
        positive += int(new > old)
        print(
            f"REGION {name} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH positive={positive}/14", flush=True)

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    base_ecology = ecological_statistics(
        baseline, masks, observation, area, land
    )
    new_ecology = ecological_statistics(
        proposed, masks, observation, area, land
    )
    for name in masks:
        old = float(base_ecology[name]["ratio"])
        new = float(new_ecology[name]["ratio"])
        print(
            f"ECOLOGY {name} baseline={old:.9f} candidate={new:.9f} delta={new-old:+.9f} "
            f"phase={new_ecology[name]['phase_months']}",
            flush=True,
        )
    del observation, area, land, masks, base_ecology, new_ecology
    gc.collect()

    full_gain = float(
        proposed_scores["global"]["overall_score"]
        - baseline_scores["global"]["overall_score"]
    )
    original_functions = {
        "footprint": model._local_fire_footprint,
        "annual_closure": model._annual_regime_closure,
        "fragmentation": model._fragmented_managed_recurrence_brake,
    }
    for label, original in original_functions.items():
        setattr(
            model,
            {
                "footprint": "_local_fire_footprint",
                "annual_closure": "_annual_regime_closure",
                "fragmentation": "_fragmented_managed_recurrence_brake",
            }[label],
            lambda prediction, data, p, enabled: prediction,
        )
        try:
            without, runtime = timed_prediction(model, data)
        finally:
            setattr(
                model,
                {
                    "footprint": "_local_fire_footprint",
                    "annual_closure": "_annual_regime_closure",
                    "fragmentation": "_fragmented_managed_recurrence_brake",
                }[label],
                original,
            )
        without_scores = evaluator.score(without)
        without_candidate = validate_prediction(candidate(model, without, data))
        without_candidate_scores = evaluator.score(without_candidate)
        incumbent_contribution = float(
            baseline_scores["global"]["overall_score"]
            - without_scores["global"]["overall_score"]
        )
        gain_without = float(
            without_candidate_scores["global"]["overall_score"]
            - without_scores["global"]["overall_score"]
        )
        print(
            f"NONDUP component={label} incumbent_contribution={incumbent_contribution:+.9f} "
            f"candidate_gain_full={full_gain:+.9f} candidate_gain_without={gain_without:+.9f} "
            f"interaction={full_gain-gain_without:+.9f} runtime={runtime:.3f}s rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del without, without_candidate
        gc.collect()

    prefix = 96
    expected_prefix = proposed[:prefix].copy()
    del proposed, baseline
    gc.collect()
    for values in data.values():
        values[prefix:] *= np.float32(0.5)
    perturbed_base, perturbed_runtime = timed_prediction(model, data)
    perturbed_candidate = validate_prediction(candidate(model, perturbed_base, data))
    difference = float(
        np.max(np.abs(perturbed_candidate[:prefix] - expected_prefix))
    )
    print(
        f"PREFIX future_half_after={prefix} max_abs_difference={difference:.12g} "
        f"runtime={perturbed_runtime:.3f}s rss_mb={rss_mb():.1f}",
        flush=True,
    )
    print(
        f"DONE elapsed={time.perf_counter() - total_started:.3f}s peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
