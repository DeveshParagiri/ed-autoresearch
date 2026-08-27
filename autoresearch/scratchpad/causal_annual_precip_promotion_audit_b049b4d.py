"""Promotion audit for replacing future-leaking annual rainfall with EMA12.

Scratch only.  This does not edit or officially evaluate the canonical model.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.temperature_pathway_blend import ecological_ratios  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


ORIGINAL_COMMIT = "b049b4d"


def load_original_model():
    source = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{ORIGINAL_COMMIT}")
    exec(compile(source, f"{ORIGINAL_COMMIT}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def metric_text(score: dict[str, float]) -> str:
    return (
        f"overall={score['overall_score']:.9f} bias={score['bias_score']:.9f} "
        f"rmse={score['rmse_score']:.9f} seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}"
    )


def main() -> int:
    model = load_model()
    original_model = load_original_model()
    data = load_inputs(original_model.INPUTS)
    params = dict(model.PARAMS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    original_annual = data["annual_precipitation"].copy()

    original = validate_prediction(
        original_model.predict(data, dict(original_model.PARAMS), None)
    )
    original_scores = evaluator.score(original)
    print("ORIGINAL " + metric_text(original_scores["global"]), flush=True)

    causal_annual = 12.0 * model._antecedent(
        data["monthly_precipitation"], 1.0 - np.exp(-1.0 / 12.0)
    )
    candidate = validate_prediction(model.predict(data, params, None))
    candidate_scores = evaluator.score(candidate)
    print("CAUSAL " + metric_text(candidate_scores["global"]), flush=True)
    print(
        "GLOBAL_DELTA "
        + " ".join(
            f"{name}={candidate_scores['global'][key] - original_scores['global'][key]:+.9f}"
            for name, key in (
                ("overall", "overall_score"),
                ("bias", "bias_score"),
                ("rmse", "rmse_score"),
                ("seasonal", "seasonal_cycle_score"),
                ("spatial", "spatial_distribution_score"),
            )
        ),
        flush=True,
    )
    for region in sorted(name for name in candidate_scores if name != "global"):
        old = original_scores[region]
        new = candidate_scores[region]
        print(
            f"REGION {region} original={old['overall_score']:.9f} "
            f"causal={new['overall_score']:.9f} "
            f"delta={new['overall_score'] - old['overall_score']:+.9f}",
            flush=True,
        )

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    # Preserve the established regime definitions, which were written using
    # the prepared annual-rain field.  Only the predictions differ here.
    data["annual_precipitation"] = original_annual
    original_ecology = ecological_ratios(original, data, observed, area, land)
    candidate_ecology = ecological_ratios(candidate, data, observed, area, land)
    for regime in original_ecology:
        print(
            f"ECOLOGY {regime} original={original_ecology[regime]:.9f} "
            f"causal={candidate_ecology[regime]:.9f} "
            f"delta={candidate_ecology[regime] - original_ecology[regime]:+.9f}",
            flush=True,
        )
    del observed
    gc.collect()

    original_prefix = original[:96].copy()
    candidate_prefix = candidate[:96].copy()
    del original, candidate
    gc.collect()

    # Mutate every future primitive input.  Recompute the derived annualized
    # rainfall from the perturbed monthly series rather than perturbing that
    # derived state independently.
    for name, values in data.items():
        values[96:] *= 0.5
    causal_perturbed = 12.0 * model._antecedent(
        data["monthly_precipitation"], 1.0 - np.exp(-1.0 / 12.0)
    )
    annual_prefix_delta = float(
        np.max(np.abs(causal_annual[:96] - causal_perturbed[:96]))
    )
    candidate_future = validate_prediction(model.predict(data, params, None))
    candidate_prefix_delta = float(
        np.max(np.abs(candidate_prefix - candidate_future[:96]))
    )
    del candidate_future
    gc.collect()

    # The old prepared field also appears prefix invariant when mutated as an
    # opaque input.  That test cannot detect that each early-month value was
    # constructed from later months in the same calendar year; report it only
    # as a contrast with the valid derived-state test above.
    original_future = validate_prediction(
        original_model.predict(data, dict(original_model.PARAMS), None)
    )
    original_prefix_delta = float(
        np.max(np.abs(original_prefix - original_future[:96]))
    )
    print(f"CAUSAL_ANNUAL_PREFIX_MAX_ABS={annual_prefix_delta:.12g}", flush=True)
    print(f"CAUSAL_PREDICTION_PREFIX_MAX_ABS={candidate_prefix_delta:.12g}", flush=True)
    print(f"OPAQUE_ORIGINAL_INPUT_PREFIX_MAX_ABS={original_prefix_delta:.12g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
