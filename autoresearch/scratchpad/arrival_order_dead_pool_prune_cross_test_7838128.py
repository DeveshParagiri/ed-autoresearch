"""Exact arrival-order cross-test after the dead-fuel-pool prune.

The supported secondary footprint made arrival order slightly negative in the
9f957d7 operating-point audit.  This diagnostic fixes the dead-pool-pruned
7838128 incumbent and compares four structural choices: current arrival before
secondary footprint, removal, arrival after the footprint, and the single
predeclared half-strength bracket at the current placement.  No coefficient is
searched and no canonical or official artifact is changed.
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

from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "7838128"
EXPECTED_MODEL_BLOB = "de74aa63e2d99b1f1416c4c0fc6f35255966bc33"
EXPECTED_OVERALL = 0.719748275
METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)


def load_pinned():
    source = subprocess.run(
        ("git", "show", f"{PINNED}:autoresearch/model.py"),
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
    module = types.ModuleType("ed_fire_pinned_7838128_arrival_cross")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def checked(prediction, full_grid: bool):
    array = np.asarray(prediction, dtype=np.float64)
    if full_grid:
        return validate_prediction(array)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("sampled prediction contains non-finite values")
    return array


def predict_swapped(model, data, params, full_grid: bool):
    """Apply supported secondary footprint before arrival redistribution."""
    original_arrival = model._ignition_combustibility_arrival_order
    original_secondary = model._secondary_open_footprint

    def identity(prediction, data_, params_, enabled_):
        return prediction

    def secondary_then_arrival(prediction, data_, params_, enabled_):
        expanded = original_secondary(prediction, data_, params_, enabled_)
        return original_arrival(expanded, data_, params_, enabled_)

    model._ignition_combustibility_arrival_order = identity
    model._secondary_open_footprint = secondary_then_arrival
    try:
        return checked(model.predict(data, params, None), full_grid)
    finally:
        model._ignition_combustibility_arrival_order = original_arrival
        model._secondary_open_footprint = original_secondary


def predictions(model, data, *, full_grid: bool = True) -> dict[str, np.ndarray]:
    params = dict(model.PARAMS)
    components = tuple(model.COMPONENTS)
    current = checked(model.predict(data, params, None), full_grid)
    removed = checked(
        model.predict(
            data,
            params,
            tuple(name for name in components if name != "arrival_order"),
        ),
        full_grid,
    )
    swapped = predict_swapped(model, data, params, full_grid)
    weaker_params = dict(params)
    weaker_params["arrival_order_strength"] = -0.125
    weaker = checked(model.predict(data, weaker_params, None), full_grid)
    return {
        "current_before": current,
        "removed": removed,
        "after_secondary": swapped,
        "weaker_before": weaker,
    }


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    variants = predictions(model, data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    all_scores = {
        name: evaluator.score(prediction)
        for name, prediction in variants.items()
    }
    reference = all_scores["current_before"]
    reference_global = reference["global"]
    if abs(reference_global["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(
            f"failed incumbent reproduction {reference_global['overall_score']:.9f}"
        )
    for name, scores in all_scores.items():
        global_score = scores["global"]
        breadth = sum(
            scores[region]["overall_score"] > reference[region]["overall_score"]
            for region in scores
            if region != "global"
        )
        print(
            f"VARIANT name={name} "
            + " ".join(
                f"{label}={global_score[key]:.9f}" for label, key in METRICS
            )
            + f" delta={global_score['overall_score']-reference_global['overall_score']:+.9f} "
            f"regional_breadth={breadth}/14",
            flush=True,
        )
        for region in sorted(key for key in scores if key != "global"):
            old = reference[region]["overall_score"]
            new = scores[region]["overall_score"]
            print(
                f"REGION variant={name} name={region} full={old:.9f} "
                f"candidate={new:.9f} delta={new-old:+.9f}",
                flush=True,
            )

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    for name, prediction in variants.items():
        ecology = ecological_statistics(
            prediction, masks, observation, area, land
        )
        for regime, values in ecology.items():
            print(
                f"ECOLOGY variant={name} regime={regime} "
                f"cells={values['cells']} ratio={float(values['ratio']):.9f} "
                f"phase={values['phase_months']}",
                flush=True,
            )
    del observation, fine
    gc.collect()

    rows, columns = np.nonzero(land)
    index = np.linspace(0, rows.size - 1, 96, dtype=np.int64)
    prefix_rows, prefix_columns = rows[index], columns[index]
    prefix_data = {
        name: np.asarray(values[:, prefix_rows, prefix_columns])[:, None, :]
        for name, values in data.items()
    }
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    original_prefix = predictions(model, prefix_data, full_grid=False)
    changed_prefix = predictions(model, perturbed, full_grid=False)
    for name in original_prefix:
        delta = float(
            np.max(
                np.abs(original_prefix[name][:96] - changed_prefix[name][:96])
            )
        )
        print(f"PREFIX variant={name} max_abs={delta:.12g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
