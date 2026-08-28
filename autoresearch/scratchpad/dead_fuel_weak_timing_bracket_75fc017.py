"""Exact weak dead-fuel timing bracket on the reservoir-stage incumbent.

The dead-fuel allocator was removed because its former strength of 3 became
redundant with the pathway and litter reservoirs.  This diagnostic restores
the unchanged causal function at its original stage immediately before
conditional allocation, while keeping the supported secondary footprint at
the new reservoir stage.  Strengths are a fixed, predeclared bracket rather
than an optimization.  Geographic and ecological masks are audit-only.
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


PINNED = "75fc017"
EXPECTED_MODEL_BLOB = "f526cbfa0a9747b78bf71506c665e4b1fd3c8605"
EXPECTED_OVERALL = 0.719756369
STRENGTHS = (0.0, 0.25, 0.5, 1.0, 1.5)
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
    module = types.ModuleType("ed_fire_pinned_75fc017_dead_timing")
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


def predict_strength(model, data, strength: float, *, full_grid: bool = True):
    """Restore the unchanged allocator immediately before allocation."""
    params = dict(model.PARAMS)
    params["dead_fuel_pool_w"] = strength
    original_allocation = model._conditional_fire_allocation

    def dead_then_allocation(prediction, data_, params_, enabled_):
        with_dead = model._dead_fuel_pool_response(
            prediction,
            data_,
            params_,
            set(enabled_) | {"dead_fuel_pool"},
        )
        return original_allocation(with_dead, data_, params_, enabled_)

    model._conditional_fire_allocation = dead_then_allocation
    try:
        return checked(model.predict(data, params, None), full_grid)
    finally:
        model._conditional_fire_allocation = original_allocation


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    variants = {
        strength: predict_strength(model, data, strength)
        for strength in STRENGTHS
    }
    scores = {
        strength: evaluator.score(prediction)
        for strength, prediction in variants.items()
    }
    reference = scores[0.0]
    reference_global = reference["global"]
    if abs(reference_global["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(
            "failed incumbent reproduction "
            f"{reference_global['overall_score']:.9f}"
        )

    for strength in STRENGTHS:
        current = scores[strength]
        global_score = current["global"]
        breadth = sum(
            current[region]["overall_score"]
            > reference[region]["overall_score"]
            for region in current
            if region != "global"
        )
        print(
            f"VARIANT strength={strength:.2f} "
            + " ".join(
                f"{label}={global_score[key]:.9f}" for label, key in METRICS
            )
            + f" delta={global_score['overall_score']-reference_global['overall_score']:+.9f} "
            f"regional_breadth={breadth}/14",
            flush=True,
        )
        for region in sorted(key for key in current if key != "global"):
            old = reference[region]["overall_score"]
            new = current[region]["overall_score"]
            print(
                f"REGION strength={strength:.2f} name={region} "
                f"absent={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
                flush=True,
            )

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    for strength, prediction in variants.items():
        ecology = ecological_statistics(
            prediction, masks, observation, area, land
        )
        for regime, values in ecology.items():
            print(
                f"ECOLOGY strength={strength:.2f} regime={regime} "
                f"cells={values['cells']} ratio={float(values['ratio']):.9f} "
                f"phase={values['phase_months']}",
                flush=True,
            )
    del fine, observation
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
    for strength in STRENGTHS:
        original = predict_strength(
            model, prefix_data, strength, full_grid=False
        )
        changed = predict_strength(
            model, perturbed, strength, full_grid=False
        )
        delta = float(np.max(np.abs(original[:96] - changed[:96])))
        print(
            f"PREFIX strength={strength:.2f} max_abs={delta:.12g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
