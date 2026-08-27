"""Sensitivity of the current model to the frozen ED ecosystem-state input.

Read-only diagnostic.  Perturbs only the eight variables stored in ed.nc and
reports score change plus prediction-space dependence.  No candidate model or
official result is changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_model, validate_model, validate_prediction  # noqa: E402


ED_INPUTS = (
    "gpp",
    "aboveground_biomass",
    "soil_carbon",
    "leaf_area_index",
    "natural_canopy_height",
    "secondary_canopy_height",
    "natural_vegetation_fraction",
    "secondary_vegetation_fraction",
)
FRACTIONS = {"natural_vegetation_fraction", "secondary_vegetation_fraction"}


def main() -> int:
    model = load_model()
    names, _ = validate_model(model)
    data = load_inputs(names)
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    baseline_score = evaluator.score(baseline)["global"]["overall_score"]
    area = np.cos(np.deg2rad(-89.5 + np.arange(180, dtype=np.float64)))[None, :, None]
    baseline_area = float(np.sum(baseline * area))

    if "--future-causality" in sys.argv:
        changed = dict(data)
        for name in names:
            values = np.asarray(data[name], dtype=np.float32).copy()
            values[96:] *= 0.50
            if name in FRACTIONS or name.startswith("luh2_"):
                values = np.clip(values, 0.0, 1.0)
            changed[name] = values
        prediction = validate_prediction(
            model.predict(changed, dict(model.PARAMS), None)
        )
        past = baseline[:96]
        altered_past = prediction[:96]
        relative_l1 = float(np.sum(np.abs(altered_past - past) * area)) / (
            float(np.sum(past * area)) + 1e-12
        )
        maximum = float(np.max(np.abs(altered_past - past)))
        print(
            "future inputs x0.50 after month 96"
            f"\tpast_normalized_l1={relative_l1:.6f}"
            f"\tpast_max_abs={maximum:.6f}",
            flush=True,
        )
        return 0

    def report(label: str, changed: dict[str, np.ndarray]) -> None:
        prediction = validate_prediction(model.predict(changed, dict(model.PARAMS), None))
        score = evaluator.score(prediction)["global"]["overall_score"]
        l1 = float(np.sum(np.abs(prediction - baseline) * area)) / (baseline_area + 1e-12)
        area_change = float(np.sum(prediction * area)) / (baseline_area + 1e-12) - 1.0
        correlation = float(np.corrcoef(baseline.ravel(), prediction.ravel())[0, 1])
        print(
            f"{label}\toverall={score:.4f}\tdelta={score-baseline_score:+.4f}"
            f"\tnormalized_l1={l1:.4f}\tarea_delta={area_change:+.4f}"
            f"\tprediction_corr={correlation:.6f}",
            flush=True,
        )

    print(f"baseline\toverall={baseline_score:.4f}", flush=True)
    if "--bundle-only" not in sys.argv and "--temporal-only" not in sys.argv:
        for name in ED_INPUTS:
            for factor in (0.9, 1.1):
                changed = dict(data)
                values = np.asarray(data[name], dtype=np.float32) * factor
                if name in FRACTIONS:
                    values = np.clip(values, 0.0, 1.0)
                changed[name] = values
                report(f"{name} x{factor:.1f}", changed)

    if "--temporal-only" not in sys.argv:
        for factor in (0.50, 0.75, 0.90, 1.10, 1.25, 1.50):
            changed = dict(data)
            for name in ED_INPUTS:
                values = np.asarray(data[name], dtype=np.float32) * factor
                if name in FRACTIONS:
                    values = np.clip(values, 0.0, 1.0)
                changed[name] = values
            report(f"all ED state x{factor:.2f}", changed)

    if "--bundle-only" not in sys.argv:
        for name in ED_INPUTS:
            changed = dict(data)
            mean = np.asarray(data[name], dtype=np.float32).reshape(
                16, 12, 180, 360
            ).mean(axis=(0, 1))
            changed[name] = np.broadcast_to(mean, (192, 180, 360))
            report(f"{name} temporal variation removed", changed)

    changed = dict(data)
    for name in ED_INPUTS:
        mean = np.asarray(data[name], dtype=np.float32).reshape(16, 12, 180, 360).mean(axis=(0, 1))
        changed[name] = np.broadcast_to(mean, (192, 180, 360))
    report("all ED temporal variation removed", changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
