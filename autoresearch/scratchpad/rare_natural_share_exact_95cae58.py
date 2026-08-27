"""Exact two-point correction of the rare-onset natural-cover share.

The earlier falsification summed LUH2 primary and secondary, but prepared LUH2
secondary is not a compositional cover fraction.  This compares the same safe
squared rain-establishment gate using either ED vegetation cover or LUH2 primary
alone.  No canonical or official state is changed.
"""

from __future__ import annotations

import gc
import sys
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
from autoresearch.scratchpad.rare_natural_onset_exact_95cae58 import (  # noqa: E402
    onset_wrapper,
)
from autoresearch.scratchpad.rare_natural_onset_fuel_refinement_95cae58 import (  # noqa: E402
    BASELINE_REGIONAL,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    original = model._rare_lightning_ignition
    state = {"scale": 0.003, "fuel_power": 2.0, "natural_mode": "ed_state"}
    model._rare_lightning_ignition = onset_wrapper(model, original, state)
    results = []
    predictions = {}
    try:
        for mode in ("ed_state", "primary_only"):
            state["natural_mode"] = mode
            prediction = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
            scores = evaluator.score(prediction)
            print(
                metric_line(
                    f"rare_natural_onset_squared:scale=.003:natural={mode}",
                    scores["global"],
                ),
                flush=True,
            )
            results.append((float(scores["global"]["overall_score"]), mode, scores))
            predictions[mode] = prediction.copy()
            del prediction
            gc.collect()

        with Dataset(GFED5_PATH) as dataset:
            reference = np.asarray(dataset.variables["burntArea"][:192])
        observation = (
            reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4))
            / 100.0
        )
        del reference
        area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
        land = load_land_mask()
        masks = regime_masks(data)
        for overall, mode, scores in results:
            positive = 0
            for name in sorted(BASELINE_REGIONAL):
                delta = float(scores[name]["overall_score"]) - BASELINE_REGIONAL[name]
                positive += int(delta > 0.0)
                print(
                    f"REGION natural={mode} {name} "
                    f"score={scores[name]['overall_score']:.9f} delta={delta:+.9f}",
                    flush=True,
                )
            print(
                f"REGIONAL_BREADTH natural={mode} positive={positive}/14",
                flush=True,
            )
            ecology = ecological_statistics(
                predictions[mode], masks, observation, area, land
            )
            for name, values in ecology.items():
                print(
                    f"ECOLOGY natural={mode} {name} "
                    f"ratio={float(values['ratio']):.9f} "
                    f"phase={values['phase_months']}",
                    flush=True,
                )

        best_overall, best_mode, _ = max(results, key=lambda row: row[0])
        expected_prefix = predictions[best_mode][:96].copy()
        del predictions, observation, area, land
        gc.collect()
        for values in data.values():
            values[96:] *= np.float32(0.5)
        state["natural_mode"] = best_mode
        perturbed = validate_prediction(
            model.predict(data, dict(model.PARAMS), None)
        )
        difference = float(
            np.max(np.abs(perturbed[:96] - expected_prefix))
        )
        print(
            f"BEST_VALID natural={best_mode} overall={best_overall:.9f} "
            f"prefix_max_abs_difference={difference:.12g}",
            flush=True,
        )
    finally:
        model._rare_lightning_ignition = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
