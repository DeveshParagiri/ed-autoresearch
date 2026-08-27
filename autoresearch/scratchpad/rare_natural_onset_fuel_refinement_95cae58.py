"""Two-point exact refinement of rare-onset dry-limb fuel support.

This runs only the squared annual rain-establishment form requested after the
linear source bracket.  It is scratch diagnostic work, not an official model.
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
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


BASELINE_REGIONAL = {
    "aust": 0.694427952,
    "boas": 0.605967176,
    "bona": 0.721736204,
    "ceam": 0.482286537,
    "ceas": 0.568253278,
    "eqas": 0.538699445,
    "euro": 0.536061068,
    "mide": 0.495173288,
    "nhaf": 0.740110657,
    "nhsa": 0.516030529,
    "seas": 0.671854064,
    "shaf": 0.761591848,
    "shsa": 0.502292982,
    "tena": 0.459173548,
}


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    original = model._rare_lightning_ignition
    state = {"scale": 0.0, "fuel_power": 2.0}
    model._rare_lightning_ignition = onset_wrapper(model, original, state)
    results = []
    predictions = {}
    try:
        for scale in (0.0015, 0.003):
            state["scale"] = scale
            prediction = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
            scores = evaluator.score(prediction)
            print(
                metric_line(
                    f"rare_natural_onset_squared_fuel:scale={scale:g}",
                    scores["global"],
                ),
                flush=True,
            )
            results.append((float(scores["global"]["overall_score"]), scale, scores))
            predictions[scale] = prediction.copy()
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
        for overall, scale, scores in results:
            positive = 0
            for name in sorted(BASELINE_REGIONAL):
                delta = float(scores[name]["overall_score"]) - BASELINE_REGIONAL[name]
                positive += int(delta > 0.0)
                print(
                    f"REGION scale={scale:g} {name} "
                    f"score={scores[name]['overall_score']:.9f} delta={delta:+.9f}",
                    flush=True,
                )
            print(
                f"REGIONAL_BREADTH scale={scale:g} positive={positive}/14",
                flush=True,
            )
            ecology = ecological_statistics(
                predictions[scale], masks, observation, area, land
            )
            for name, values in ecology.items():
                print(
                    f"ECOLOGY scale={scale:g} {name} "
                    f"ratio={float(values['ratio']):.9f} "
                    f"phase={values['phase_months']}",
                    flush=True,
                )

        best_overall, best_scale, _ = max(results, key=lambda row: row[0])
        expected_prefix = predictions[best_scale][:96].copy()
        del predictions, observation, area, land
        gc.collect()
        for values in data.values():
            values[96:] *= np.float32(0.5)
        state["scale"] = best_scale
        perturbed = validate_prediction(
            model.predict(data, dict(model.PARAMS), None)
        )
        difference = float(
            np.max(np.abs(perturbed[:96] - expected_prefix))
        )
        print(
            f"BEST_SQUARED scale={best_scale:g} overall={best_overall:.9f} "
            f"prefix_max_abs_difference={difference:.12g}",
            flush=True,
        )
    finally:
        model._rare_lightning_ignition = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
