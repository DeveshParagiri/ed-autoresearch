"""Exact pair-removal audit for the weakest current physical components."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_model,
    validate_model,
    validate_prediction,
)


WEAK = (
    "arrival_order",
    "dead_fuel_pool",
    "phenology",
    "secondary_open_footprint",
)
ANCHOR = "rare_ignition"


def main() -> int:
    model = load_model()
    inputs, components = validate_model(model)
    data = load_inputs(inputs)
    evaluator = GFED5Evaluator(GFED5_PATH)

    def score(removed):
        enabled = tuple(name for name in components if name not in removed)
        prediction = validate_prediction(
            model.predict(data, dict(model.PARAMS), enabled)
        )
        return evaluator.score(prediction)["global"]

    full = score(frozenset())
    singles = {
        name: score(frozenset((name,))) for name in (*WEAK, ANCHOR)
    }
    pairs = list(itertools.combinations(WEAK, 2))
    pairs += [(name, ANCHOR) for name in WEAK]

    print(f"full\t{full['overall_score']:.9f}", flush=True)
    print(
        "pair\toverall_without\tdelta_from_full\tinclusion_interaction\t"
        "d_bias\td_rmse\td_seasonal\td_spatial",
        flush=True,
    )
    for first, second in pairs:
        without = score(frozenset((first, second)))
        interaction = (
            full["overall_score"]
            - singles[first]["overall_score"]
            - singles[second]["overall_score"]
            + without["overall_score"]
        )
        values = (
            without["overall_score"],
            without["overall_score"] - full["overall_score"],
            interaction,
            without["bias_score"] - full["bias_score"],
            without["rmse_score"] - full["rmse_score"],
            without["seasonal_cycle_score"] - full["seasonal_cycle_score"],
            without["spatial_distribution_score"]
            - full["spatial_distribution_score"],
        )
        print(
            f"{first}+{second}\t"
            + "\t".join(f"{value:+.9f}" for value in values),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
