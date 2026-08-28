"""Exact operating-point component removals for canonical model 121c83c.

This is a pruning diagnostic, not a Shapley approximation.  It evaluates the
full committed model and each of its fifteen one-component removals using the
fixed canonical parameters and the exact fast GFED5 evaluator.
"""

from __future__ import annotations

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


def main() -> int:
    model = load_model()
    inputs, components = validate_model(model)
    data = load_inputs(inputs)
    evaluator = GFED5Evaluator(GFED5_PATH)

    def score(enabled):
        prediction = validate_prediction(
            model.predict(data, dict(model.PARAMS), enabled)
        )
        return evaluator.score(prediction)["global"]

    full = score(None)
    print(
        "component\toverall_without\td_overall\td_bias\td_rmse\t"
        "d_seasonal\td_spatial",
        flush=True,
    )
    rows = []
    for component in components:
        enabled = tuple(name for name in components if name != component)
        without = score(enabled)
        row = (
            component,
            without["overall_score"],
            full["overall_score"] - without["overall_score"],
            full["bias_score"] - without["bias_score"],
            full["rmse_score"] - without["rmse_score"],
            full["seasonal_cycle_score"] - without["seasonal_cycle_score"],
            full["spatial_distribution_score"]
            - without["spatial_distribution_score"],
        )
        rows.append(row)
        print(
            row[0] + "\t" + "\t".join(f"{value:.9f}" for value in row[1:]),
            flush=True,
        )
    print("ranked by current contribution", flush=True)
    for row in sorted(rows, key=lambda item: item[2]):
        print(f"{row[0]}\t{row[2]:+.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
