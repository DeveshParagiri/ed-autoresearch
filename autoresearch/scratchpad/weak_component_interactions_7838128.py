"""Exact focused pruning interactions after dead-fuel removal at 7838128.

The audit removes only declared mechanistic components. It never edits or
officially evaluates the canonical model.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402
from unrepresented_state_audit_9f957d7 import antecedent  # noqa: E402


PINNED = "7838128"
FOCUS = (
    "arrival_order",
    "phenology",
    "secondary_open_footprint",
    "rare_ignition",
    "curing",
)
PAIR_TESTS = (
    ("arrival_order", "phenology"),
    ("arrival_order", "secondary_open_footprint"),
    ("arrival_order", "rare_ignition"),
    ("arrival_order", "curing"),
    ("phenology", "secondary_open_footprint"),
    ("phenology", "rare_ignition"),
)


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 12.0
    )
    enabled = set(model.COMPONENTS)
    configurations = [("base", ())]
    configurations.extend((f"drop:{name}", (name,)) for name in FOCUS)
    configurations.extend(
        (f"drop:{first}+{second}", (first, second))
        for first, second in PAIR_TESTS
    )

    rows = []
    for label, removed in configurations:
        prediction = validate_prediction(
            model.predict(
                data,
                dict(model.PARAMS),
                components=tuple(sorted(enabled - set(removed))),
            )
        )
        score = evaluator.score(prediction)
        ecology = ecological_ratios(prediction, prepared, observed, area, land)
        rows.append((label, removed, score, ecology))
        global_score = score["global"]
        print(
            f"RESULT {label} overall={global_score['overall_score']:.9f} "
            f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
            f"seasonal={global_score['seasonal_cycle_score']:.9f} "
            f"spatial={global_score['spatial_distribution_score']:.9f}",
            flush=True,
        )
        del prediction
        gc.collect()

    base_label, _, base_score, base_ecology = rows[0]
    del base_label
    base_overall = base_score["global"]["overall_score"]
    deltas = {}
    print("AUDIT")
    for label, removed, score, ecology in rows[1:]:
        delta = score["global"]["overall_score"] - base_overall
        deltas[removed] = delta
        regional = {
            region: score[region]["overall_score"] - base_score[region]["overall_score"]
            for region in score
            if region != "global"
        }
        print(
            f"{label} delta={delta:+.9f} "
            f"regions={sum(value > 0.0 for value in regional.values())}/{len(regional)} "
            + "regional="
            + ",".join(
                f"{region}:{value:+.6f}" for region, value in sorted(regional.items())
            )
        )
        print(
            "ecology="
            + ",".join(
                f"{name}:{base_ecology[name]:.5f}->{ecology[name]:.5f}"
                for name in base_ecology
            )
        )

    print("PAIR_INTERACTIONS")
    for first, second in PAIR_TESTS:
        pair = (first, second)
        interaction = (
            deltas[pair] - deltas[(first,)] - deltas[(second,)]
        )
        print(
            f"{first}+{second} pair_delta={deltas[pair]:+.9f} "
            f"interaction={interaction:+.9f}"
        )


if __name__ == "__main__":
    main()
