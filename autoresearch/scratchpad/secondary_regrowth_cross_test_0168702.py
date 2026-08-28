"""Cross-test the weak secondary footprint after cross-stratum propagation."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402
from secondary_regrowth_footprint_33ac854 import (  # noqa: E402
    MONTH_DAYS,
    candidate,
    losses,
    secondary_regrowth_states,
)


PINNED = "0168702"


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
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    state = secondary_regrowth_states(data)["structural"]
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    weight = area * observed_annual
    ranking = np.argsort(weight.ravel())[::-1]
    coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
    cells = ranking[: int(np.searchsorted(coverage, 0.90) + 1)]
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    base_annual, base_cycle = losses(incumbent, observed, area, cells, folds)
    base_score = evaluator.score(incumbent)["global"]
    print(f"BASE overall={base_score['overall_score']:.9f}")
    survivors = []
    for strength in (0.25, 0.5):
        trial = candidate(incumbent, state, strength)
        annual, cycle = losses(trial, observed, area, cells, folds)
        annual_gain = base_annual - annual
        cycle_gain = base_cycle - cycle
        held = bool(
            np.all(annual_gain > 0.0)
            and -cycle_gain.sum() <= 0.05 * annual_gain.sum()
        )
        print(
            f"strength={strength:g} held={held} annual_gain="
            + ",".join(f"{value:+.6f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.6f}" for value in cycle_gain)
        )
        if held:
            survivors.append((strength, trial))
    if not survivors:
        print("EXACT skipped: no stable held survivor")
        return
    # The .5 strength was already the conservative exact survivor on 33ac854;
    # prefer it here if its held sign survives the small parent mechanism.
    strength, trial = survivors[-1]
    global_score = evaluator.score(validate_prediction(trial))["global"]
    print(
        f"EXACT strength={strength:g} overall={global_score['overall_score']:.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f}"
    )


if __name__ == "__main__":
    main()
