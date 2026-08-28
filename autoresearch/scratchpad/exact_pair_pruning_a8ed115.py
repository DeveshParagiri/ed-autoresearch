"""Exact full-land audit of sampled pair-removal survivors at a8ed115.

The pair choices come from ``operating_point_pair_pruning_a8ed115.py`` and are
the narrow, interpretable survivors with negative operating-point interaction.
All interventions are applied in memory.  This script never edits the
canonical model or runs the official evaluator command.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_subterm_loo_ca6848f import (  # noqa: E402
    metric_line,
    score_selected,
    selected_ecological_statistics,
    selected_regime_masks,
)
from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    load_observed,
    load_selected,
)
from autoresearch.scratchpad.operating_point_pair_pruning_a8ed115 import (  # noqa: E402
    run_late,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask, load_model  # noqa: E402


EXPECTED_MODEL_BLOB = "731e1ee048fd1099dffe75d11a738fd9125f8064"
EXPECTED_BASE = 0.719021686
PAIRS = (
    ("regime_capacity", "local_footprint"),
    ("dead_fuel_pool", "annual_regime_closure"),
    ("state_fire_season", "annual_regime_closure"),
    ("annual_regime_closure", "surface_seasonality"),
)


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def main() -> int:
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"model blob changed: {blob}")
    model = load_model()
    evaluator = GFED5Evaluator(GFED5_PATH)
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))[rows, columns]
    print(
        f"DESIGN cells={rows.size} model_blob={blob} exact_reference={EXPECTED_BASE:.9f}",
        flush=True,
    )
    data = load_selected(model.INPUTS, rows, columns)
    observation = load_observed(rows, columns)
    masks = selected_regime_masks(data)

    baseline = run_late(model, data, frozenset())
    baseline_prediction = baseline[:, None, :]
    baseline_scores = score_selected(
        evaluator, baseline_prediction, rows, columns
    )
    observed_score = float(baseline_scores["global"]["overall_score"])
    if abs(observed_score - EXPECTED_BASE) > 5e-8:
        raise RuntimeError(f"baseline mismatch: {observed_score:.12f}")
    baseline_ecology = selected_ecological_statistics(
        baseline, masks, observation, area
    )
    print(
        metric_line("BASE", baseline_scores, baseline_scores)
        + f"\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    del baseline, baseline_prediction
    gc.collect()

    for first, second in PAIRS:
        label = f"{first}+{second}"
        candidate = run_late(model, data, frozenset((first, second)))
        scores = score_selected(evaluator, candidate[:, None, :], rows, columns)
        ecology = selected_ecological_statistics(
            candidate, masks, observation, area
        )
        print(
            metric_line(label, scores, baseline_scores)
            + f"\trss_mb={rss_mb():.1f}",
            flush=True,
        )
        for name in masks:
            old = float(baseline_ecology[name]["ratio"])
            new = float(ecology[name]["ratio"])
            print(
                f"ECOLOGY pair={label} regime={name} ratio={old:.9f}->{new:.9f} "
                f"delta={new-old:+.9f} phase={baseline_ecology[name]['phase_months']}->{ecology[name]['phase_months']}",
                flush=True,
            )
        del candidate, scores, ecology
        gc.collect()
    print(f"DONE peak_rss_mb={rss_mb():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
