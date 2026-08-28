"""Exact operating-point leave-one-component-out audit at 9f957d7.

This is a pruning diagnostic, not Shapley attribution.  It evaluates the full
15-component supported-secondary model and each single-component removal with
all parameters and downstream operators fixed.  Components whose removal
improves Overall, or whose contribution is smallest at this operating point,
receive regional and ecological audits.  No canonical or official artifact is
changed.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path
from typing import Mapping

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


PINNED = "9f957d7"
EXPECTED_MODEL_BLOB = "d08831585a66a90ab0d15080fb655871d1c8167c"
EXPECTED_OVERALL = 0.719646904
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
    module = types.ModuleType("ed_fire_pinned_9f957d7_loo")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def delta_text(without: Mapping[str, float], full: Mapping[str, float]) -> str:
    return " ".join(
        f"d_{label}={without[key]-full[key]:+.9f}"
        for label, key in METRICS
    )


def main() -> int:
    model = load_pinned()
    components = tuple(model.COMPONENTS)
    if len(components) != 15:
        raise RuntimeError(f"expected 15 components, got {len(components)}")
    for required in ("arrival_order", "secondary_open_footprint"):
        if required not in components:
            raise RuntimeError(f"missing required component {required}")
    data = load_inputs(model.INPUTS)
    params = dict(model.PARAMS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    full_prediction = validate_prediction(model.predict(data, params, None))
    full_scores = evaluator.score(full_prediction)
    full_global = full_scores["global"]
    if abs(full_global["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(
            f"failed incumbent reproduction {full_global['overall_score']:.9f}"
        )
    print(
        f"FULL commit={PINNED} components={len(components)} "
        + " ".join(
            f"{label}={full_global[key]:.9f}" for label, key in METRICS
        ),
        flush=True,
    )

    rows: list[tuple[str, dict[str, dict[str, float]]]] = []
    for component in components:
        enabled = tuple(name for name in components if name != component)
        prediction = validate_prediction(model.predict(data, params, enabled))
        scores = evaluator.score(prediction)
        rows.append((component, scores))
        print(
            f"DROP component={component} "
            f"without={scores['global']['overall_score']:.9f} "
            + delta_text(scores["global"], full_global),
            flush=True,
        )
        del prediction
        gc.collect()

    ranked = sorted(
        rows,
        key=lambda row: row[1]["global"]["overall_score"]
        - full_global["overall_score"],
        reverse=True,
    )
    print("RANKED_REMOVAL", flush=True)
    for component, scores in ranked:
        contribution = (
            full_global["overall_score"] - scores["global"]["overall_score"]
        )
        print(
            f"component={component} contribution={contribution:+.9f} "
            + delta_text(scores["global"], full_global),
            flush=True,
        )

    negative = [
        component
        for component, scores in ranked
        if scores["global"]["overall_score"] > full_global["overall_score"]
    ]
    weakest = [component for component, _ in ranked[:3]]
    suspicious = tuple(dict.fromkeys((*negative, *weakest)))
    print(
        "SUSPICIOUS components=" + ",".join(suspicious),
        flush=True,
    )

    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks = regime_masks(data)
    full_ecology = ecological_statistics(
        full_prediction, masks, observation, area, land
    )
    score_by_component = {component: scores for component, scores in rows}
    for component in suspicious:
        scores = score_by_component[component]
        print(f"REGIONAL_WARNING component={component}", flush=True)
        for region in sorted(key for key in scores if key != "global"):
            old = full_scores[region]["overall_score"]
            new = scores[region]["overall_score"]
            print(
                f"REGION component={component} name={region} "
                f"full={old:.9f} without={new:.9f} delta={new-old:+.9f}",
                flush=True,
            )

        enabled = tuple(name for name in components if name != component)
        prediction = validate_prediction(model.predict(data, params, enabled))
        ecology = ecological_statistics(
            prediction, masks, observation, area, land
        )
        print(f"ECOLOGICAL_WARNING component={component}", flush=True)
        for regime in masks:
            old = full_ecology[regime]
            new = ecology[regime]
            print(
                f"ECOLOGY component={component} regime={regime} "
                f"cells={old['cells']} full_ratio={float(old['ratio']):.9f} "
                f"without_ratio={float(new['ratio']):.9f} "
                f"delta={float(new['ratio'])-float(old['ratio']):+.9f} "
                f"full_phase={old['phase_months']} without_phase={new['phase_months']}",
                flush=True,
            )
        del prediction
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
