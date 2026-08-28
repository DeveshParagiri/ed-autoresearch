"""Exact Overall-first replay of corrected limiting-factor candidates.

The four configurations were named before exact scoring after the corrected
whole-cell held screen.  A mild single-fold timing tradeoff is allowed here;
acceptance requires positive exact Overall and no severe new ecological
pathology.  This is scratch-only and never invokes the recording evaluator.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.corrected_softmin_limiting_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    EXPECTED_MODEL_BLOB,
    PINNED,
    candidate,
    load_pinned,
)
from autoresearch.scratchpad.land_cover_interface_exact_121c83c import (  # noqa: E402
    METRICS,
    REGIONS,
    audit_masks,
    ecology,
    global_area_ratio,
    severe_pathology,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


CONFIGS = (
    ("softmin_beta25_w1", "softmin", 25.0, 1.00),
    ("softmin_beta8_w1", "softmin", 8.0, 1.00),
    ("hardmin_w010", "softmin", np.inf, 0.10),
    ("harmonic_w010", "harmonic", -1.0, 0.10),
)


def main() -> int:
    model = load_pinned()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(
            f"current model blob {current_blob} differs from pinned {EXPECTED_MODEL_BLOB}"
        )

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_scores = evaluator.score(incumbent)
    base_global = base_scores["global"]
    if abs(base_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {base_global['overall_score']:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks, fractional_countries = audit_masks(data)
    base_ecology = ecology(
        incumbent,
        masks,
        fractional_countries,
        observation,
        area,
        land,
    )
    base_area_ratio = global_area_ratio(incumbent, observation, area, land)

    print(
        f"BASE pinned={PINNED} model_blob={current_blob} "
        + " ".join(
            f"{label}={base_global[key]:.9f}" for label, key in METRICS
        )
        + f" area_ratio={base_area_ratio:.9f}",
        flush=True,
    )
    print(
        "BASE_ECOLOGY "
        + ",".join(
            f"{name}:{float(values['ratio']):.9f}"
            for name, values in base_ecology.items()
        ),
        flush=True,
    )

    rows, columns = np.where(land)
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    probe_data = {
        name: np.asarray(
            values[:, rows[probe], columns[probe]],
            dtype=np.float64,
        )[:, None, :]
        for name, values in data.items()
    }
    changed = {name: values.copy() for name, values in probe_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123

    accepted = []
    for label, family, sharpness, blend in CONFIGS:
        trial = validate_prediction(
            candidate(model, data, family, sharpness, blend)
        )
        scores = evaluator.score(trial)
        global_scores = scores["global"]
        ratio = global_area_ratio(trial, observation, area, land)
        trial_ecology = ecology(
            trial,
            masks,
            fractional_countries,
            observation,
            area,
            land,
        )
        pathologies = severe_pathology(base_ecology, trial_ecology)

        before = candidate(model, probe_data, family, sharpness, blend)
        after = candidate(model, changed, family, sharpness, blend)
        prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
        if prefix_max != 0.0:
            raise RuntimeError(f"prefix causality failed for {label}: {prefix_max}")

        print(
            f"EXACT label={label} "
            + " ".join(
                f"{metric}={global_scores[key]:.9f}" for metric, key in METRICS
            )
            + " deltas="
            + ",".join(
                f"{metric}:{global_scores[key]-base_global[key]:+.9f}"
                for metric, key in METRICS
            )
            + f" area_ratio={ratio:.9f} area_delta={ratio-base_area_ratio:+.9f} "
            + f"prefix_max={prefix_max:.12g}",
            flush=True,
        )
        print(
            f"REGIONS label={label} "
            + ",".join(
                f"{region}:{scores[region]['overall_score']-base_scores[region]['overall_score']:+.9f}"
                for region in REGIONS
            ),
            flush=True,
        )
        print(
            f"ECOLOGY label={label} "
            + ",".join(
                f"{name}:{float(base_ecology[name]['ratio']):.9f}"
                f"->{float(values['ratio']):.9f}"
                for name, values in trial_ecology.items()
            )
            + " severe=" + (",".join(pathologies) if pathologies else "none"),
            flush=True,
        )

        overall_delta = global_scores["overall_score"] - base_global["overall_score"]
        if overall_delta > 0.0 and not pathologies:
            accepted.append((overall_delta, label))

    if not accepted:
        print(
            "DECISION accept=0 reason=no_positive_overall_ecologically_safe_candidate",
            flush=True,
        )
        return 0
    accepted.sort(reverse=True)
    delta, label = accepted[0]
    print(
        f"DECISION accept=1 label={label} overall_delta={delta:+.9f} "
        f"rule=overall_first_no_severe_ecology",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
