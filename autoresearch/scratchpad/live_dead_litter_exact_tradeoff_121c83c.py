"""Exact Overall-first replay of the strongest live-to-dead litter forms.

Six fixed forms from the held experiment are replayed despite limited raw-cycle
fold reversals.  Acceptance requires positive exact Overall and no severe new
ecological pathology.  The live/dead balance is evaluated in spatial chunks
with the exact original equations to limit memory without changing the local
pointwise result.  This script never invokes the recording evaluator.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.land_cover_interface_exact_121c83c import (  # noqa: E402
    METRICS,
    REGIONS,
    audit_masks,
    ecology,
    global_area_ratio,
    severe_pathology,
)
from autoresearch.scratchpad.live_dead_litter_mass_balance_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    EXPECTED_MODEL_BLOB,
    PINNED,
    apply_formulation,
    litter_state,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


CONFIGS = tuple(
    (f"load_replacement_w{blend:.2f}", "load_replacement", blend)
    for blend in (0.10, 0.25, 0.50)
) + tuple(
    (f"relative_allocation_w{blend:.2f}", "relative_allocation", blend)
    for blend in (0.10, 0.25, 0.50)
)


@dataclass
class CompactState:
    old_fine: np.ndarray
    litter_load: np.ndarray
    readiness: np.ndarray
    fine_share: np.ndarray


def full_litter_state_chunked(
    data: dict[str, np.ndarray],
    incumbent: np.ndarray,
    chunk_size: int = 4096,
) -> tuple[CompactState, float]:
    """Run the exact litter equations per independent spatial chunk."""
    shape = incumbent.shape
    time, cells = shape[0], int(np.prod(shape[1:]))
    incumbent_flat = np.asarray(incumbent).reshape(time, cells)
    output = {
        name: np.empty((time, cells), dtype=np.float32)
        for name in ("old_fine", "litter_load", "readiness", "fine_share")
    }
    max_chunk_closure = 0.0
    for start in range(0, cells, chunk_size):
        stop = min(start + chunk_size, cells)
        selected = {
            name: np.asarray(values).reshape(time, cells)[:, start:stop][:, None, :]
            for name, values in data.items()
        }
        state = litter_state(selected, incumbent_flat[:, start:stop])
        for name in output:
            output[name][:, start:stop] = getattr(state, name)
        max_chunk_closure = max(max_chunk_closure, state.closure)
        del selected, state
    return (
        CompactState(
            **{name: values.reshape(shape) for name, values in output.items()}
        ),
        float(max_chunk_closure),
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
    probe_incumbent = model.predict(probe_data, dict(model.PARAMS), None)[:, 0, :]
    probe_state = litter_state(probe_data, probe_incumbent)
    changed = {name: values.copy() for name, values in probe_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)[:, 0, :]
    changed_state = litter_state(changed, changed_incumbent)

    for label, formulation, blend in CONFIGS:
        before, _ = apply_formulation(
            formulation,
            probe_incumbent,
            probe_state,
            blend,
        )
        after, _ = apply_formulation(
            formulation,
            changed_incumbent,
            changed_state,
            blend,
        )
        prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
        if prefix_max != 0.0:
            raise RuntimeError(f"prefix causality failed for {label}: {prefix_max}")
        print(
            f"PREFIX label={label} max_abs={prefix_max:.12g} "
            f"state_closure={probe_state.closure:.12g} "
            f"changed_state_closure={changed_state.closure:.12g}",
            flush=True,
        )

    full_state, max_chunk_closure = full_litter_state_chunked(data, incumbent)
    print(
        f"STATE max_chunk_relative_closure={max_chunk_closure:.12g} "
        f"old_fine_mean={full_state.old_fine.mean():.9f} "
        f"litter_mean={full_state.litter_load.mean():.9f} "
        f"readiness_mean={full_state.readiness.mean():.9f} "
        f"fine_share_mean={full_state.fine_share.mean():.9f}",
        flush=True,
    )

    accepted = []
    for label, formulation, blend in CONFIGS:
        trial, _ = apply_formulation(
            formulation,
            incumbent,
            full_state,
            blend,
        )
        trial = validate_prediction(trial)
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
            + f"mass_closure={max_chunk_closure:.12g}",
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
        del trial, scores, trial_ecology
        gc.collect()

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
