"""Exact replay of the six held-stable causal litter timing channels.

The held experiment factors a positive litter multiplier into slow capacity
and fast timing parts with a one-sided exponential mean.  All three declared
strengths at both declared horizons improved annual, normalized-allocation,
and raw-cycle losses in every whole-cell fold.  This script replays those six
mechanistic, target-blind timing laws on the full grid and audits regions,
ecological regimes, Congo, global area, reconstruction, and prefix causality.
It never invokes the recording evaluator or changes the canonical model.
"""

from __future__ import annotations

import gc
import subprocess
import sys
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
from autoresearch.scratchpad.live_dead_litter_exact_tradeoff_121c83c import (  # noqa: E402
    full_litter_state_chunked,
)
from autoresearch.scratchpad.live_dead_litter_mass_balance_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    EXPECTED_MODEL_BLOB,
    PINNED,
    litter_state,
    load_pinned,
)
from autoresearch.scratchpad.two_channel_capacity_timing_121c83c_20260828_a import (  # noqa: E402
    CompactLitter,
    factorized_predictions,
    raw_factor,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


CONFIGS = tuple(
    (strength, tau) for strength in (0.05, 0.10, 0.20) for tau in (12.0, 24.0)
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
        incumbent, masks, fractional_countries, observation, area, land
    )
    base_area_ratio = global_area_ratio(incumbent, observation, area, land)

    full_state, full_closure = full_litter_state_chunked(data, incumbent)
    litter = CompactLitter(
        old_fine=full_state.old_fine,
        litter_load=full_state.litter_load,
        fine_share=full_state.fine_share,
        max_closure=full_closure,
    )
    print(
        f"BASE pinned={PINNED} model_blob={current_blob} "
        + " ".join(f"{label}={base_global[key]:.9f}" for label, key in METRICS)
        + f" area_ratio={base_area_ratio:.9f} mass_closure={full_closure:.12g}",
        flush=True,
    )

    winners: list[tuple[float, float, float]] = []
    for strength, tau in CONFIGS:
        factor = raw_factor("litter", strength, data, litter)
        predictions, base_error, factor_error = factorized_predictions(
            incumbent, factor, tau
        )
        trial = validate_prediction(predictions["timing"])
        scores = evaluator.score(trial)
        global_scores = scores["global"]
        trial_ecology = ecology(
            trial, masks, fractional_countries, observation, area, land
        )
        pathologies = severe_pathology(base_ecology, trial_ecology)
        ratio = global_area_ratio(trial, observation, area, land)
        delta = global_scores["overall_score"] - base_global["overall_score"]
        label = f"litter_timing_b{strength:.2f}_tau{tau:.0f}"
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
            + f"base_recon={base_error:.12g} factor_recon={factor_error:.12g}",
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
        if delta > 0.0 and not pathologies:
            winners.append((delta, strength, tau))
        del factor, predictions, trial, scores, trial_ecology
        gc.collect()

    # Repeat the future-counterfactual test through the exact timing channel.
    rows, columns = np.where(land)
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    before_data = {
        name: np.asarray(values[:, rows[probe], columns[probe]])[:, None, :]
        for name, values in data.items()
    }
    after_data = {name: values.copy() for name, values in before_data.items()}
    for values in after_data.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    prefix_max = 0.0
    before_incumbent = model.predict(before_data, dict(model.PARAMS), None)[:, 0, :]
    after_incumbent = model.predict(after_data, dict(model.PARAMS), None)[:, 0, :]
    before_state_raw = litter_state(before_data, before_incumbent)
    after_state_raw = litter_state(after_data, after_incumbent)
    before_state = CompactLitter(
        old_fine=before_state_raw.old_fine,
        litter_load=before_state_raw.litter_load,
        fine_share=before_state_raw.fine_share,
        max_closure=before_state_raw.closure,
    )
    after_state = CompactLitter(
        old_fine=after_state_raw.old_fine,
        litter_load=after_state_raw.litter_load,
        fine_share=after_state_raw.fine_share,
        max_closure=after_state_raw.closure,
    )
    before_flat = {name: values[:, 0, :] for name, values in before_data.items()}
    after_flat = {name: values[:, 0, :] for name, values in after_data.items()}
    for strength, tau in CONFIGS:
        before_factor = raw_factor("litter", strength, before_flat, before_state)
        after_factor = raw_factor("litter", strength, after_flat, after_state)
        before_prediction = factorized_predictions(
            before_incumbent, before_factor, tau
        )[0]["timing"]
        after_prediction = factorized_predictions(
            after_incumbent, after_factor, tau
        )[0]["timing"]
        local = float(
            np.max(np.abs(before_prediction[:96] - after_prediction[:96]))
        )
        prefix_max = max(prefix_max, local)
        print(
            f"PREFIX strength={strength:.2f} tau={tau:.0f} max_abs={local:.12g}",
            flush=True,
        )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix causality failed: {prefix_max}")

    if not winners:
        print("DECISION accept=0 reason=no_positive_exact_ecologically_safe_candidate")
        return 0
    winners.sort(key=lambda item: item[0])
    delta, strength, tau = winners[-1]
    print(
        f"DECISION accept=1 strength={strength:.2f} tau={tau:.0f} "
        f"overall_delta={delta:+.9f} "
        f"rule=all_fold_all_metric_then_exact_overall_ecology_prefix",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
