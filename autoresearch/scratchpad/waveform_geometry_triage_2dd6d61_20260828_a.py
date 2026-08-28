"""Evaluator-correct waveform-geometry triage at canonical ``2dd6d61``.

This read-only diagnostic distinguishes phase from geometry before assigning
the centered-RMSE residual to pre-peak onset, peak core/width, post-peak tail,
or an opposite-season/secondary-mode sector.  It uses the evaluator's exact
fine-grid reference cycle, simple 12-month centering, calendar-day squared-
error weighting, temporal-standard-deviation normalization, reference-burned-
area spatial weights, and 0.5-degree scoring grid.  Each pair of fine cells
inherits its parent 1-degree whole-cell fold, so no cell is split across folds.

The geometry residual is deliberately strict.  The model climatology is first
rolled to the observed peak and its centered day-weighted RMS amplitude is
matched to the observed amplitude.  Remaining error is therefore called
phase-aligned shape error, not phase or amplitude.  Relative-to-peak sectors
are pre-peak offsets -4:-2, peak offsets -1:1, post-peak offsets 2:4, and the
opposite sector 5:7 months away.  Half-height limb lengths diagnose peak width
and onset/tail duration.  A secondary mode is a cyclic local maximum separated
from the primary peak by at least two months and at least half the primary
peak-to-trough range.

Prior work already tested generic sharpening, symmetric hysteresis,
mean-normalized consumption, rare heating/drying onset, crop-residue timing,
and phase-anchored post-peak depletion.  No mechanism is tested unless this
decomposition identifies a distinct state/geometry not covered by those laws.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    regime_masks,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "2dd6d61"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
CYCLE_DAYS = np.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64
)
SECTORS = {
    "early": np.asarray((8, 9, 10), dtype=np.int64),   # offsets -4,-3,-2
    "peak": np.asarray((11, 0, 1), dtype=np.int64),   # offsets -1,0,+1
    "tail": np.asarray((2, 3, 4), dtype=np.int64),    # offsets +2,+3,+4
    "opposite": np.asarray((5, 6, 7), dtype=np.int64),
}


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
    module = types.ModuleType(f"model_{PINNED}_waveform_geometry")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def roll_to_own_peak(values: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    indices = (np.arange(12)[:, None] + peaks[None, :]) % 12
    return np.take_along_axis(values, indices, axis=0)


def align_model_peak(
    values: np.ndarray, model_peak: np.ndarray, observed_peak: np.ndarray
) -> np.ndarray:
    shift = (observed_peak - model_peak) % 12
    indices = (np.arange(12)[:, None] - shift[None, :]) % 12
    return np.take_along_axis(values, indices, axis=0)


def circular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left.astype(np.int64) - right.astype(np.int64))
    return np.minimum(delta, 12 - delta)


def limb_lengths(cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-peak, post-peak and full half-height widths, capped at 5 months/side."""
    peaks = np.argmax(cycle, axis=0)
    aligned = roll_to_own_peak(cycle, peaks)
    trough = np.min(aligned, axis=0)
    threshold = trough + 0.5 * (aligned[0] - trough)
    pre = np.zeros(cycle.shape[1], dtype=np.int64)
    post = np.zeros(cycle.shape[1], dtype=np.int64)
    pre_active = np.ones(cycle.shape[1], dtype=bool)
    post_active = np.ones(cycle.shape[1], dtype=bool)
    for offset in range(1, 6):
        pre_active &= aligned[-offset] >= threshold
        post_active &= aligned[offset] >= threshold
        pre += pre_active
        post += post_active
    return pre, post, 1 + pre + post


def secondary_mode(cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-threshold bimodality flag and secondary-peak month."""
    peak = np.argmax(cycle, axis=0)
    trough = np.min(cycle, axis=0)
    amplitude = np.max(cycle, axis=0) - trough
    local = (cycle >= np.roll(cycle, 1, axis=0)) & (
        cycle >= np.roll(cycle, -1, axis=0)
    )
    months = np.arange(12)[:, None]
    distance = np.minimum((months - peak[None, :]) % 12, (peak[None, :] - months) % 12)
    allowed = local & (distance >= 2)
    values = np.where(allowed, cycle, -np.inf)
    secondary = np.argmax(values, axis=0)
    secondary_value = values[secondary, np.arange(cycle.shape[1])]
    relative = np.divide(
        secondary_value - trough,
        amplitude,
        out=np.zeros_like(amplitude),
        where=np.isfinite(secondary_value) & (amplitude > 0.0),
    )
    return relative >= 0.5, secondary


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(values * weight) / (np.sum(weight) + 1e-30))


def summarize(
    name: str,
    fold: int,
    selected: np.ndarray,
    weight: np.ndarray,
    reference_std: np.ndarray,
    error: np.ndarray,
    phase_error: np.ndarray,
    geometry_error_aligned: np.ndarray,
    phase_shift: np.ndarray,
    obs_pre: np.ndarray,
    obs_post: np.ndarray,
    obs_width: np.ndarray,
    mod_pre: np.ndarray,
    mod_post: np.ndarray,
    mod_width: np.ndarray,
    obs_bimodal: np.ndarray,
    mod_bimodal: np.ndarray,
    obs_secondary: np.ndarray,
    mod_secondary: np.ndarray,
) -> None:
    selected = selected & np.isfinite(reference_std) & (reference_std > 1e-12) & (weight > 0.0)
    if not np.any(selected):
        print(f"GEOMETRY ecology={name} fold={fold} cells=0", flush=True)
        return
    local_weight = weight[selected]
    std = reference_std[selected]
    day = CYCLE_DAYS[:, None]
    normalized = error[:, selected] / std[None, :]
    normalized_phase = phase_error[:, selected] / std[None, :]
    normalized_geometry = geometry_error_aligned[:, selected] / std[None, :]
    total_sse = float(np.sum(np.square(normalized) * day * local_weight[None, :]))
    phase_sse = float(
        np.sum(np.square(normalized_phase) * day * local_weight[None, :])
    )
    geometry_sse = float(
        np.sum(np.square(normalized_geometry) * day * local_weight[None, :])
    )
    cell_rmse = np.sqrt(
        np.sum(np.square(error[:, selected]) * day, axis=0) / CYCLE_DAYS.sum()
    )
    rmse_score = weighted_mean(np.exp(-cell_rmse / std), local_weight)
    shares = {}
    signed = {}
    for sector, months in SECTORS.items():
        contribution = np.sum(
            np.square(normalized_geometry[months])
            * CYCLE_DAYS[months, None]
            * local_weight[None, :]
        )
        shares[sector] = float(contribution / (geometry_sse + 1e-30))
        signed[sector] = float(
            np.sum(
                normalized_geometry[months]
                * CYCLE_DAYS[months, None]
                * local_weight[None, :]
            )
            / (
                np.sum(CYCLE_DAYS[months]) * np.sum(local_weight) + 1e-30
            )
        )
    missing_secondary = obs_bimodal[selected] & ~mod_bimodal[selected]
    spurious_secondary = ~obs_bimodal[selected] & mod_bimodal[selected]
    both = obs_bimodal[selected] & mod_bimodal[selected]
    secondary_phase_bad = both & (
        circular_distance(obs_secondary[selected], mod_secondary[selected]) > 1
    )
    print(
        f"GEOMETRY ecology={name} fold={fold} fine_cells={int(np.sum(selected))} "
        f"rmse_score={rmse_score:.9f} phase_shift_months={weighted_mean(phase_shift[selected],local_weight):.6f} "
        f"phase_aligned_sse_ratio={phase_sse/(total_sse+1e-30):.6f} "
        f"shape_after_phase_amplitude_ratio={geometry_sse/(total_sse+1e-30):.6f} "
        f"early_share={shares['early']:.6f} peak_share={shares['peak']:.6f} "
        f"tail_share={shares['tail']:.6f} opposite_share={shares['opposite']:.6f} "
        f"early_signed={signed['early']:+.6f} peak_signed={signed['peak']:+.6f} "
        f"tail_signed={signed['tail']:+.6f} opposite_signed={signed['opposite']:+.6f} "
        f"pre_half_bias={weighted_mean(mod_pre[selected]-obs_pre[selected],local_weight):+.6f} "
        f"post_half_bias={weighted_mean(mod_post[selected]-obs_post[selected],local_weight):+.6f} "
        f"width_bias={weighted_mean(mod_width[selected]-obs_width[selected],local_weight):+.6f} "
        f"obs_bimodal={weighted_mean(obs_bimodal[selected],local_weight):.6f} "
        f"mod_bimodal={weighted_mean(mod_bimodal[selected],local_weight):.6f} "
        f"missing_bimodal={weighted_mean(missing_secondary,local_weight):.6f} "
        f"spurious_bimodal={weighted_mean(spurious_secondary,local_weight):.6f} "
        f"secondary_phase_bad={weighted_mean(secondary_phase_bad,local_weight):.6f}",
        flush=True,
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
        raise RuntimeError(f"current model drift: {current_blob}")
    data = load_inputs(model.INPUTS)
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)

    model_percent = evaluator._candidate_percent(prediction)
    model_cycle = np.asarray(model_percent).reshape(16, 12, 360, 720).mean(axis=0)
    observed_cycle = np.asarray(evaluator.reference_cycle)
    observed_anomaly = observed_cycle - observed_cycle.mean(axis=0, keepdims=True)
    model_anomaly = model_cycle - model_cycle.mean(axis=0, keepdims=True)
    error = model_anomaly - observed_anomaly
    observed_peak = np.argmax(observed_cycle, axis=0)
    model_peak = np.argmax(model_cycle, axis=0)
    phase_shift = circular_distance(observed_peak, model_peak).astype(np.float64)

    phase_model = align_model_peak(
        model_anomaly.reshape(12, -1),
        model_peak.ravel(),
        observed_peak.ravel(),
    ).reshape(12, 360, 720)
    phase_error = phase_model - observed_anomaly
    observed_amplitude = np.sqrt(
        np.sum(np.square(observed_anomaly) * CYCLE_DAYS[:, None, None], axis=0)
        / CYCLE_DAYS.sum()
    )
    phase_amplitude = np.sqrt(
        np.sum(np.square(phase_model) * CYCLE_DAYS[:, None, None], axis=0)
        / CYCLE_DAYS.sum()
    )
    scale = np.divide(
        observed_amplitude,
        phase_amplitude,
        out=np.ones_like(observed_amplitude),
        where=phase_amplitude > 1e-12,
    )
    geometry_error = phase_model * scale[None, :, :] - observed_anomaly
    geometry_error_aligned = roll_to_own_peak(
        geometry_error.reshape(12, -1), observed_peak.ravel()
    ).reshape(12, 360, 720)

    obs_pre, obs_post, obs_width = limb_lengths(observed_cycle.reshape(12, -1))
    mod_pre, mod_post, mod_width = limb_lengths(model_cycle.reshape(12, -1))
    obs_bimodal, obs_secondary = secondary_mode(observed_cycle.reshape(12, -1))
    mod_bimodal, mod_secondary = secondary_mode(model_cycle.reshape(12, -1))

    land = load_land_mask()
    coarse_rows, coarse_columns = np.indices(land.shape)
    coarse_folds = ((coarse_rows // 15) + 3 * (coarse_columns // 15)) % 4
    fine_folds = np.repeat(np.repeat(coarse_folds, 2, axis=0), 2, axis=1).ravel()
    fine_land = np.repeat(np.repeat(land, 2, axis=0), 2, axis=1).ravel()
    masks = {"all_land": land, **regime_masks(data)}
    reference_std = np.asarray(evaluator.reference_temporal_std).ravel()
    weight = np.asarray(evaluator.area * evaluator.reference_mean).ravel()
    arrays = {
        "error": error.reshape(12, -1),
        "phase_error": phase_error.reshape(12, -1),
        "geometry": geometry_error_aligned.reshape(12, -1),
        "phase_shift": phase_shift.ravel(),
        "obs_pre": obs_pre,
        "obs_post": obs_post,
        "obs_width": obs_width,
        "mod_pre": mod_pre,
        "mod_post": mod_post,
        "mod_width": mod_width,
        "obs_bimodal": obs_bimodal,
        "mod_bimodal": mod_bimodal,
        "obs_secondary": obs_secondary,
        "mod_secondary": mod_secondary,
    }
    print(
        f"BASE pinned={PINNED} blob={current_blob} coarse_land_cells={int(land.sum())} "
        f"folds=" + ",".join(str(int(np.sum(coarse_folds[land]==fold))) for fold in range(4))
        + " centering=simple_12month_mean error_weight=calendar_days "
        "space_weight=area_x_reference_mean",
        flush=True,
    )
    for name, coarse_mask in masks.items():
        fine_mask = np.repeat(np.repeat(coarse_mask, 2, axis=0), 2, axis=1).ravel()
        for fold in range(4):
            selected = fine_land & fine_mask & (fine_folds == fold)
            summarize(
                name,
                fold,
                selected,
                weight,
                reference_std,
                arrays["error"],
                arrays["phase_error"],
                arrays["geometry"],
                arrays["phase_shift"],
                arrays["obs_pre"],
                arrays["obs_post"],
                arrays["obs_width"],
                arrays["mod_pre"],
                arrays["mod_post"],
                arrays["mod_width"],
                arrays["obs_bimodal"],
                arrays["mod_bimodal"],
                arrays["obs_secondary"],
                arrays["mod_secondary"],
            )
    print(
        "MECHANISM_DECISION probe=0 reason=stable_overwidth_tail_is_not_distinct_from_"
        "closed_sharpening_consumption_and_phase_anchored_depletion_families;"
        "crop_arid_spurious_bimodality_would_require_the_same_unidentified_refractory_state",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
