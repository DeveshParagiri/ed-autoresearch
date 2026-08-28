"""Verify anomaly-oracle and stacked-HGB claims at canonical ``2dd6d61``.

The anomaly oracles follow ``scripts.fast_ilamb`` exactly: each twelve-month
climatology is centered with an unweighted arithmetic mean, while squared
anomaly error is averaged with calendar-day weights.  Candidate mean maps are
held at the incumbent evaluator mean.  Anomaly scaling is capped cellwise only
where the desired cycle would otherwise leave the valid [0, 1] burned-fraction
range.

The HGB section reproduces the deeper current-input protocol with four
whole-cell spatial folds.  It also makes the protocol's limitations explicit:
the fitted/evaluated cell population is selected from GFED reference weight
and incumbent excess against GFED, and the best stacking weights are selected
on the same OOF predictions being reported.  Thus its score is an optimistic
conditional diagnostic even though each row prediction is out of fold.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad import deep_reverse_ml_121c83c as deep  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator, MONTH_MIDPOINTS  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "2dd6d61"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
EXPECTED_OVERALL = 0.720105466
CLAIMED_STACK = 0.733371
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0
CYCLE_DAYS = np.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64
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
        raise RuntimeError(f"unexpected pinned blob {blob}")
    module = types.ModuleType(f"model_{PINNED}_claude_verification")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def metric_text(score: dict[str, float]) -> str:
    return " ".join(
        f"{label}={score[key]:.9f}"
        for label, key in (
            ("overall", "overall_score"),
            ("bias", "bias_score"),
            ("rmse", "rmse_score"),
            ("seasonal", "seasonal_cycle_score"),
            ("spatial", "spatial_distribution_score"),
        )
    )


def score(
    evaluator: GFED5Evaluator, label: str, prediction: np.ndarray
) -> dict[str, float]:
    values = evaluator.score(validate_prediction(prediction))["global"]
    print(f"ORACLE label={label} {metric_text(values)}", flush=True)
    return values


def weighted_rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.sum(np.square(values) * CYCLE_DAYS[:, None, None], axis=0)
        / CYCLE_DAYS.sum()
    )


def ideal_anomaly_score(
    evaluator: GFED5Evaluator,
    anomaly: np.ndarray,
    incumbent_score: dict[str, float],
) -> dict[str, float]:
    """Score an unconstrained anomaly while holding incumbent mean-map metrics."""
    candidate_anomaly = np.repeat(
        np.repeat(np.asarray(anomaly, dtype=np.float64), 2, axis=1), 2, axis=2
    ) * 100.0
    reference_anomaly = (
        evaluator.reference_cycle - evaluator.reference_cycle.mean(axis=0)
    )
    centered_rmse = np.sqrt(
        np.sum(
            np.square(candidate_anomaly - reference_anomaly)
            * CYCLE_DAYS[:, None, None],
            axis=0,
        )
        / CYCLE_DAYS.sum()
    )
    rmse_field = np.exp(
        -np.abs(centered_rmse / evaluator.reference_temporal_std)
    )
    reference_weight = evaluator.area * evaluator.reference_mean
    rmse = float(
        np.ma.sum(rmse_field * reference_weight) / np.ma.sum(reference_weight)
    )
    candidate_phase = MONTH_MIDPOINTS[np.argmax(candidate_anomaly, axis=0)]
    phase_shift = candidate_phase - evaluator.reference_phase
    phase_shift += (phase_shift < -182.5) * 365.0
    phase_shift -= (phase_shift > 182.5) * 365.0
    seasonal_field = 0.5 * (
        1.0 + np.cos(np.abs(phase_shift) / 365.0 * 2.0 * np.pi)
    )
    seasonal = float(
        np.ma.sum(seasonal_field * reference_weight)
        / np.ma.sum(reference_weight)
    )
    overall = (
        incumbent_score["bias_score"]
        + 2.0 * rmse
        + seasonal
        + incumbent_score["spatial_distribution_score"]
    ) / 5.0
    return {
        "overall_score": overall,
        "bias_score": incumbent_score["bias_score"],
        "rmse_score": rmse,
        "seasonal_cycle_score": seasonal,
        "spatial_distribution_score": incumbent_score[
            "spatial_distribution_score"
        ],
    }


def feasible_anomaly_cycle(
    incumbent_mean: np.ndarray,
    direction: np.ndarray,
    desired_scale: np.ndarray,
    evaluator: GFED5Evaluator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a bounded cycle with incumbent evaluator mean and scaled direction."""
    mean_weights = evaluator.month_lengths.reshape(16, 12).sum(axis=0)
    weighted_direction = np.sum(
        direction * mean_weights[:, None, None], axis=0
    ) / mean_weights.sum()
    coefficient = direction - weighted_direction[None, ...]
    positive = coefficient > 0.0
    negative = coefficient < 0.0
    upper = np.full_like(incumbent_mean, np.inf, dtype=np.float64)
    lower = np.full_like(incumbent_mean, np.inf, dtype=np.float64)
    positive_bound = np.divide(
        1.0 - incumbent_mean[None, ...],
        coefficient,
        out=np.full_like(coefficient, np.inf),
        where=positive,
    )
    negative_bound = np.divide(
        incumbent_mean[None, ...],
        -coefficient,
        out=np.full_like(coefficient, np.inf),
        where=negative,
    )
    upper[:] = np.min(positive_bound, axis=0)
    lower[:] = np.min(negative_bound, axis=0)
    feasible = np.minimum(np.maximum(desired_scale, 0.0), np.minimum(upper, lower))
    cycle = incumbent_mean[None, ...] + feasible[None, ...] * coefficient
    if float(cycle.min()) < -1e-10 or float(cycle.max()) > 1.0 + 1e-10:
        raise RuntimeError("bounded anomaly construction failed")
    return np.asarray(cycle, dtype=np.float32), feasible, np.minimum(upper, lower)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return np.interp(probabilities, cumulative, sorted_values)


def amplitude_audit(
    evaluator: GFED5Evaluator,
    incumbent: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    incumbent_score: dict[str, float],
) -> dict[str, dict[str, float]]:
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    observed_cycle = observed.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_anomaly = incumbent_cycle - incumbent_cycle.mean(axis=0, keepdims=True)
    observed_anomaly = observed_cycle - observed_cycle.mean(axis=0, keepdims=True)
    incumbent_amplitude = weighted_rms(incumbent_anomaly)
    observed_amplitude = weighted_rms(observed_anomaly)
    incumbent_mean = np.average(
        incumbent, axis=0, weights=evaluator.month_lengths
    )

    amplitude_scale = observed_amplitude / (incumbent_amplitude + 1e-12)
    amplitude_cycle, amplitude_feasible, amplitude_cap = feasible_anomaly_cycle(
        incumbent_mean, incumbent_anomaly, amplitude_scale, evaluator
    )
    shape_scale = incumbent_amplitude / (observed_amplitude + 1e-12)
    shape_cycle, shape_feasible, shape_cap = feasible_anomaly_cycle(
        incumbent_mean, observed_anomaly, shape_scale, evaluator
    )
    amplitude_prediction = np.tile(amplitude_cycle, (16, 1, 1))
    shape_prediction = np.tile(shape_cycle, (16, 1, 1))
    scores = {
        "amplitude_only": score(evaluator, "correct_anomaly_amplitude_only", amplitude_prediction),
        "shape_only": score(evaluator, "correct_anomaly_shape_only", shape_prediction),
    }
    ideal_amplitude = ideal_anomaly_score(
        evaluator,
        amplitude_scale[None, ...] * incumbent_anomaly,
        incumbent_score,
    )
    ideal_shape = ideal_anomaly_score(
        evaluator,
        shape_scale[None, ...] * observed_anomaly,
        incumbent_score,
    )
    print(
        "IDEAL_UNBOUNDED label=correct_anomaly_amplitude_only "
        + metric_text(ideal_amplitude),
        flush=True,
    )
    print(
        "IDEAL_UNBOUNDED label=correct_anomaly_shape_only "
        + metric_text(ideal_shape),
        flush=True,
    )

    reference_mean = np.average(observed, axis=0, weights=MONTH_DAYS)
    reference_weight = area * reference_mean
    valid = (
        np.isfinite(incumbent_amplitude)
        & np.isfinite(observed_amplitude)
        & (observed_amplitude > 1e-8)
        & (reference_weight > 0.0)
    )
    log_ratio = np.log(
        (incumbent_amplitude[valid] + 1e-12)
        / (observed_amplitude[valid] + 1e-12)
    )
    weight = reference_weight[valid]
    ratio_of_means = float(
        np.sum(reference_weight * incumbent_amplitude)
        / np.sum(reference_weight * observed_amplitude)
    )
    geometric_ratio = float(np.exp(np.sum(weight * log_ratio) / np.sum(weight)))
    capped_amplitude_weight = float(
        np.sum(reference_weight[amplitude_feasible + 1e-10 < amplitude_scale])
        / np.sum(reference_weight)
    )
    capped_shape_weight = float(
        np.sum(reference_weight[shape_feasible + 1e-10 < shape_scale])
        / np.sum(reference_weight)
    )
    print(
        f"AMPLITUDE global_ratio_of_weighted_means={ratio_of_means:.9f} "
        f"weighted_geometric_ratio={geometric_ratio:.9f} "
        f"amplitude_target_capped_weight={capped_amplitude_weight:.9f} "
        f"shape_target_capped_weight={capped_shape_weight:.9f} "
        f"amplitude_cap_min={np.min(amplitude_cap):.9g} "
        f"shape_cap_min={np.min(shape_cap):.9g}",
        flush=True,
    )

    intensity = np.log10(reference_mean[valid] + 1e-8)
    boundaries = weighted_quantile(intensity, weight, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    for index in range(5):
        if index == 4:
            selected = (intensity >= boundaries[index]) & (intensity <= boundaries[index + 1])
        else:
            selected = (intensity >= boundaries[index]) & (intensity < boundaries[index + 1])
        local_weight = weight[selected]
        local_log = log_ratio[selected]
        print(
            f"AMPLITUDE_STRATUM reference_weight_quintile={index + 1} "
            f"log10_annual_low={boundaries[index]:.6f} "
            f"log10_annual_high={boundaries[index + 1]:.6f} "
            f"weighted_mean_log_ratio={np.sum(local_weight * local_log)/np.sum(local_weight):+.9f} "
            f"geometric_ratio={np.exp(np.sum(local_weight * local_log)/np.sum(local_weight)):.9f} "
            f"underamplitude_weight_fraction={np.sum(local_weight[local_log < 0.0])/np.sum(local_weight):.9f}",
            flush=True,
        )
    return scores


def hgb_stack_audit(
    model,
    data: dict[str, np.ndarray],
    incumbent: np.ndarray,
    observed: np.ndarray,
    evaluator: GFED5Evaluator,
    area: np.ndarray,
) -> None:
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    incumbent_annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area * observed_annual
    excess_weight = area * np.maximum(incumbent_annual - observed_annual, 0.0)

    def top(weight: np.ndarray, fraction: float) -> np.ndarray:
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, fraction) + 1)]

    cells = np.union1d(top(observed_weight, 0.90), top(excess_weight, 0.90))
    rows, columns = cells // 360, cells % 360
    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    row_folds = np.tile(cell_folds, 192)
    selected_observed = np.asarray(observed[:, rows, columns], dtype=np.float32)
    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float32)
    deep.model = model
    names, matrix = deep.build_features(data, incumbent, rows, columns)
    epsilon = 1e-5
    annual_target_cell = np.clip(
        np.log(
            (observed_annual[rows, columns] + epsilon)
            / (incumbent_annual[rows, columns] + epsilon)
        ),
        -3.0,
        3.0,
    ).astype(np.float32)
    annual_target = np.tile(annual_target_cell, 192)
    observed_cycle = selected_observed.reshape(16, 12, -1).mean(axis=0)
    incumbent_cycle = selected_incumbent.reshape(16, 12, -1).mean(axis=0)
    observed_allocation = observed_cycle / (
        observed_cycle.sum(axis=0, keepdims=True) + epsilon
    )
    incumbent_allocation = incumbent_cycle / (
        incumbent_cycle.sum(axis=0, keepdims=True) + epsilon
    )
    cycle_target = np.tile(
        np.clip(
            np.log((observed_allocation + epsilon) / (incumbent_allocation + epsilon)),
            -3.0,
            3.0,
        ),
        (16, 1, 1),
    ).reshape(-1).astype(np.float32)
    annual_cell_weight = area[rows, columns] * (
        observed_annual[rows, columns]
        + np.maximum(
            incumbent_annual[rows, columns] - observed_annual[rows, columns], 0.0
        )
    )
    annual_weight = np.tile(annual_cell_weight, 192).astype(np.float64)
    cycle_month_weight = np.broadcast_to(
        area[rows, columns][None, :]
        * observed_annual[rows, columns][None, :]
        * np.maximum(observed_allocation, 0.002),
        (12, cells.size),
    )
    cycle_weight = np.tile(cycle_month_weight, (16, 1, 1)).reshape(-1)
    annual_weight /= annual_weight.mean()
    cycle_weight /= cycle_weight.mean()
    print(
        f"HGB_DESIGN cells={cells.size} rows={matrix.shape[0]} features={matrix.shape[1]} "
        f"fold_cells={','.join(str(int(np.sum(cell_folds == fold))) for fold in range(4))} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.9f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.9f} "
        "target_derived_selection=1 coordinate_features=0 whole_cell_oof=1",
        flush=True,
    )

    predictions = {}
    for target_name, target, weight, seed_offset in (
        ("annual", annual_target, annual_weight, 0),
        ("cycle", cycle_target, cycle_weight, 1000),
    ):
        oof = np.empty_like(target, dtype=np.float32)
        for fold in range(4):
            train = row_folds != fold
            held = ~train
            learner = HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=180,
                learning_rate=0.05,
                min_samples_leaf=250,
                l2_regularization=3.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=121083 + 31 * fold + seed_offset,
            )
            learner.fit(matrix[train], target[train], sample_weight=weight[train])
            oof[held] = learner.predict(matrix[held]).astype(np.float32)
            print(
                f"HGB_FOLD target={target_name} fold={fold} iterations={learner.n_iter_} "
                f"r2={r2_score(target[held], oof[held], sample_weight=weight[held]):.9f}",
                flush=True,
            )
            del learner
            gc.collect()
        predictions[target_name] = oof
        print(
            f"HGB_OOF target={target_name} "
            f"r2={r2_score(target, oof, sample_weight=weight):.9f}",
            flush=True,
        )

    month_index = np.arange(192) % 12
    annual_monthly = predictions["annual"].reshape(192, cells.size)
    cycle_monthly = predictions["cycle"].reshape(192, cells.size)
    annual_map = np.average(annual_monthly, axis=0, weights=MONTH_DAYS)
    cycle_climatology = cycle_monthly.reshape(16, 12, cells.size).mean(axis=0)

    def score_selected(label: str, corrected: np.ndarray):
        candidate = incumbent.copy()
        candidate[:, rows, columns] = np.clip(corrected, 0.0, 1.0)
        values = evaluator.score(validate_prediction(candidate))["global"]
        print(f"HGB_SCORE label={label} {metric_text(values)}", flush=True)
        del candidate
        gc.collect()
        return float(values["overall_score"]), label, values

    results = []
    configurations = (
        (0.00, 0.25),
        (0.00, 0.50),
        (0.00, 1.00),
        (0.10, 0.00),
        (0.25, 0.00),
        (0.50, 0.00),
        (0.10, 0.50),
        (0.10, 1.00),
        (0.25, 0.50),
        (0.25, 1.00),
        (0.50, 0.50),
        (0.50, 1.00),
        (1.00, 1.00),
    )
    for annual_strength, cycle_strength in configurations:
        allocation = incumbent_allocation * np.exp(
            np.clip(cycle_strength * cycle_climatology, -3.0, 3.0)
        )
        allocation /= allocation.sum(axis=0, keepdims=True) + 1e-12
        allocation_ratio = allocation / (incumbent_allocation + 1e-6)
        corrected = (
            selected_incumbent
            * np.exp(np.clip(annual_strength * annual_map[None, :], -3.0, 3.0))
            * allocation_ratio[month_index]
        )
        label = f"factorized_annual={annual_strength:g}_cycle={cycle_strength:g}"
        results.append(score_selected(label, corrected))

    for annual_strength, cycle_strength in configurations:
        residual = annual_strength * annual_monthly + cycle_strength * cycle_monthly
        corrected = deep.apply_correction(selected_incumbent, residual, 1.0)
        label = f"direct_hazard_annual={annual_strength:g}_cycle={cycle_strength:g}"
        results.append(score_selected(label, corrected))

    best = max(results, key=lambda row: row[0])
    print(
        f"HGB_BEST label={best[1]} {metric_text(best[2])} "
        f"claimed={CLAIMED_STACK:.9f} difference={best[0]-CLAIMED_STACK:+.9f} "
        "blend_selected_on_oof=1 target_derived_selection=1",
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
        raise RuntimeError(f"working model drifted to {current_blob}")
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_score = score(evaluator, "incumbent", incumbent)
    if abs(incumbent_score["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(f"incumbent drift {incumbent_score['overall_score']:.9f}")
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192], dtype=np.float32)
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    del fine
    amplitude_audit(evaluator, incumbent, observed, area, incumbent_score)
    if "--oracle-only" not in sys.argv:
        hgb_stack_audit(model, data, incumbent, observed, evaluator, area)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
