"""Exogenous-only continuation of the bf42d58 residual triage.

The companion diagnostic finds a modest annual OOF ceiling dominated by the
incumbent's own previous fire.  This script removes every incumbent prediction,
hazard, and fire-memory feature.  It therefore tests whether coupled-valid
climate, vegetation, land state, and their causal memories contain a new
physical interaction rather than a learned calibration of current error.

All learners remain scratch-only.  Coordinates define four held blocks and
never enter a feature.  No region, neighbour, target-derived runtime field,
future value, or completed-record climatology is used.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_residual_seasonal_memory_triage_bf42d58 import (  # noqa: E402
    EPS,
    EXPECTED_OVERALL,
    build_annual_features,
    build_current_cache,
    build_monthly_features,
    causal_center,
    ecology_masks,
    ecology_ratios,
    fit_oof,
    load_observation,
    metric_text,
    pinned_model,
    score_variant,
    select_high_weight,
    selected_data,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask, validate_prediction  # noqa: E402


def keep_exogenous(name: str) -> bool:
    return not any(
        token in name
        for token in ("previous_fire", "log_hazard", "hazard_ema")
    )


def main() -> int:
    started = time.perf_counter()
    if "--compact" in sys.argv:
        fit_oof.__globals__["print_ranks"] = lambda *_args, **_kwargs: None
    model = pinned_model()
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    baseline_grid = build_current_cache(model, land)
    base_scores = evaluator.score(validate_prediction(baseline_grid))
    base_global = base_scores["global"]
    if abs(float(base_global["overall_score"]) - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(f"current mismatch: {metric_text(base_global)}")
    print(f"BASE {metric_text(base_global)}", flush=True)

    observation_grid = load_observation()
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    rows, columns, _, retained = select_high_weight(observation_grid, area_grid)
    count = rows.size
    data = selected_data(model, rows, columns)
    baseline = np.asarray(baseline_grid[:, rows, columns], dtype=np.float32)
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float32)
    monthly_names, monthly_x, monthly_base_count, _ = build_monthly_features(
        data, baseline
    )
    annual_names, annual_x, annual_base_count = build_annual_features(
        data, baseline, monthly_names, monthly_x, monthly_base_count
    )
    annual_base_indices = np.asarray(
        [index for index, name in enumerate(annual_names[:annual_base_count]) if keep_exogenous(name)],
        dtype=np.int64,
    )
    annual_full_indices = np.asarray(
        [index for index, name in enumerate(annual_names) if keep_exogenous(name)],
        dtype=np.int64,
    )
    monthly_base_indices = np.asarray(
        [index for index, name in enumerate(monthly_names[:monthly_base_count]) if keep_exogenous(name)],
        dtype=np.int64,
    )
    monthly_full_indices = np.asarray(
        [index for index, name in enumerate(monthly_names) if keep_exogenous(name)],
        dtype=np.int64,
    )
    print(
        f"DESIGN cells={count} retained={retained:.9f} annual_base={annual_base_indices.size} "
        f"annual_memory={annual_full_indices.size} monthly_base={monthly_base_indices.size} "
        f"monthly_memory={monthly_full_indices.size} incumbent_features=0 coordinates_features=0",
        flush=True,
    )

    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    annual_folds = np.tile(cell_folds, (15, 1)).reshape(-1)
    monthly_folds = np.tile(cell_folds, (180, 1)).reshape(-1)
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    pred_year = baseline.reshape(16, 12, count).sum(axis=1)
    obs_year = observation.reshape(16, 12, count).sum(axis=1)
    annual_target = np.clip(
        np.log((obs_year[1:] + EPS) / (pred_year[1:] + EPS)), -3.0, 3.0
    ).reshape(-1).astype(np.float32)
    obs_floor = float(np.sum(obs_year[1:] * area[None, :])) / (
        15.0 * float(np.sum(area))
    )
    annual_weights = (
        area[None, :] * (obs_year[1:] + 0.02 * obs_floor)
    ).reshape(-1)
    annual_weights /= annual_weights.mean()

    base_alloc = baseline.reshape(16, 12, count) / (pred_year[:, None, :] + EPS)
    obs_alloc = observation.reshape(16, 12, count) / (obs_year[:, None, :] + EPS)
    seasonal_target = np.clip(
        np.log((obs_alloc[1:] + EPS) / (base_alloc[1:] + EPS)), -3.0, 3.0
    ).reshape(-1).astype(np.float32)
    seasonal_weights = (
        area[None, None, :]
        * (obs_year[1:, None, :] + 0.02 * obs_floor)
        * (obs_alloc[1:] + 0.02 / 12.0)
    ).reshape(-1)
    seasonal_weights /= seasonal_weights.mean()

    annual_oof = {}
    for label, indices in (
        ("annual_exogenous", annual_base_indices),
        ("annual_exogenous_memory", annual_full_indices),
    ):
        names = tuple(annual_names[index] for index in indices)
        annual_oof[label] = fit_oof(
            annual_x[:, indices], annual_target, annual_weights,
            annual_folds, names, label, 180,
        ).reshape(15, count)

    monthly_used = monthly_x.reshape(192, count, -1)[12:].reshape(180 * count, -1)
    seasonal_oof = {}
    for label, indices in (
        ("season_exogenous", monthly_base_indices),
        ("season_exogenous_memory", monthly_full_indices),
    ):
        names = tuple(monthly_names[index] for index in indices)
        raw = fit_oof(
            monthly_used[:, indices], seasonal_target, seasonal_weights,
            monthly_folds, names, label, 800,
        ).reshape(180, count)
        seasonal_oof[label] = causal_center(raw)
    del annual_x, monthly_x, monthly_used
    gc.collect()

    variants: list[tuple[float, str, dict[str, dict[str, float]], np.ndarray]] = []
    for label, correction in annual_oof.items():
        for blend in (0.25, 0.50, 1.0):
            corrected = baseline.copy()
            factor = np.exp(np.clip(blend * correction, -2.0, 2.0)).astype(np.float32)
            corrected[12:] *= np.repeat(factor[:, None, :], 12, axis=1).reshape(180, count)
            scores = score_variant(evaluator, baseline_grid, rows, columns, corrected)
            full_label = f"{label}:blend={blend:g}"
            variants.append((float(scores["global"]["overall_score"]), full_label, scores, corrected))
            print(f"SCORE label={full_label} {metric_text(scores['global'])}", flush=True)

    for label, correction in seasonal_oof.items():
        for blend in (0.10, 0.25, 0.50):
            corrected = baseline.copy()
            corrected[12:] *= np.exp(np.clip(blend * correction, -2.0, 2.0)).astype(np.float32)
            scores = score_variant(evaluator, baseline_grid, rows, columns, corrected)
            full_label = f"{label}:blend={blend:g}"
            variants.append((float(scores["global"]["overall_score"]), full_label, scores, corrected))
            print(f"SCORE label={full_label} {metric_text(scores['global'])}", flush=True)

    variants.sort(key=lambda item: item[0], reverse=True)
    print("TOP_EXACT_OOF", flush=True)
    for overall, label, scores, _ in variants:
        print(
            f"TOP label={label} delta={overall-float(base_global['overall_score']):+.9f} "
            f"{metric_text(scores['global'])}", flush=True,
        )

    best_overall, best_label, best_scores, best_selected = variants[0]
    full_rows, full_columns = np.nonzero(land)
    mask_names = (
        "monthly_precipitation", "air_temperature", "leaf_area_index",
        "natural_canopy_height", "aboveground_biomass",
        "natural_vegetation_fraction", "luh2_primary_fraction",
        "luh2_cropland_fraction", "luh2_rangeland_fraction",
    )
    full_mean = {
        name: np.asarray(selected_input(name, full_rows, full_columns)[:, 0, :], dtype=np.float32).mean(axis=0)
        for name in mask_names
    }
    masks = ecology_masks(full_mean, full_rows.size)
    best_grid = np.array(baseline_grid, dtype=np.float32, copy=True)
    best_grid[:, rows, columns] = np.clip(best_selected, 0.0, 1.0)
    baseline_land = np.asarray(baseline_grid[:, full_rows, full_columns], dtype=np.float32)
    best_land = best_grid[:, full_rows, full_columns]
    obs_land = observation_grid[:, full_rows, full_columns]
    land_area = area_grid[full_rows, full_columns]
    base_ecology = ecology_ratios(baseline_land, obs_land, land_area, masks)
    best_ecology = ecology_ratios(best_land, obs_land, land_area, masks)
    for name in masks:
        print(
            f"ECOLOGY best={best_label} name={name} baseline={base_ecology[name]:.9f} "
            f"oof={best_ecology[name]:.9f} delta={best_ecology[name]-base_ecology[name]:+.9f}",
            flush=True,
        )
    positive = 0
    for name in sorted(key for key in best_scores if key != "global"):
        delta = float(best_scores[name]["overall_score"] - base_scores[name]["overall_score"])
        positive += int(delta > 0.0)
        print(f"REGION best={best_label} name={name} delta={delta:+.9f}", flush=True)
    print(f"REGIONAL_BREADTH best={best_label} positive={positive}/14", flush=True)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        f"DONE best={best_label} best_overall={best_overall:.9f} "
        f"delta={best_overall-float(base_global['overall_score']):+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f} peak_rss_gib={peak/(1024.0**3):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
