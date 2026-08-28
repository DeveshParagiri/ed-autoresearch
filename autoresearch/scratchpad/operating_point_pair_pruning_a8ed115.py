"""Current operating-point pair-removal screen for model a8ed115.

This scratch-only pruning round measures actual one-at-a-time and pair removal
counterfactuals on 768 whole score-dominant cells.  It does not average across
all component coalitions.  Two scopes are screened independently: every pair
of active public ``COMPONENTS`` and every pair of tightly named late operators.
The latter are replaced by identity in memory or have their local coefficient
set to zero.  No canonical file, ledger, progress figure, or official
evaluation is touched.
"""

from __future__ import annotations

import gc
import itertools
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    format_metrics,
    load_observed,
    load_selected,
    metrics,
    select_cells,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


EXPECTED_MODEL_BLOB = "731e1ee048fd1099dffe75d11a738fd9125f8064"
EXPECTED_BASE = 0.719021686

# Label -> (parameter overrides, identity-replaced functions).  Every entry is
# local and physically legible; no private inner learned surface is removed.
LATE_INTERVENTIONS: dict[str, tuple[dict[str, float], tuple[str, ...]]] = {
    "ecological_brakes": ({}, ("_ecological_regime_brakes",)),
    "pathway_event_mix": ({"pathway_mix_w": 0.0}, ()),
    "regime_capacity": ({}, ("_ecological_fire_capacity",)),
    "seasonal_rain_capacity": ({}, ("_seasonal_rainfall_capacity",)),
    "state_fire_season": ({}, ("_state_dependent_fire_season",)),
    "rare_ignition": ({}, ("_rare_lightning_ignition",)),
    "crop_management": ({}, ("_rain_conditioned_crop_management",)),
    "dead_fuel_pool": ({}, ("_dead_fuel_pool_response",)),
    "conditional_allocation": ({}, ("_conditional_fire_allocation",)),
    "greenup_brake": ({}, ("_live_fuel_greenup_brake",)),
    "surface_bank": ({}, ("_surface_fire_opportunity_bank",)),
    "local_footprint": ({}, ("_local_fire_footprint",)),
    "annual_regime_closure": ({}, ("_annual_regime_closure",)),
    "multi_pathway_bank": ({}, ("_multi_pathway_opportunity_bank",)),
    "fuel_recovery": ({}, ("_pathway_fuel_recovery_reservoir",)),
    "secondary_litter": ({}, ("_secondary_fuel_litter_banks",)),
    "fragment_recurrence": ({}, ("_fragmented_managed_recurrence_brake",)),
    "surface_seasonality": ({}, ("_surface_seasonality_capacity",)),
}


def objective(values: Sequence[float], baseline: Sequence[float]) -> float:
    """Small screening loss spanning magnitude, allocation, and raw cycle."""
    alloc, annual, raw_cycle, _phase, area_ratio = values
    base_alloc, base_annual, base_raw, _base_phase, _base_ratio = baseline
    return float(
        alloc / base_alloc
        + 2.0 * annual / base_annual
        + raw_cycle / base_raw
        + 0.5 * abs(np.log(max(area_ratio, 1e-8)))
    )


def run_components(model, data, removed: frozenset[str]) -> np.ndarray:
    enabled = tuple(name for name in model.COMPONENTS if name not in removed)
    return np.asarray(
        model.predict(data, dict(model.PARAMS), enabled), dtype=np.float32
    )[:, 0, :]


def run_late(model, data, removed: frozenset[str]) -> np.ndarray:
    params = dict(model.PARAMS)
    functions: set[str] = set()
    for label in removed:
        overrides, replacements = LATE_INTERVENTIONS[label]
        params.update(overrides)
        functions.update(replacements)
    originals = {name: getattr(model, name) for name in functions}
    identity = lambda prediction, data_, params_, enabled_: prediction
    try:
        for name in functions:
            setattr(model, name, identity)
        return np.asarray(model.predict(data, params, None), dtype=np.float32)[:, 0, :]
    finally:
        for name, function in originals.items():
            setattr(model, name, function)


def screen_scope(
    scope: str,
    labels: Sequence[str],
    runner,
    model,
    data,
    observation,
    area,
    reference_weight,
    folds,
    baseline_metrics,
) -> list[dict[str, object]]:
    base_objective = objective(baseline_metrics, baseline_metrics)
    singles: dict[str, tuple[tuple[float, ...], float]] = {}
    for label in labels:
        prediction = runner(model, data, frozenset((label,)))
        values, fold_values = metrics(
            prediction, observation, area, reference_weight, folds
        )
        loss = objective(values, baseline_metrics)
        singles[label] = (values, loss)
        fold_breadth = sum(
            objective(current, baseline) < objective(baseline, baseline)
            for current, baseline in zip(fold_values, BASELINE_FOLDS)
        )
        print(
            f"SINGLE scope={scope} name={label} loss_delta={loss-base_objective:+.9f} "
            f"fold_wins={fold_breadth}/4 {format_metrics(values)}",
            flush=True,
        )
        del prediction
        gc.collect()

    pairs: list[dict[str, object]] = []
    for first, second in itertools.combinations(labels, 2):
        prediction = runner(model, data, frozenset((first, second)))
        values, fold_values = metrics(
            prediction, observation, area, reference_weight, folds
        )
        loss = objective(values, baseline_metrics)
        synergy = loss - singles[first][1] - singles[second][1] + base_objective
        fold_breadth = sum(
            objective(current, baseline) < objective(baseline, baseline)
            for current, baseline in zip(fold_values, BASELINE_FOLDS)
        )
        metric_wins = sum(
            current < base
            for current, base in zip(values[:3], baseline_metrics[:3])
        )
        pairs.append(
            {
                "scope": scope,
                "first": first,
                "second": second,
                "values": values,
                "loss": loss,
                "delta": loss - base_objective,
                "synergy": synergy,
                "fold_wins": fold_breadth,
                "metric_wins": metric_wins,
            }
        )
        del prediction
        gc.collect()

    print(f"PAIR_RANK scope={scope}", flush=True)
    for row in sorted(pairs, key=lambda item: (item["delta"], item["synergy"]))[:24]:
        print(
            f"PAIR scope={scope} names={row['first']}+{row['second']} "
            f"loss_delta={row['delta']:+.9f} synergy={row['synergy']:+.9f} "
            f"fold_wins={row['fold_wins']}/4 metric_wins={row['metric_wins']}/3 "
            f"{format_metrics(row['values'])}",
            flush=True,
        )
    return pairs


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
    rows, cols, area, reference_weight, retained = select_cells(evaluator)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.9f} "
        f"model_blob={blob} exact_reference={EXPECTED_BASE:.9f}",
        flush=True,
    )
    data = load_selected(model.INPUTS, rows, cols)
    observation = load_observed(rows, cols)
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    global BASELINE_FOLDS
    baseline_metrics, BASELINE_FOLDS = metrics(
        baseline, observation, area, reference_weight, folds
    )
    print(
        f"BASE loss={objective(baseline_metrics, baseline_metrics):.9f} "
        + format_metrics(baseline_metrics),
        flush=True,
    )

    component_pairs = screen_scope(
        "components",
        tuple(model.COMPONENTS),
        run_components,
        model,
        data,
        observation,
        area,
        reference_weight,
        folds,
        baseline_metrics,
    )
    late_pairs = screen_scope(
        "late",
        tuple(LATE_INTERVENTIONS),
        run_late,
        model,
        data,
        observation,
        area,
        reference_weight,
        folds,
        baseline_metrics,
    )
    all_pairs = component_pairs + late_pairs
    plausible = [
        row
        for row in all_pairs
        if row["delta"] < 0.0 and row["fold_wins"] >= 3 and row["metric_wins"] >= 2
    ]
    print(f"PLAUSIBLE count={len(plausible)}", flush=True)
    for row in sorted(plausible, key=lambda item: item["delta"]):
        print(
            f"SURVIVOR scope={row['scope']} names={row['first']}+{row['second']} "
            f"loss_delta={row['delta']:+.9f} synergy={row['synergy']:+.9f} "
            f"fold_wins={row['fold_wins']}/4 metric_wins={row['metric_wins']}/3",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
