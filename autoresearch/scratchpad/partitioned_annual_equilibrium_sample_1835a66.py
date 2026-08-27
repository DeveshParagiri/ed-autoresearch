"""Sampled falsification of a partitioned annual fire equilibrium.

The incumbent annual closure contains two physically distinct operations: a
warm persistent-fire brake and a cold thaw source.  This diagnostic keeps the
cold source bit-for-bit identical and replaces only the warm operation by a
continuous mixture of natural-open and managed-open pathways.  Coordinates
select held blocks and observations score candidates; neither enters an
equation.  No canonical file or official evaluator is modified.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    MONTH_DAYS,
    format_metrics,
    load_observed,
    load_selected,
    metrics,
)
from autoresearch.scratchpad.managed_recurrence_ecology_sample_f45c0ce import (  # noqa: E402
    ecology_masks,
    select_stratified,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def annual_log_square(prediction: np.ndarray, observed: np.ndarray) -> np.ndarray:
    pred = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs = np.average(observed, axis=0, weights=MONTH_DAYS)
    return np.square(np.log((pred + 1e-5) / (obs + 1e-5)))


def print_comparison(
    label: str,
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    weight: np.ndarray,
    folds: np.ndarray,
    baseline_square: np.ndarray,
) -> tuple[float, ...]:
    overall, held = metrics(prediction, observed, area, weight, folds)
    square = annual_log_square(prediction, observed)
    changes = []
    for fold in range(4):
        selected = folds == fold
        changes.append(
            float(
                np.sqrt(np.average(square[selected], weights=weight[selected]))
                - np.sqrt(
                    np.average(
                        baseline_square[selected], weights=weight[selected]
                    )
                )
            )
        )
    print(
        f"CANDIDATE {label} {format_metrics(overall)} "
        f"fold_annual_deltas={','.join(f'{value:+.8f}' for value in changes)} "
        f"stable={all(value < 0.0 for value in changes)}",
        flush=True,
    )
    return overall


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    masks, land = ecology_masks()
    rows, cols, full_area, full_weight = select_stratified(
        evaluator, masks, land, per_mask=384, global_count=1536
    )
    area = full_area[rows, cols]
    weight = full_weight[rows, cols]
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_observed_weight="
        f"{float(weight.sum() / full_weight.sum()):.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )

    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = np.asarray(load_observed(rows, cols), dtype=np.float64)
    parameters = dict(model.PARAMS)

    def predict(overrides=None):
        current = dict(parameters)
        if overrides:
            current.update(overrides)
        return np.asarray(model.predict(data, current, None), dtype=np.float64)[:, 0, :]

    full = predict()
    no_closure = predict(
        {"persistent_warm_open_brake": 0.0, "cold_thaw_source": 0.0}
    )
    warm_only = predict({"cold_thaw_source": 0.0})
    cold_only = predict({"persistent_warm_open_brake": 0.0})
    base_square = annual_log_square(full, observed)
    print_comparison(
        "incumbent_full", full, observed, area, weight, folds, base_square
    )
    print_comparison(
        "no_annual_closure",
        no_closure,
        observed,
        area,
        weight,
        folds,
        base_square,
    )
    print_comparison(
        "warm_only", warm_only, observed, area, weight, folds, base_square
    )
    print_comparison(
        "cold_only", cold_only, observed, area, weight, folds, base_square
    )

    original_closure = model._annual_regime_closure

    def field(current_data, name):
        return np.asarray(current_data[name], dtype=np.float64)

    def pathway_states(prediction, current_data):
        hazard = -np.log1p(
            -np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0 - 1e-7)
        )
        recurrence_hazard = antecedent(hazard, 12.0)
        recurrence = recurrence_hazard / (recurrence_hazard + 0.01)

        rangeland = np.clip(
            field(current_data, "luh2_rangeland_fraction"), 0.0, 1.0
        )
        pasture = np.clip(
            field(current_data, "luh2_pasture_fraction"), 0.0, 1.0
        )
        managed_open = np.clip(rangeland + pasture, 0.0, 1.0)
        managed_access = managed_open / (managed_open + 0.15)

        natural = np.clip(
            field(current_data, "natural_vegetation_fraction"), 0.0, 1.0
        )
        canopy = np.clip(
            field(current_data, "natural_canopy_height"), 0.0, None
        )
        natural_open = natural * 8.0 / (canopy + 8.0)
        managed_share = managed_open / (
            managed_open + natural_open + 0.10
        )
        natural_share = natural_open / (
            managed_open + natural_open + 0.10
        )

        rain = np.clip(
            field(current_data, "monthly_precipitation"), 0.0, None
        )
        annual_rain = 12.0 * antecedent(rain, 12.0)
        fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
            -annual_rain / 3000.0
        )
        temperature = field(current_data, "air_temperature")
        temperature12 = antecedent(temperature, 12.0)
        warm = model._rising(temperature12, 0.25, 18.0)
        support = managed_access * fuel * warm
        return hazard, recurrence, managed_share, natural_share, support

    # Get pathway shares at the actual input to the closure, rather than from
    # the final prediction.  Temporarily capturing this argument avoids copying
    # the complete canonical stage sequence into the diagnostic.
    captured = {}

    def capture_closure(prediction, current_data, p, enabled):
        captured["prediction"] = np.asarray(prediction, dtype=np.float64).copy()
        captured["data"] = current_data
        return original_closure(prediction, current_data, p, enabled)

    model._annual_regime_closure = capture_closure
    try:
        reproduced = predict()
    finally:
        model._annual_regime_closure = original_closure
    print(
        f"CAPTURE_REPRO max_abs={float(np.max(np.abs(reproduced-full))):.12g}",
        flush=True,
    )
    closure_input = captured["prediction"]
    prepared = captured["data"]
    h0, recurrence, managed_share, natural_share, support = pathway_states(
        closure_input, prepared
    )
    cell_weights = weight / (np.mean(weight) + 1e-30)
    annual_managed = np.average(managed_share[:, 0, :], axis=0, weights=MONTH_DAYS)
    annual_natural = np.average(natural_share[:, 0, :], axis=0, weights=MONTH_DAYS)
    annual_recurrence = np.average(recurrence[:, 0, :], axis=0, weights=MONTH_DAYS)
    print(
        "PATHWAYS "
        f"managed_share={float(np.average(annual_managed, weights=cell_weights)):.8f} "
        f"natural_share={float(np.average(annual_natural, weights=cell_weights)):.8f} "
        f"recurrence={float(np.average(annual_recurrence, weights=cell_weights)):.8f}",
        flush=True,
    )

    candidate_predictions = {}

    def make_closure(kind: str, strength: float):
        def partitioned(prediction, current_data, p, enabled):
            base = np.asarray(prediction, dtype=np.float64)
            warm_parameters = dict(p)
            warm_parameters["cold_thaw_source"] = 0.0
            cold_parameters = dict(p)
            cold_parameters["persistent_warm_open_brake"] = 0.0
            warm_result = original_closure(
                base, current_data, warm_parameters, enabled
            )
            cold_result = original_closure(
                base, current_data, cold_parameters, enabled
            )
            base_hazard, rec, q_managed, _, managed_support = pathway_states(
                base, current_data
            )
            warm_hazard = -np.log1p(
                -np.clip(warm_result, 0.0, 1.0 - 1e-7)
            )
            cold_hazard = -np.log1p(
                -np.clip(cold_result, 0.0, 1.0 - 1e-7)
            )
            warm_delta = np.zeros_like(base_hazard)
            active = base_hazard > 1e-15
            warm_delta[active] = np.log(
                np.clip(
                    warm_hazard[active] / base_hazard[active], 1e-12, 1e12
                )
            )
            cold_source = np.maximum(cold_hazard - base_hazard, 0.0)
            gap = 1.0 - rec

            if kind == "brake_redistribution":
                # Strictly redistributes the incumbent warm-brake budget: low-
                # recurrence managed-open cells surrender a bounded fraction
                # of that brake, but no new source or brake is introduced.
                log_adjustment = warm_delta * (
                    1.0 - strength * q_managed * gap
                )
            elif kind == "gap_equilibrium":
                # Natural and recurrent managed pathways retain the incumbent
                # brake.  Only the low-recurrence managed share is replaced by
                # a rain-built, warm fuel-capacity response.
                managed_delta = (
                    rec * warm_delta
                    + strength * gap * managed_support
                )
                log_adjustment = (
                    (1.0 - q_managed) * warm_delta
                    + q_managed * managed_delta
                )
            elif kind == "symmetric_equilibrium":
                # Falsification control: it also imposes new suppression on
                # recurrent managed fire, the branch implicated by the exact
                # stacked-candidate failure.
                managed_delta = (
                    rec * warm_delta
                    + strength * (1.0 - 2.0 * rec) * managed_support
                )
                log_adjustment = (
                    (1.0 - q_managed) * warm_delta
                    + q_managed * managed_delta
                )
            else:
                raise ValueError(kind)
            adjusted = (
                base_hazard * np.exp(np.clip(log_adjustment, -5.0, 5.0))
                + cold_source
            )
            return np.asarray(
                1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)),
                dtype=np.float32,
            )

        return partitioned

    designs = (
        ("brake_redistribution", 0.5),
        ("brake_redistribution", 1.0),
        ("gap_equilibrium", 0.25),
        ("gap_equilibrium", 0.5),
        ("symmetric_equilibrium", 0.25),
    )
    for kind, strength in designs:
        label = f"{kind}:k={strength:g}"
        model._annual_regime_closure = make_closure(kind, strength)
        try:
            candidate = predict()
        finally:
            model._annual_regime_closure = original_closure
        candidate_predictions[label] = candidate
        print_comparison(
            label,
            candidate,
            observed,
            area,
            weight,
            folds,
            base_square,
        )

    base_annual = np.average(full, axis=0, weights=MONTH_DAYS)
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    for label, candidate in candidate_predictions.items():
        candidate_annual = np.average(candidate, axis=0, weights=MONTH_DAYS)
        print(f"ECOLOGY_CANDIDATE {label}", flush=True)
        for name, mask in masks.items():
            selected = mask[rows, cols]
            selected_area = area[selected]
            denominator = float(observed_annual[selected] @ selected_area)
            old = float(base_annual[selected] @ selected_area) / (
                denominator + 1e-12
            )
            new = float(candidate_annual[selected] @ selected_area) / (
                denominator + 1e-12
            )
            print(
                f"ECOLOGY {name} ratio={old:.8f}->{new:.8f} "
                f"delta={new-old:+.8f}",
                flush=True,
            )

    # Decompose where each incumbent branch earns annual-log error, by pathway
    # share and held block.  Positive improvement means the branch helps.
    no_square = annual_log_square(no_closure, observed)
    warm_square = annual_log_square(warm_only, observed)
    cold_square = annual_log_square(cold_only, observed)
    full_square = annual_log_square(full, observed)
    for branch, before, after in (
        ("warm", no_square, warm_square),
        ("cold", no_square, cold_square),
        ("warm_given_cold", cold_square, full_square),
        ("cold_given_warm", warm_square, full_square),
    ):
        improvement = before - after
        values = []
        for fold in range(4):
            selected = folds == fold
            values.append(
                float(
                    np.average(improvement[selected], weights=weight[selected])
                )
            )
        print(
            f"DECOMPOSE branch={branch} held_improvement="
            + ",".join(f"{value:+.9f}" for value in values),
            flush=True,
        )
        for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            selected = (annual_managed >= lower) & (annual_managed < upper)
            if np.any(selected):
                value = float(
                    np.average(improvement[selected], weights=weight[selected])
                )
                print(
                    f"DECOMPOSE_PATH branch={branch} managed_share={lower:.2f}:{upper:.2f} "
                    f"cells={int(selected.sum())} improvement={value:+.9f}",
                    flush=True,
                )

    model._annual_regime_closure = original_closure
    del data, candidate_predictions, captured, prepared
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
