"""Exact proxy bracket for a rare natural heating-onset ignition source.

The candidate is inserted inside the existing rare-ignition component, before
all later canonical closures.  It uses only current or causal trailing local
state and one globally shared equation.  This scratch diagnostic does not edit
the canonical model or invoke the official evaluator.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.clean_exogenous_rebuild_b867ed7 import metric_line  # noqa: E402
from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


SCALES = (0.0, 0.0005, 0.0015, 0.003)


def trailing_sum(values: np.ndarray, months: int = 12) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    running = np.zeros(values.shape[1:], dtype=np.float64)
    for time in range(values.shape[0]):
        running += values[time]
        if time >= months:
            running -= values[time - months]
        output[time] = running
    return output


def onset_wrapper(model, original, scale_state):
    def rare_natural_onset(prediction, data, p, enabled):
        incumbent = original(prediction, data, p, enabled)
        scale = float(scale_state["scale"])
        fuel_power = float(scale_state.get("fuel_power", 1.0))
        if scale <= 0.0 or "rare_ignition" not in enabled:
            return incumbent

        temperature = np.asarray(data["air_temperature"], dtype=np.float64)
        dryness = np.clip(
            np.asarray(data["dryness"], dtype=np.float64), 0.0, None
        )
        rain = np.clip(
            np.asarray(data["monthly_precipitation"], dtype=np.float64),
            0.0,
            None,
        )
        lightning = np.clip(
            np.asarray(data["lightning_flash_rate"], dtype=np.float64),
            0.0,
            None,
        )
        alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
        alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
        temperature_3 = model._antecedent(temperature, alpha_3)
        dryness_3 = model._antecedent(dryness, alpha_3)
        lightning_3 = model._antecedent(lightning, alpha_3)
        rain_12 = model._antecedent(rain, alpha_12)

        heat_onset = model._rising(
            temperature - temperature_3, 1.0 / 1.5, 0.5
        )
        dryness_departure = (dryness - dryness_3) / (
            dryness + dryness_3 + 100.0
        )
        drying_onset = model._rising(dryness_departure, 25.0, 0.01)
        combustion = np.sqrt(
            dryness / (dryness + 250.0) * 1.0 / (1.0 + rain / 35.0)
        )
        lightning_departure = np.maximum(
            (lightning - lightning_3)
            / (lightning + lightning_3 + 0.002),
            0.0,
        )
        lightning_arrival = model._rising(
            lightning_departure, 10.0, 0.05
        )
        lightning_access = lightning_3 / (lightning_3 + 0.02)
        ignition_timing = lightning_access * (
            0.35 + 0.65 * lightning_arrival
        )

        natural_mode = str(scale_state.get("natural_mode", "invalid_luh2_sum"))
        primary = np.clip(
            np.asarray(data["luh2_primary_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        crop = np.clip(
            np.asarray(data["luh2_cropland_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        pasture = np.clip(
            np.asarray(data["luh2_pasture_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        rangeland = np.clip(
            np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        if natural_mode == "ed_state":
            natural_cover = np.clip(
                np.asarray(data["natural_vegetation_fraction"], dtype=np.float64)
                + np.asarray(
                    data["secondary_vegetation_fraction"], dtype=np.float64
                ),
                0.0,
                1.0,
            )
        elif natural_mode == "primary_only":
            natural_cover = primary
        elif natural_mode == "invalid_luh2_sum":
            # Retained only to reproduce the completed falsification bracket.
            # LUH2 secondary is not a compositional land fraction here and
            # this mode must never be promoted into the canonical model.
            secondary = np.clip(
                np.asarray(data["luh2_secondary_fraction"], dtype=np.float64),
                0.0,
                1.0,
            )
            natural_cover = np.clip(primary + secondary, 0.0, 1.0)
        else:
            raise ValueError(f"unknown natural_mode: {natural_mode}")
        managed_cover = np.clip(crop + pasture + rangeland, 0.0, 1.0)
        natural_share = natural_cover / (
            natural_cover + managed_cover + 0.1
        )

        annualized_rain = 12.0 * rain_12
        rain_built_fuel = (
            np.power(
                annualized_rain / (annualized_rain + 250.0), fuel_power
            )
            * np.exp(-annualized_rain / 3000.0)
        )
        incumbent_hazard = -np.log1p(
            -np.clip(np.asarray(incumbent, dtype=np.float64), 0.0, 1.0 - 1e-7)
        )
        opportunity_gap = 1.0 / (
            1.0 + trailing_sum(incumbent_hazard) / 0.1
        )
        source = (
            natural_share
            * heat_onset
            * drying_onset
            * combustion
            * ignition_timing
            * rain_built_fuel
            * opportunity_gap
        )
        return np.asarray(
            np.clip(
                1.0
                - (1.0 - incumbent)
                * np.exp(-np.clip(scale * source, 0.0, 50.0)),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )

    return rare_natural_onset


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    evaluator = GFED5Evaluator(GFED5_PATH)
    original = model._rare_lightning_ignition
    scale_state = {"scale": 0.0}
    model._rare_lightning_ignition = onset_wrapper(
        model, original, scale_state
    )

    rows = []
    best_prediction = None
    try:
        for scale in SCALES:
            scale_state["scale"] = scale
            prediction = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
            scores = evaluator.score(prediction)
            global_scores = dict(scores["global"])
            print(metric_line(f"rare_natural_onset:scale={scale:g}", global_scores), flush=True)
            rows.append((float(global_scores["overall_score"]), scale, scores))
            if best_prediction is None or rows[-1][0] > max(row[0] for row in rows[:-1]):
                del best_prediction
                best_prediction = prediction.copy()
            del prediction
            gc.collect()

        baseline_overall, _, baseline_scores = rows[0]
        best_overall, best_scale, best_scores = max(rows, key=lambda row: row[0])
        print(
            f"BEST scale={best_scale:g} overall={best_overall:.9f} "
            f"delta={best_overall - baseline_overall:+.9f}",
            flush=True,
        )
        positive_regions = 0
        for name in sorted(key for key in best_scores if key != "global"):
            delta = (
                float(best_scores[name]["overall_score"])
                - float(baseline_scores[name]["overall_score"])
            )
            positive_regions += int(delta > 0.0)
            print(
                f"REGION {name} baseline={baseline_scores[name]['overall_score']:.9f} "
                f"winner={best_scores[name]['overall_score']:.9f} delta={delta:+.9f}",
                flush=True,
            )
        print(f"REGIONAL_BREADTH positive={positive_regions}/14", flush=True)

        if best_overall > baseline_overall + 1e-12:
            assert best_prediction is not None
            with Dataset(GFED5_PATH) as dataset:
                reference = np.asarray(dataset.variables["burntArea"][:192])
            observation = (
                reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4))
                / 100.0
            )
            del reference
            area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
            land = load_land_mask()
            ecology = ecological_statistics(
                best_prediction,
                regime_masks(data),
                observation,
                area,
                land,
            )
            for name, values in ecology.items():
                print(
                    f"ECOLOGY {name} cells={values['cells']} "
                    f"ratio={float(values['ratio']):.9f} "
                    f"phase={values['phase_months']}",
                    flush=True,
                )

            prefix = 96
            expected_prefix = best_prediction[:prefix].copy()
            del best_prediction, observation, area, land
            gc.collect()
            for values in data.values():
                values[prefix:] *= np.float32(0.5)
            scale_state["scale"] = best_scale
            perturbed = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
            difference = float(
                np.max(np.abs(perturbed[:prefix] - expected_prefix))
            )
            print(
                f"PREFIX future_half_after={prefix} "
                f"max_abs_difference={difference:.12g}",
                flush=True,
            )
    finally:
        model._rare_lightning_ignition = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
