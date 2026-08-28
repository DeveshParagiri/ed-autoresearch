"""Held replacement probe for rare-natural-onset timing at ``2dd6d61``.

The requested reverse-ML distillation is a probabilistic union of absolute
warmth and rapid warming, gated by a fresh lightning pulse.  Inspection of the
active canonical code found a narrower mapping than the prose hypothesis: the
installed onset timing factor is

    H_t I_t,
    H_t = sigmoid((T_t - T3_t - 0.5) / 1.5),
    I_t = L3_t / (L3_t + 0.02) * (0.35 + 0.65 P_t),
    P_t = sigmoid((p_t - 0.05) / 0.10),
    p_t = max((L_t - L3_t) / (L_t + L3_t + 0.002), 0).

The onset source does not directly contain the absolute thermal window; that
window occurs in the preceding broad rare-ignition paths.  To recover the
actual incumbent exactly at zero blend, this probe replaces only ``H I`` by

    B_t(b) = (1-b) H_t I_t + b S_t,
    S_t = P_t * (A_t + H_t - A_t H_t),
    A_t = sigmoid((T_t - 5) / 3),

for the fixed blends b=(0.10, 0.25, 0.50, 1.00).  The active squared rain-fuel,
drying-onset, combustion, primary-natural-share, and trailing-opportunity gates
are copied without change.  No new source or multiplier is introduced.

All runtime terms are current inputs or point-local causal antecedents.  GFED
and coordinates enter only post-prediction held losses, ecology, exact scoring,
and disjoint whole-cell fold assignment.  This file never edits canonical or
ledger artifacts and never invokes an official evaluation.
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

from autoresearch.scratchpad.clean_exogenous_rebuild_b867ed7 import (  # noqa: E402
    metric_line,
)
from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    held_losses,
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
BLENDS = (0.10, 0.25, 0.50, 1.00)


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
    module = types.ModuleType(f"model_{PINNED}_pulse_thermal_union")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def timing_terms(
    model, data: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    temperature_3 = model._antecedent(temperature, alpha_3)
    lightning_3 = model._antecedent(lightning, alpha_3)
    warming = model._rising(
        temperature - temperature_3, 1.0 / 1.5, 0.5
    )
    absolute_warmth = model._rising(temperature, 1.0 / 3.0, 5.0)
    lightning_departure = np.maximum(
        (lightning - lightning_3) / (lightning + lightning_3 + 0.002), 0.0
    )
    lightning_pulse = model._rising(lightning_departure, 10.0, 0.05)
    incumbent_access = lightning_3 / (lightning_3 + 0.02)
    incumbent_timing = warming * incumbent_access * (
        0.35 + 0.65 * lightning_pulse
    )
    union_timing = lightning_pulse * (
        absolute_warmth
        + warming
        - absolute_warmth * warming
    )
    return incumbent_timing, union_timing, {
        "warming": warming,
        "absolute_warmth": absolute_warmth,
        "lightning_pulse": lightning_pulse,
        "lightning_access": incumbent_access,
    }


def replacement_wrapper(model, original, state: dict[str, float]):
    def rare_natural_union(prediction, data, p, enabled):
        blend = float(state["blend"])
        if blend <= 0.0:
            return original(prediction, data, p, enabled)

        onset_scale = float(max(p.get("rare_natural_onset_scale", 0.0), 0.0))
        if "rare_ignition" not in enabled or onset_scale <= 0.0:
            return original(prediction, data, p, enabled)

        without_onset = dict(p)
        without_onset["rare_natural_onset_scale"] = 0.0
        incumbent = original(prediction, data, without_onset, enabled)
        incumbent_timing, union_timing, terms = timing_terms(model, data)
        timing = (1.0 - blend) * incumbent_timing + blend * union_timing

        rain = np.clip(
            np.asarray(data["monthly_precipitation"], dtype=np.float64),
            0.0,
            None,
        )
        dryness = np.clip(
            np.asarray(data["dryness"], dtype=np.float64), 0.0, None
        )
        alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
        dryness_3 = model._antecedent(dryness, alpha_3)
        dryness_departure = (dryness - dryness_3) / (
            dryness + dryness_3 + 100.0
        )
        drying_onset = model._rising(dryness_departure, 25.0, 0.01)
        combustion = np.sqrt(
            dryness / (dryness + 250.0) * 1.0 / (1.0 + rain / 35.0)
        )

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
        managed = np.clip(crop + pasture + rangeland, 0.0, 1.0)
        natural_share = primary / (primary + managed + 0.1)
        annual_rain = np.clip(
            np.asarray(data["annual_precipitation"], dtype=np.float64),
            0.0,
            None,
        )
        onset_fuel = np.power(
            annual_rain / (annual_rain + 250.0), 2.0
        ) * np.exp(-annual_rain / 3000.0)

        incumbent_hazard = -np.log1p(
            -np.clip(
                np.asarray(incumbent, dtype=np.float64),
                0.0,
                1.0 - 1e-7,
            )
        )
        trailing_hazard = np.empty_like(incumbent_hazard)
        accumulator = np.zeros_like(incumbent_hazard[0])
        for time in range(incumbent_hazard.shape[0]):
            accumulator += incumbent_hazard[time]
            if time >= 12:
                accumulator -= incumbent_hazard[time - 12]
            trailing_hazard[time] = accumulator
        opportunity_gap = 1.0 / (1.0 + trailing_hazard / 0.1)
        source = (
            natural_share
            * timing
            * drying_onset
            * combustion
            * onset_fuel
            * opportunity_gap
        )
        del incumbent_timing, union_timing, terms
        return np.asarray(
            np.clip(
                1.0
                - (1.0 - incumbent)
                * np.exp(-np.clip(onset_scale * source, 0.0, 50.0)),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )

    return rare_natural_union


def load_observation(
    evaluator: GFED5Evaluator,
) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    return observation, area


def area_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
) -> float:
    model_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    return float(np.sum(model_annual * area) / np.sum(obs_annual * area))


def prefix_test(
    model,
    full_data: Mapping[str, np.ndarray],
    rows: np.ndarray,
    columns: np.ndarray,
    state: dict[str, float],
) -> float:
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    before = {
        name: np.asarray(values[:, rows[probe], columns[probe]])[:, None, :]
        for name, values in full_data.items()
    }
    after = {name: values.copy() for name, values in before.items()}
    for values in after.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    maximum = 0.0
    for blend in BLENDS:
        state["blend"] = blend
        before_prediction = validate_prediction(
            model.predict(before, dict(model.PARAMS), None)
        )
        after_prediction = validate_prediction(
            model.predict(after, dict(model.PARAMS), None)
        )
        difference = float(
            np.max(np.abs(before_prediction[:96] - after_prediction[:96]))
        )
        print(
            f"PREFIX blend={blend:.2f} cutoff=96 cells=64 max_abs={difference:.12g}",
            flush=True,
        )
        maximum = max(maximum, difference)
    return maximum


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
    land = load_land_mask()
    rows, columns = np.where(land)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    evaluator = GFED5Evaluator(GFED5_PATH)
    observation_grid, area_grid = load_observation(evaluator)
    observation = np.asarray(
        observation_grid[:, rows, columns], dtype=np.float64
    )
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    observed_annual = np.average(observation, axis=0, weights=MONTH_DAYS)

    direct_baseline = validate_prediction(
        model.predict(data, dict(model.PARAMS), None)
    )
    baseline = np.asarray(
        direct_baseline[:, rows, columns], dtype=np.float64
    )
    base_losses = held_losses(
        baseline, observation, area, observed_annual, folds
    )
    base_scores = evaluator.score(direct_baseline)
    masks = regime_masks(data)
    base_ecology = ecological_statistics(
        direct_baseline, masks, observation_grid, area_grid, land
    )
    base_area = area_ratio(direct_baseline, observation_grid, area_grid)

    selected_data = {
        name: np.asarray(values[:, rows, columns])
        for name, values in data.items()
    }
    incumbent_timing, union_timing, timing = timing_terms(model, selected_data)
    print(
        f"BASE pinned={PINNED} blob={current_blob} land_cells={rows.size} folds="
        + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(metric_line("BASE_EXACT", base_scores["global"]), flush=True)
    print(
        "BASE_HELD annual="
        + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation="
        + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle="
        + ",".join(f"{value:.9f}" for value in base_losses[2])
        + f" area_ratio={base_area:.9f}",
        flush=True,
    )
    print(
        f"TIMING incumbent_mean={incumbent_timing.mean():.9f} "
        f"union_mean={union_timing.mean():.9f} "
        f"union_gt_incumbent={np.mean(union_timing > incumbent_timing):.9f} "
        f"corr={np.corrcoef(incumbent_timing.ravel(),union_timing.ravel())[0,1]:.9f} "
        f"pulse_mean={timing['lightning_pulse'].mean():.9f} "
        f"absolute_warmth_mean={timing['absolute_warmth'].mean():.9f} "
        f"warming_mean={timing['warming'].mean():.9f} "
        f"access_mean={timing['lightning_access'].mean():.9f}",
        flush=True,
    )
    del incumbent_timing, union_timing, timing, selected_data
    gc.collect()

    original = model._rare_lightning_ignition
    state = {"blend": 0.0}
    model._rare_lightning_ignition = replacement_wrapper(model, original, state)
    records = []
    try:
        recovered = validate_prediction(
            model.predict(data, dict(model.PARAMS), None)
        )
        recovery = float(
            np.max(
                np.abs(
                    np.asarray(recovered, dtype=np.float64)
                    - np.asarray(direct_baseline, dtype=np.float64)
                )
            )
        )
        print(f"RECOVERY blend=0 max_abs={recovery:.12g}", flush=True)
        if recovery != 0.0:
            raise RuntimeError(f"zero blend does not recover incumbent: {recovery}")
        del recovered
        gc.collect()

        for blend in BLENDS:
            state["blend"] = blend
            prediction = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
            selected_prediction = np.asarray(
                prediction[:, rows, columns], dtype=np.float64
            )
            losses = held_losses(
                selected_prediction,
                observation,
                area,
                observed_annual,
                folds,
            )
            gains = tuple(
                base_losses[index] - losses[index] for index in range(3)
            )
            stable = tuple(bool(np.all(gain > 0.0)) for gain in gains)
            aggregate = float(
                sum(
                    np.sum(gains[index] / base_losses[index])
                    for index in range(3)
                )
            )
            candidate_area = area_ratio(prediction, observation_grid, area_grid)
            ecology = ecological_statistics(
                prediction, masks, observation_grid, area_grid, land
            )
            pathologies = []
            for name in base_ecology:
                before = float(base_ecology[name]["ratio"])
                after = float(ecology[name]["ratio"])
                if (
                    after < 0.25
                    or after > 4.0
                    or after / before < 0.75
                    or after / before > 1.25
                ):
                    pathologies.append(name)

            exact = None
            region_breadth = None
            if aggregate > 0.0:
                exact = evaluator.score(prediction)
                region_breadth = sum(
                    int(
                        exact[name]["overall_score"]
                        > base_scores[name]["overall_score"]
                    )
                    for name in exact
                    if name != "global"
                )
                print(
                    metric_line(f"EXACT blend={blend:.2f}", exact["global"])
                    + f" delta={exact['global']['overall_score']-base_scores['global']['overall_score']:+.9f}"
                    + f" regions_positive={region_breadth}/14",
                    flush=True,
                )

            print(
                f"HELD blend={blend:.2f} annual_stable={int(stable[0])} "
                f"allocation_stable={int(stable[1])} raw_stable={int(stable[2])} "
                f"aggregate={aggregate:+.9f} annual_gain="
                + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain="
                + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain="
                + ",".join(f"{value:+.9f}" for value in gains[2])
                + f" area_ratio={candidate_area:.9f} area_drift={candidate_area-base_area:+.9f} "
                + "severe="
                + (",".join(pathologies) if pathologies else "none"),
                flush=True,
            )
            print(
                f"ECOLOGY blend={blend:.2f} "
                + ",".join(
                    f"{name}:{float(base_ecology[name]['ratio']):.9f}->"
                    f"{float(ecology[name]['ratio']):.9f}"
                    for name in base_ecology
                ),
                flush=True,
            )
            records.append(
                (aggregate, blend, stable, candidate_area, exact, pathologies)
            )
            del prediction, selected_prediction, ecology
            gc.collect()

        best = max(records, key=lambda record: record[0])
        print(
            f"SUMMARY best_blend={best[1]:.2f} aggregate={best[0]:+.9f} "
            f"gates=" + ",".join(str(int(value)) for value in best[2])
            + f" exact_run={int(best[4] is not None)} severe="
            + (",".join(best[5]) if best[5] else "none"),
            flush=True,
        )
        prefix_max = prefix_test(model, data, rows, columns, state)
        print(f"PREFIX_SUMMARY max_abs={prefix_max:.12g}", flush=True)
        if prefix_max != 0.0:
            raise RuntimeError(f"prefix causality failed: {prefix_max}")
    finally:
        model._rare_lightning_ignition = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
