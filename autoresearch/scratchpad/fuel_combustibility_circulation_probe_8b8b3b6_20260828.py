"""Bounded held probe of fuel-production/combustibility circulation.

The only changed operation is the release selector inside the incumbent finite
surface-opportunity bank.  For bounded current productivity ``P`` and
combustibility ``C``, the signed state is

    J[t] = P[t-1] C[t] - C[t-1] P[t].

The signed physical form, its reversed-sign falsification control, and an
unsigned transition-amplitude control modulate the incumbent release
opportunity by ``clip(1 + kappa * surface_share * direction, .75, 1.25)``.
Storage, pathway share, hazard input, and every downstream canonical stage are
unchanged.  Coordinates and GFED are used only for held folds and audits.

The initial strict all-fold gate is retained as a held stability diagnostic.
Under the clarified Overall-first rule, exact full-grid proxy replay covers all
signed and unsigned brackets because aggregate annual, allocation, and
raw-cycle evidence is positive and ecology is benign.  This script writes no
official or canonical artifact.

Observed on 2026-08-28 from canonical model source cdb03c0, whose model blob
0d05b1c75489fbdde6a1996aa993ed1e67657c71 is unchanged from 8b8b3b6, the
four-fold gains in annual/allocation/raw-cycle loss were as follows.  Signed
kappa .10 gave annual [+1.959e-6,+0.592e-6,-0.275e-6,+0.314e-6], allocation
[+8.778e-6,+5.152e-6,-0.783e-6,+4.626e-6], and raw cycle
[+10.735e-6,+11.211e-6,+4.012e-6,+6.124e-6].  Signed kappa .25 gave annual
[+5.368e-6,+1.893e-6,-1.096e-6,+1.302e-6], allocation
[+21.137e-6,+12.150e-6,-2.135e-6,+10.770e-6], and raw cycle
[+26.093e-6,+27.177e-6,+9.837e-6,+14.691e-6].  Signed kappa .50 gave annual
[+10.826e-6,+4.100e-6,-2.670e-6,+2.901e-6], allocation
[+39.710e-6,+22.012e-6,-4.792e-6,+19.039e-6], and raw cycle
[+49.862e-6,+51.694e-6,+19.048e-6,+27.439e-6].  The normalized aggregate
shape gains for signed/unsigned/reverse were respectively
.000981552/.000944774/-.001031358 at .10,
.002367206/.002281502/-.002675704 at .25, and
.004462291/.004311257/-.005700191 at .50.  Thus the signed direction narrowly
beat both controls and the reverse sign was consistently harmful, but the
allocation failure in fold 2 persisted at every bracket and the unsigned
control retained 96--97% of the signed gain.  No bracket passed that strict
held diagnostic in the initial pass.  Every prefix delta was exactly zero.  At
the strongest signed bracket the held-cell ecological ratios changed
only slightly: intact tropical closed .822142 to .822154, temperate closed
.878983 to .879000, boreal 1.627064 to 1.627178, tropical open 1.028298 to
1.028327, productive rangeland .672304 to .672374, cropland .971006 to .971007,
and arid low fuel 1.131521 to 1.131512.
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

from autoresearch.scratchpad.additive_pathway_replacement_a8ed115 import (  # noqa: E402
    ecological_ratios_selected,
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


PINNED = "cdb03c0"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
EXPECTED_OVERALL = 0.720105466
KAPPAS = (0.10, 0.25, 0.50)
MODES = ("signed", "unsigned", "reverse")
REGIONS = (
    "bona", "tena", "ceam", "nhsa", "shsa", "euro", "mide",
    "nhaf", "shaf", "boas", "ceas", "seas", "eqas", "aust",
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
        raise RuntimeError(f"unexpected pinned model blob {blob}")
    module = types.ModuleType(f"model_{PINNED}_fuel_combustibility_circulation")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def make_surface_bank(model, mode: str, kappa: float, diagnostics: dict[str, float]):
    if mode not in MODES:
        raise ValueError(mode)

    def surface_bank(
        prediction: np.ndarray,
        data: Mapping[str, np.ndarray],
        p: Mapping[str, float],
        enabled: set[str],
    ) -> np.ndarray:
        if "surface_opportunity_bank" not in enabled:
            return prediction
        strength = float(np.clip(p.get("surface_bank_w", 0.0), 0.0, 1.0))
        if strength <= 0.0:
            return prediction

        alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
        alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
        alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
        gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
        gpp_3 = model._antecedent(gpp, alpha_3)
        gpp_12 = model._antecedent(gpp, alpha_12)
        fine_fuel = gpp_12 / (gpp_12 + 0.35)
        curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)

        rain = np.clip(
            np.asarray(data["monthly_precipitation"], dtype=np.float64),
            0.0,
            None,
        )
        rain_6 = model._antecedent(rain, alpha_6)
        rain_deficit = np.maximum(
            (rain_6 - rain) / (rain_6 + rain + 10.0), 0.0
        )
        dryness = np.clip(
            np.asarray(data["dryness"], dtype=np.float64), 0.0, None
        )
        combustion = dryness / (dryness + 500.0)
        temperature = np.asarray(data["air_temperature"], dtype=np.float64)
        thermal = model._rising(temperature, 1.0 / 3.0, 5.0)

        productivity = gpp / (gpp + 0.35)
        trajectory_combustibility = (
            combustion / (1.0 + rain / 35.0) * thermal
        )
        previous_productivity = np.empty_like(productivity)
        previous_combustibility = np.empty_like(trajectory_combustibility)
        previous_productivity[0] = productivity[0]
        previous_combustibility[0] = trajectory_combustibility[0]
        previous_productivity[1:] = productivity[:-1]
        previous_combustibility[1:] = trajectory_combustibility[:-1]
        circulation = (
            previous_productivity * trajectory_combustibility
            - previous_combustibility * productivity
        )
        if mode == "signed":
            direction = circulation
        elif mode == "unsigned":
            direction = np.abs(circulation)
        else:
            direction = -circulation

        natural = np.clip(
            np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        rangeland = np.clip(
            np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        pasture = np.clip(
            np.asarray(data["luh2_pasture_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        crop = np.clip(
            np.asarray(data["luh2_cropland_fraction"], dtype=np.float64),
            0.0,
            1.0,
        )
        canopy = np.clip(
            np.asarray(data["natural_canopy_height"], dtype=np.float64),
            0.0,
            None,
        )
        biomass = np.clip(
            np.asarray(data["aboveground_biomass"], dtype=np.float64),
            0.0,
            None,
        )
        open_cover = np.clip(
            rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
        )
        surface_capacity = (1.0 - crop) * fine_fuel * open_cover
        woody_capacity = (
            natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
        )
        crop_capacity = crop * fine_fuel
        surface_share = surface_capacity / (
            0.05 + surface_capacity + woody_capacity + crop_capacity
        )
        modulator = np.clip(
            1.0 + float(kappa) * surface_share * direction, 0.75, 1.25
        )

        hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
        bank = np.zeros_like(hazard[0])
        hazard_state = hazard[0].copy()
        allocated = np.empty_like(hazard)
        release_rate = float(max(p.get("surface_bank_release", 8.0), 0.0))
        input_sum = np.zeros_like(hazard[0])
        output_sum = np.zeros_like(hazard[0])
        for time in range(hazard.shape[0]):
            relative_opportunity = hazard[time] / (
                hazard[time] + hazard_state + 1e-8
            )
            physical_window = np.sqrt(
                np.clip(
                    fine_fuel[time]
                    * combustion[time]
                    * rain_deficit[time],
                    0.0,
                    1.0,
                )
            )
            physical_window *= 0.25 + 0.75 * curing[time] / (
                curing[time] + 0.05
            )
            release_opportunity = (
                relative_opportunity * physical_window * modulator[time]
            )
            release_fraction = 1.0 - np.exp(
                -(1.0 / 24.0 + release_rate * release_opportunity)
            )
            stored = strength * surface_share[time] * hazard[time]
            bank += stored
            released = release_fraction * bank
            bank -= released
            allocated[time] = hazard[time] - stored + released
            hazard_state += alpha_12 * (hazard[time] - hazard_state)
            input_sum += hazard[time]
            output_sum += allocated[time]

        diagnostics.clear()
        diagnostics.update(
            circulation_mean=float(np.mean(circulation)),
            circulation_abs_mean=float(np.mean(np.abs(circulation))),
            circulation_abs_p95=float(np.quantile(np.abs(circulation), 0.95)),
            modulator_mean=float(np.mean(modulator)),
            modulator_min=float(np.min(modulator)),
            modulator_max=float(np.max(modulator)),
            final_bank_fraction=float(
                np.sum(bank) / max(float(np.sum(input_sum)), 1e-15)
            ),
            closure_max_abs=float(
                np.max(np.abs(input_sum - output_sum - bank))
            ),
        )
        return np.asarray(
            1.0 - np.exp(-np.clip(allocated, 0.0, 50.0)), dtype=np.float32
        )

    return surface_bank


def predict_variant(
    model,
    data: Mapping[str, np.ndarray],
    mode: str,
    kappa: float,
) -> tuple[np.ndarray, dict[str, float]]:
    original = model._surface_fire_opportunity_bank
    diagnostics: dict[str, float] = {}
    model._surface_fire_opportunity_bank = make_surface_bank(
        model, mode, kappa, diagnostics
    )
    try:
        prediction = checked_prediction(
            model.predict(data, dict(model.PARAMS), None)
        )
    finally:
        model._surface_fire_opportunity_bank = original
    return prediction, diagnostics


def checked_prediction(prediction: np.ndarray) -> np.ndarray:
    """Apply the official validator on-grid and equivalent checks off-grid."""
    array = np.asarray(prediction)
    if array.shape == (192, 180, 360):
        return validate_prediction(array)
    if array.ndim != 3 or array.shape[0] != 192:
        raise RuntimeError(f"unexpected reduced prediction shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RuntimeError("reduced prediction contains nonfinite values")
    if float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0:
        raise RuntimeError("reduced prediction is outside [0, 1]")
    return array


def gain_text(gains: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    labels = ("annual", "allocation", "raw_cycle")
    return " ".join(
        label + "_gain=" + ",".join(f"{value:+.9f}" for value in values)
        for label, values in zip(labels, gains)
    )


def global_area_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    land: np.ndarray,
) -> float:
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    weight = area * land
    return float(np.sum(pred_annual * weight) / np.sum(obs_annual * weight))


def main() -> int:
    head = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"current model drifted to {current_blob}")

    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_scores = evaluator.score(incumbent)
    incumbent_global = incumbent_scores["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError(
            f"incumbent score drift {incumbent_global['overall_score']:.9f}"
        )

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual_grid = np.average(observation, axis=0, weights=MONTH_DAYS)
    pred_annual_grid = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area_grid * obs_annual_grid
    excess_weight = area_grid * np.maximum(
        pred_annual_grid - obs_annual_grid, 0.0
    )

    def top(weight: np.ndarray) -> np.ndarray:
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, 0.90) + 1)]

    cells = np.union1d(top(observed_weight), top(excess_weight))
    rows, columns = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    selected_data = {
        name: np.asarray(values[:, rows, columns], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    selected_incumbent = np.asarray(
        incumbent[:, rows, columns], dtype=np.float64
    )
    selected_observation = np.asarray(
        observation[:, rows, columns], dtype=np.float64
    )
    selected_area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    selected_obs_annual = np.asarray(
        obs_annual_grid[rows, columns], dtype=np.float64
    )
    local_baseline = checked_prediction(
        model.predict(selected_data, dict(model.PARAMS), None)
    )[:, 0, :]
    local_match = float(np.max(np.abs(local_baseline - selected_incumbent)))
    if local_match > 2e-6:
        raise RuntimeError(f"selected baseline mismatch {local_match}")
    base_losses = held_losses(
        selected_incumbent,
        selected_observation,
        selected_area,
        selected_obs_annual,
        folds,
    )
    base_ecology = ecological_ratios_selected(
        selected_incumbent,
        selected_observation,
        selected_data,
        selected_area,
    )
    print(
        f"BASE head={head} model_source={PINNED} blob={current_blob} "
        f"overall={incumbent_global['overall_score']:.9f} cells={cells.size} "
        f"folds=" + ",".join(str(int(np.sum(folds == fold))) for fold in range(4))
        + f" local_match={local_match:.12g}",
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{v:.9f}" for v in base_losses[0])
        + " allocation=" + ",".join(f"{v:.9f}" for v in base_losses[1])
        + " raw_cycle=" + ",".join(f"{v:.9f}" for v in base_losses[2]),
        flush=True,
    )
    print(
        "POLICY overall_first_exact=signed_and_unsigned_all_brackets;"
        "strict_held_diagnostic=allocation_and_raw_positive_all_folds,"
        "aggregate_annual_positive,signed_shape_better_than_controls;"
        "kappa=.10,.25,.50;clip=.75,1.25",
        flush=True,
    )

    records: dict[tuple[str, float], dict[str, object]] = {}
    for kappa in KAPPAS:
        for mode in MODES:
            local_trial, diagnostics = predict_variant(
                model, selected_data, mode, kappa
            )
            local_trial = np.asarray(local_trial[:, 0, :], dtype=np.float64)
            losses = held_losses(
                local_trial,
                selected_observation,
                selected_area,
                selected_obs_annual,
                folds,
            )
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            shape_gain = float(
                np.sum(gains[1] / base_losses[1])
                + np.sum(gains[2] / base_losses[2])
            )
            annual_gain = float(np.sum(gains[0] / base_losses[0]))
            trial_ecology = ecological_ratios_selected(
                local_trial,
                selected_observation,
                selected_data,
                selected_area,
            )
            records[(mode, kappa)] = {
                "prediction": local_trial,
                "gains": gains,
                "shape_gain": shape_gain,
                "annual_gain": annual_gain,
                "diagnostics": diagnostics,
                "ecology": trial_ecology,
            }
            print(
                f"HELD mode={mode} kappa={kappa:.2f} "
                f"shape_gain={shape_gain:+.9f} annual_relative_gain={annual_gain:+.9f} "
                + gain_text(gains)
                + " diag="
                + ",".join(f"{name}:{value:.9g}" for name, value in diagnostics.items()),
                flush=True,
            )
            print(
                f"HELD_ECOLOGY mode={mode} kappa={kappa:.2f} "
                + ",".join(
                    f"{name}:{base_ecology[name]:.6f}->{trial_ecology[name]:.6f}"
                    for name in base_ecology
                ),
                flush=True,
            )

    for kappa in KAPPAS:
        signed = records[("signed", kappa)]
        gains = signed["gains"]
        assert isinstance(gains, tuple)
        controls = (
            float(records[("unsigned", kappa)]["shape_gain"]),
            float(records[("reverse", kappa)]["shape_gain"]),
        )
        shape_stable = bool(np.all(gains[1] > 0.0) and np.all(gains[2] > 0.0))
        annual_positive = float(signed["annual_gain"]) > 0.0
        signed_wins = float(signed["shape_gain"]) > max(controls)
        exact_gate = bool(shape_stable and annual_positive and signed_wins)
        print(
            f"HELD_DIAGNOSTIC kappa={kappa:.2f} shape_stable={int(shape_stable)} "
            f"annual_positive={int(annual_positive)} signed_wins={int(signed_wins)} "
            f"strict_pass={int(exact_gate)} signed_shape={float(signed['shape_gain']):+.9f} "
            f"unsigned_shape={controls[0]:+.9f} reverse_shape={controls[1]:+.9f}",
            flush=True,
        )

    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    changed_data = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed_data.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    prefix_max = 0.0
    for kappa in KAPPAS:
        for mode in MODES:
            before = predict_variant(model, prefix_data, mode, kappa)[0]
            after = predict_variant(model, changed_data, mode, kappa)[0]
            delta = float(np.max(np.abs(before[:96] - after[:96])))
            prefix_max = max(prefix_max, delta)
            print(
                f"PREFIX mode={mode} kappa={kappa:.2f} max_abs={delta:.12g}",
                flush=True,
            )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix failure {prefix_max}")

    land = load_land_mask()
    masks = regime_masks(data)
    base_full_ecology = ecological_statistics(
        incumbent, masks, observation, area_grid, land
    )
    base_area = global_area_ratio(incumbent, observation, area_grid, land)
    exact_records: dict[tuple[str, float], dict[str, float]] = {}
    for kappa in KAPPAS:
        for mode in ("signed", "unsigned"):
            full_candidate, full_diagnostics = predict_variant(
                model, data, mode, kappa
            )
            full_scores = evaluator.score(full_candidate)
            full_global = full_scores["global"]
            trial_full_ecology = ecological_statistics(
                full_candidate, masks, observation, area_grid, land
            )
            trial_area = global_area_ratio(
                full_candidate, observation, area_grid, land
            )
            exact_records[(mode, kappa)] = {
                "overall": float(full_global["overall_score"]),
                "delta_overall": float(
                    full_global["overall_score"]
                    - incumbent_global["overall_score"]
                ),
            }
            print(
                f"EXACT mode={mode} kappa={kappa:.2f} "
                f"overall={full_global['overall_score']:.9f} "
                f"delta_overall={full_global['overall_score']-incumbent_global['overall_score']:+.9f} "
                f"bias={full_global['bias_score']:.9f} "
                f"delta_bias={full_global['bias_score']-incumbent_global['bias_score']:+.9f} "
                f"rmse={full_global['rmse_score']:.9f} "
                f"delta_rmse={full_global['rmse_score']-incumbent_global['rmse_score']:+.9f} "
                f"seasonal={full_global['seasonal_cycle_score']:.9f} "
                f"delta_seasonal={full_global['seasonal_cycle_score']-incumbent_global['seasonal_cycle_score']:+.9f} "
                f"spatial={full_global['spatial_distribution_score']:.9f} "
                f"delta_spatial={full_global['spatial_distribution_score']-incumbent_global['spatial_distribution_score']:+.9f} "
                f"area_ratio={base_area:.9f}->{trial_area:.9f}",
                flush=True,
            )
            print(
                f"REGIONS mode={mode} kappa={kappa:.2f} "
                + ",".join(
                    f"{name}:{full_scores[name]['overall_score']-incumbent_scores[name]['overall_score']:+.9f}"
                    for name in REGIONS
                ),
                flush=True,
            )
            print(
                f"ECOLOGY mode={mode} kappa={kappa:.2f} "
                + ",".join(
                    f"{name}:{float(base_full_ecology[name]['ratio']):.6f}"
                    f"->{float(trial_full_ecology[name]['ratio']):.6f}"
                    for name in base_full_ecology
                ),
                flush=True,
            )
            print(
                f"EXACT_DIAG mode={mode} kappa={kappa:.2f} "
                + ",".join(
                    f"{name}:{value:.9g}"
                    for name, value in full_diagnostics.items()
                ),
                flush=True,
            )
            del full_candidate
            gc.collect()

    for kappa in KAPPAS:
        signed_delta = exact_records[("signed", kappa)]["delta_overall"]
        unsigned_delta = exact_records[("unsigned", kappa)]["delta_overall"]
        print(
            f"EXACT_CONTROL kappa={kappa:.2f} "
            f"signed_delta={signed_delta:+.9f} "
            f"unsigned_delta={unsigned_delta:+.9f} "
            f"signed_minus_unsigned={signed_delta-unsigned_delta:+.9f}",
            flush=True,
        )
    chosen_mode, chosen_kappa = max(
        exact_records,
        key=lambda key: exact_records[key]["overall"],
    )
    chosen = exact_records[(chosen_mode, chosen_kappa)]
    accept = bool(chosen["delta_overall"] > 0.0)
    print(
        f"DECISION exact=1 accept={int(accept)} mode={chosen_mode} "
        f"kappa={chosen_kappa:.2f} overall={chosen['overall']:.9f} "
        f"delta_overall={chosen['delta_overall']:+.9f} "
        f"prefix={prefix_max:.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
