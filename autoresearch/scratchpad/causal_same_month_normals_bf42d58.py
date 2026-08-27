"""Held-cell screen for causal same-calendar-month climate normals.

The diagnostic tree in Entry 109 included a broad ``calendar_mean`` feature,
but no small physical response was isolated from it.  This script tests that
missing mechanism directly.  Each ED site keeps twelve independent running
states for rain, dryness, temperature, GPP, and LAI.  The state used in month
``t`` contains only the same calendar month from *prior* years; the current
value is incorporated only after the response for ``t`` has been constructed.

The tested equations are globally shared, pointwise, and target two physical
roles: weather surprise relative to the locally expected season, and a stable
seasonal combustion opportunity carried by prior-year climate and fuel.  A
small tree is used only to ask whether nonlinear anomaly summaries add held-
cell information.  No fitted response is installed or evaluated officially.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    annual_loss,
    area_ratio,
    cycle_loss,
    ecological_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    select_high_weight,
)
from scripts.runtime import load_land_mask  # noqa: E402


EXPECTED_COMMIT = "bf42d58b4c0e60fc3408b9e94e9cd34a0a581214"
EXPECTED_MODEL_BLOB = "a1966275d22874d1c71c45c7b8a8f5c8e473358d"
EPS = np.float32(1e-6)


def pinned_model():
    source = subprocess.run(
        ("git", "show", f"{EXPECTED_COMMIT}:autoresearch/model.py"),
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
    module = types.ModuleType("ed_fire_pinned_bf42d58")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def lagged_calendar_normal(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prior-year same-month mean, spread, and evidence weight.

    The current value is deliberately excluded from the returned state.  The
    first occurrence of each month is neutral: its normal equals the current
    value and its confidence is zero.  This makes the response suitable for a
    cold start and exactly prefix causal.
    """
    values = np.asarray(values, dtype=np.float32)
    mean = np.zeros((12, values.shape[1]), dtype=np.float64)
    m2 = np.zeros_like(mean)
    count = np.zeros(12, dtype=np.int64)
    normal = np.empty_like(values, dtype=np.float32)
    spread = np.empty_like(values, dtype=np.float32)
    confidence = np.empty_like(values, dtype=np.float32)
    for time_index in range(values.shape[0]):
        month = time_index % 12
        if count[month] == 0:
            normal[time_index] = values[time_index]
            spread[time_index] = 0.0
            confidence[time_index] = 0.0
        else:
            normal[time_index] = mean[month]
            spread[time_index] = np.sqrt(
                np.maximum(m2[month] / count[month], 0.0)
            )
            confidence[time_index] = count[month] / (count[month] + 2.0)
        count[month] += 1
        delta = values[time_index] - mean[month]
        mean[month] += delta / count[month]
        m2[month] += delta * (values[time_index] - mean[month])
    return normal, spread, confidence


def ema(values: np.ndarray, months: float = 12.0) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(values[0], dtype=np.float32).copy()
    output = np.empty_like(values, dtype=np.float32)
    for time_index in range(values.shape[0]):
        state += alpha * (values[time_index] - state)
        output[time_index] = state
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def relative_allocator(
    prediction: np.ndarray,
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply a bounded signal while conserving a causal annual reference."""
    factor = np.exp(np.clip(float(strength) * signal, -2.0, 2.0))
    candidate = np.clip(prediction * factor, 0.0, 1.0)
    base_state = np.asarray(prediction[0], dtype=np.float64).copy()
    candidate_state = np.asarray(candidate[0], dtype=np.float64).copy()
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    output = np.empty_like(candidate, dtype=np.float32)
    for time_index in range(candidate.shape[0]):
        base_state += alpha * (prediction[time_index] - base_state)
        candidate_state += alpha * (candidate[time_index] - candidate_state)
        output[time_index] = np.clip(
            candidate[time_index] * base_state / (candidate_state + 1e-12),
            0.0,
            1.0,
        )
    return output


def hazard_capacity(
    prediction: np.ndarray,
    signal: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Scale finite event hazard without manufacturing a separate ignition."""
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    adjusted = hazard * np.exp(np.clip(float(strength) * signal, -2.0, 2.0))
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def states(data: dict[str, np.ndarray], baseline: np.ndarray):
    field = {
        name: np.asarray(values[:, 0, :], dtype=np.float32)
        for name, values in data.items()
    }
    rain = np.clip(field["monthly_precipitation"], 0.0, None)
    dryness = np.clip(field["dryness"], 0.0, None)
    temperature = field["air_temperature"]
    gpp = np.clip(field["gpp"], 0.0, None)
    lai = np.clip(field["leaf_area_index"], 0.0, None)

    rain_normal, rain_spread, confidence = lagged_calendar_normal(rain)
    dryness_normal, dryness_spread, _ = lagged_calendar_normal(dryness)
    temperature_normal, temperature_spread, _ = lagged_calendar_normal(temperature)
    gpp_normal, gpp_spread, _ = lagged_calendar_normal(gpp)
    lai_normal, lai_spread, _ = lagged_calendar_normal(lai)

    rain_deficit = np.clip(
        (rain_normal - rain) / (rain_normal + rain + 10.0), -1.0, 1.0
    )
    dryness_surplus = np.clip(
        (dryness - dryness_normal)
        / (dryness + dryness_normal + 100.0),
        -1.0,
        1.0,
    )
    temperature_surplus = np.clip(
        (temperature - temperature_normal)
        / (temperature_spread + 5.0),
        -1.0,
        1.0,
    )
    gpp_curing = np.clip(
        (gpp_normal - gpp) / (gpp_normal + gpp + 0.2), -1.0, 1.0
    )
    lai_curing = np.clip(
        (lai_normal - lai) / (lai_normal + lai + 0.5), -1.0, 1.0
    )

    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    expected_combustion = (
        dryness_normal
        / (dryness_normal + 250.0)
        / (1.0 + rain_normal / 35.0)
    )
    expected_fuel = gpp_normal / (gpp_normal + 0.35)
    expected_thermal = sigmoid((temperature_normal - 8.0) / 4.0)
    expected_opportunity = (
        expected_combustion * expected_fuel * expected_thermal
    )
    expected_reference = ema(expected_opportunity, 12.0)
    expected_phase = np.log(
        (expected_opportunity + 0.02) / (expected_reference + 0.02)
    )

    trailing_fire = np.empty_like(baseline, dtype=np.float32)
    accumulator = np.zeros(baseline.shape[1], dtype=np.float64)
    for time_index in range(baseline.shape[0]):
        accumulator += baseline[time_index]
        if time_index >= 12:
            accumulator -= baseline[time_index - 12]
        trailing_fire[time_index] = accumulator * 12.0 / min(time_index + 1, 12)
    recurrent = trailing_fire / (trailing_fire + 0.04)

    natural = np.clip(field["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(field["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(field["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(field["secondary_canopy_height"], 0.0, None)
    pasture = np.clip(field["luh2_pasture_fraction"], 0.0, 1.0)
    rangeland = np.clip(field["luh2_rangeland_fraction"], 0.0, 1.0)
    crop = np.clip(field["luh2_cropland_fraction"], 0.0, 1.0)
    urban = np.clip(field["luh2_urban_fraction"], 0.0, 1.0)
    open_cover = np.clip(
        natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0)
        + pasture
        + rangeland,
        0.0,
        2.0,
    )
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface = np.clip(open_cover * expected_fuel * continuity, 0.0, 1.0)

    weather_surprise = confidence * surface * (
        0.4 * rain_deficit
        + 0.4 * dryness_surplus
        + 0.2 * temperature_surplus
    )
    curing_surprise = confidence * surface * combustion * np.maximum(
        0.7 * gpp_curing + 0.3 * lai_curing, 0.0
    )
    stable_phase = confidence * recurrent * surface * expected_phase
    coherent_surprise = confidence * recurrent * surface * (
        0.45 * rain_deficit
        + 0.35 * dryness_surplus
        + 0.20 * temperature_surplus
        + 0.30 * np.maximum(gpp_curing, 0.0)
    )
    variability = {
        "rain_cv_prior": rain_spread / (rain_normal + 1.0),
        "dryness_cv_prior": dryness_spread / (dryness_normal + 1.0),
        "temperature_spread_prior": temperature_spread,
        "gpp_cv_prior": gpp_spread / (gpp_normal + 0.05),
        "lai_cv_prior": lai_spread / (lai_normal + 0.1),
    }
    vegetation_variability = np.sqrt(
        np.maximum(
            variability["gpp_cv_prior"] * variability["lai_cv_prior"], 0.0
        )
    )
    vegetation_reliability = 1.0 / (1.0 + 5.0 * vegetation_variability)
    stable_fuel_capacity = (
        confidence * surface * recurrent * vegetation_reliability
    )
    stable_fuel_contrast = (
        confidence * surface * recurrent * (vegetation_reliability - 0.65)
    )
    stable_fuel_gap = (
        confidence * surface * (1.0 - recurrent) * vegetation_reliability
    )
    anomalous_heat_capacity = (
        confidence
        * surface
        * recurrent
        * np.maximum(temperature_surplus, 0.0)
    )
    anomalous_dry_capacity = (
        confidence
        * surface
        * recurrent
        * np.maximum(dryness_surplus, 0.0)
    )
    return {
        "confidence": confidence,
        "rain_deficit": rain_deficit,
        "dryness_surplus": dryness_surplus,
        "temperature_surplus": temperature_surplus,
        "gpp_curing": gpp_curing,
        "lai_curing": lai_curing,
        "expected_opportunity": expected_opportunity,
        "expected_phase": expected_phase,
        "surface": surface,
        "recurrent": recurrent,
        "weather_surprise": weather_surprise,
        "curing_surprise": curing_surprise,
        "stable_phase": stable_phase,
        "coherent_surprise": coherent_surprise,
        "stable_fuel_capacity": stable_fuel_capacity,
        "stable_fuel_contrast": stable_fuel_contrast,
        "stable_fuel_gap": stable_fuel_gap,
        "anomalous_heat_capacity": anomalous_heat_capacity,
        "anomalous_dry_capacity": anomalous_dry_capacity,
        **variability,
    }


def weighted_mae(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * np.abs(actual - predicted)) / np.sum(weight))


def ml_diagnostic(
    state: dict[str, np.ndarray],
    baseline: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    cell_folds: np.ndarray,
) -> None:
    """Compare shallow held-cell residual learners with and without new state."""
    count = baseline.shape[1]
    base_cycle = baseline.reshape(16, 12, count).mean(axis=0)
    obs_cycle = observation.reshape(16, 12, count).mean(axis=0)
    target = np.log((obs_cycle + EPS) / (base_cycle + EPS)).reshape(-1)
    weight = (
        area[None, :] * (obs_cycle + 0.02 * obs_cycle.mean())
    ).reshape(-1)
    folds = np.broadcast_to(cell_folds[None, :], (12, count)).reshape(-1)

    base_names = ("log_burn", "share", "surface", "recurrent")
    annual = base_cycle.sum(axis=0)
    base_features = (
        np.log(base_cycle + EPS),
        base_cycle / (annual[None, :] + EPS),
        state["surface"].reshape(16, 12, count).mean(axis=0),
        state["recurrent"].reshape(16, 12, count).mean(axis=0),
    )
    new_names = (
        "rain_deficit_positive",
        "rain_deficit_negative",
        "dryness_surplus_positive",
        "temperature_surplus_positive",
        "gpp_curing_positive",
        "lai_curing_positive",
        "expected_phase",
        "weather_surprise",
        "curing_surprise",
        "stable_phase",
        "coherent_surprise",
        "rain_cv_prior",
        "dryness_cv_prior",
        "temperature_spread_prior",
        "gpp_cv_prior",
        "lai_cv_prior",
    )
    new_fields = (
        np.maximum(state["rain_deficit"], 0.0),
        np.minimum(state["rain_deficit"], 0.0),
        np.maximum(state["dryness_surplus"], 0.0),
        np.maximum(state["temperature_surplus"], 0.0),
        np.maximum(state["gpp_curing"], 0.0),
        np.maximum(state["lai_curing"], 0.0),
        state["expected_phase"],
        state["weather_surprise"],
        state["curing_surprise"],
        state["stable_phase"],
        state["coherent_surprise"],
        state["rain_cv_prior"],
        state["dryness_cv_prior"],
        state["temperature_spread_prior"],
        state["gpp_cv_prior"],
        state["lai_cv_prior"],
    )
    new_cycles = tuple(
        values.reshape(16, 12, count).mean(axis=0) for values in new_fields
    )
    for name in (
        "lai_cv_prior",
        "gpp_cv_prior",
        "expected_phase",
        "temperature_surplus_positive",
        "dryness_surplus_positive",
    ):
        values = new_cycles[new_names.index(name)].reshape(-1)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 6)))
        if edges.size < 3:
            continue
        bins = np.clip(np.digitize(values, edges[1:-1]), 0, edges.size - 2)
        for bin_index in range(edges.size - 1):
            chosen = bins == bin_index
            if not chosen.any():
                continue
            fold_means = []
            for fold in range(4):
                held = chosen & (folds == fold)
                fold_means.append(
                    float(np.sum(weight[held] * target[held]) / np.sum(weight[held]))
                    if held.any()
                    else 0.0
                )
            signed_mean = float(np.sum(weight[chosen] * target[chosen]) / np.sum(weight[chosen]))
            print(
                f"ML_BIN name={name} bin={bin_index} "
                f"low={edges[bin_index]:.9g} high={edges[bin_index+1]:.9g} "
                f"residual={signed_mean:+.9f} "
                + " ".join(
                    f"fold{fold}={value:+.9f}" for fold, value in enumerate(fold_means)
                ),
                flush=True,
            )
    x_base = np.column_stack([values.reshape(-1) for values in base_features])
    x_augmented = np.column_stack(
        [*(values.reshape(-1) for values in base_features),
         *(values.reshape(-1) for values in new_cycles)]
    )
    split_counts: Counter[str] = Counter()
    permutation_costs: dict[str, list[float]] = {
        name: [] for name in new_names
    }
    base_losses: list[float] = []
    augmented_losses: list[float] = []
    for fold in range(4):
        train = folds != fold
        test = folds == fold
        kwargs = dict(
            learning_rate=0.06,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=60,
            l2_regularization=2.0,
            random_state=2200 + fold,
        )
        base_model = HistGradientBoostingRegressor(**kwargs).fit(
            x_base[train], target[train], sample_weight=weight[train]
        )
        augmented_model = HistGradientBoostingRegressor(**kwargs).fit(
            x_augmented[train], target[train], sample_weight=weight[train]
        )
        base_loss = weighted_mae(
            target[test], base_model.predict(x_base[test]), weight[test]
        )
        augmented_loss = weighted_mae(
            target[test], augmented_model.predict(x_augmented[test]), weight[test]
        )
        rng = np.random.default_rng(4400 + fold)
        for feature_offset, name in enumerate(new_names, start=len(base_names)):
            shuffled = x_augmented[test].copy()
            shuffled[:, feature_offset] = shuffled[
                rng.permutation(shuffled.shape[0]), feature_offset
            ]
            shuffled_loss = weighted_mae(
                target[test], augmented_model.predict(shuffled), weight[test]
            )
            permutation_costs[name].append(shuffled_loss - augmented_loss)
        base_losses.append(base_loss)
        augmented_losses.append(augmented_loss)
        tree = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=120,
            random_state=3300 + fold,
        ).fit(x_augmented[train], target[train], sample_weight=weight[train])
        all_names = (*base_names, *new_names)
        for index in tree.tree_.feature:
            if index >= 0:
                split_counts[all_names[index]] += 1
        print(
            f"ML_FOLD fold={fold} base_mae={base_loss:.9f} "
            f"augmented_mae={augmented_loss:.9f} "
            f"delta={augmented_loss-base_loss:+.9f}",
            flush=True,
        )
    print(
        f"ML_TOTAL base_mae={np.mean(base_losses):.9f} "
        f"augmented_mae={np.mean(augmented_losses):.9f} "
        f"folds_improved={sum(a < b for a,b in zip(augmented_losses,base_losses))}/4 "
        f"shallow_splits={dict(split_counts)}",
        flush=True,
    )
    for name, values in sorted(
        permutation_costs.items(), key=lambda item: -float(np.mean(item[1]))
    ):
        print(
            f"ML_IMPORTANCE name={name} mean_cost={np.mean(values):+.9f} "
            f"positive_folds={sum(value > 0.0 for value in values)}/4 "
            + " ".join(
                f"fold{fold}={value:+.9f}" for fold, value in enumerate(values)
            ),
            flush=True,
        )


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"moving canonical model {current_blob}")

    observation_grid = load_observation()
    area_grid = one_degree_area()
    rows, columns, cell_weight, retained = select_high_weight(
        observation_grid, area_grid
    )
    land = load_land_mask()
    keep = land[rows, columns]
    rows, columns, cell_weight = rows[keep], columns[keep], cell_weight[keep]
    print(
        f"DESIGN cells={rows.size} retained_fire_weight={retained:.9f} "
        "held_coordinates_only=1 same_month_state_uses_prior_years_only=1",
        flush=True,
    )
    data = {
        name: selected_input(name, rows, columns) for name in model.INPUTS
    }
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    cell_folds = ((rows // 12) + 3 * (columns // 12)) % 4
    del observation_grid, area_grid, land
    gc.collect()

    state = states(data, baseline)
    ml_diagnostic(state, baseline, observation, area, cell_folds)
    variants: dict[str, np.ndarray] = {"incumbent": baseline}
    signals = {
        "weather_surprise": state["weather_surprise"],
        "curing_surprise": state["curing_surprise"],
        "stable_phase": state["stable_phase"],
        "coherent_surprise": state["coherent_surprise"],
    }
    for name, signal in signals.items():
        for strength in (0.25, 0.5, 1.0, 2.0, 4.0):
            variants[f"{name}_{strength:g}"] = relative_allocator(
                baseline, signal, strength
            )
    capacity_signals = {
        "stable_fuel_capacity": state["stable_fuel_capacity"],
        "stable_fuel_contrast": state["stable_fuel_contrast"],
        "stable_fuel_gap": state["stable_fuel_gap"],
        "anomalous_heat_capacity": state["anomalous_heat_capacity"],
        "anomalous_dry_capacity": state["anomalous_dry_capacity"],
    }
    for name, signal in capacity_signals.items():
        for strength in (0.1, 0.25, 0.5, 1.0, 2.0):
            variants[f"{name}_{strength:g}"] = hazard_capacity(
                baseline, signal, strength
            )

    annual_results = {
        name: annual_loss(value, observation, area, cell_folds)
        for name, value in variants.items()
    }
    cycle_results = {
        name: cycle_loss(value, observation, area, cell_folds)
        for name, value in variants.items()
    }
    reference_annual = annual_results["incumbent"]
    reference_cycle = cycle_results["incumbent"]
    for name in variants:
        annual = annual_results[name]
        cycle = cycle_results[name]
        print(
            f"VARIANT name={name} annual={annual[0]:.9f} "
            f"annual_delta={annual[0]-reference_annual[0]:+.9f} "
            f"annual_folds={sum(a < b for a,b in zip(annual[1],reference_annual[1]))}/4 "
            + " ".join(
                f"annual_fold{fold}_delta={annual[1][fold]-reference_annual[1][fold]:+.9f}"
                for fold in range(4)
            )
            + " "
            f"cycle={cycle[0]:.9f} "
            f"cycle_delta={cycle[0]-reference_cycle[0]:+.9f} "
            f"cycle_folds={sum(a < b for a,b in zip(cycle[1],reference_cycle[1]))}/4 "
            + " ".join(
                f"cycle_fold{fold}_delta={cycle[1][fold]-reference_cycle[1][fold]:+.9f}"
                for fold in range(4)
            ),
            flush=True,
        )

    means = {
        name: np.asarray(values[:, 0, :], dtype=np.float32).mean(axis=0)
        for name, values in data.items()
    }
    ecology = ecological_masks(means)
    ranked = sorted(
        (name for name in variants if name != "incumbent"),
        key=lambda name: (
            annual_results[name][0] - reference_annual[0]
            + 12.0 * (cycle_results[name][0] - reference_cycle[0])
        ),
    )[:4]
    for name in ("incumbent", *ranked):
        for regime, mask in ecology.items():
            if mask.any():
                print(
                    f"ECOLOGY name={name} regime={regime} cells={int(mask.sum())} "
                    f"ratio={area_ratio(variants[name],observation,area,mask):.9f}",
                    flush=True,
                )

    perturbed = {
        name: values.copy() for name, values in data.items()
    }
    for values in perturbed.values():
        values[96:] *= 0.5
    perturbed_baseline = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed_state = states(perturbed, perturbed_baseline)
    for name in (*signals, *capacity_signals):
        prefix_delta = float(
            np.max(np.abs(state[name][:96] - perturbed_state[name][:96]))
        )
        print(f"PREFIX signal={name} max_abs={prefix_delta:.12g}", flush=True)
    print(
        f"DONE wall_seconds={time.perf_counter()-started:.3f} "
        f"best={ranked[0] if ranked else 'none'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
