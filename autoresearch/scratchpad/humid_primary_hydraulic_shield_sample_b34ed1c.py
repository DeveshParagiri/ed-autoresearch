"""Held-block screen of hydraulic shielding in humid primary mosaics.

The committed wet-forest brake is a current-state intersection of warm air,
causal annual rainfall, tall canopy, high LAI, and natural cover.  The annual
closure instead suppresses persistent high-fire *open* warm systems with low
lightning variability.  Prior humid-short dry-window and organic-horizon
experiments found real but tiny or null effects.  This diagnostic therefore
tests only states absent from those mechanisms: finite root-zone water
storage, continuous optical woody shielding, and their intersection.

The prediction equations use only coupled-valid current or prefix-causal local
state.  Coordinates define held spatial folds and observations define the
post-prediction warm-humid-primary audit and residual target; neither enters a
candidate state.  The script runs the current model only on selected cells,
does not invoke the official evaluator command, and edits no tracked file.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    MONTH_DAYS,
    antecedent,
    format_metrics,
    load_observed,
    load_selected,
    metrics,
    input_index,
    weighted_corr,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


MONTH_WEIGHTS = MONTH_DAYS / MONTH_DAYS.sum()


def field(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(data[name][:, 0, :], dtype=np.float64)


def rising(values: np.ndarray, center: float, width: float) -> np.ndarray:
    z = np.clip((values - center) / width, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def falling(values: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 - rising(values, center, width)


def trailing_annual(rain: np.ndarray) -> np.ndarray:
    output = np.empty_like(rain)
    accumulator = np.zeros_like(rain[0])
    for time in range(rain.shape[0]):
        accumulator += rain[time]
        if time >= 12:
            accumulator -= rain[time - 12]
        output[time] = accumulator * 12.0 / min(time + 1, 12)
    return output


def warm_humid_primary_mask(
    data: Mapping[str, np.ndarray],
    rain: np.ndarray,
    temperature12: np.ndarray,
    combustion: np.ndarray,
) -> np.ndarray:
    """Reproduce Entry 142's ordered third complement regime."""
    annual_rain = trailing_annual(rain).mean(axis=0)
    temperature = temperature12.mean(axis=0)
    dry_fraction = combustion.mean(axis=0)

    def mean(name: str) -> np.ndarray:
        return field(data, name).mean(axis=0)

    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    pasture = mean("luh2_pasture_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    urban = mean("luh2_urban_fraction")
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    open_cover = np.clip(pasture + rangeland + residual, 0.0, 1.0)
    canopy = mean("natural_canopy_height")
    lai = mean("leaf_area_index")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    established = (
        (
            (temperature >= 20.0)
            & (annual_rain >= 1200.0)
            & (canopy >= 20.0)
            & (lai >= 3.0)
            & (natural >= 0.7)
            & (primary >= 0.5)
        )
        | (
            (temperature >= 5.0)
            & (temperature < 20.0)
            & (canopy >= 15.0)
            & (lai >= 2.5)
            & (natural >= 0.6)
        )
        | ((temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6))
        | (
            (temperature >= 20.0)
            & (annual_rain >= 500.0)
            & (annual_rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        )
        | (
            (rangeland >= 0.4)
            & (annual_rain >= 250.0)
            & (annual_rain < 1500.0)
            & (biomass >= 0.2)
        )
        | (crop >= 0.5)
        | ((annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0))
    )
    complement = ~established
    seasonal_open = (
        complement
        & (temperature >= 18.0)
        & (annual_rain >= 400.0)
        & (annual_rain < 1600.0)
        & (dry_fraction >= 0.18)
        & (open_cover >= 0.20)
    )
    remaining = complement & ~seasonal_open
    seasonal_primary = (
        remaining
        & (temperature >= 18.0)
        & (annual_rain >= 600.0)
        & (annual_rain < 1800.0)
        & (dry_fraction >= 0.18)
        & (primary >= 0.25)
    )
    remaining &= ~seasonal_primary
    return remaining & (temperature >= 18.0) & (annual_rain >= 1200.0) & (primary >= 0.25)


def regime_from_means(states: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the ordered Entry 142 regime from full-grid mean states."""
    annual_rain = states["annual_rain"]
    temperature = states["temperature"]
    dry_fraction = states["dry_fraction"]
    primary = states["luh2_primary_fraction"]
    crop = states["luh2_cropland_fraction"]
    pasture = states["luh2_pasture_fraction"]
    rangeland = states["luh2_rangeland_fraction"]
    urban = states["luh2_urban_fraction"]
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    open_cover = np.clip(pasture + rangeland + residual, 0.0, 1.0)
    canopy = states["natural_canopy_height"]
    lai = states["leaf_area_index"]
    biomass = states["aboveground_biomass"]
    natural = states["natural_vegetation_fraction"]
    established = (
        (
            (temperature >= 20.0) & (annual_rain >= 1200.0)
            & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7)
            & (primary >= 0.5)
        )
        | (
            (temperature >= 5.0) & (temperature < 20.0)
            & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6)
        )
        | ((temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6))
        | (
            (temperature >= 20.0) & (annual_rain >= 500.0)
            & (annual_rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0)
            & (natural >= 0.5)
        )
        | (
            (rangeland >= 0.4) & (annual_rain >= 250.0)
            & (annual_rain < 1500.0) & (biomass >= 0.2)
        )
        | (crop >= 0.5)
        | ((annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0))
    )
    complement = ~established
    seasonal_open = (
        complement & (temperature >= 18.0) & (annual_rain >= 400.0)
        & (annual_rain < 1600.0) & (dry_fraction >= 0.18)
        & (open_cover >= 0.20)
    )
    remaining = complement & ~seasonal_open
    seasonal_primary = (
        remaining & (temperature >= 18.0) & (annual_rain >= 600.0)
        & (annual_rain < 1800.0) & (dry_fraction >= 0.18)
        & (primary >= 0.25)
    )
    remaining &= ~seasonal_primary
    return remaining & (temperature >= 18.0) & (annual_rain >= 1200.0) & (primary >= 0.25)


def full_regime_mask() -> np.ndarray:
    """Stream full-grid means without retaining any 192-month field."""
    index = input_index()

    def simple_mean(name: str, transform=None) -> np.ndarray:
        total = np.zeros((180, 360), dtype=np.float64)
        with Dataset(index[name]) as dataset:
            variable = dataset.variables[name]
            for time in range(192):
                values = np.asarray(variable[time], dtype=np.float64)
                if transform is not None:
                    values = transform(values)
                total += values / 192.0
        return total

    rain_history: list[np.ndarray] = []
    rain_sum = np.zeros((180, 360), dtype=np.float64)
    annual_rain_mean = np.zeros_like(rain_sum)
    with Dataset(index["monthly_precipitation"]) as dataset:
        variable = dataset.variables["monthly_precipitation"]
        for time in range(192):
            current = np.clip(np.asarray(variable[time], dtype=np.float64), 0.0, None)
            rain_history.append(current)
            rain_sum += current
            if len(rain_history) > 12:
                rain_sum -= rain_history.pop(0)
            annual_rain_mean += rain_sum * (12.0 / len(rain_history)) / 192.0

    alpha12 = 1.0 - np.exp(-1.0 / 12.0)
    temperature_mean = np.zeros((180, 360), dtype=np.float64)
    with Dataset(index["air_temperature"]) as dataset:
        variable = dataset.variables["air_temperature"]
        state = np.asarray(variable[0], dtype=np.float64)
        for time in range(192):
            state += alpha12 * (np.asarray(variable[time], dtype=np.float64) - state)
            temperature_mean += state / 192.0

    states = {
        "annual_rain": annual_rain_mean,
        "temperature": temperature_mean,
        "dry_fraction": simple_mean(
            "dryness", lambda values: np.clip(values, 0.0, None) / (np.clip(values, 0.0, None) + 500.0)
        ),
    }
    for name in (
        "luh2_primary_fraction",
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
        "natural_canopy_height",
        "leaf_area_index",
        "aboveground_biomass",
        "natural_vegetation_fraction",
    ):
        states[name] = simple_mean(name)
    return regime_from_means(states)


def causal_states(
    data: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return existing controls and physically distinct shielding states."""
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    lightning = np.clip(field(data, "lightning_flash_rate"), 0.0, None)
    canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    lai = np.clip(field(data, "leaf_area_index"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)
    natural = np.clip(field(data, "natural_vegetation_fraction"), 0.0, 1.0)
    primary = np.clip(field(data, "luh2_primary_fraction"), 0.0, 1.0)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)

    rain12 = antecedent(rain, 12.0)
    temperature24 = antecedent(temperature, 24.0)
    annual_rain = 12.0 * rain12
    combustion = dryness / (dryness + 500.0)

    # Exact current wet-forest state, reconstructed as a diagnostic control.
    wet_forest = (
        rising(temperature, 20.0, 2.0)
        * rising(annual_rain, 1200.0, 250.0)
        * rising(canopy, 15.0, 3.0)
        * rising(lai, 2.5, 0.5)
        * natural
    )

    lightning_variability = np.empty_like(lightning)
    for time in range(lightning.shape[0]):
        lightning_variability[time] = lightning[max(0, time - 11) : time + 1].std(axis=0)
    annual_fire = np.empty_like(prediction)
    annual_fire[0] = prediction[0]
    for time in range(1, prediction.shape[0]):
        annual_fire[time] = prediction[max(0, time - 12) : time].mean(axis=0)
    open_cover = np.clip(rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0)
    warm_closure = (
        falling(lightning_variability, 0.025, 0.005)
        * rising(annual_fire, 0.007, 0.003)
        * rising(temperature24, 18.0, 4.0)
        * rising(annual_rain, 900.0, 220.0)
        * open_cover
        * rising(natural, 0.2, 0.1)
        * rising(biomass, 0.075, 0.05)
    )

    # Root-zone storage is finite and initialized from month zero.  Rain fills
    # a 300 mm store while heat and atmospheric dryness set monthly demand.
    thermal = rising(temperature, 10.0, 4.0)
    demand = 50.0 * (0.25 + 0.75 * thermal) * (0.20 + 0.80 * combustion)
    store = np.minimum(300.0, rain[0]).copy()
    root_wetness = np.empty_like(rain)
    for time in range(rain.shape[0]):
        if time > 0:
            store = np.clip(store + rain[time] - demand[time], 0.0, 300.0)
        root_wetness[time] = store / 300.0

    # Beer-Lambert leaf shielding and finite canopy stature are continuous,
    # rather than threshold versions of the incumbent canopy/LAI gate.
    optical_depth = 1.0 - np.exp(-0.50 * lai)
    stature = canopy / (canopy + 10.0)
    woody_mass = 0.25 + 0.75 * biomass / (biomass + 2.0)
    primary_matrix = primary * (1.0 - crop)
    optical_primary = np.clip(
        primary_matrix * optical_depth * stature * woody_mass,
        0.0,
        1.0,
    )
    root_primary = np.clip(primary_matrix * root_wetness, 0.0, 1.0)
    hydraulic_shield = np.clip(optical_primary * root_wetness, 0.0, 1.0)

    controls = {
        "wet_forest": wet_forest,
        "warm_closure": warm_closure,
    }
    candidates = {
        "root_primary": root_primary,
        "optical_primary": optical_primary,
        "hydraulic_shield": hydraulic_shield,
    }
    return controls, candidates


def weighted_fit(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    sqrt_weight = np.sqrt(np.clip(weight, 0.0, None))
    xw = design * sqrt_weight[:, None]
    yw = target * sqrt_weight
    ridge = 1e-8 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)


def weighted_rmse(target: np.ndarray, fitted: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(target - fitted), weights=weight)))


def held_screen(
    label: str,
    candidate: np.ndarray,
    controls: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    base_oof = np.empty_like(target)
    extended_oof = np.empty_like(target)
    coefficients: list[float] = []
    deltas: list[float] = []
    for fold in range(4):
        train = folds != fold
        held = ~train
        base_beta = weighted_fit(controls[train], target[train], weight[train])
        extended = np.column_stack((controls, candidate))
        extended_beta = weighted_fit(extended[train], target[train], weight[train])
        base_oof[held] = controls[held] @ base_beta
        extended_oof[held] = extended[held] @ extended_beta
        base_rmse = weighted_rmse(target[held], base_oof[held], weight[held])
        extended_rmse = weighted_rmse(target[held], extended_oof[held], weight[held])
        coefficients.append(float(extended_beta[-1]))
        deltas.append(extended_rmse - base_rmse)
    total_delta = weighted_rmse(target, extended_oof, weight) - weighted_rmse(
        target, base_oof, weight
    )
    print(
        f"HELD {label} rmse_delta={total_delta:+.8f} "
        f"betas={','.join(f'{value:+.5f}' for value in coefficients)} "
        f"fold_deltas={','.join(f'{value:+.8f}' for value in deltas)}",
        flush=True,
    )
    return total_delta, tuple(coefficients), tuple(deltas)


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    full_mask = full_regime_mask()
    coarse_mean = (
        np.asarray(evaluator.reference_mean, dtype=np.float64)
        .reshape(180, 2, 360, 2)
        .mean(axis=(1, 3))
        / 100.0
    )
    coarse_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    global_weight = coarse_area * coarse_mean
    regime_cells = np.flatnonzero(full_mask.ravel())
    global_control = np.argsort(global_weight.ravel())[::-1][:1536]
    cells = np.union1d(regime_cells, global_control)
    rows, cols = cells // 360, cells % 360
    area = coarse_area[rows, cols]
    reference_weight = global_weight[rows, cols]
    retained = float(reference_weight.sum() / global_weight.sum())
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()
    prediction = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float64
    )[:, 0, :]
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature12 = antecedent(field(data, "air_temperature"), 12.0)
    combustion = np.clip(field(data, "dryness"), 0.0, None)
    combustion = combustion / (combustion + 500.0)
    mask = full_mask[rows, cols]
    regime_global_share = retained * reference_weight[mask].sum() / reference_weight.sum()
    print(
        f"REGIME cells={int(mask.sum())} sampled_global_observed_weight={regime_global_share:.8f} "
        f"estimated_regime_coverage={regime_global_share / 0.09825934:.8f}",
        flush=True,
    )
    if mask.sum() < 40 or np.unique(folds[mask]).size != 4:
        raise RuntimeError("selected humid-primary sample is too small for four held blocks")

    controls, candidates = causal_states(data, prediction)
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    annual_target = np.clip(
        np.log((obs_annual + 1e-5) / (pred_annual + 1e-5)), -3.0, 3.0
    )
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_allocation = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-6)
    obs_allocation = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-6)
    cycle_target = np.clip(
        np.log((obs_allocation + 1e-4) / (pred_allocation + 1e-4)), -3.0, 3.0
    )

    annual_controls = np.column_stack(
        (
            np.ones(mask.sum()),
            np.average(controls["wet_forest"][:, mask], axis=0, weights=MONTH_DAYS),
            np.average(controls["warm_closure"][:, mask], axis=0, weights=MONTH_DAYS),
        )
    )
    annual_weight = reference_weight[mask]
    annual_folds = folds[mask]
    annual_y = annual_target[mask]
    cycle_controls = np.column_stack(
        (
            np.ones(12 * mask.sum()),
            (
                controls["wet_forest"][:, mask].reshape(16, 12, -1).mean(axis=0)
                - controls["wet_forest"][:, mask].mean(axis=0)[None, :]
            ).reshape(-1),
            (
                controls["warm_closure"][:, mask].reshape(16, 12, -1).mean(axis=0)
                - controls["warm_closure"][:, mask].mean(axis=0)[None, :]
            ).reshape(-1),
        )
    )
    cycle_y = cycle_target[:, mask].reshape(-1)
    cycle_weight = (
        reference_weight[mask][None, :]
        * (obs_allocation[:, mask] + 0.01 / 12.0)
    ).reshape(-1)
    cycle_folds = np.broadcast_to(annual_folds[None, :], (12, mask.sum())).reshape(-1)

    annual_results = {}
    for name, state in candidates.items():
        annual_state = np.average(state[:, mask], axis=0, weights=MONTH_DAYS)
        raw = weighted_corr(annual_state, annual_y, annual_weight)
        print(f"RAW_ANNUAL {name} corr={raw:+.8f}", flush=True)
        annual_results[name] = held_screen(
            f"annual:{name}",
            annual_state,
            annual_controls,
            annual_y,
            annual_weight,
            annual_folds,
        )
        state_cycle = state[:, mask].reshape(16, 12, -1).mean(axis=0)
        state_anomaly = state_cycle - state[:, mask].mean(axis=0)[None, :]
        print(
            f"RAW_CYCLE {name} corr={weighted_corr(state_anomaly, cycle_target[:, mask], cycle_weight.reshape(12, -1)):+.8f}",
            flush=True,
        )
        held_screen(
            f"cycle:{name}",
            state_anomaly.reshape(-1),
            cycle_controls,
            cycle_y,
            cycle_weight,
            cycle_folds,
        )

    # Only nominate a mechanism if it has the suppressive sign in every fold
    # and improves held annual RMSE in aggregate and in at least three folds.
    eligible = []
    for name, (delta, betas, fold_deltas) in annual_results.items():
        if all(beta < 0.0 for beta in betas) and delta < 0.0 and sum(value < 0.0 for value in fold_deltas) >= 3:
            eligible.append((delta, name, betas))
    if eligible:
        delta, name, betas = min(eligible)
        print(
            f"DECISION smallest={name} held_delta={delta:+.8f} "
            f"median_suppressive_beta={np.median(betas):+.8f}",
            flush=True,
        )
    else:
        print("DECISION no_stable_increment_beyond_existing_controls", flush=True)

    # The only candidate with the physically required suppressive coefficient
    # in all four held folds is the minimal root-primary state.  Test three
    # deliberately weak, fixed hazard brakes; this is a sampled falsification,
    # not coefficient fitting or an exact/full-grid evaluation.
    baseline_all, baseline_all_folds = metrics(
        prediction, observed, area, reference_weight, folds
    )
    baseline_regime, baseline_regime_folds = metrics(
        prediction[:, mask], observed[:, mask], area[mask],
        reference_weight[mask], folds[mask],
    )
    print("BASE_ALL " + format_metrics(baseline_all), flush=True)
    print("BASE_REGIME " + format_metrics(baseline_regime), flush=True)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    for strength in (0.10, 0.25, 0.50):
        candidate = 1.0 - np.exp(
            -np.clip(hazard * np.exp(-strength * candidates["root_primary"]), 0.0, 50.0)
        )
        current_all, current_all_folds = metrics(
            candidate, observed, area, reference_weight, folds
        )
        current_regime, current_regime_folds = metrics(
            candidate[:, mask], observed[:, mask], area[mask],
            reference_weight[mask], folds[mask],
        )
        names = ("alloc_rmse", "annual_log_rmse", "raw_cycle_rmse", "phase", "area_ratio")
        print(
            f"BRACKET w={strength:.2f} ALL "
            + " ".join(
                f"{name}={current_all[index] - baseline_all[index]:+.8f}"
                for index, name in enumerate(names)
            ),
            flush=True,
        )
        print(
            f"BRACKET w={strength:.2f} REGIME "
            + " ".join(
                f"{name}={current_regime[index] - baseline_regime[index]:+.8f}"
                for index, name in enumerate(names)
            ),
            flush=True,
        )
        for fold in range(4):
            print(
                f"BRACKET_FOLD w={strength:.2f} fold={fold} "
                f"all_annual={current_all_folds[fold][1] - baseline_all_folds[fold][1]:+.8f} "
                f"regime_annual={current_regime_folds[fold][1] - baseline_regime_folds[fold][1]:+.8f}",
                flush=True,
            )

    # Candidate states are exactly prefix-causal: changing all future state
    # arrays cannot alter any earlier state already constructed.
    split = 96
    changed = {name: np.asarray(values).copy() for name, values in data.items()}
    for name in ("monthly_precipitation", "air_temperature", "dryness"):
        changed[name][split:] *= 1.5
    changed_controls, changed_candidates = causal_states(changed, prediction)
    prefix_max = max(
        float(np.max(np.abs(candidates[name][:split] - changed_candidates[name][:split])))
        for name in candidates
    )
    print(f"PREFIX max={prefix_max:.12e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
