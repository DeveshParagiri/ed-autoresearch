"""Low-memory held-block diagnostic for the cold-mixed fire deficit.

The diagnostic uses every one-degree cell in the cold-mixed complement from
Entry 142.  Coordinates define four spatial validation folds but are never
features.  Predictors are current local state, causal memories, and compact
mechanistic interactions derived from inputs already used by the coupled-ready
canonical model.  Annual log-propensity and normalized monthly-cycle residuals
are fitted separately.  The learners are diagnostic only and never enter
``model.py`` or the official ledger.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_half_greenup_exact_80368d8 import (  # noqa: E402
    ecology_masks,
)
from autoresearch.scratchpad.phenology_stage_split_80368d8 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    causal_mean_states,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    MASK_INPUTS,
    one_degree_area,
    selected_input,
)
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
)


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    output = np.empty_like(values, dtype=np.float32)
    state = np.asarray(values[0], dtype=np.float32).copy()
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def snow_state(
    rain: np.ndarray, temperature: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal snow cover and melt release as diagnostic features only."""
    cover = np.empty_like(rain, dtype=np.float32)
    melt_release = np.empty_like(rain, dtype=np.float32)
    store = np.asarray(
        rain[0] * sigmoid((1.0 - temperature[0]) / 2.0), dtype=np.float32
    )
    for time in range(rain.shape[0]):
        previous_cover = store / (store + 18.0)
        snowfall = rain[time] * sigmoid((1.0 - temperature[time]) / 2.0)
        melt = sigmoid((temperature[time] - 1.0) / 2.0)
        melt_release[time] = previous_cover * melt
        store = np.clip((store + snowfall) * (1.0 - melt), 0.0, 500.0)
        cover[time] = store / (store + 18.0)
    return cover, melt_release


def monthly_features(data: dict[str, np.ndarray], prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Construct a compact set of causal local-state and physical interaction features."""
    def field(name: str) -> np.ndarray:
        return np.asarray(data[name][:, 0, :], dtype=np.float32)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    temperature = field("air_temperature")
    dryness = np.clip(field("dryness"), 0.0, None)
    gpp = np.clip(field("gpp"), 0.0, None)
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    lai = np.clip(field("leaf_area_index"), 0.0, None)
    biomass = np.clip(field("aboveground_biomass"), 0.0, None)
    canopy = np.clip(field("natural_canopy_height"), 0.0, None)
    natural = np.clip(field("natural_vegetation_fraction"), 0.0, 1.0)
    secondary = np.clip(field("secondary_vegetation_fraction"), 0.0, 1.0)
    soil_carbon = np.clip(field("soil_carbon"), 0.0, None)
    primary = np.clip(field("luh2_primary_fraction"), 0.0, 1.0)
    crop = np.clip(field("luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field("luh2_urban_fraction"), 0.0, 1.0)

    rain3, rain12 = ema(rain, 3.0), ema(rain, 12.0)
    temperature3, temperature24 = ema(temperature, 3.0), ema(temperature, 24.0)
    dryness3, dryness12 = ema(dryness, 3.0), ema(dryness, 12.0)
    gpp3, gpp12 = ema(gpp, 3.0), ema(gpp, 12.0)
    lightning3, lightning12 = ema(lightning, 3.0), ema(lightning, 12.0)
    incumbent = np.asarray(prediction[:, 0, :], dtype=np.float32)
    incumbent12 = ema(incumbent, 12.0)

    annual_rain = 12.0 * rain12
    warming3 = temperature - temperature3
    rain_deficit = np.maximum((rain12 - rain) / (rain12 + rain + 10.0), 0.0)
    drying3 = (dryness - dryness3) / (dryness + dryness3 + 100.0)
    gpp_decline = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 1e-3), 0.0)
    lightning_arrival = np.maximum(
        (lightning - lightning3) / (lightning + lightning3 + 0.002), 0.0
    )
    combustion = np.sqrt(
        dryness / (dryness + 250.0) * 1.0 / (1.0 + rain / 35.0)
    )
    cold_background = sigmoid((5.0 - temperature24) / 3.0)
    managed = np.clip(crop + pasture + rangeland + urban, 0.0, 1.0)
    primary_share = primary / (primary + managed + 0.1)
    open_carrier = np.clip(
        pasture + rangeland + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    fine_fuel = gpp12 / (gpp12 + 0.35)
    ignition = lightning12 / (lightning12 + 0.02)
    opportunity_gap = 1.0 / (1.0 + incumbent12 / 0.01)
    snow_cover, melt_release = snow_state(rain, temperature)

    # These named interactions distinguish already-tested hypotheses.  The
    # first mirrors cold-thaw opportunity already present in the model; the
    # second tests post-melt litter exposure without adding snow-derived fuel;
    # the third represents persistent open fine fuel during a brief warm/dry
    # combustion window, rather than a generic cold ignition floor or crown fire.
    thaw_opportunity = (
        cold_background
        * sigmoid((warming3 - 0.5) / 1.5)
        * combustion
        * ignition
        * open_carrier
        * fine_fuel
        * opportunity_gap
    )
    post_melt_exposure = (
        melt_release
        * sigmoid(drying3 / 0.04)
        * combustion
        * ignition
        * open_carrier
        * fine_fuel
        * opportunity_gap
    )
    warm_dry_open_fuel = (
        cold_background
        * sigmoid((temperature - 4.0) / 3.0)
        * combustion
        * open_carrier
        * fine_fuel
        * (0.25 + 0.75 * ignition)
    )

    return {
        "temperature": temperature,
        "temperature24": temperature24,
        "warming3": warming3,
        "rain": rain,
        "rain12": rain12,
        "annual_rain": annual_rain,
        "rain_deficit": rain_deficit,
        "dryness": dryness,
        "dryness12": dryness12,
        "drying3": drying3,
        "gpp": gpp,
        "gpp12": gpp12,
        "gpp_decline": gpp_decline,
        "lightning": lightning,
        "lightning12": lightning12,
        "lightning_arrival": lightning_arrival,
        "lai": lai,
        "biomass": biomass,
        "canopy": canopy,
        "natural": natural,
        "secondary": secondary,
        "soil_carbon": soil_carbon,
        "primary_share": primary_share,
        "managed": managed,
        "open_carrier": open_carrier,
        "fine_fuel": fine_fuel,
        "combustion": combustion,
        "cold_background": cold_background,
        "snow_cover": snow_cover,
        "melt_release": melt_release,
        "incumbent": incumbent,
        "incumbent12": incumbent12,
        "opportunity_gap": opportunity_gap,
        "thaw_opportunity": thaw_opportunity,
        "post_melt_exposure": post_melt_exposure,
        "warm_dry_open_fuel": warm_dry_open_fuel,
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mx, my = weighted_mean(x, weights), weighted_mean(y, weights)
    dx, dy = x - mx, y - my
    denominator = np.sqrt(
        weighted_mean(dx * dx, weights) * weighted_mean(dy * dy, weights)
    )
    return weighted_mean(dx * dy, weights) / max(float(denominator), 1e-12)


def fit_held_blocks(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    names: tuple[str, ...],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.zeros_like(y, dtype=np.float64)
    importances = np.zeros((4, x.shape[1]), dtype=np.float64)
    for fold in range(4):
        train = folds != fold
        held = folds == fold
        learner = GradientBoostingRegressor(
            n_estimators=90,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=30,
            max_features=0.75,
            loss="huber",
            random_state=1949 + fold,
        )
        learner.fit(x[train], y[train], sample_weight=weights[train])
        prediction[held] = learner.predict(x[held])
        importances[fold] = learner.feature_importances_
        baseline = weighted_mean(np.abs(y[held]), weights[held])
        corrected = weighted_mean(np.abs(y[held] - prediction[held]), weights[held])
        print(
            f"FOLD task={label} fold={fold} n_train={int(train.sum())} "
            f"n_held={int(held.sum())} baseline_mae={baseline:.9f} "
            f"corrected_mae={corrected:.9f} delta={corrected - baseline:+.9f}",
            flush=True,
        )
    rank = np.argsort(importances.mean(axis=0))[::-1]
    print(f"IMPORTANCE task={label}", flush=True)
    for index in rank[:15]:
        top_count = int(
            sum(index in np.argsort(row)[-8:] for row in importances)
        )
        print(
            f"{names[index]} mean={importances[:, index].mean():.9f} "
            f"std={importances[:, index].std():.9f} top8_folds={top_count}",
            flush=True,
        )
    return prediction, importances


def binned_shape(
    feature: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    name: str,
    task: str,
) -> None:
    edges = np.quantile(feature, np.linspace(0.0, 1.0, 7))
    edges = np.unique(edges)
    if edges.size < 3:
        return
    print(f"SHAPE task={task} feature={name}", flush=True)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (feature >= lower) & (feature <= upper if upper == edges[-1] else feature < upper)
        if not np.any(mask):
            continue
        print(
            f"lo={lower:.8g} hi={upper:.8g} n={int(mask.sum())} "
            f"target={weighted_mean(target[mask], weights[mask]):+.9f}",
            flush=True,
        )


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

    mask_data = load_inputs(MASK_INPUTS)
    land = load_land_mask()
    states = causal_mean_states(mask_data, model)
    established = ecology_masks(mask_data, model, land)
    established_union = np.any(np.stack(tuple(established.values()), axis=0), axis=0)
    cold_mixed = land & ~established_union & (states["temperature"] < 5.0)
    rows, columns = np.nonzero(cold_mixed)
    print(
        f"COLD_MIXED cells={rows.size} model_blob={blob} "
        "coordinates_used_only_for_folds=1",
        flush=True,
    )

    sampled: dict[str, np.ndarray] = {}
    for name in model.INPUTS:
        if name in mask_data:
            sampled[name] = np.asarray(
                mask_data[name][:, rows, columns][:, None, :], dtype=np.float32
            )
        else:
            sampled[name] = selected_input(name, rows, columns)
    del mask_data, states, established, established_union, cold_mixed, land
    gc.collect()

    current = np.asarray(model.predict(sampled, dict(model.PARAMS), None), dtype=np.float32)
    features = monthly_features(sampled, current)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )[:, rows, columns]
    del reference, sampled
    gc.collect()

    area = one_degree_area()[rows, columns]
    pred_cycle = current[:, 0, :].reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation.reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    ratio = float(np.sum(pred_annual * area)) / max(float(np.sum(obs_annual * area)), 1e-12)
    obs_monthly = np.sum(obs_cycle * area[None, :], axis=1)
    pred_monthly = np.sum(pred_cycle * area[None, :], axis=1)
    print(
        f"BASE ratio={ratio:.9f} model_peak={MONTHS[int(np.argmax(pred_monthly))]} "
        f"obs_peak={MONTHS[int(np.argmax(obs_monthly))]}",
        flush=True,
    )

    block_lat = rows // 15
    block_lon = columns // 15
    cell_folds = (block_lat + 2 * block_lon) % 4
    positive = obs_annual[obs_annual > 1e-8]
    floor = 0.02 * float(np.median(positive)) if positive.size else 1e-6
    annual_weights = area * (obs_annual + floor)
    annual_target = np.clip(
        np.log((obs_annual + 1e-6) / (pred_annual + 1e-6)), -4.0, 4.0
    )

    feature_names = tuple(features)
    dynamic_std = {
        "temperature",
        "warming3",
        "rain",
        "rain_deficit",
        "dryness",
        "drying3",
        "gpp",
        "gpp_decline",
        "lightning",
        "lightning_arrival",
        "combustion",
        "snow_cover",
        "melt_release",
        "incumbent",
        "thaw_opportunity",
        "post_melt_exposure",
        "warm_dry_open_fuel",
    }
    annual_columns: list[np.ndarray] = []
    annual_names: list[str] = []
    for name, values in features.items():
        annual_columns.append(values.mean(axis=0))
        annual_names.append(f"mean:{name}")
        if name in dynamic_std:
            annual_columns.append(values.std(axis=0))
            annual_names.append(f"std:{name}")
    annual_x = np.column_stack(annual_columns).astype(np.float32)
    annual_oof, annual_importance = fit_held_blocks(
        annual_x,
        annual_target,
        annual_weights,
        cell_folds,
        tuple(annual_names),
        "annual",
    )
    print(
        f"ANNUAL oof_corr={weighted_corr(annual_oof, annual_target, annual_weights):+.9f}",
        flush=True,
    )
    for blend in (0.25, 0.5, 1.0):
        corrected = pred_annual * np.exp(blend * np.clip(annual_oof, -3.0, 3.0))
        baseline_loss = weighted_mean(np.abs(annual_target), annual_weights)
        corrected_target = np.log((obs_annual + 1e-6) / (corrected + 1e-6))
        corrected_loss = weighted_mean(np.abs(corrected_target), annual_weights)
        corrected_ratio = float(np.sum(corrected * area)) / max(
            float(np.sum(obs_annual * area)), 1e-12
        )
        print(
            f"ANNUAL_BLEND blend={blend:g} loss={corrected_loss:.9f} "
            f"delta={corrected_loss - baseline_loss:+.9f} ratio={corrected_ratio:.9f}",
            flush=True,
        )

    annual_rank = np.argsort(annual_importance.mean(axis=0))[::-1]
    for index in annual_rank[:6]:
        binned_shape(
            annual_x[:, index],
            annual_target,
            annual_weights,
            annual_names[index],
            "annual",
        )

    # The cycle learner sees month-of-year climatologies of causal online
    # features; no month number or coordinate is a feature.
    pred_norm = pred_cycle / np.maximum(pred_annual[None, :], 1e-8)
    obs_norm = obs_cycle / np.maximum(obs_annual[None, :], 1e-8)
    cycle_target = (obs_norm - pred_norm).T.reshape(-1)
    cycle_columns = []
    for values in features.values():
        cycle_columns.append(values.reshape(16, 12, -1).mean(axis=0).T.reshape(-1))
    cycle_x = np.column_stack(cycle_columns).astype(np.float32)
    cycle_weights = np.repeat(annual_weights, 12)
    cycle_folds = np.repeat(cell_folds, 12)
    cycle_oof, cycle_importance = fit_held_blocks(
        cycle_x,
        cycle_target,
        cycle_weights,
        cycle_folds,
        feature_names,
        "cycle",
    )
    baseline_cycle = weighted_mean(np.abs(cycle_target), cycle_weights)
    for blend in (0.25, 0.5, 1.0):
        corrected_norm = pred_norm.T + blend * cycle_oof.reshape(-1, 12)
        corrected_norm = np.clip(corrected_norm, 0.0, None)
        corrected_norm /= np.maximum(corrected_norm.sum(axis=1, keepdims=True), 1e-12)
        residual = obs_norm.T - corrected_norm
        loss = weighted_mean(np.abs(residual.reshape(-1)), cycle_weights)
        aggregate = np.sum(corrected_norm.T * obs_annual[None, :] * area[None, :], axis=1)
        aggregate /= max(float(aggregate.sum()), 1e-12)
        obs_aggregate = obs_monthly / max(float(obs_monthly.sum()), 1e-12)
        aggregate_l1 = 0.5 * float(np.sum(np.abs(aggregate - obs_aggregate)))
        print(
            f"CYCLE_BLEND blend={blend:g} loss={loss:.9f} "
            f"delta={loss - baseline_cycle:+.9f} aggregate_l1={aggregate_l1:.9f} "
            f"peak={MONTHS[int(np.argmax(aggregate))]}/{MONTHS[int(np.argmax(obs_aggregate))]}",
            flush=True,
        )

    cycle_rank = np.argsort(cycle_importance.mean(axis=0))[::-1]
    for index in cycle_rank[:6]:
        binned_shape(
            cycle_x[:, index],
            cycle_target,
            cycle_weights,
            feature_names[index],
            "cycle",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
