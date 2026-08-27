"""Held-cell diagnostic and bounded physical test for managed temperate fire timing.

The TENA box is used only to report the motivating failure after prediction.
The learner and candidate mechanism contain no coordinate, country, region, or
calendar feature.  The candidate diverts a finite share of cultivated hazard
during active crop growth, removes the harvested fraction, and releases the
retained residue during causal pre-greenup warming or post-growth senescence.
It is a scratch experiment only and never edits ``model.py`` or the ledger.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from scripts.runtime import GFED5_PATH, load_land_mask, load_model  # noqa: E402


EXPECTED_MODEL_BLOB = "c56b96a1cbd57e4342b14f4cc13ea541830703e7"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -50.0, 50.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def rolling_std(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        output[time] = values[max(0, time - 11) : time + 1].std(axis=0)
    return output


def observation_one_degree() -> np.ndarray:
    output = np.empty((192, 180, 360), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for time in range(192):
            output[time] = np.asarray(variable[time], dtype=np.float32).reshape(
                180, 2, 360, 2
            ).mean(axis=(1, 3)) / 100.0
    return output


def temporal_moments(name: str) -> tuple[np.ndarray, np.ndarray]:
    values = selected_input(name, np.arange(180).repeat(360), np.tile(np.arange(360), 180))
    values = values[:, 0, :].reshape(192, 180, 360)
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    del values
    return mean, std


def select_top(
    mask: np.ndarray,
    weight: np.ndarray,
    quota: int,
    used: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero(mask & ~used)
    if not candidates.size:
        return candidates
    top_count = min(candidates.size, 3 * quota // 4)
    order = np.argsort(weight.ravel()[candidates])[::-1]
    top = candidates[order[:top_count]]
    remainder = candidates[order[top_count:]]
    random_count = min(remainder.size, quota - top.size)
    random = rng.choice(remainder, size=random_count, replace=False) if random_count else np.empty(0, dtype=int)
    chosen = np.concatenate((top, random))
    used.ravel()[chosen] = True
    return chosen


def build_sample_masks(
    land: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    temperature, temperature_std = temporal_moments("air_temperature")
    rain, _ = temporal_moments("monthly_precipitation")
    crop, _ = temporal_moments("luh2_cropland_fraction")
    pasture, _ = temporal_moments("luh2_pasture_fraction")
    rangeland, _ = temporal_moments("luh2_rangeland_fraction")
    natural, _ = temporal_moments("natural_vegetation_fraction")
    canopy, _ = temporal_moments("natural_canopy_height")
    lai, _ = temporal_moments("leaf_area_index")
    biomass, _ = temporal_moments("aboveground_biomass")
    annual_rain = 12.0 * rain
    managed = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    temperate_managed = (
        land
        & (temperature > 4.0)
        & (temperature < 20.0)
        & (temperature_std > 4.0)
        & (annual_rain > 250.0)
        & (annual_rain < 1600.0)
        & (managed > 0.20)
    )
    masks = {
        "temperate_managed": temperate_managed,
        "intact_tropical_closed": land & (temperature >= 20.0) & (annual_rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7),
        "temperate_closed": land & (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": land & (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": land & (temperature >= 20.0) & (annual_rain >= 500.0) & (annual_rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": land & (rangeland >= 0.4) & (annual_rain >= 250.0) & (annual_rain < 1500.0) & (biomass >= 0.2),
        "cropland": land & (crop >= 0.5),
        "arid_low_fuel": land & (annual_rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }
    obs_annual = observation.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    weight = obs_annual * area
    rng = np.random.default_rng(52027)
    used = np.zeros_like(land)
    names = tuple(masks)
    quotas = (1400, 220, 220, 220, 220, 320, 480, 220)
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for group, (name, quota) in enumerate(zip(names, quotas, strict=True)):
        chosen = select_top(masks[name], weight, quota, used, rng)
        indices.append(chosen)
        labels.append(np.full(chosen.size, group, dtype=np.int8))
        print(f"SAMPLE name={name} cells={chosen.size} observed_weight={weight.ravel()[chosen].sum():.9e}", flush=True)
    flat = np.concatenate(indices)
    return flat // 360, flat % 360, names, {name: np.concatenate(labels) == index for index, name in enumerate(names)}


def sampled_observation(
    observation: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    return np.asarray(observation[:, rows, columns][:, None, :], dtype=np.float32)


def feature_bank(
    data: Mapping[str, np.ndarray], prediction: np.ndarray
) -> dict[str, np.ndarray]:
    def field(name: str) -> np.ndarray:
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    temperature = field("air_temperature")
    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    dryness = np.clip(field("dryness"), 0.0, None)
    gpp = np.clip(field("gpp"), 0.0, None)
    lai = np.clip(field("leaf_area_index"), 0.0, None)
    crop = np.clip(field("luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    natural = np.clip(field("natural_vegetation_fraction"), 0.0, 1.0)
    canopy = np.clip(field("natural_canopy_height"), 0.0, None)
    temperature3, temperature24 = ema(temperature, 3.0), ema(temperature, 24.0)
    rain6, rain12 = ema(rain, 6.0), ema(rain, 12.0)
    gpp3, gpp12, lai3 = ema(gpp, 3.0), ema(gpp, 12.0), ema(lai, 3.0)
    combustion = np.sqrt(dryness / (dryness + 250.0) / (1.0 + rain / 35.0))
    rain_deficit = np.maximum((rain6 - rain) / (rain6 + rain + 10.0), 0.0)
    greenup = np.maximum((gpp - gpp3) / (gpp + gpp3 + 0.1), 0.0)
    senescence = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.1), 0.0)
    lai_fall = np.maximum((lai3 - lai) / (lai3 + lai + 0.2), 0.0)
    warming = sigmoid((temperature - temperature3 - 0.5) / 1.5)
    cooling = sigmoid((temperature3 - temperature - 0.5) / 1.5)
    seasonal = sigmoid((rolling_std(temperature) - 4.0) / 1.5)
    temperate = sigmoid((temperature24 - 4.0) / 3.0) * sigmoid((20.0 - temperature24) / 3.0)
    managed = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    open_state = np.clip(managed + natural * 8.0 / (canopy + 8.0), 0.0, 1.0)
    fine_fuel = gpp12 / (gpp12 + 0.35)
    eligible = temperate * seasonal * crop * fine_fuel
    pre_greenup = eligible * warming * (1.0 - gpp / (gpp + 0.25)) * combustion
    post_growth = eligible * cooling * np.maximum(senescence, lai_fall) * combustion
    summer_growth = eligible * gpp / (gpp + 0.25) * (1.0 - np.maximum(senescence, lai_fall))
    return {
        "incumbent_hazard": -np.log1p(-np.clip(prediction[:, 0, :], 0.0, 1.0 - 1e-7)),
        "temperature": temperature,
        "temperature24": temperature24,
        "temperature_range": rolling_std(temperature),
        "warming": warming,
        "cooling": cooling,
        "rain": rain,
        "rain_deficit": rain_deficit,
        "dryness": dryness,
        "combustion": combustion,
        "gpp": gpp,
        "gpp12": gpp12,
        "greenup": greenup,
        "senescence": senescence,
        "lai_fall": lai_fall,
        "crop": crop,
        "managed": managed,
        "open_state": open_state,
        "fine_fuel": fine_fuel,
        "pre_greenup_residue": pre_greenup,
        "post_growth_residue": post_growth,
        "summer_live_crop": summer_growth,
    }


def held_cell_screen(
    features: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> None:
    names = tuple(features)
    x = np.column_stack(
        [values.reshape(16, 12, -1).mean(axis=0).ravel() for values in features.values()]
    ).astype(np.float32)
    pred = prediction[:, 0, :].reshape(16, 12, -1).mean(axis=0)
    obs = observation[:, 0, :].reshape(16, 12, -1).mean(axis=0)
    floor = 2e-5
    target = np.log((obs + floor) / (pred + floor))
    target -= target.mean(axis=0, keepdims=True)
    y = target.ravel().astype(np.float32)
    weight = ((obs + pred + floor) * area[None, :]).ravel()
    cell_folds = (rows // 15 + 2 * (columns // 20)) % 4
    folds = np.tile(cell_folds, 12)
    importance = np.empty((4, len(names)), dtype=np.float64)
    for fold in range(4):
        train, held = folds != fold, folds == fold
        learner = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=50,
            max_features=0.8,
            loss="huber",
            random_state=9181 + fold,
        )
        learner.fit(x[train], y[train], sample_weight=weight[train])
        learned = learner.predict(x[held])
        base = np.average(np.abs(y[held]), weights=weight[held])
        corrected = np.average(np.abs(y[held] - learned), weights=weight[held])
        importance[fold] = learner.feature_importances_
        print(f"ML_FOLD fold={fold} base_mae={base:.9f} corrected_mae={corrected:.9f} delta={corrected-base:+.9f}", flush=True)
    ranking = np.argsort(importance.mean(axis=0))[::-1]
    for index in ranking[:14]:
        top8 = sum(index in np.argsort(row)[-8:] for row in importance)
        print(f"ML_IMPORTANCE name={names[index]} mean={importance[:, index].mean():.9f} std={importance[:, index].std():.9f} top8_folds={top8}", flush=True)


def apply_residue_budget(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    strength: float,
    retention: float,
    release_gain: float,
) -> np.ndarray:
    features = feature_bank(data, prediction)
    crop = features["crop"]
    fine_fuel = features["fine_fuel"]
    natural = np.clip(np.asarray(data["natural_vegetation_fraction"][:, 0, :]), 0.0, 1.0)
    canopy = np.clip(np.asarray(data["natural_canopy_height"][:, 0, :]), 0.0, None)
    biomass = np.clip(np.asarray(data["aboveground_biomass"][:, 0, :]), 0.0, None)
    woody = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    crop_share = crop_capacity / (0.05 + crop_capacity + woody + (1.0 - crop) * fine_fuel * features["open_state"])

    temperature = features["temperature"]
    temperature24 = features["temperature24"]
    seasonal = sigmoid((features["temperature_range"] - 4.0) / 1.5)
    temperate = sigmoid((temperature24 - 4.0) / 3.0) * sigmoid((20.0 - temperature24) / 3.0)
    productive = ema(np.clip(np.asarray(data["monthly_precipitation"][:, 0, :]), 0.0, None), 12.0)
    productive = productive / (productive + 30.0)
    eligibility = temperate * seasonal * productive

    gpp = np.clip(np.asarray(data["gpp"][:, 0, :]), 0.0, None)
    live_growth = gpp / (gpp + 0.25)
    capture_gate = eligibility * live_growth * (1.0 - np.maximum(features["senescence"], features["lai_fall"]))
    thermal = sigmoid((temperature - 3.0) / 2.5)
    dormant = 1.0 - gpp / (gpp + 0.25)
    spring = features["warming"] * dormant
    autumn = features["cooling"] * np.maximum(features["senescence"], features["lai_fall"])
    readiness = np.clip((spring + autumn) * features["combustion"] * thermal, 0.0, 1.0)

    hazard = -np.log1p(-np.clip(prediction[:, 0, :], 0.0, 1.0 - 1e-7))
    output = np.empty_like(hazard)
    initial_release = 1.0 - np.exp(-(1.0 / 18.0 + release_gain * readiness[0]))
    initial_capture = strength * crop_share[0] * capture_gate[0] * hazard[0]
    bank = retention * initial_capture * (1.0 - initial_release) / np.maximum(initial_release, 1e-4)
    for time in range(hazard.shape[0]):
        captured = np.clip(strength * crop_share[time] * capture_gate[time], 0.0, 0.95) * hazard[time]
        bank += retention * captured
        release = (1.0 - np.exp(-(1.0 / 18.0 + release_gain * readiness[time]))) * bank
        bank -= release
        output[time] = hazard[time] - captured + release
    return np.asarray(1.0 - np.exp(-np.clip(output, 0.0, 50.0)), dtype=np.float32)[:, None, :]


def diagnostics(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    chosen: np.ndarray,
) -> dict[str, float | str]:
    pred = prediction[:, 0, chosen].reshape(16, 12, -1).mean(axis=0)
    obs = observation[:, 0, chosen].reshape(16, 12, -1).mean(axis=0)
    weight = area[chosen]
    pred_month = np.sum(pred * weight[None, :], axis=1)
    obs_month = np.sum(obs * weight[None, :], axis=1)
    pred_norm = pred_month / max(float(pred_month.sum()), 1e-12)
    obs_norm = obs_month / max(float(obs_month.sum()), 1e-12)
    cell_weight = (obs.sum(axis=0) + 1e-6) * weight
    cell_l1 = 0.5 * np.sum(np.abs(pred / (pred.sum(axis=0, keepdims=True) + 1e-9) - obs / (obs.sum(axis=0, keepdims=True) + 1e-9)), axis=0)
    model_peak = np.argmax(pred, axis=0)
    observed_peak = np.argmax(obs, axis=0)
    phase = np.abs(model_peak - observed_peak)
    phase = np.minimum(phase, 12 - phase)
    phase_score = 0.5 * (1.0 + np.cos(phase / 12.0 * 2.0 * np.pi))
    return {
        "ratio": float(pred_month.sum() / max(float(obs_month.sum()), 1e-12)),
        "seasonal_l1": float(0.5 * np.sum(np.abs(pred_norm - obs_norm))),
        "cell_l1": float(np.average(cell_l1, weights=cell_weight)),
        "phase_score": float(np.average(phase_score, weights=cell_weight)),
        "model_peak": MONTHS[int(np.argmax(pred_month))],
        "obs_peak": MONTHS[int(np.argmax(obs_month))],
    }


def main() -> int:
    blob = subprocess.run(["git", "hash-object", "autoresearch/model.py"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"moving model: expected {EXPECTED_MODEL_BLOB}, got {blob}")
    model = load_model()
    land = load_land_mask()
    observation = observation_one_degree()
    area_grid = one_degree_area()
    rows, columns, names, groups = build_sample_masks(land, observation, area_grid)
    data = {name: selected_input(name, rows, columns) for name in model.INPUTS}
    obs = sampled_observation(observation, rows, columns)
    area = area_grid[rows, columns]
    del land, area_grid
    gc.collect()
    incumbent = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float32)
    features = feature_bank(data, incumbent)
    held_cell_screen(features, incumbent, obs, area, rows, columns)
    del features
    baseline = {name: diagnostics(incumbent, obs, area, groups[name]) for name in names}
    for name in names:
        print("BASE " + name + " " + " ".join(f"{key}={value}" for key, value in baseline[name].items()), flush=True)

    configs = (
        (0.40, 1.00, 10.0),
        (0.65, 1.00, 10.0),
        (0.65, 0.50, 10.0),
        (0.85, 0.50, 16.0),
        (0.85, 0.25, 16.0),
    )
    folds = (rows // 15 + 2 * (columns // 20)) % 4
    for strength, retention, release in configs:
        candidate = apply_residue_budget(incumbent, data, strength, retention, release)
        label = f"s{strength:g}_ret{retention:g}_rel{release:g}"
        for name in names:
            result = diagnostics(candidate, obs, area, groups[name])
            delta = float(result["cell_l1"]) - float(baseline[name]["cell_l1"])
            print("RESULT " + label + " group=" + name + " " + " ".join(f"{key}={value}" for key, value in result.items()) + f" cell_l1_delta={delta:+.9f}", flush=True)
        for fold in range(4):
            chosen = folds == fold
            before = diagnostics(incumbent, obs, area, chosen)
            after = diagnostics(candidate, obs, area, chosen)
            print(f"FOLD_RESULT {label} fold={fold} cell_l1_delta={float(after['cell_l1'])-float(before['cell_l1']):+.9f} ratio={after['ratio']}", flush=True)
        del candidate
        gc.collect()

    # Prefix causality is tested on the strongest fully conserved version.  A
    # large perturbation to every future input must leave the first 96 months
    # bit-identical through both the incumbent and the residue allocator.
    perturbed = {
        name: np.asarray(values, dtype=np.float32).copy()
        for name, values in data.items()
    }
    for values in perturbed.values():
        values[96:] *= np.float32(0.5)
    future_incumbent = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )
    prefix_candidate = apply_residue_budget(incumbent, data, 0.65, 1.0, 10.0)
    future_candidate = apply_residue_budget(
        future_incumbent, perturbed, 0.65, 1.0, 10.0
    )
    print(
        "PREFIX "
        f"incumbent_max_abs={np.max(np.abs(incumbent[:96] - future_incumbent[:96])):.9e} "
        f"candidate_max_abs={np.max(np.abs(prefix_candidate[:96] - future_candidate[:96])):.9e}",
        flush=True,
    )
    del perturbed, future_incumbent, prefix_candidate, future_candidate
    gc.collect()

    # Exact post-prediction TENA audit using all valid cells in the box.  The
    # box never enters either the learner or candidate equation above.
    lat = -89.5 + np.arange(180)
    lon = -179.5 + np.arange(360)
    tena = load_land_mask() & (lat[:, None] > 30.25) & (lat[:, None] <= 49.75) & (lon[None, :] > -125.25) & (lon[None, :] <= -66.25)
    flat = np.flatnonzero(tena)
    tr, tc = flat // 360, flat % 360
    tena_data = {name: selected_input(name, tr, tc) for name in model.INPUTS}
    tena_obs = sampled_observation(observation, tr, tc)
    tena_area = one_degree_area()[tr, tc]
    tena_base = np.asarray(model.predict(tena_data, dict(model.PARAMS), None), dtype=np.float32)
    all_cells = np.ones(flat.size, dtype=bool)
    print("TENA_BASE " + " ".join(f"{key}={value}" for key, value in diagnostics(tena_base, tena_obs, tena_area, all_cells).items()), flush=True)
    for strength, retention, release in configs:
        candidate = apply_residue_budget(tena_base, tena_data, strength, retention, release)
        label = f"s{strength:g}_ret{retention:g}_rel{release:g}"
        print("TENA_RESULT " + label + " " + " ".join(f"{key}={value}" for key, value in diagnostics(candidate, tena_obs, tena_area, all_cells).items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
