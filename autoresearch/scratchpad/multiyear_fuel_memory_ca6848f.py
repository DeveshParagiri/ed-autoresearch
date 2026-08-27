"""Held-cell screen for causal multi-year fuel and drought memory.

Diagnostic only, pinned to the official .719 model blob.  The learned surface
never enters ``model.py``: it asks whether 24--60 month rain/GPP state adds
annual-map information after the incumbent prediction and its short causal
states.  Two physical translations are tested separately as finite pointwise
stocks with globally shared equations: wet-year open-fuel carryover and
compound drought deadwood.  Coordinates construct held spatial folds only.
No region, calendar, neighbour, future summary, or invalid forcing is a
runtime predictor.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


MODEL_BLOB = "ca6848f2db28af24a06cd9f06e3adcdecaf7fcc0"
EPS = np.float32(1e-6)


def pinned_model():
    source = subprocess.run(
        ("git", "cat-file", "blob", MODEL_BLOB),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("ed_fire_multiyear_pinned")
    module.__file__ = f"git-blob:{MODEL_BLOB}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(values[0], dtype=np.float32).copy()
    output = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        state += alpha * (values[index] - state)
        output[index] = state
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def observation_grid() -> np.ndarray:
    output = np.empty((192, 180, 360), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for row in range(180):
            slab = np.ma.asarray(variable[:192, 2 * row : 2 * row + 2, :])
            if np.ma.getmaskarray(slab).any():
                raise ValueError("masked GFED observation")
            output[:, row, :] = np.asarray(slab, dtype=np.float32).reshape(
                192, 2, 360, 2
            ).mean(axis=(1, 3)) / 100.0
    return output


def select_high_weight(
    observation: np.ndarray,
    area: np.ndarray,
    land: np.ndarray,
    fraction: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, float]:
    annual = observation.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    weight = area * annual * land
    land_flat = np.flatnonzero(land.ravel())
    order = land_flat[np.argsort(weight.ravel()[land_flat])[::-1]]
    cumulative = np.cumsum(weight.ravel()[order]) / max(float(weight.sum()), 1e-12)
    count = int(np.searchsorted(cumulative, fraction) + 1)
    selected = order[:count]
    retained = float(weight.ravel()[selected].sum() / weight.sum())
    return selected // 360, selected % 360, retained


def finite_states(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def field(name: str) -> np.ndarray:
        return np.asarray(data[name][:, 0, :], dtype=np.float32)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    gpp = np.clip(field("gpp"), 0.0, None)
    dryness = np.clip(field("dryness"), 0.0, None)
    temperature = field("air_temperature")
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    natural = np.clip(field("natural_vegetation_fraction"), 0.0, 1.0)
    secondary = np.clip(field("secondary_vegetation_fraction"), 0.0, 1.0)
    canopy = np.clip(field("natural_canopy_height"), 0.0, None)
    secondary_canopy = np.clip(field("secondary_canopy_height"), 0.0, None)
    biomass = np.clip(field("aboveground_biomass"), 0.0, None)
    crop = np.clip(field("luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field("luh2_urban_fraction"), 0.0, 1.0)

    rain12, rain24, rain36, rain60 = (
        ema(rain, months) for months in (12.0, 24.0, 36.0, 60.0)
    )
    gpp3, gpp12, gpp24, gpp36, gpp60 = (
        ema(gpp, months) for months in (3.0, 12.0, 24.0, 36.0, 60.0)
    )
    dryness12 = ema(dryness, 12.0)
    temperature24 = ema(temperature, 24.0)
    lightning12 = ema(lightning, 12.0)

    wet_departure = np.maximum(
        (rain24 - rain60) / (rain24 + rain60 + 10.0), 0.0
    )
    drought_departure = np.maximum(
        (rain60 - rain24) / (rain24 + rain60 + 10.0), 0.0
    )
    gpp_surplus = np.maximum(
        (gpp24 - gpp60) / (gpp24 + gpp60 + 0.10), 0.0
    )
    gpp_deficit = np.maximum(
        (gpp60 - gpp24) / (gpp24 + gpp60 + 0.10), 0.0
    )

    wet_stock = np.zeros_like(rain[0], dtype=np.float32)
    drought_stock = np.zeros_like(rain[0], dtype=np.float32)
    deadwood_stock = np.zeros_like(rain[0], dtype=np.float32)
    wet_history = np.empty_like(rain, dtype=np.float32)
    drought_history = np.empty_like(rain, dtype=np.float32)
    deadwood_history = np.empty_like(rain, dtype=np.float32)
    wet_decay = np.float32(np.exp(-1.0 / 36.0))
    drought_decay = np.float32(np.exp(-1.0 / 48.0))
    deadwood_decay = np.float32(np.exp(-1.0 / 60.0))
    for index in range(rain.shape[0]):
        wet_stock *= wet_decay
        recharge = 1.0 - np.exp(
            -wet_departure[index]
            * gpp12[index]
            / (gpp12[index] + 0.35)
            / 6.0
        )
        wet_stock += recharge * (1.0 - wet_stock)
        np.clip(wet_stock, 0.0, 1.0, out=wet_stock)

        previous_drought = drought_stock.copy()
        drought_stock *= drought_decay
        loading = 1.0 - np.exp(
            -drought_departure[index] * (0.25 + 0.75 * gpp_deficit[index]) / 6.0
        )
        drought_stock += loading * (1.0 - drought_stock)
        np.clip(drought_stock, 0.0, 1.0, out=drought_stock)
        mortality = np.maximum(drought_stock - previous_drought, 0.0)
        deadwood_stock *= deadwood_decay
        deadwood_stock += mortality * (1.0 - deadwood_stock)
        np.clip(deadwood_stock, 0.0, 1.0, out=deadwood_stock)

        wet_history[index] = wet_stock
        drought_history[index] = drought_stock
        deadwood_history[index] = deadwood_stock

    fine_fuel = gpp12 / (gpp12 + 0.35)
    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    continuity = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    surface = np.clip(natural_open + secondary_open + managed_open, 0.0, 2.0)
    surface *= fine_fuel * continuity
    woody = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    combustion = dryness / (dryness + 250.0) / (1.0 + rain / 35.0)
    thermal = sigmoid((temperature - 5.0) / 4.0)
    ignition = lightning12 / (lightning12 + 0.02)
    current_drydown = np.maximum(
        (rain12 - rain) / (rain12 + rain + 10.0), 0.0
    )

    return {
        "rain": rain,
        "gpp": gpp,
        "dryness": dryness,
        "temperature": temperature,
        "lightning": lightning,
        "rain12": rain12,
        "rain24": rain24,
        "rain36": rain36,
        "rain60": rain60,
        "gpp3": gpp3,
        "gpp12": gpp12,
        "gpp24": gpp24,
        "gpp36": gpp36,
        "gpp60": gpp60,
        "dryness12": dryness12,
        "temperature24": temperature24,
        "lightning12": lightning12,
        "wet_departure": wet_departure,
        "drought_departure": drought_departure,
        "gpp_surplus": gpp_surplus,
        "gpp_deficit": gpp_deficit,
        "wet_stock": wet_history,
        "drought_stock": drought_history,
        "deadwood_stock": deadwood_history,
        "surface": surface,
        "woody": woody,
        "combustion": combustion,
        "thermal": thermal,
        "ignition": ignition,
        "current_drydown": current_drydown,
        "crop": crop,
        "rangeland": rangeland,
        "natural": natural,
        "primary": np.clip(field("luh2_primary_fraction"), 0.0, 1.0),
        "canopy": canopy,
        "biomass": biomass,
        "lai": np.clip(field("leaf_area_index"), 0.0, None),
        "soil": np.clip(field("soil_carbon"), 0.0, None),
    }


def build_features(
    states: dict[str, np.ndarray],
    baseline: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray]:
    count = baseline.shape[-1]
    baseline_year = baseline.reshape(16, 12, count).sum(axis=1)
    previous_december = np.arange(11, 180, 12)

    base: dict[str, np.ndarray] = {
        "log_incumbent_year": np.log(baseline_year[1:] + EPS),
        "log_incumbent_prior": np.log(baseline_year[:-1] + EPS),
    }
    for name in (
        "rain12",
        "gpp3",
        "gpp12",
        "dryness12",
        "temperature24",
        "lightning12",
        "surface",
        "woody",
        "combustion",
        "current_drydown",
        "crop",
        "rangeland",
        "primary",
        "canopy",
        "biomass",
        "lai",
        "soil",
    ):
        base[name] = states[name][previous_december]

    long: dict[str, np.ndarray] = {}
    for name in (
        "rain24",
        "rain36",
        "rain60",
        "gpp24",
        "gpp36",
        "gpp60",
        "wet_departure",
        "drought_departure",
        "gpp_surplus",
        "gpp_deficit",
        "wet_stock",
        "drought_stock",
        "deadwood_stock",
    ):
        long[name] = states[name][previous_december]
    long["wet_open_fuel"] = (
        states["wet_stock"] * states["surface"]
    )[previous_december]
    long["wet_open_release"] = (
        states["wet_stock"]
        * states["surface"]
        * states["combustion"]
        * states["current_drydown"]
        * states["thermal"]
    )[previous_december]
    long["drought_woody_fuel"] = (
        states["deadwood_stock"] * states["woody"]
    )[previous_december]
    long["drought_woody_release"] = (
        states["deadwood_stock"]
        * states["woody"]
        * states["combustion"]
        * states["thermal"]
        * states["ignition"]
    )[previous_december]

    base_names = tuple(base)
    long_names = tuple(long)
    x_base = np.column_stack([base[name].reshape(-1) for name in base_names]).astype(
        np.float32
    )
    x_long = np.column_stack([long[name].reshape(-1) for name in long_names]).astype(
        np.float32
    )
    return base_names, x_base, long_names, x_long


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.abs(values) * weights) / np.sum(weights))


def tree_feature_counts(
    learner: HistGradientBoostingRegressor,
    names: tuple[str, ...],
) -> Counter:
    output: Counter = Counter()
    for stage in learner._predictors:
        nodes = stage[0].nodes
        for node in nodes:
            if not node["is_leaf"]:
                output[names[int(node["feature_idx"])]] += 1
    return output


def held_screen(
    x_base: np.ndarray,
    x_long: np.ndarray,
    names: tuple[str, ...],
    target: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof_base = np.zeros_like(target)
    oof_long = np.zeros_like(target)
    ridge_signs: dict[str, list[int]] = {name: [] for name in names[x_base.shape[1] :]}
    feature_counts: list[Counter] = []
    for fold in range(4):
        train, held = folds != fold, folds == fold
        base_model = HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=3.0,
            min_samples_leaf=350,
            early_stopping=False,
            random_state=8800 + fold,
        )
        long_model = HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=3.0,
            min_samples_leaf=350,
            early_stopping=False,
            random_state=8900 + fold,
        )
        base_model.fit(x_base[train], target[train], sample_weight=weights[train])
        full = np.column_stack((x_base, x_long))
        long_model.fit(full[train], target[train], sample_weight=weights[train])
        oof_base[held] = base_model.predict(x_base[held]).astype(np.float32)
        oof_long[held] = long_model.predict(full[held]).astype(np.float32)
        feature_counts.append(tree_feature_counts(long_model, names))

        mean = np.average(full[train], axis=0, weights=weights[train])
        scale = np.sqrt(
            np.average(np.square(full[train] - mean), axis=0, weights=weights[train])
        ) + 1e-6
        ridge = Ridge(alpha=20.0)
        ridge.fit(
            (full[train] - mean) / scale,
            target[train],
            sample_weight=weights[train],
        )
        for offset, name in enumerate(names[x_base.shape[1] :], x_base.shape[1]):
            ridge_signs[name].append(int(np.sign(ridge.coef_[offset])))

        raw = weighted_mae(target[held], weights[held])
        base_loss = weighted_mae(target[held] - oof_base[held], weights[held])
        long_loss = weighted_mae(target[held] - oof_long[held], weights[held])
        print(
            f"FOLD fold={fold} raw={raw:.9f} short={base_loss:.9f} "
            f"long={long_loss:.9f} incremental={long_loss-base_loss:+.9f}",
            flush=True,
        )
        del base_model, long_model, full, ridge
        gc.collect()

    raw = weighted_mae(target, weights)
    short = weighted_mae(target - oof_base, weights)
    long = weighted_mae(target - oof_long, weights)
    print(
        f"OOF raw={raw:.9f} short={short:.9f} long={long:.9f} "
        f"incremental={long-short:+.9f}",
        flush=True,
    )
    total = sum(feature_counts, Counter())
    for name in names[x_base.shape[1] :]:
        stability = sum(name in counts for counts in feature_counts)
        signs = ridge_signs[name]
        print(
            f"LONG_FEATURE name={name} tree_splits={total[name]} "
            f"tree_folds={stability}/4 ridge_signs={','.join(f'{value:+d}' for value in signs)}",
            flush=True,
        )
    return oof_base, oof_long


def annual_loss(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    cell_folds: np.ndarray,
) -> tuple[float, tuple[float, ...], float]:
    count = prediction.shape[-1]
    pred_year = prediction.reshape(16, 12, count)[1:].sum(axis=1)
    obs_year = observation.reshape(16, 12, count)[1:].sum(axis=1)
    floor = float(np.sum(obs_year * area[None, :])) / (
        15.0 * float(np.sum(area))
    )
    weight = area[None, :] * (obs_year + 0.02 * floor)
    residual = np.abs(np.log((obs_year + EPS) / (pred_year + EPS)))
    total = float(np.sum(weight * residual) / np.sum(weight))
    held = tuple(
        float(
            np.sum(weight[:, cell_folds == fold] * residual[:, cell_folds == fold])
            / np.sum(weight[:, cell_folds == fold])
        )
        for fold in range(4)
    )
    ratio = float(np.sum(pred_year * area[None, :])) / max(
        float(np.sum(obs_year * area[None, :])), 1e-12
    )
    return total, held, ratio


def cycle_l1(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
) -> float:
    count = prediction.shape[-1]
    pred = prediction.reshape(16, 12, count).mean(axis=0)
    obs = observation.reshape(16, 12, count).mean(axis=0)
    pred_month = np.sum(pred * area[None, :], axis=1)
    obs_month = np.sum(obs * area[None, :], axis=1)
    pred_month /= max(float(pred_month.sum()), 1e-12)
    obs_month /= max(float(obs_month.sum()), 1e-12)
    return 0.5 * float(np.sum(np.abs(pred_month - obs_month)))


def ecology_masks(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rain = 12.0 * states["rain"].mean(axis=0)
    temperature = states["temperature"].mean(axis=0)
    lai = states["lai"].mean(axis=0)
    canopy = states["canopy"].mean(axis=0)
    biomass = states["biomass"].mean(axis=0)
    primary = states["primary"].mean(axis=0)
    crop = states["crop"].mean(axis=0)
    rangeland = states["rangeland"].mean(axis=0)
    natural = states["natural"].mean(axis=0)
    return {
        "intact_tropical_closed": (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0) & (biomass >= 0.2),
        "crop": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    count = prediction.shape[-1]
    pred = prediction.reshape(16, 12, count).mean(axis=0).sum(axis=0)
    obs = observation.reshape(16, 12, count).mean(axis=0).sum(axis=0)
    output = {}
    for name, mask in masks.items():
        if not np.any(mask):
            output[name] = float("nan")
        else:
            output[name] = float(np.sum(pred[mask] * area[mask])) / max(
                float(np.sum(obs[mask] * area[mask])), 1e-12
            )
    return output


def candidate_from_modifier(
    baseline: np.ndarray,
    modifier: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    adjusted = hazard * (1.0 + strength * np.clip(modifier, 0.0, 1.0))
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def brake_from_modifier(
    baseline: np.ndarray,
    modifier: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    adjusted = hazard * np.exp(-strength * np.clip(modifier, 0.0, 1.0))
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observation = observation_grid()
    rows, columns, retained = select_high_weight(observation, area_grid, land)
    sampled_observation = observation[:, rows, columns][:, None, :]
    area = area_grid[rows, columns].astype(np.float64)
    del observation, area_grid, land
    gc.collect()

    data = {name: selected_input(name, rows, columns) for name in model.INPUTS}
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )
    states = finite_states(data)
    count = rows.size
    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    folds = np.tile(cell_folds, (15, 1)).reshape(-1)

    pred_year = baseline.reshape(16, 12, count)[1:].sum(axis=1)
    obs_year = sampled_observation.reshape(16, 12, count)[1:].sum(axis=1)
    target = np.clip(np.log((obs_year + EPS) / (pred_year + EPS)), -4.0, 4.0).reshape(-1)
    floor = float(np.sum(obs_year * area[None, :])) / (15.0 * float(np.sum(area)))
    weights = (area[None, :] * (obs_year + 0.02 * floor)).reshape(-1)
    weights /= weights.mean()
    base_names, x_base, long_names, x_long = build_features(states, baseline)
    names = base_names + long_names
    print(
        f"DESIGN blob={MODEL_BLOB} cells={count} years=15 retained_obs_weight={retained:.6f} "
        f"short_features={len(base_names)} long_features={len(long_names)} coordinates_features=0",
        flush=True,
    )
    oof_short, oof_long = held_screen(
        x_base, x_long, names, target, weights, folds
    )

    base_loss, base_folds, base_ratio = annual_loss(
        baseline, sampled_observation, area, cell_folds
    )
    base_cycle = cycle_l1(baseline, sampled_observation, area)
    masks = ecology_masks(states)
    base_ecology = ecology_ratios(
        baseline, sampled_observation, area, masks
    )
    print(
        f"BASE annual_loss={base_loss:.9f} ratio={base_ratio:.9f} cycle_l1={base_cycle:.9f} "
        + " ".join(f"fold{i}={value:.9f}" for i, value in enumerate(base_folds)),
        flush=True,
    )

    # Diagnostic score leverage: apply only held-cell/year corrections, with no
    # target fit ever evaluated on its own training cell.
    for family, oof in (("short", oof_short), ("long", oof_long)):
        for strength in (0.10, 0.25, 0.50):
            correction = np.exp(
                np.clip(strength * oof.reshape(15, count), -1.0, 1.0)
            )
            learned = baseline.copy().reshape(16, 12, count)
            learned[1:] *= correction[:, None, :]
            learned = np.clip(learned, 0.0, 1.0).reshape(192, 1, count)
            loss, held, ratio = annual_loss(
                learned, sampled_observation, area, cell_folds
            )
            print(
                f"ML_HEADROOM family={family} strength={strength:g} annual_loss={loss:.9f} "
                f"delta={loss-base_loss:+.9f} ratio={ratio:.9f} "
                f"cycle_l1={cycle_l1(learned, sampled_observation, area):.9f} "
                + " ".join(
                    f"fold{i}_delta={value-base_folds[i]:+.9f}"
                    for i, value in enumerate(held)
                ),
                flush=True,
            )

    mechanisms = {
        "wet_open_carryover": ("source", (
            states["wet_stock"]
            * states["surface"]
            * states["combustion"]
            * (0.25 + 0.75 * states["current_drydown"])
            * states["thermal"]
        )),
        "drought_deadwood_source": ("source", (
            states["deadwood_stock"]
            * states["woody"]
            * states["combustion"]
            * states["thermal"]
            * states["ignition"]
        )),
        # Repeated moisture deficits can collapse living fine-fuel continuity
        # for several years.  This is deliberately a recovery-limited brake,
        # distinct from treating drought mortality as an extra fire source.
        "drought_woody_damage_brake": ("brake", (
            states["deadwood_stock"]
            * states["woody"]
            * states["combustion"]
            * states["thermal"]
            * states["ignition"]
        )),
        "drought_surface_damage_brake": ("brake", (
            states["drought_stock"]
            * states["gpp_deficit"]
            * states["surface"]
        )),
    }
    best = None
    for family, (mode, modifier) in mechanisms.items():
        for strength in (0.25, 0.50, 1.0, 2.0):
            if mode == "source":
                candidate = candidate_from_modifier(
                    baseline, modifier[:, None, :], strength
                )
            else:
                candidate = brake_from_modifier(
                    baseline, modifier[:, None, :], strength
                )
            loss, held, ratio = annual_loss(
                candidate, sampled_observation, area, cell_folds
            )
            cycle = cycle_l1(candidate, sampled_observation, area)
            ecology = ecology_ratios(candidate, sampled_observation, area, masks)
            deltas = tuple(value - base_folds[i] for i, value in enumerate(held))
            print(
                f"MECHANISM family={family} mode={mode} strength={strength:g} annual_loss={loss:.9f} "
                f"delta={loss-base_loss:+.9f} ratio={ratio:.9f} cycle_l1={cycle:.9f} "
                f"cycle_delta={cycle-base_cycle:+.9f} "
                + " ".join(f"fold{i}_delta={value:+.9f}" for i, value in enumerate(deltas)),
                flush=True,
            )
            print(
                f"ECOLOGY family={family} strength={strength:g} "
                + " ".join(
                    f"{name}={base_ecology[name]:.6f}->{ecology[name]:.6f}"
                    for name in masks
                ),
                flush=True,
            )
            record = (loss, family, strength, deltas, cycle - base_cycle, ecology)
            if best is None or record[0] < best[0]:
                best = record

    # Directly perturb only future forcing on a small pointwise subset and
    # recompute the full candidate, including the incumbent model.
    prefix_count = min(96, count)
    prefix_data = {
        name: np.asarray(values[:, :, :prefix_count], dtype=np.float32).copy()
        for name, values in data.items()
    }
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= np.float32(0.5)
    prefix_base = np.asarray(
        model.predict(prefix_data, dict(model.PARAMS), None), dtype=np.float32
    )
    perturbed_base = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )
    prefix_states = finite_states(prefix_data)
    perturbed_states = finite_states(perturbed)
    prefix_modifier = (
        prefix_states["wet_stock"]
        * prefix_states["surface"]
        * prefix_states["combustion"]
        * (0.25 + 0.75 * prefix_states["current_drydown"])
        * prefix_states["thermal"]
    )
    perturbed_modifier = (
        perturbed_states["wet_stock"]
        * perturbed_states["surface"]
        * perturbed_states["combustion"]
        * (0.25 + 0.75 * perturbed_states["current_drydown"])
        * perturbed_states["thermal"]
    )
    prefix_candidate = candidate_from_modifier(
        prefix_base, prefix_modifier[:, None, :], 1.0
    )
    perturbed_candidate = candidate_from_modifier(
        perturbed_base, perturbed_modifier[:, None, :], 1.0
    )
    print(
        f"PREFIX cells={prefix_count} baseline_max={float(np.max(np.abs(prefix_base[:96]-perturbed_base[:96]))):.12g} "
        f"modifier_max={float(np.max(np.abs(prefix_modifier[:96]-perturbed_modifier[:96]))):.12g} "
        f"candidate_max={float(np.max(np.abs(prefix_candidate[:96]-perturbed_candidate[:96]))):.12g}",
        flush=True,
    )
    assert np.array_equal(prefix_candidate[:96], perturbed_candidate[:96])

    assert best is not None
    print(
        f"BEST family={best[1]} strength={best[2]:g} annual_delta={best[0]-base_loss:+.9f} "
        f"folds_all_improve={int(all(value < 0.0 for value in best[3]))} "
        f"cycle_delta={best[4]:+.9f}",
        flush=True,
    )
    print(
        f"RESOURCE elapsed_s={time.perf_counter()-started:.3f} "
        f"maxrss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
