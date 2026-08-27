"""Held-block diagnosis of the clean warm seasonal-open fire carrier.

The clean nine-regime partition is rebuilt with streaming two-dimensional
states.  Coordinates are used only to choose held geographic blocks and never
enter a learner or candidate equation.  Shallow forests diagnose annual
propensity and normalized-cycle residuals separately; no learned surface is a
model candidate.  A cycle correction is normalized within every cell-year so
it cannot add annual burned-area mass.
"""

from __future__ import annotations

import gc
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    load_observed,
    load_selected,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, INPUTS_DIR, load_model  # noqa: E402


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def input_index() -> dict[str, Path]:
    result = {}
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if variable.dimensions == ("time", "lat", "lon"):
                    result[name] = path
    return result


def stream_mean(name: str, index: dict[str, Path]) -> np.ndarray:
    total = np.zeros((180, 360), dtype=np.float64)
    with Dataset(index[name]) as dataset:
        variable = dataset.variables[name]
        for time in range(192):
            total += np.asarray(variable[time], dtype=np.float64)
    return total / 192.0


def stream_max(name: str, index: dict[str, Path], absolute: bool = False) -> np.ndarray:
    result = np.full((180, 360), -np.inf, dtype=np.float64)
    with Dataset(index[name]) as dataset:
        variable = dataset.variables[name]
        for time in range(192):
            field = np.asarray(variable[time], dtype=np.float64)
            if absolute:
                field = np.abs(field)
            result = np.maximum(result, field)
    return result


def stream_antecedent_mean(name: str, months: float, index: dict[str, Path]) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    total = np.zeros((180, 360), dtype=np.float64)
    state = None
    with Dataset(index[name]) as dataset:
        variable = dataset.variables[name]
        for time in range(192):
            field = np.asarray(variable[time], dtype=np.float64)
            if state is None:
                state = field.copy()
            state += alpha * (field - state)
            total += state
    return total / 192.0


def clean_warm_open_mask() -> np.ndarray:
    index = input_index()
    rain = 12.0 * stream_antecedent_mean("monthly_precipitation", 12.0, index)
    temperature = stream_antecedent_mean("air_temperature", 12.0, index)
    dryness = stream_mean("dryness", index)
    dry_fraction = np.zeros((180, 360), dtype=np.float64)
    with Dataset(index["dryness"]) as dataset:
        variable = dataset.variables["dryness"]
        for time in range(192):
            field = np.clip(np.asarray(variable[time], dtype=np.float64), 0.0, None)
            dry_fraction += field / (field + 500.0)
    dry_fraction /= 192.0

    primary = stream_mean("luh2_primary_fraction", index)
    crop = stream_mean("luh2_cropland_fraction", index)
    pasture = stream_mean("luh2_pasture_fraction", index)
    rangeland = stream_mean("luh2_rangeland_fraction", index)
    urban = stream_mean("luh2_urban_fraction", index)
    classified = np.clip(primary + crop + pasture + rangeland + urban, 0.0, 1.0)
    residual = np.clip(1.0 - classified, 0.0, 1.0)
    open_cover = np.clip(pasture + rangeland + residual, 0.0, 1.0)

    canopy = stream_mean("natural_canopy_height", index)
    lai = stream_mean("leaf_area_index", index)
    biomass = stream_mean("aboveground_biomass", index)
    natural = stream_mean("natural_vegetation_fraction", index)
    annual_installed = stream_max("annual_precipitation", index)
    temperature_max = stream_max("air_temperature", index, absolute=True)
    natural_max = stream_max("natural_vegetation_fraction", index)
    secondary_max = stream_max("secondary_vegetation_fraction", index)
    land = (
        (annual_installed > 0.0)
        | (temperature_max > 1e-6)
        | (natural_max > 0.0)
        | (secondary_max > 0.0)
    )

    established = (
        (
            (temperature >= 20.0)
            & (rain >= 1200.0)
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
            & (rain >= 500.0)
            & (rain < 1500.0)
            & (canopy >= 5.0)
            & (canopy < 20.0)
            & (natural >= 0.5)
        )
        | (
            (rangeland >= 0.4)
            & (rain >= 250.0)
            & (rain < 1500.0)
            & (biomass >= 0.2)
        )
        | (crop >= 0.5)
        | ((rain < 250.0) & (biomass < 0.3) & (lai < 1.0))
    )
    complement = land & ~established
    warm_open = (
        complement
        & (temperature >= 18.0)
        & (rain >= 400.0)
        & (rain < 1600.0)
        & (dry_fraction >= 0.18)
        & (open_cover >= 0.20)
    )
    del dryness
    return warm_open


def select_regime_cells(evaluator: GFED5Evaluator, mask: np.ndarray, count: int = 1536):
    reference_mean = np.asarray(evaluator.reference_mean, dtype=np.float64)
    coarse_mean = reference_mean.reshape(180, 2, 360, 2).mean(axis=(1, 3)) / 100.0
    coarse_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    weight = coarse_area * coarse_mean
    candidates = np.flatnonzero(mask.ravel())
    order = candidates[np.argsort(weight.ravel()[candidates])[::-1]]
    cells = order[: min(count, order.size)]
    rows, cols = cells // 360, cells % 360
    selected_weight = float(weight.ravel()[cells].sum())
    return (
        rows,
        cols,
        coarse_area[rows, cols],
        weight[rows, cols],
        float(weight[mask].sum() / weight.sum()),
        selected_weight / float(weight[mask].sum()),
        selected_weight / float(weight.sum()),
        int(candidates.size),
    )


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def stage_predictions(model, raw_data):
    p = dict(model.PARAMS)
    enabled = set(model.COMPONENTS)
    data = dict(raw_data)
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float32)
    data["annual_precipitation"] = 12.0 * model._antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    rate = model._fire_rate(data, p, enabled)
    if "cropland" in enabled and p.get("crop_k", 0.0) > 0.0:
        crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
        rate *= 1.0 / (1.0 + p["crop_k"] * np.power(crop, p["crop_n"]))
    if "curing" in enabled:
        rate *= model._curing(data, p)
    if "lag" in enabled:
        rate = model._lag(rate, p)
    prediction = model._transform(rate, p)
    stages = [("transform", np.asarray(prediction, dtype=np.float32).copy())]
    functions = (
        ("regime_brakes", model._ecological_regime_brakes),
        ("pathway_event", model._pathway_event_scaling),
        ("ecological_capacity", model._ecological_fire_capacity),
        ("seasonal_rain_capacity", model._seasonal_rainfall_capacity),
        ("fire_season", model._state_dependent_fire_season),
        ("rare_ignition", model._rare_lightning_ignition),
        ("crop_management", model._rain_conditioned_crop_management),
        ("dead_fuel_pool", model._dead_fuel_pool_response),
        ("conditional_allocation", model._conditional_fire_allocation),
        ("greenup_brake", model._live_fuel_greenup_brake),
        ("surface_bank", model._surface_fire_opportunity_bank),
        ("footprint", model._local_fire_footprint),
        ("annual_closure", model._annual_regime_closure),
        ("multi_path_bank", model._multi_pathway_opportunity_bank),
        ("fuel_recovery", model._pathway_fuel_recovery_reservoir),
        ("secondary_litter", model._secondary_fuel_litter_banks),
        ("fragmentation", model._fragmented_managed_recurrence_brake),
    )
    for label, function in functions:
        prediction = function(prediction, data, p, enabled)
        stages.append((label, np.asarray(prediction, dtype=np.float32).copy()))
    return data, stages


def cycle_metrics(prediction, observed, area, cell_weight):
    prediction = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-8)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-8)
    alloc_rmse = float(
        np.sqrt(
            np.average(
                np.mean(np.square(pred_alloc - obs_alloc), axis=0),
                weights=cell_weight,
            )
        )
    )
    annual_log_rmse = float(
        np.sqrt(
            np.average(
                np.square(np.log((pred_annual + 1e-5) / (obs_annual + 1e-5))),
                weights=cell_weight,
            )
        )
    )
    pred_month = np.sum(pred_cycle * area[None, :], axis=1)
    obs_month = np.sum(obs_cycle * area[None, :], axis=1)
    pred_norm = pred_month / pred_month.sum()
    obs_norm = obs_month / obs_month.sum()
    l1 = 0.5 * float(np.abs(pred_norm - obs_norm).sum())
    ratio = float(pred_annual @ area / (obs_annual @ area))
    return {
        "alloc_rmse": alloc_rmse,
        "annual_log_rmse": annual_log_rmse,
        "l1": l1,
        "ratio": ratio,
        "peak": MONTHS[int(np.argmax(pred_month))],
        "obs_peak": MONTHS[int(np.argmax(obs_month))],
    }


def build_features(data, incumbent):
    def field(name):
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    rain3, rain6, rain12 = (antecedent(rain, months) for months in (3.0, 6.0, 12.0))
    dryness = np.clip(field("dryness"), 0.0, None)
    dryness3 = antecedent(dryness, 3.0)
    temperature = field("air_temperature")
    temperature3 = antecedent(temperature, 3.0)
    temperature12 = antecedent(temperature, 12.0)
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    lightning3 = antecedent(lightning, 3.0)
    lightning12 = antecedent(lightning, 12.0)
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    hazard12 = antecedent(hazard, 12.0)
    features = {
        "rain": rain,
        "rain3": rain3,
        "rain6": rain6,
        "rain12": rain12,
        "annual_rain": 12.0 * rain12,
        "rain_deficit6": (rain6 - rain) / (rain6 + rain + 10.0),
        "rain_deficit12": (rain12 - rain) / (rain12 + rain + 10.0),
        "wet_departure": (rain - rain12) / (rain + rain12 + 10.0),
        "dryness": dryness,
        "dryness3": dryness3,
        "dryness_departure": (dryness - dryness3) / (dryness + dryness3 + 100.0),
        "temperature": temperature,
        "temperature3": temperature3,
        "temperature12": temperature12,
        "warming3": temperature - temperature3,
        "warming12": temperature - temperature12,
        "lightning": lightning,
        "lightning3": lightning3,
        "lightning12": lightning12,
        "lightning_departure": (lightning - lightning3) / (lightning + lightning3 + 0.002),
        "incumbent_hazard": hazard,
        "hazard12": hazard12,
    }
    for name in (
        "luh2_primary_fraction",
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
    ):
        features[name] = field(name)
    names = tuple(features)
    matrix = np.column_stack([features[name].reshape(-1) for name in names]).astype(np.float32)
    return names, matrix


def tree_pairs(model, names):
    pairs = Counter()
    for estimator in model.estimators_:
        tree = estimator.tree_
        root = int(tree.feature[0])
        if root < 0:
            continue
        for child in (int(tree.children_left[0]), int(tree.children_right[0])):
            feature = int(tree.feature[child]) if child >= 0 else -1
            if feature >= 0:
                pairs[tuple(sorted((names[root], names[feature])))] += 1
    return pairs


def oof_fit(names, matrix, target, rows, cols, cell_weight, label):
    ntime, ncells = target.shape
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    row_cell = np.tile(np.arange(ncells), ntime)
    target_flat = target.reshape(-1)
    weight_flat = np.tile(cell_weight / np.mean(cell_weight), ntime)
    prediction = np.empty_like(target_flat, dtype=np.float64)
    importances = []
    pair_counts = Counter()
    rng = np.random.default_rng(23051991)
    for fold in range(4):
        train = np.flatnonzero(folds[row_cell] != fold)
        held = np.flatnonzero(folds[row_cell] == fold)
        if train.size > 120000:
            train = rng.choice(train, 120000, replace=False)
        learner = RandomForestRegressor(
            n_estimators=96,
            max_depth=3,
            min_samples_leaf=400,
            max_features=0.8,
            bootstrap=True,
            n_jobs=4,
            random_state=100 + fold,
        )
        learner.fit(matrix[train], target_flat[train], sample_weight=weight_flat[train])
        prediction[held] = learner.predict(matrix[held])
        importances.append(learner.feature_importances_)
        pair_counts.update(tree_pairs(learner, names))
        top = np.argsort(learner.feature_importances_)[::-1][:6]
        print(
            f"ML_FOLD target={label} fold={fold} train={train.size} held={held.size} top="
            + ",".join(f"{names[i]}:{learner.feature_importances_[i]:.4f}" for i in top),
            flush=True,
        )
    mean_importance = np.mean(np.stack(importances), axis=0)
    top = np.argsort(mean_importance)[::-1][:10]
    print(
        f"ML_IMPORTANCE target={label} "
        + " ".join(f"{names[i]}={mean_importance[i]:.6f}" for i in top),
        flush=True,
    )
    print(
        f"ML_PAIRS target={label} "
        + " ".join(f"{left}*{right}={count}" for (left, right), count in pair_counts.most_common(10)),
        flush=True,
    )
    return prediction.reshape(ntime, ncells), mean_importance, pair_counts


def conserve_cell_year_mass(baseline, factor):
    candidate = np.asarray(baseline, dtype=np.float64) * factor
    output = candidate.reshape(16, 12, -1)
    base = np.asarray(baseline, dtype=np.float64).reshape(16, 12, -1)
    days = MONTH_DAYS.reshape(16, 12)[:, :, None]
    base_mass = np.sum(base * days, axis=1, keepdims=True)
    new_mass = np.sum(output * days, axis=1, keepdims=True)
    output *= base_mass / (new_mass + 1e-12)
    return np.clip(output.reshape(baseline.shape), 0.0, 1.0)


def partial_grid(matrix, target, weight, names, pair):
    left, right = (names.index(pair[0]), names.index(pair[1]))
    x, y = matrix[:, left], matrix[:, right]
    edges_x = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 5)))
    edges_y = np.unique(np.quantile(y, np.linspace(0.0, 1.0, 5)))
    if edges_x.size < 5 or edges_y.size < 5:
        return
    grid = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            selected = (
                (x >= edges_x[i])
                & (x <= edges_x[i + 1] if i == 3 else x < edges_x[i + 1])
                & (y >= edges_y[j])
                & (y <= edges_y[j + 1] if j == 3 else y < edges_y[j + 1])
            )
            if np.any(selected):
                grid[i, j] = np.average(target[selected], weights=weight[selected])
    print(
        f"PARTIAL pair={pair[0]}*{pair[1]} x_edges={np.array2string(edges_x, precision=4)} "
        f"y_edges={np.array2string(edges_y, precision=4)}",
        flush=True,
    )
    print(np.array2string(grid, precision=5), flush=True)


def main() -> int:
    mask = clean_warm_open_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, cell_weight, regime_share, retained_regime, retained_global, regime_cells = select_regime_cells(evaluator, mask)
    print(
        f"DESIGN regime_cells={regime_cells} selected={rows.size} "
        f"regime_obs_share={regime_share:.8f} retained_regime={retained_regime:.8f} "
        f"retained_global={retained_global:.8f}",
        flush=True,
    )
    model = load_model()
    raw_data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    prepared, stages = stage_predictions(model, raw_data)
    incumbent = stages[-1][1][:, 0, :]
    direct = np.asarray(model.predict(raw_data, dict(model.PARAMS), None), dtype=np.float32)[:, 0, :]
    print(f"STAGE_REPRO max_abs={float(np.max(np.abs(incumbent - direct))):.12g}", flush=True)
    del direct

    for label, values in stages:
        current = cycle_metrics(values[:, 0, :], observed, area, cell_weight)
        print(
            f"STAGE {label} ratio={current['ratio']:.6f} annual_log_rmse={current['annual_log_rmse']:.6f} "
            f"alloc_rmse={current['alloc_rmse']:.6f} l1={current['l1']:.6f} "
            f"peak={current['peak']} obs_peak={current['obs_peak']}",
            flush=True,
        )

    names, matrix = build_features(prepared, incumbent)
    pred_cycle = incumbent.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-8)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-8)
    cycle_target = np.tile(
        np.log((obs_alloc + 1e-4) / (pred_alloc + 1e-4)),
        (16, 1, 1),
    ).reshape(192, -1)
    pred_annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    annual_cell_target = np.log((obs_annual + 1e-5) / (pred_annual + 1e-5))
    annual_target = np.broadcast_to(annual_cell_target[None, :], incumbent.shape)

    cycle_hat, _, cycle_pairs = oof_fit(
        names, matrix, cycle_target, rows, cols, cell_weight, "cycle"
    )
    annual_hat_monthly, _, annual_pairs = oof_fit(
        names, matrix, annual_target, rows, cols, cell_weight, "annual"
    )
    baseline = cycle_metrics(incumbent, observed, area, cell_weight)
    print(
        f"OOF_BASE ratio={baseline['ratio']:.6f} annual_log_rmse={baseline['annual_log_rmse']:.6f} "
        f"alloc_rmse={baseline['alloc_rmse']:.6f} l1={baseline['l1']:.6f} peak={baseline['peak']}",
        flush=True,
    )
    for blend in (0.25, 0.5, 1.0):
        cycle_candidate = conserve_cell_year_mass(
            incumbent,
            np.exp(np.clip(blend * cycle_hat, -1.5, 1.5)),
        )
        values = cycle_metrics(cycle_candidate, observed, area, cell_weight)
        mass_error = float(
            np.max(
                np.abs(
                    np.sum(cycle_candidate.reshape(16, 12, -1) * MONTH_DAYS.reshape(16, 12)[:, :, None], axis=1)
                    - np.sum(incumbent.reshape(16, 12, -1) * MONTH_DAYS.reshape(16, 12)[:, :, None], axis=1)
                )
            )
        )
        print(
            f"OOF_CYCLE blend={blend:g} ratio={values['ratio']:.6f} "
            f"annual_log_rmse={values['annual_log_rmse']:.6f} alloc_rmse={values['alloc_rmse']:.6f} "
            f"l1={values['l1']:.6f} peak={values['peak']} mass_max_abs={mass_error:.12g}",
            flush=True,
        )

    annual_hat = np.mean(annual_hat_monthly, axis=0)
    for blend in (0.25, 0.5, 1.0):
        annual_candidate = np.clip(
            incumbent * np.exp(np.clip(blend * annual_hat[None, :], -1.5, 1.5)),
            0.0,
            1.0,
        )
        values = cycle_metrics(annual_candidate, observed, area, cell_weight)
        print(
            f"OOF_ANNUAL blend={blend:g} ratio={values['ratio']:.6f} "
            f"annual_log_rmse={values['annual_log_rmse']:.6f} alloc_rmse={values['alloc_rmse']:.6f} "
            f"l1={values['l1']:.6f} peak={values['peak']}",
            flush=True,
        )

    row_weight = np.tile(cell_weight / np.mean(cell_weight), 192)
    if cycle_pairs:
        partial_grid(matrix, cycle_target.reshape(-1), row_weight, names, cycle_pairs.most_common(1)[0][0])
    if annual_pairs:
        partial_grid(matrix, annual_target.reshape(-1), row_weight, names, annual_pairs.most_common(1)[0][0])
    del stages, matrix, raw_data, prepared
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
