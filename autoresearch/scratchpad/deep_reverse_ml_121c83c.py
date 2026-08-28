"""Deeper held-block reverse-ML diagnostic for the 121c83c incumbent.

This is diagnosis, not a candidate model.  The two depth-four learners predict
annual-mean and normalized-monthly-cycle residuals separately.  Every feature
is either a coupled-valid ``model.INPUTS`` value, the incumbent fire
opportunity, or a point-local current/prefix-causal state built from them.
Coordinates are used only to assign whole cells to spatial folds.  No region,
cell identity, future summary, or benchmark-derived value is a feature, and no
learned prediction or coefficient is eligible for ``model.py``.
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "121c83c"
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    """Causal exponential state initialized from the first coupled timestep."""
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(values[0], dtype=np.float32).copy()
    result = np.empty_like(values, dtype=np.float32)
    for step in range(values.shape[0]):
        state += alpha * (values[step] - state)
        result[step] = state
    return result


def rising(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values - center) / scale, -50.0, 50.0)))


def tree_counts(regressor, names: tuple[str, ...]):
    features = Counter()
    pairs = Counter()
    for stage in regressor._predictors:
        nodes = stage[0].nodes
        stack = [0]
        while stack:
            node = stack.pop()
            if nodes["is_leaf"][node]:
                continue
            feature = names[int(nodes["feature_idx"][node])]
            features[feature] += 1
            for child_name in ("left", "right"):
                child = int(nodes[child_name][node])
                if not nodes["is_leaf"][child]:
                    child_feature = names[int(nodes["feature_idx"][child])]
                    if child_feature != feature:
                        pairs[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)
    return features, pairs


def weighted_r2(target, prediction, weight) -> float:
    return float(r2_score(target, prediction, sample_weight=weight))


def build_features(data, opportunity, rows, cols):
    """Return a monthly matrix containing no cross-cell or future information."""
    names: list[str] = []
    columns: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (192, rows.size):
            raise ValueError(f"{name} has unexpected shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} is not finite")
        names.append(name)
        columns.append(values.reshape(-1))

    selected = {
        name: np.asarray(data[name][:, rows, cols], dtype=np.float32)
        for name in data
        if name in model.INPUTS
    }
    rain = np.clip(selected["monthly_precipitation"], 0.0, None)
    dryness = np.clip(selected["dryness"], 0.0, None)
    temperature = selected["air_temperature"]
    gpp = np.clip(selected["gpp"], 0.0, None)
    lightning = np.clip(selected["lightning_flash_rate"], 0.0, None)
    lai = np.clip(selected["leaf_area_index"], 0.0, None)
    fire = np.asarray(opportunity[:, rows, cols], dtype=np.float32)
    hazard = -np.log1p(-np.clip(fire, 0.0, 1.0 - 1e-7))

    rain3, rain12 = antecedent(rain, 3.0), antecedent(rain, 12.0)
    dry3, dry12 = antecedent(dryness, 3.0), antecedent(dryness, 12.0)
    temp3, temp12 = antecedent(temperature, 3.0), antecedent(temperature, 12.0)
    gpp3, gpp12 = antecedent(gpp, 3.0), antecedent(gpp, 12.0)
    light3, light12 = antecedent(lightning, 3.0), antecedent(lightning, 12.0)
    lai3, lai12 = antecedent(lai, 3.0), antecedent(lai, 12.0)
    hazard3, hazard12 = antecedent(hazard, 3.0), antecedent(hazard, 12.0)

    add("log_rain", np.log1p(rain))
    add("log_dryness", np.log1p(dryness))
    add("temperature", temperature)
    add("log_gpp", np.log1p(gpp))
    add("log_lightning", np.log1p(1000.0 * lightning))
    add("log_lai", np.log1p(lai))
    add("log_opportunity", np.log1p(1000.0 * hazard))
    add("log_opportunity_ema3", np.log1p(1000.0 * hazard3))
    add("log_opportunity_ema12", np.log1p(1000.0 * hazard12))

    add("rain_ema12", rain12)
    add("rain_deficit3", np.maximum((rain3 - rain) / (rain3 + rain + 10.0), 0.0))
    add("rain_deficit12", np.maximum((rain12 - rain) / (rain12 + rain + 10.0), 0.0))
    add("rain_wet_anomaly", np.maximum((rain - rain12) / (rain + rain12 + 10.0), 0.0))
    add("dryness_ema12", dry12)
    add("dryness_rise3", np.maximum((dryness - dry3) / (dryness + dry3 + 100.0), 0.0))
    add(
        "dryness_variability12",
        np.sqrt(np.maximum(antecedent(dryness * dryness, 12.0) - dry12 * dry12, 0.0))
        / (dry12 + 1.0),
    )
    add("temperature_ema12", temp12)
    add("warming3", temperature - temp3)
    add(
        "temperature_variability12",
        np.sqrt(np.maximum(antecedent(temperature * temperature, 12.0) - temp12 * temp12, 0.0)),
    )
    add("gpp_ema12", gpp12)
    add("gpp_curing", np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.2), 0.0))
    add("gpp_greenup", np.maximum((gpp - gpp3) / (gpp3 + gpp + 0.2), 0.0))
    add("lightning_ema12", light12)
    add("lightning_pulse3", np.maximum((lightning - light3) / (lightning + light3 + 0.002), 0.0))
    add("lai_ema12", lai12)
    add("lai_senescence", np.maximum((lai3 - lai) / (lai3 + lai + 0.2), 0.0))
    add("lai_greenup", np.maximum((lai - lai3) / (lai3 + lai + 0.2), 0.0))
    add("opportunity_relative", hazard / (hazard12 + 0.002))
    add("opportunity_rise3", np.maximum((hazard - hazard3) / (hazard + hazard3 + 0.002), 0.0))
    add(
        "opportunity_variability12",
        np.sqrt(np.maximum(antecedent(hazard * hazard, 12.0) - hazard12 * hazard12, 0.0))
        / (hazard12 + 0.002),
    )

    for name in (
        "luh2_cropland_fraction",
        "luh2_rangeland_fraction",
        "luh2_primary_fraction",
        "luh2_pasture_fraction",
        "luh2_urban_fraction",
        "natural_vegetation_fraction",
        "secondary_vegetation_fraction",
    ):
        add(name, np.clip(selected[name], 0.0, 1.0))
    for name in (
        "aboveground_biomass",
        "soil_carbon",
        "natural_canopy_height",
        "secondary_canopy_height",
    ):
        add(f"log_{name}", np.log1p(np.clip(selected[name], 0.0, None)))

    crop = np.clip(selected["luh2_cropland_fraction"], 0.0, 1.0)
    rangeland = np.clip(selected["luh2_rangeland_fraction"], 0.0, 1.0)
    pasture = np.clip(selected["luh2_pasture_fraction"], 0.0, 1.0)
    urban = np.clip(selected["luh2_urban_fraction"], 0.0, 1.0)
    natural = np.clip(selected["natural_vegetation_fraction"], 0.0, 1.0)
    secondary = np.clip(selected["secondary_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(selected["natural_canopy_height"], 0.0, None)
    secondary_canopy = np.clip(selected["secondary_canopy_height"], 0.0, None)
    biomass = np.clip(selected["aboveground_biomass"], 0.0, None)
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    fine_fuel = gpp12 / (gpp12 + 0.35)
    open_cover = np.clip(
        rangeland
        + pasture
        + natural * 8.0 / (canopy + 8.0)
        + secondary * 8.0 / (secondary_canopy + 8.0),
        0.0,
        2.0,
    )
    surface = (1.0 - crop) * fine_fuel * open_cover * continuity
    woody = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    total = 0.05 + surface + woody + crop_capacity
    combustion = (
        dryness / (dryness + 250.0)
        / (1.0 + rain / 35.0)
        * rising(temperature, 5.0, 3.0)
    )
    add("fine_fuel", fine_fuel)
    add("continuity", continuity)
    add("surface_share", surface / total)
    add("woody_share", woody / total)
    add("crop_share", crop_capacity / total)
    add("combustion", combustion)
    add("natural_ignition", lightning / (lightning + 0.02) * combustion * (surface + woody) / total)

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return tuple(names), matrix


def losses(prediction, observed, area, cells, folds):
    rows, cols = cells // 360, cells % 360
    pred = np.asarray(prediction[:, rows, cols], dtype=np.float64)
    obs = np.asarray(observed[:, rows, cols], dtype=np.float64)
    obs_ann = np.average(obs, axis=0, weights=MONTH_DAYS)
    pred_ann = np.average(pred, axis=0, weights=MONTH_DAYS)
    weight = area[rows, cols] * obs_ann
    obs_cycle = obs.reshape(16, 12, -1).mean(axis=0)
    pred_cycle = pred.reshape(16, 12, -1).mean(axis=0)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-12)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-12)
    annual = []
    cycle = []
    for fold in range(4):
        held = folds == fold
        annual.append(
            np.sqrt(
                np.sum(weight[held] * np.square(np.log(obs_ann[held] + 1e-5) - np.log(pred_ann[held] + 1e-5)))
                / (np.sum(weight[held]) + 1e-15)
            )
        )
        cycle.append(
            np.sum(weight[held][None, :] * np.abs(obs_alloc[:, held] - pred_alloc[:, held]))
            / (np.sum(weight[held]) + 1e-15)
        )
    return np.asarray(annual), np.asarray(cycle)


def apply_correction(prediction, residual, strength):
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    factor = np.exp(np.clip(strength * residual, -1.0, 1.0))
    return np.asarray(1.0 - np.exp(-np.clip(hazard * factor, 0.0, 50.0)), dtype=np.float32)


def partial_interactions(label, matrix, weight, folds, names, models, pair_counts):
    common = set(pair_counts[0])
    for counts in pair_counts[1:]:
        common &= set(counts)
    ranked = sorted(common, key=lambda pair: sum(counts[pair] for counts in pair_counts), reverse=True)
    print(f"{label}_STABLE_INTERACTIONS count={len(ranked)}")
    rng = np.random.default_rng(121083)
    for pair in ranked[:20]:
        left, right = names.index(pair[0]), names.index(pair[1])
        contrasts = []
        left_effects = []
        right_effects = []
        corner_effects = []
        grids = []
        for fold, learner in enumerate(models):
            train = folds != fold
            held_index = np.flatnonzero(folds == fold)
            probability = weight[held_index].astype(np.float64)
            probability /= probability.sum()
            sample_index = rng.choice(held_index, size=min(8000, held_index.size), replace=False, p=probability)
            sample = matrix[sample_index].copy()
            left_low, left_high = np.quantile(matrix[train, left], (0.25, 0.75))
            right_low, right_high = np.quantile(matrix[train, right], (0.25, 0.75))
            predictions = {}
            for left_key, left_value in (("l", left_low), ("h", left_high)):
                for right_key, right_value in (("l", right_low), ("h", right_high)):
                    probe = sample.copy()
                    probe[:, left] = left_value
                    probe[:, right] = right_value
                    predictions[left_key + right_key] = learner.predict(probe)
            contrasts.append(float(np.mean(predictions["hh"] - predictions["hl"] - predictions["lh"] + predictions["ll"])))
            left_effects.append(float(np.mean(0.5 * (predictions["hl"] + predictions["hh"] - predictions["ll"] - predictions["lh"]))))
            right_effects.append(float(np.mean(0.5 * (predictions["lh"] + predictions["hh"] - predictions["ll"] - predictions["hl"]))))
            center = float(np.mean(predictions["ll"]))
            corner_effects.append(tuple(float(np.mean(predictions[key])) - center for key in ("ll", "lh", "hl", "hh")))
            grids.append((left_low, left_high, right_low, right_high))
        stable_sign = min(contrasts) > 0.0 or max(contrasts) < 0.0
        print(
            f"PAIR {pair[0]}*{pair[1]} counts={','.join(str(counts[pair]) for counts in pair_counts)} "
            f"interaction={','.join(f'{value:+.5f}' for value in contrasts)} stable_sign={stable_sign} "
            f"left_effect={','.join(f'{value:+.5f}' for value in left_effects)} "
            f"right_effect={','.join(f'{value:+.5f}' for value in right_effects)} "
            f"corners_ll_lh_hl_hh={';'.join(','.join(f'{value:+.5f}' for value in corner) for corner in corner_effects)} "
            f"q25q75={';'.join(','.join(f'{value:.5g}' for value in grid) for grid in grids)}"
        )


def main() -> None:
    global model
    model = load_pinned()
    current_blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    pinned_blob = subprocess.run(
        ["git", "rev-parse", f"{PINNED}:autoresearch/model.py"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if current_blob != pinned_blob:
        raise RuntimeError(f"current model blob {current_blob} differs from {PINNED} blob {pinned_blob}")

    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_ann = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_ann = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    obs_weight = area * obs_ann
    excess_weight = area * np.maximum(pred_ann - obs_ann, 0.0)

    def top(weight, fraction):
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, fraction) + 1)]

    cells = np.union1d(top(obs_weight, 0.90), top(excess_weight, 0.90))
    rows, cols = cells // 360, cells % 360
    cell_folds = ((rows // 15) + 3 * (cols // 15)) % 4
    row_folds = np.tile(cell_folds, 192)
    selected_obs = np.asarray(observed[:, rows, cols], dtype=np.float32)
    selected_pred = np.asarray(incumbent[:, rows, cols], dtype=np.float32)
    names, matrix = build_features(data, incumbent, rows, cols)
    print(
        f"IDENTITY pinned={PINNED} model_blob={current_blob} cells={cells.size} rows={matrix.shape[0]} "
        f"features={matrix.shape[1]} observed_coverage={obs_weight.ravel()[cells].sum()/obs_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f} "
        f"fold_cells={','.join(str(int(np.sum(cell_folds == fold))) for fold in range(4))}"
    )

    eps = 1e-5
    annual_target_cell = np.clip(
        np.log((obs_ann[rows, cols] + eps) / (pred_ann[rows, cols] + eps)), -3.0, 3.0
    ).astype(np.float32)
    annual_target = np.tile(annual_target_cell, 192)
    obs_cycle = selected_obs.reshape(16, 12, -1).mean(axis=0)
    pred_cycle = selected_pred.reshape(16, 12, -1).mean(axis=0)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + eps)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + eps)
    cycle_target = np.tile(
        np.clip(np.log((obs_alloc + eps) / (pred_alloc + eps)), -3.0, 3.0), (16, 1, 1)
    ).reshape(-1).astype(np.float32)
    annual_cell_weight = area[rows, cols] * (
        obs_ann[rows, cols] + np.maximum(pred_ann[rows, cols] - obs_ann[rows, cols], 0.0)
    )
    annual_weight = np.tile(annual_cell_weight, 192).astype(np.float64)
    cycle_month_weight = np.broadcast_to(
        area[rows, cols][None, :] * obs_ann[rows, cols][None, :] * np.maximum(obs_alloc, 0.002),
        (12, cells.size),
    )
    cycle_weight = np.tile(cycle_month_weight, (16, 1, 1)).reshape(-1).astype(np.float64)
    annual_weight /= annual_weight.mean()
    cycle_weight /= cycle_weight.mean()

    base_annual, base_cycle = losses(incumbent, observed, area, cells, cell_folds)
    print(
        "BASE_LOSS annual=" + ",".join(f"{value:.7f}" for value in base_annual)
        + " cycle=" + ",".join(f"{value:.7f}" for value in base_cycle)
    )
    for target_name, target, weight in (
        ("ANNUAL", annual_target, annual_weight),
        ("CYCLE", cycle_target, cycle_weight),
    ):
        oof = np.empty_like(target, dtype=np.float32)
        models = []
        feature_counts = []
        pair_counts = []
        for fold in range(4):
            train = row_folds != fold
            held = ~train
            learner = HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=180,
                learning_rate=0.05,
                min_samples_leaf=250,
                l2_regularization=3.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=121083 + 31 * fold + (0 if target_name == "ANNUAL" else 1000),
            )
            learner.fit(matrix[train], target[train], sample_weight=weight[train])
            oof[held] = learner.predict(matrix[held]).astype(np.float32)
            features, pairs = tree_counts(learner, names)
            models.append(learner)
            feature_counts.append(features)
            pair_counts.append(pairs)
            print(
                f"{target_name}_FOLD fold={fold} iterations={learner.n_iter_} "
                f"r2={weighted_r2(target[held], oof[held], weight[held]):.7f} "
                f"top_features={','.join(name for name, _ in features.most_common(8))}"
            )
        print(f"{target_name}_OOF_R2 value={weighted_r2(target, oof, weight):.7f}")
        partial_interactions(target_name, matrix, weight, row_folds, names, models, pair_counts)
        residual = oof.reshape(192, cells.size)
        for strength in (0.10, 0.25, 0.50, 1.0):
            corrected_selected = apply_correction(selected_pred, residual, strength)
            trial = incumbent.copy()
            trial[:, rows, cols] = corrected_selected
            annual_loss, cycle_loss = losses(trial, observed, area, cells, cell_folds)
            annual_gain = base_annual - annual_loss
            cycle_gain = base_cycle - cycle_loss
            score = evaluator.score(validate_prediction(trial))["global"]
            print(
                f"{target_name}_HEADROOM strength={strength:g} annual_gain="
                + ",".join(f"{value:+.7f}" for value in annual_gain)
                + " cycle_gain=" + ",".join(f"{value:+.7f}" for value in cycle_gain)
                + f" annual_all={bool(np.all(annual_gain > 0.0))} cycle_all={bool(np.all(cycle_gain > 0.0))} "
                + f"overall={score['overall_score']:.9f} bias={score['bias_score']:.9f} "
                + f"rmse={score['rmse_score']:.9f} seasonal={score['seasonal_cycle_score']:.9f} "
                + f"spatial={score['spatial_distribution_score']:.9f}"
            )


if __name__ == "__main__":
    main()
