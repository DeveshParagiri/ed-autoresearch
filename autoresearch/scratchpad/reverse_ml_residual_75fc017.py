"""Held-whole-cell reverse-ML diagnostic on mechanistic incumbent 75fc017.

Annual and normalized-cycle learners are separate shallow trees. Features use
only valid inputs plus site-local current or prefix-causal summaries. The target
is never available to a candidate equation; coordinates assign folds only and
are absent from the feature matrix. Incumbent prediction is excluded as a
feature. No learned surface is canonical or officially evaluated.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "75fc017"
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
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def split_structure(regressor, names):
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
            for key in ("left", "right"):
                child = int(nodes[key][node])
                if not nodes["is_leaf"][child]:
                    child_feature = names[int(nodes["feature_idx"][child])]
                    pairs[tuple(sorted((feature, child_feature)))] += 1
                stack.append(child)
    return features, pairs


def fit_oof(X, y, weights, folds, names, min_leaf):
    prediction = np.zeros_like(y, dtype=np.float64)
    models = []
    feature_counts = []
    pair_counts = []
    for fold in range(4):
        train = folds != fold
        held = ~train
        regressor = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=min_leaf,
            l2_regularization=2.0,
            random_state=1701 + fold,
        )
        regressor.fit(X[train], y[train], sample_weight=weights[train])
        prediction[held] = regressor.predict(X[held])
        features, pairs = split_structure(regressor, names)
        models.append(regressor)
        feature_counts.append(features)
        pair_counts.append(pairs)
    return prediction, models, feature_counts, pair_counts


def print_stability(label, X, weights, folds, names, models, feature_counts, pair_counts):
    stable_features = set(names)
    stable_pairs = set(pair_counts[0])
    for counts in feature_counts:
        stable_features &= set(counts)
    for counts in pair_counts[1:]:
        stable_pairs &= set(counts)
    ranked_features = sorted(
        stable_features,
        key=lambda name: sum(counts[name] for counts in feature_counts),
        reverse=True,
    )
    ranked_pairs = sorted(
        stable_pairs,
        key=lambda pair: sum(counts[pair] for counts in pair_counts),
        reverse=True,
    )
    print(f"{label}_STABLE_FEATURES")
    for name in ranked_features[:15]:
        index = names.index(name)
        partial = []
        for fold, model in enumerate(models):
            train = folds != fold
            held = ~train
            low, high = np.quantile(X[train, index], (0.25, 0.75))
            probe_low = X[held].copy()
            probe_high = X[held].copy()
            probe_low[:, index] = low
            probe_high[:, index] = high
            delta = model.predict(probe_high) - model.predict(probe_low)
            partial.append(
                float(np.average(delta, weights=np.maximum(weights[held], 1e-12)))
            )
        print(
            f"{name} splits={','.join(str(count[name]) for count in feature_counts)} "
            f"partial={','.join(f'{value:+.4f}' for value in partial)}"
        )
    print(f"{label}_STABLE_INTERACTIONS")
    for pair in ranked_pairs[:15]:
        print(
            f"{pair[0]}*{pair[1]} counts="
            + ",".join(str(count[pair]) for count in pair_counts)
        )


def score_patch(evaluator, incumbent, rows, cols, corrected):
    trial = incumbent.copy()
    trial[:, rows, cols] = np.clip(corrected, 0.0, 1.0)
    return evaluator.score(validate_prediction(trial))["global"]


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    base = evaluator.score(incumbent)["global"]
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_annual = np.average(incumbent, axis=0, weights=MONTH_DAYS)
    observed_weight = area * obs_annual
    excess_weight = area * np.maximum(pred_annual - obs_annual, 0.0)

    def top(weight, fraction):
        order = np.argsort(weight.ravel())[::-1]
        cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
        return order[: int(np.searchsorted(cumulative, fraction) + 1)]

    cells = np.union1d(top(observed_weight, 0.90), top(excess_weight, 0.90))
    rows, cols = cells // 360, cells % 360
    count = len(cells)
    folds_cell = ((rows // 15) + 3 * (cols // 15)) % 4
    selected = {
        name: np.asarray(data[name][:, rows, cols], dtype=np.float64)
        for name in model.INPUTS
    }
    pred = np.asarray(incumbent[:, rows, cols], dtype=np.float64)
    obs = np.asarray(observed[:, rows, cols], dtype=np.float64)
    area_cell = area[rows, cols]
    obs_ann = obs_annual[rows, cols]
    pred_ann = pred_annual[rows, cols]
    print(
        f"BASE overall={base['overall_score']:.9f} cells={count} "
        f"observed_coverage={observed_weight.ravel()[cells].sum()/observed_weight.sum():.6f} "
        f"excess_coverage={excess_weight.ravel()[cells].sum()/excess_weight.sum():.6f}"
    )

    dynamic = {}
    for name in model.INPUTS:
        dynamic[name] = selected[name]
    rain = np.clip(selected["monthly_precipitation"], 0.0, None)
    gpp = np.clip(selected["gpp"], 0.0, None)
    dryness = np.clip(selected["dryness"], 0.0, None)
    temperature = selected["air_temperature"]
    lightning = np.clip(selected["lightning_flash_rate"], 0.0, None)
    rain3, rain6, rain12 = antecedent(rain, 3), antecedent(rain, 6), antecedent(rain, 12)
    gpp3, gpp12 = antecedent(gpp, 3), antecedent(gpp, 12)
    dry12 = antecedent(dryness, 12)
    temp3, temp12 = antecedent(temperature, 3), antecedent(temperature, 12)
    light3, light12 = antecedent(lightning, 3), antecedent(lightning, 12)
    dynamic.update(
        {
            "annual_rain_causal": 12.0 * rain12,
            "rain_deficit_3": np.maximum((rain3-rain)/(rain3+rain+10), 0),
            "rain_deficit_6": np.maximum((rain6-rain)/(rain6+rain+10), 0),
            "rain_deficit_12": np.maximum((rain12-rain)/(rain12+rain+10), 0),
            "rain_wet_anomaly": np.maximum((rain-rain12)/(rain+rain12+10), 0),
            "rain_variability": np.sqrt(np.maximum(antecedent(rain*rain,12)-rain12*rain12,0)),
            "gpp_12": gpp12,
            "gpp_curing": np.maximum((gpp3-gpp)/(gpp3+gpp+0.2), 0),
            "gpp_greenup": np.maximum((gpp-gpp3)/(gpp3+gpp+0.2), 0),
            "dryness_12": dry12,
            "dryness_variability": np.sqrt(np.maximum(antecedent(dryness*dryness,12)-dry12*dry12,0))/(dry12+1),
            "temperature_12": temp12,
            "warming_3": temperature-temp3,
            "temperature_variability": np.sqrt(np.maximum(antecedent(temperature*temperature,12)-temp12*temp12,0)),
            "lightning_12": light12,
            "lightning_pulse": np.maximum((lightning-light3)/(lightning+light3+0.002),0),
        }
    )
    crop = np.clip(selected["luh2_cropland_fraction"], 0, 1)
    range_ = np.clip(selected["luh2_rangeland_fraction"], 0, 1)
    pasture = np.clip(selected["luh2_pasture_fraction"], 0, 1)
    urban = np.clip(selected["luh2_urban_fraction"], 0, 1)
    natural = np.clip(selected["natural_vegetation_fraction"], 0, 1)
    secondary = np.clip(selected["secondary_vegetation_fraction"], 0, 1)
    canopy = np.clip(selected["natural_canopy_height"], 0, None)
    secondary_canopy = np.clip(selected["secondary_canopy_height"], 0, None)
    biomass = np.clip(selected["aboveground_biomass"], 0, None)
    continuity = 1/(1+2*crop**1.5+5*urban)
    fine_fuel = gpp12/(gpp12+0.35)
    open_cover = np.clip(range_+pasture+natural*8/(canopy+8)+secondary*8/(secondary_canopy+8),0,2)
    surface = (1-crop)*fine_fuel*open_cover*continuity
    woody = (natural*canopy/(canopy+8)+secondary*secondary_canopy/(secondary_canopy+8))*biomass/(biomass+1)
    crop_capacity = crop*fine_fuel
    total = 0.05+surface+woody+crop_capacity
    dynamic.update(
        {
            "fine_fuel": fine_fuel,
            "continuity": continuity,
            "surface_share": surface/total,
            "woody_share": woody/total,
            "crop_share": crop_capacity/total,
            "secondary_open": secondary*8/(secondary_canopy+8),
        }
    )

    names = list(dynamic)
    annual_X = np.column_stack([dynamic[name].mean(axis=0) for name in names]).astype(np.float32)
    cycle_X = np.column_stack(
        [dynamic[name].reshape(16,12,count).mean(axis=0).reshape(-1) for name in names]
    ).astype(np.float32)
    eps = 1e-5
    annual_y = np.clip(np.log(obs_ann+eps)-np.log(pred_ann+eps), -3, 3)
    annual_weight = area_cell*(obs_ann+np.maximum(pred_ann-obs_ann,0))
    obs_cycle = obs.reshape(16,12,count).mean(axis=0)
    pred_cycle = pred.reshape(16,12,count).mean(axis=0)
    obs_alloc = obs_cycle/(obs_cycle.sum(axis=0,keepdims=True)+eps)
    pred_alloc = pred_cycle/(pred_cycle.sum(axis=0,keepdims=True)+eps)
    cycle_y = np.clip(np.log(obs_alloc+eps)-np.log(pred_alloc+eps),-3,3).reshape(-1)
    cycle_weight = (np.broadcast_to(area_cell*obs_ann,(12,count))*np.maximum(obs_alloc,0.002)).reshape(-1)
    cycle_folds = np.tile(folds_cell, 12)

    annual_oof, annual_models, annual_features, annual_pairs = fit_oof(
        annual_X, annual_y, annual_weight, folds_cell, names, 35
    )
    cycle_oof, cycle_models, cycle_features, cycle_pairs = fit_oof(
        cycle_X, cycle_y, cycle_weight, cycle_folds, names, 100
    )
    print_stability("ANNUAL",annual_X,annual_weight,folds_cell,names,annual_models,annual_features,annual_pairs)
    print_stability("CYCLE",cycle_X,cycle_weight,cycle_folds,names,cycle_models,cycle_features,cycle_pairs)

    cycle_oof = cycle_oof.reshape(12,count)
    for blend in (0.1,0.25,0.5):
        annual_corrected = pred*np.exp(np.clip(blend*annual_oof[None,:],-1,1))
        annual_score = score_patch(evaluator,incumbent,rows,cols,annual_corrected)
        factor = np.exp(np.clip(blend*cycle_oof,-1,1))
        cycle_corrected = pred.copy()
        for year in range(16):
            block = pred[year*12:(year+1)*12]*factor
            block *= pred[year*12:(year+1)*12].sum(axis=0,keepdims=True)/(block.sum(axis=0,keepdims=True)+1e-12)
            cycle_corrected[year*12:(year+1)*12]=block
        cycle_score = score_patch(evaluator,incumbent,rows,cols,cycle_corrected)
        combined = annual_corrected.copy()
        for year in range(16):
            block = annual_corrected[year*12:(year+1)*12]*factor
            block *= annual_corrected[year*12:(year+1)*12].sum(axis=0,keepdims=True)/(block.sum(axis=0,keepdims=True)+1e-12)
            combined[year*12:(year+1)*12]=block
        combined_score=score_patch(evaluator,incumbent,rows,cols,combined)
        print(f"OOF blend={blend:g} annual={annual_score['overall_score']:.9f} cycle={cycle_score['overall_score']:.9f} combined={combined_score['overall_score']:.9f}")


if __name__ == "__main__":
    main()
