"""Pinned BOAS spatial-residual diagnosis for the exact 0.718882578 incumbent.

The learner is scratch-only and uses no coordinates, regions, neighbours, or
benchmark-derived runtime state.  BOAS and ecological masks are diagnostics.
The script writes only an untracked float32 prediction cache; it never edits
``model.py``, runs the official evaluator, or records the result ledger.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


MODEL_BLOB = "c56b96a1cbd57e4342b14f4cc13ea541830703e7"
EXPECTED_OVERALL = 0.718882578
CACHE = ROOT / "autoresearch/scratchpad/canonical_c56b96a_chunked.npy"


def load_model():
    source = subprocess.check_output(
        ("git", "cat-file", "blob", MODEL_BLOB), cwd=ROOT
    )
    module = types.ModuleType("ed_fire_c56b96a")
    module.__file__ = f"git-blob:{MODEL_BLOB}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def build_cache(model, land: np.ndarray) -> np.ndarray:
    if CACHE.exists():
        result = np.load(CACHE, mmap_mode="r")
        if result.shape != (192, 180, 360) or result.dtype != np.float32:
            raise ValueError(f"bad incumbent cache: {result.shape} {result.dtype}")
        print(f"CACHE reuse={CACHE} bytes={CACHE.stat().st_size}", flush=True)
        return result

    output = np.zeros((192, 180, 360), dtype=np.float32)
    rows, columns = np.nonzero(land)
    for start in range(0, rows.size, 1536):
        stop = min(start + 1536, rows.size)
        data = {
            name: selected_input(name, rows[start:stop], columns[start:stop])
            for name in model.INPUTS
        }
        prediction = np.asarray(
            model.predict(data, dict(model.PARAMS), None), dtype=np.float32
        )[:, 0, :]
        output[:, rows[start:stop], columns[start:stop]] = prediction
        print(f"CACHE_CHUNK {start}:{stop}/{rows.size}", flush=True)
        del data, prediction
        gc.collect()
    np.save(CACHE, output, allow_pickle=False)
    return np.load(CACHE, mmap_mode="r")


def weighted_mae(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.abs(values) * weights) / np.sum(weights))


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weight = weights / np.sum(weights)
    x_centered = x - np.sum(weight * x)
    y_centered = y - np.sum(weight * y)
    covariance = np.sum(weight * x_centered * y_centered)
    variance = np.sum(weight * x_centered * x_centered) * np.sum(
        weight * y_centered * y_centered
    )
    return float(covariance / np.sqrt(max(float(variance), 1e-30)))


def main() -> int:
    model = load_model()
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_grid = build_cache(model, land)
    scores = evaluator.score(incumbent_grid)
    overall = float(scores["global"]["overall_score"])
    if abs(overall - EXPECTED_OVERALL) > 5e-8:
        raise RuntimeError(f"incumbent mismatch: {overall:.12f}")
    print(
        "BASE "
        f"overall={overall:.9f} boas_overall={scores['boas']['overall_score']:.9f} "
        f"boas_spatial={scores['boas']['spatial_distribution_score']:.9f} "
        f"boas_bias={scores['boas']['bias_score']:.9f} "
        f"boas_rmse={scores['boas']['rmse_score']:.9f} "
        f"boas_seasonal={scores['boas']['seasonal_cycle_score']:.9f}",
        flush=True,
    )

    inside_boas = ~evaluator.regions["boas"].reshape(180, 2, 360, 2).all(
        axis=(1, 3)
    )
    rows, columns = np.nonzero(land & inside_boas)
    incumbent = np.asarray(incumbent_grid[:, rows, columns], dtype=np.float32)
    observation_grid = load_observation()
    observation = np.asarray(
        observation_grid[:, rows, columns], dtype=np.float32
    )
    del observation_grid
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    pred_annual = incumbent.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    ratio = float(np.sum(pred_annual * area)) / float(np.sum(obs_annual * area))
    print(
        f"BOAS cells={rows.size} area_ratio={ratio:.9f} "
        f"raw_corr={np.corrcoef(pred_annual, obs_annual)[0, 1]:+.9f} "
        f"pred_cv={np.std(pred_annual) / max(np.mean(pred_annual), 1e-12):.6f} "
        f"obs_cv={np.std(obs_annual) / max(np.mean(obs_annual), 1e-12):.6f}",
        flush=True,
    )

    dynamic = {
        "temperature": "air_temperature",
        "rain": "monthly_precipitation",
        "dryness": "dryness",
        "gpp": "gpp",
        "lightning": "lightning_flash_rate",
    }
    static = {
        "soil_carbon": "soil_carbon",
        "biomass": "aboveground_biomass",
        "lai": "leaf_area_index",
        "canopy": "natural_canopy_height",
        "secondary_canopy": "secondary_canopy_height",
        "natural": "natural_vegetation_fraction",
        "secondary": "secondary_vegetation_fraction",
        "primary": "luh2_primary_fraction",
        "crop": "luh2_cropland_fraction",
        "pasture": "luh2_pasture_fraction",
        "rangeland": "luh2_rangeland_fraction",
    }
    features: dict[str, np.ndarray] = {}
    monthly: dict[str, np.ndarray] = {}
    for short, source in {**dynamic, **static}.items():
        values = np.asarray(selected_input(source, rows, columns)[:, 0, :])
        monthly[short] = values
        features[f"mean:{short}"] = values.mean(axis=0)
        if short in dynamic:
            features[f"std:{short}"] = values.std(axis=0)

    temperature = monthly["temperature"]
    rain = np.clip(monthly["rain"], 0.0, None)
    dryness = np.clip(monthly["dryness"], 0.0, None)
    lightning = np.clip(monthly["lightning"], 0.0, None)
    soil = np.clip(features["mean:soil_carbon"], 0.0, None)
    natural = np.clip(features["mean:natural"], 0.0, 1.0)
    canopy = np.clip(features["mean:canopy"], 0.0, None)
    cold = 1.0 / (1.0 + np.exp(np.clip((features["mean:temperature"] - 5.0) / 3.0, -30.0, 30.0)))
    forest = natural * canopy / (canopy + 10.0)
    organic = soil / (soil + 4.0)
    ignition = features["mean:lightning"] / (features["mean:lightning"] + 0.01)
    thaw_months = np.mean(
        (1.0 / (1.0 + np.exp(np.clip(-(temperature - 1.0) / 3.0, -30.0, 30.0))))
        * (1.0 / (1.0 + np.exp(np.clip((temperature - 15.0) / 3.0, -30.0, 30.0)))),
        axis=0,
    )
    combustion = np.mean(
        dryness / (dryness + 250.0) / (1.0 + rain / 35.0), axis=0
    )
    physical = {
        "cold_forest": cold * forest,
        "organic_continuity": cold * forest * organic,
        "organic_ignition": cold * forest * organic * ignition,
        "organic_thaw_event": cold * forest * organic * ignition * thaw_months * combustion,
    }
    features.update(physical)

    positive = obs_annual[obs_annual > 0.0]
    floor = 0.02 * float(np.median(positive))
    target = np.clip(
        np.log((obs_annual + floor) / (pred_annual + floor)), -4.0, 4.0
    )
    weights = area * (obs_annual + floor)
    names = tuple(features)
    x = np.column_stack([features[name] for name in names]).astype(np.float32)
    folds = (rows // 5 + 2 * (columns // 8)) % 4
    oof = np.zeros(target.shape, dtype=np.float64)
    importances = []
    top_counts: Counter[str] = Counter()
    for fold in range(4):
        train = folds != fold
        test = ~train
        learner = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=24,
            loss="huber",
            random_state=4200 + fold,
        )
        learner.fit(x[train], target[train], sample_weight=weights[train])
        oof[test] = learner.predict(x[test])
        order = np.argsort(learner.feature_importances_)[::-1]
        importances.append(learner.feature_importances_)
        for index in order[:6]:
            top_counts[names[index]] += 1
        base = weighted_mae(target[test], weights[test])
        corrected = weighted_mae(target[test] - 0.25 * oof[test], weights[test])
        print(
            f"FOLD {fold} n={int(np.sum(test))} mae={base:.9f}->{corrected:.9f} "
            f"delta={corrected - base:+.9f} "
            f"corr={weighted_corr(oof[test], target[test], weights[test]):+.9f} "
            f"top={','.join(names[index] for index in order[:6])}",
            flush=True,
        )
    mean_importance = np.mean(np.vstack(importances), axis=0)
    order = np.argsort(mean_importance)[::-1]
    print(
        f"OOF mae={weighted_mae(target, weights):.9f}->"
        f"{weighted_mae(target - 0.25 * oof, weights):.9f} "
        f"corr={weighted_corr(oof, target, weights):+.9f}",
        flush=True,
    )
    for index in order[:12]:
        print(
            f"RANK {names[index]} importance={mean_importance[index]:.9f} "
            f"top6_folds={top_counts[names[index]]}",
            flush=True,
        )
    for name, values in physical.items():
        signs = []
        for fold in range(4):
            test = folds == fold
            signs.append(weighted_corr(values[test], target[test], weights[test]))
        print(
            f"PHYSICAL {name} correlations="
            + ",".join(f"{value:+.6f}" for value in signs),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
