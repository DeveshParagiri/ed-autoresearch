"""Exact targeted subterm pruning audit for model blob ca6848f.

The incumbent differs from the pinned 39ee93e prediction only by the final
warm surface-seasonality capacity operator.  This diagnostic reconstructs that
incumbent from the cache, independently verifies it with a fresh pointwise run
over every evaluator land cell, then removes one physical subterm at a time.
It never edits the canonical model or records an official evaluation.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    load_observed,
    load_selected,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


MODEL_BLOB = "ca6848f2db28af24a06cd9f06e3adcdecaf7fcc0"
PREDECESSOR_BLOB = "39ee93ebf1155af9ae9d70e05847b9c3f086887d"
PREDECESSOR_CACHE = ROOT / "autoresearch/scratchpad/canonical_39ee93eb_chunked.npy"
EXPECTED_PREDECESSOR = 0.7183634082232968
EXPECTED_INCUMBENT = 0.718706140


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def load_blob(blob: str):
    source = subprocess.check_output(("git", "cat-file", "blob", blob), cwd=ROOT)
    module = types.ModuleType(f"model_{blob[:8]}")
    module.__file__ = f"git-blob:{blob}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def prepared_data(model, data):
    prepared = dict(data)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    prepared["annual_precipitation"] = 12.0 * model._antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float32), alpha_12
    )
    return prepared


def selected_regime_masks(data):
    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)[0]

    rain = 12.0 * mean("monthly_precipitation")
    temperature = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    return {
        "intact_tropical_closed": (
            (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0)
            & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5)
        ),
        "temperate_closed": (
            (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0)
            & (lai >= 2.5) & (natural >= 0.6)
        ),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (
            (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0)
            & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5)
        ),
        "productive_rangeland": (
            (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0)
            & (biomass >= 0.2)
        ),
        "cropland": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def selected_ecological_statistics(prediction, masks, observation, area):
    model_cycle = np.asarray(prediction).reshape(16, 12, -1).mean(axis=0)
    obs_cycle = np.asarray(observation).reshape(16, 12, -1).mean(axis=0)
    model_annual = model_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    output = {}
    for name, mask in masks.items():
        weights = area * mask
        model_total = float(np.sum(model_annual * weights))
        obs_total = float(np.sum(obs_annual * weights))
        model_monthly = np.sum(model_cycle * weights[None, :], axis=1)
        obs_monthly = np.sum(obs_cycle * weights[None, :], axis=1)
        model_peak = int(np.argmax(model_monthly))
        obs_peak = int(np.argmax(obs_monthly))
        phase = min(abs(model_peak - obs_peak), 12 - abs(model_peak - obs_peak))
        output[name] = {
            "ratio": model_total / obs_total if obs_total > 1e-9 else float("inf"),
            "phase_months": phase,
        }
    return output


def score_selected(evaluator, prediction, rows, cols):
    grid = np.zeros((192, 180, 360), dtype=np.float32)
    grid[:, rows, cols] = np.asarray(prediction, dtype=np.float32)[:, 0, :]
    scores = evaluator.score(grid)
    del grid
    return scores


def metric_line(label, scores, baseline_scores):
    current = scores["global"]
    base = baseline_scores["global"]
    regional = sorted(
        (
            float(scores[name]["overall_score"])
            - float(baseline_scores[name]["overall_score"]),
            name,
        )
        for name in scores
        if name != "global"
    )
    positive = sum(delta > 0.0 for delta, _ in regional)
    return (
        f"{label}\toverall={current['overall_score']:.9f}"
        f"\tdelta={current['overall_score'] - base['overall_score']:+.9f}"
        f"\td_bias={current['bias_score'] - base['bias_score']:+.9f}"
        f"\td_rmse={current['rmse_score'] - base['rmse_score']:+.9f}"
        f"\td_seasonal={current['seasonal_cycle_score'] - base['seasonal_cycle_score']:+.9f}"
        f"\td_spatial={current['spatial_distribution_score'] - base['spatial_distribution_score']:+.9f}"
        f"\tregions={positive}/14"
        f"\tworst={regional[0][1]}:{regional[0][0]:+.9f}"
        f"\tbest={regional[-1][1]}:{regional[-1][0]:+.9f}"
    )


def run_variant(model, data, params, replacement=None):
    original = None
    if replacement is not None:
        name, function = replacement
        original = getattr(model, name)
        setattr(model, name, function)
    try:
        return np.asarray(model.predict(data, params, None), dtype=np.float32)
    finally:
        if replacement is not None:
            setattr(model, name, original)


def main() -> int:
    current_blob = subprocess.check_output(
        ("git", "rev-parse", "HEAD:autoresearch/model.py"), cwd=ROOT, text=True
    ).strip()
    if current_blob != MODEL_BLOB:
        raise RuntimeError(f"expected model blob {MODEL_BLOB}, got {current_blob}")
    predecessor = subprocess.check_output(
        ("git", "rev-parse", PREDECESSOR_BLOB), cwd=ROOT, text=True
    ).strip()
    if predecessor != PREDECESSOR_BLOB:
        raise RuntimeError("predecessor blob did not resolve exactly")

    model = load_blob(MODEL_BLOB)
    evaluator = GFED5Evaluator(GFED5_PATH)
    land = load_land_mask()
    rows, cols = np.where(land)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))[rows, cols]
    data = load_selected(model.INPUTS, rows, cols)
    prepared = prepared_data(model, data)
    observed = load_observed(rows, cols)
    masks = selected_regime_masks(data)
    print(f"DESIGN cells={rows.size} model_blob={MODEL_BLOB[:8]} rss_mb={rss_mb():.1f}", flush=True)

    predecessor_grid = np.load(PREDECESSOR_CACHE, mmap_mode="r")
    predecessor_scores = evaluator.score(predecessor_grid)
    if abs(predecessor_scores["global"]["overall_score"] - EXPECTED_PREDECESSOR) > 5e-10:
        raise RuntimeError("pinned predecessor cache score mismatch")
    predecessor_selected = np.asarray(
        predecessor_grid[:, rows, cols][:, None, :], dtype=np.float32
    )
    del predecessor_grid
    cached_incumbent = model._surface_seasonality_capacity(
        predecessor_selected, prepared, dict(model.PARAMS), set(model.COMPONENTS)
    )
    baseline_scores = score_selected(evaluator, cached_incumbent, rows, cols)
    if abs(baseline_scores["global"]["overall_score"] - EXPECTED_INCUMBENT) > 5e-9:
        raise RuntimeError(
            f"reconstructed incumbent mismatch: {baseline_scores['global']['overall_score']:.12f}"
        )
    print(
        metric_line("BASELINE_FROM_PINNED_CACHE", baseline_scores, baseline_scores)
        + f"\tpredecessor={predecessor_scores['global']['overall_score']:.9f}",
        flush=True,
    )

    fresh = run_variant(model, data, dict(model.PARAMS))
    difference = float(np.max(np.abs(fresh - cached_incumbent)))
    fresh_scores = score_selected(evaluator, fresh, rows, cols)
    print(
        metric_line("BASELINE_FRESH", fresh_scores, baseline_scores)
        + f"\tmax_abs_vs_cache={difference:.12g}\trss_mb={rss_mb():.1f}",
        flush=True,
    )
    if difference != 0.0:
        raise RuntimeError("fresh pointwise incumbent differs from cache reconstruction")
    baseline_ecology = selected_ecological_statistics(
        fresh[:, 0, :], masks, observed, area
    )
    del fresh, cached_incumbent
    gc.collect()

    identity = lambda prediction, data_, p_, enabled_: prediction
    variants = (
        ("capacity:cold_forest", {"cold_forest_capacity": 0.0}, None),
        ("capacity:productive_range", {"productive_range_brake": 0.0}, None),
        ("capacity:surface_seasonality", {"surface_seasonality_capacity": 0.0}, None),
        ("closure:warm_open", {"persistent_warm_open_brake": 0.0}, None),
        ("closure:cold_thaw", {"cold_thaw_source": 0.0}, None),
        ("closure:cold_thaw_boost", {"cold_thaw_capacity_boost": 0.0}, None),
        ("pathway:event_mix", {"pathway_mix_w": 0.0}, None),
        ("pathway:footprint", {"fire_footprint_w": 0.0}, None),
        ("pathway:fragment_recurrence", {"fragmented_managed_recurrence_brake": 0.0}, None),
        ("bank:surface", {}, ("_surface_fire_opportunity_bank", identity)),
        ("bank:multi_pathway", {}, ("_multi_pathway_opportunity_bank", identity)),
        ("bank:multi_managed", {"managed_bank_store": 0.0}, None),
        ("bank:multi_crop", {"crop_bank_store": 0.0}, None),
        ("bank:multi_woody", {"woody_bank_store": 0.0}, None),
        ("bank:multi_background", {"background_bank_store": 0.0}, None),
        ("bank:fuel_recovery", {}, ("_pathway_fuel_recovery_reservoir", identity)),
        ("bank:recovery_surface", {"surface_fuel_recovery_months": 1e-6}, None),
        ("bank:recovery_crop", {"crop_fuel_recovery_months": 1e-6}, None),
        ("bank:recovery_woody", {"woody_fuel_recovery_months": 1e-6}, None),
        ("bank:secondary_litter", {}, ("_secondary_fuel_litter_banks", identity)),
        ("bank:litter_open", {"secondary_open_litter_store": 0.0}, None),
        ("bank:litter_woody", {"secondary_woody_litter_store": 0.0}, None),
    )
    results = []
    for label, overrides, replacement in variants:
        params = dict(model.PARAMS)
        params.update(overrides)
        candidate = run_variant(model, data, params, replacement)
        scores = score_selected(evaluator, candidate, rows, cols)
        line = metric_line(label, scores, baseline_scores)
        print(line + f"\trss_mb={rss_mb():.1f}", flush=True)
        results.append((float(scores["global"]["overall_score"]), label, scores, candidate))
        # Retain only positive predictions for a final ecology audit.
        if scores["global"]["overall_score"] <= baseline_scores["global"]["overall_score"]:
            results[-1] = (results[-1][0], label, scores, None)
            del candidate
        gc.collect()

    print("RANKED_REMOVAL_DELTA", flush=True)
    for overall, label, scores, candidate in sorted(results, reverse=True):
        delta = overall - baseline_scores["global"]["overall_score"]
        print(f"{label}\t{delta:+.9f}", flush=True)
        if candidate is None:
            continue
        ecology = selected_ecological_statistics(
            candidate[:, 0, :], masks, observed, area
        )
        for name in masks:
            old = float(baseline_ecology[name]["ratio"])
            new = float(ecology[name]["ratio"])
            print(
                f"ECOLOGY {label} {name} {old:.9f}->{new:.9f} delta={new-old:+.9f}",
                flush=True,
            )
    print(f"DONE peak_rss_mb={rss_mb():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
