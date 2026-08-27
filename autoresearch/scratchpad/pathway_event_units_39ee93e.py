"""Audit finite-Poisson units in ``_pathway_event_scaling`` at blob 39ee93e.

The incumbent converts its legacy branch from burned probability to hazard
only after multiplying probability by a dimensionless event scale, while its
pathway branch multiplies hazard directly.  This diagnostic tests two fixed,
mass-consistent interpretations without fitting coefficients or invoking the
official evaluator:

``rate_union``
    Every pathway is an independent Poisson event process acting on the same
    ground.  Dimensionless pathway scales therefore multiply the base hazard
    and the pathway mixture is also formed in hazard space.

``subgrid_tiles``
    Normalized pathway capacities are disjoint subgrid fuel-class area shares.
    Each tile has a finite Poisson burn probability, and tile probabilities are
    averaged by their area shares.  ``pathway_mix_w`` is likewise interpreted
    as a subgrid area mixture between the legacy and resolved pathways.

Coordinates are used only to select and split 768 diagnostic cells.  They do
not enter either candidate equation.  Canonical files are never edited.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    format_metrics,
    load_observed,
    load_selected,
    metrics,
    select_cells,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH  # noqa: E402


EXPECTED_BLOB = "39ee93ebf1155af9ae9d70e05847b9c3f086887d"


def pinned_model() -> types.ModuleType:
    source = subprocess.check_output(
        ("git", "cat-file", "blob", EXPECTED_BLOB), cwd=ROOT
    )
    model = types.ModuleType("pathway_event_units_pinned_model")
    model.__file__ = f"git-blob:{EXPECTED_BLOB}"
    exec(compile(source, model.__file__, "exec"), model.__dict__)
    return model


def field(data, name):
    return np.asarray(data[name], dtype=np.float64)


def event_state(model, prediction, data, p):
    """Return base hazard, normalized capacities, and event-rate scales."""
    probability = np.clip(
        np.asarray(prediction, dtype=np.float64), 0.0, 1.0 - 1e-7
    )
    base_hazard = -np.log1p(-probability)
    annual_scale = float(max(p.get("annual_scale", 1.0), 0.0))
    event_half = float(max(p.get("event_scale_half", 0.003), 1e-8))
    connected = probability / (probability + event_half)
    old_scale = 1.0 + (annual_scale - 1.0) * connected

    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    gpp_3 = model._antecedent(gpp, alpha_3)
    gpp_12 = model._antecedent(gpp, alpha_12)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)

    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    rain_6 = model._antecedent(rain, alpha_6)
    rain_12 = model._antecedent(rain, alpha_12)
    dry_6 = np.maximum((rain_6 - rain) / (rain_6 + rain + 10.0), 0.0)
    dry_12 = np.maximum((rain_12 - rain) / (rain_12 + rain + 10.0), 0.0)
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    combustion = dryness / (dryness + 500.0)

    natural = np.clip(field(data, "natural_vegetation_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel

    temperature = field(data, "air_temperature")
    temperature_12 = model._antecedent(temperature, alpha_12)
    warm_anomaly = model._rising(temperature - temperature_12, 0.5, 3.0)
    lightning = np.clip(field(data, "lightning_flash_rate"), 0.0, None)
    lightning_12 = model._antecedent(lightning, alpha_12)
    ignition = lightning_12 / (lightning_12 + 0.01)
    annual_rain = np.clip(field(data, "annual_precipitation"), 0.0, None)
    humid_closed = (
        model._rising(annual_rain, 1.0 / 250.0, 1200.0)
        * model._rising(canopy, 1.0 / 3.0, 15.0)
        * natural
    )

    surface_available = dry_6 * combustion
    woody_available = (
        dry_12 * warm_anomaly * ignition * np.exp(-3.0 * humid_closed)
    )
    residue_curing = np.maximum(
        (gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0
    )
    crop_available = residue_curing * combustion

    background = np.full_like(surface_capacity, 0.05)
    total = background + surface_capacity + woody_capacity + crop_capacity + 1e-12
    shares = (
        background / total,
        surface_capacity / total,
        woody_capacity / total,
        crop_capacity / total,
    )
    scales = (
        old_scale,
        1.0 + 1.1 * connected * (0.35 + 0.65 * surface_available),
        0.65 + 1.85 * woody_available / (woody_available + 0.015),
        0.60 + 1.20 * crop_available / (crop_available + 0.06),
    )
    return base_hazard, shares, scales


def finite_poisson(hazard):
    return -np.expm1(-np.clip(hazard, 0.0, 50.0))


def rate_union_scaler(model):
    """Sum independent pathway event rates over shared ground."""
    def scale(prediction, data, p, enabled):
        base_hazard, shares, scales = event_state(model, prediction, data, p)
        if "pathway_hazards" not in enabled:
            return np.asarray(finite_poisson(base_hazard * scales[0]), dtype=np.float32)
        mix = float(np.clip(p.get("pathway_mix_w", 0.0), 0.0, 1.0))
        resolved_scale = sum(share * event_scale for share, event_scale in zip(shares, scales))
        hazard = base_hazard * ((1.0 - mix) * scales[0] + mix * resolved_scale)
        return np.asarray(finite_poisson(hazard), dtype=np.float32)
    return scale


def subgrid_tile_scaler(model):
    """Average finite Poisson probabilities over disjoint fuel-class tiles."""
    def scale(prediction, data, p, enabled):
        base_hazard, shares, scales = event_state(model, prediction, data, p)
        legacy_probability = finite_poisson(base_hazard * scales[0])
        if "pathway_hazards" not in enabled:
            return np.asarray(legacy_probability, dtype=np.float32)
        mix = float(np.clip(p.get("pathway_mix_w", 0.0), 0.0, 1.0))
        resolved_probability = sum(
            share * finite_poisson(base_hazard * event_scale)
            for share, event_scale in zip(shares, scales)
        )
        probability = (1.0 - mix) * legacy_probability + mix * resolved_probability
        return np.asarray(np.clip(probability, 0.0, 1.0), dtype=np.float32)
    return scale


def run(model, data, replacement):
    original = model._pathway_event_scaling
    try:
        model._pathway_event_scaling = replacement
        return np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
    finally:
        model._pathway_event_scaling = original


def ecology_masks(data):
    def mean(name):
        return field(data, name).mean(axis=0)[0]

    rain = mean("annual_precipitation")
    temperature = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    return {
        "intact_tropical_closed": (temperature >= 20.0) & (rain >= 1200.0) & (canopy >= 20.0) & (lai >= 3.0) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5.0) & (temperature < 20.0) & (canopy >= 15.0) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5.0) & (canopy >= 10.0) & (natural >= 0.6),
        "tropical_open": (temperature >= 20.0) & (rain >= 500.0) & (rain < 1500.0) & (canopy >= 5.0) & (canopy < 20.0) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250.0) & (rain < 1500.0) & (biomass >= 0.2),
        "crop": crop >= 0.5,
        "arid_low_fuel": (rain < 250.0) & (biomass < 0.3) & (lai < 1.0),
    }


def ecology_ratios(prediction, observed, area, masks):
    pred_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    obs_annual = observed.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
    result = {}
    for name, mask in masks.items():
        denominator = float(np.sum(obs_annual[mask] * area[mask]))
        result[name] = (
            int(np.sum(mask)),
            float(np.sum(pred_annual[mask] * area[mask])) / max(denominator, 1e-12),
        )
    return result


def main():
    model = pinned_model()
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()

    captured = {}
    original = model._pathway_event_scaling

    def capture(prediction, data_, p_, enabled_):
        captured["incoming"] = np.asarray(prediction, dtype=np.float64).copy()
        captured["data"] = data_
        return original(prediction, data_, p_, enabled_)

    baseline = run(model, data, capture)
    candidates = {
        "rate_union": run(model, data, rate_union_scaler(model)),
        "subgrid_tiles": run(model, data, subgrid_tile_scaler(model)),
    }
    baseline_metrics, baseline_folds = metrics(
        baseline, observed, area, reference_weight, folds
    )
    print(
        f"DESIGN model_blob={EXPECTED_BLOB} cells={rows.size} "
        f"retained_reference_weight={retained:.8f} no_fitted_parameters=1"
    )
    print("CANONICAL " + format_metrics(baseline_metrics))

    base_hazard, shares, scales = event_state(
        model, captured["incoming"], captured["data"], model.PARAMS
    )
    mix = float(model.PARAMS["pathway_mix_w"])
    current_old_probability = np.clip(
        captured["incoming"] * scales[0], 0.0, 1.0 - 1e-7
    )
    current_old_hazard = -np.log1p(-current_old_probability)
    resolved_scale = sum(share * event_scale for share, event_scale in zip(shares, scales))
    current_stage = finite_poisson(
        (1.0 - mix) * current_old_hazard + mix * base_hazard * resolved_scale
    )
    rate_stage = finite_poisson(
        base_hazard * ((1.0 - mix) * scales[0] + mix * resolved_scale)
    )
    tile_stage = (1.0 - mix) * finite_poisson(base_hazard * scales[0]) + mix * sum(
        share * finite_poisson(base_hazard * event_scale)
        for share, event_scale in zip(shares, scales)
    )
    share_sum = sum(shares)
    print(
        f"UNIT_AUDIT share_sum_max_error={np.max(np.abs(share_sum - 1.0)):.12g} "
        f"current_vs_rate_mean_abs={np.mean(np.abs(current_stage - rate_stage)):.10f} "
        f"current_vs_rate_p99_abs={np.quantile(np.abs(current_stage - rate_stage), 0.99):.10f} "
        f"rate_vs_tiles_mean_abs={np.mean(np.abs(rate_stage - tile_stage)):.10f} "
        f"rate_identity_max_error={np.max(np.abs(finite_poisson(base_hazard) - captured['incoming'])):.12g}"
    )

    masks = ecology_masks(captured["data"])
    base_ecology = ecology_ratios(baseline, observed, area, masks)
    for label, prediction in candidates.items():
        candidate_metrics, candidate_folds = metrics(
            prediction, observed, area, reference_weight, folds
        )
        delta = tuple(
            candidate - incumbent
            for candidate, incumbent in zip(candidate_metrics, baseline_metrics)
        )
        improving = (
            sum(c[0] < b[0] for c, b in zip(candidate_folds, baseline_folds)),
            sum(c[1] < b[1] for c, b in zip(candidate_folds, baseline_folds)),
            sum(c[2] < b[2] for c, b in zip(candidate_folds, baseline_folds)),
            sum(c[3] > b[3] for c, b in zip(candidate_folds, baseline_folds)),
        )
        print(label.upper() + " " + format_metrics(candidate_metrics))
        print(
            f"DELTA {label} alloc={delta[0]:+.8f} annual_log={delta[1]:+.8f} "
            f"raw_cycle={delta[2]:+.8f} phase={delta[3]:+.8f} "
            f"area_ratio={delta[4]:+.8f} improving_folds="
            f"alloc:{improving[0]}/4,annual:{improving[1]}/4,"
            f"raw_cycle:{improving[2]}/4,phase:{improving[3]}/4"
        )
        candidate_ecology = ecology_ratios(prediction, observed, area, masks)
        print(
            "ECOLOGY "
            + label
            + " "
            + ",".join(
                f"{name}[n={base_ecology[name][0]}]:"
                f"{base_ecology[name][1]:.5f}->{candidate_ecology[name][1]:.5f}"
                for name in masks
            )
        )
        for fold, (base_fold, candidate_fold) in enumerate(
            zip(baseline_folds, candidate_folds)
        ):
            fold_delta = tuple(
                candidate - incumbent
                for candidate, incumbent in zip(candidate_fold, base_fold)
            )
            print(
                f"FOLD {label} {fold} alloc={fold_delta[0]:+.8f} "
                f"annual_log={fold_delta[1]:+.8f} raw_cycle={fold_delta[2]:+.8f} "
                f"phase={fold_delta[3]:+.8f} area_ratio={fold_delta[4]:+.8f}"
            )


if __name__ == "__main__":
    main()
