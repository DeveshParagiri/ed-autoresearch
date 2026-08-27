"""Low-memory audit of active operator allocation in model blob 39ee93e.

This diagnostic runs the canonical pointwise model on the 768 land cells with
the greatest GFED5 reference weight. It does not fit coefficients or invoke
the official evaluator. Coordinates select diagnostic cells only. The script
checks whether the four pathway shares used by the multipath bank are a true
partition, measures internal stage removals, and tests one algebraic repair:
managed hazard is limited to its non-crop share of the surface pathway so the
bank cannot store the same hazard through overlapping managed and crop shares.
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


def field(data, name):
    return np.asarray(data[name], dtype=np.float64)


def capacities(model, data):
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    gpp_12 = model._antecedent(gpp, alpha_12)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    natural = np.clip(field(data, "natural_vegetation_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)

    open_natural = natural * 8.0 / (canopy + 8.0)
    managed_open = np.clip(rangeland + pasture, 0.0, 1.0)
    open_cover = np.clip(managed_open + open_natural, 0.0, 1.0)
    surface = (1.0 - crop) * fine_fuel * open_cover
    managed_current = managed_open * fine_fuel
    managed_partitioned = (1.0 - crop) * fine_fuel * managed_open
    woody = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    background = np.full_like(surface, 0.05)
    total = background + surface + woody + crop_capacity
    current = (
        managed_current / total,
        crop_capacity / total,
        woody / total,
        background / total,
    )
    partitioned = (
        managed_partitioned / total,
        crop_capacity / total,
        woody / total,
        background / total,
    )
    return current, partitioned


def readiness(model, data, prediction, p):
    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    rain_6 = model._antecedent(rain, alpha_6)
    rain_12 = model._antecedent(rain, alpha_12)
    deficit_6 = np.maximum((rain_6 - rain) / (rain_6 + rain + 10.0), 0.0)
    deficit_12 = np.maximum((rain_12 - rain) / (rain_12 + rain + 10.0), 0.0)
    wet_anomaly = np.maximum((rain - rain_12) / (rain + rain_12 + 10.0), 0.0)
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    gpp_3 = model._antecedent(gpp, alpha_3)
    curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)
    curing_gate = curing / (curing + 0.05)
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    combustion = dryness / (dryness + 500.0)
    temperature = field(data, "air_temperature")
    temperature_3 = model._antecedent(temperature, alpha_3)
    temperature_12 = model._antecedent(temperature, alpha_12)
    thermal = model._rising(temperature, 0.25, 5.0)
    warm_3 = model._rising(temperature - temperature_3, 0.5, 1.0)
    warm_12 = model._rising(temperature - temperature_12, 0.5, 2.0)
    lightning = np.clip(field(data, "lightning_flash_rate"), 0.0, None)
    bucket = model._fuel_moisture_bucket(
        rain,
        temperature,
        dryness,
        max(p["managed_moisture_capacity"], 1e-6),
        max(p["managed_moisture_drydown"], 1e-6),
        p["managed_moisture_threshold"],
    )
    blend = float(np.clip(p["managed_moisture_blend"], 0.0, 1.0))
    managed_dryness = (1.0 - blend) * deficit_6 + blend * bucket
    managed = managed_dryness * combustion * curing_gate * thermal
    crop = combustion * curing_gate * thermal * warm_3
    woody = (
        warm_12
        * thermal
        * (1.0 - wet_anomaly)
        * model._drying_window_flash_occupancy(
            lightning, deficit_12, thermal, wet_anomaly
        )
    )
    background = deficit_12 * combustion * thermal
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    annual_hazard = model._trailing_annual_hazard(hazard)
    gate = 0.2 / (annual_hazard + 0.2)
    return hazard, (managed, crop, woody, background), gate


def multipath_with_shares(model, prediction, data, p, shares, drop_managed=False):
    hazard, ready, managed_gate = readiness(model, data, prediction, p)
    adjusted = hazard.copy()
    settings = (
        ("managed", p["managed_bank_store"], p["managed_bank_release"], managed_gate),
        ("crop", p["crop_bank_store"], p["crop_bank_release"], 1.0),
        ("woody", p["woody_bank_store"], p["woody_bank_release"], 1.0),
        ("background", p["background_bank_store"], p["background_bank_release"], 1.0),
    )
    for index, (name, store, release, gate) in enumerate(settings):
        if drop_managed and name == "managed":
            continue
        adjusted += model._pathway_bank_delta(
            hazard, shares[index], ready[index], store, release, gate
        )
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
    )


def run_variant(model, data, replacements):
    originals = {}
    try:
        for name, function in replacements.items():
            originals[name] = getattr(model, name)
            setattr(model, name, function)
        return np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
    finally:
        for name, function in originals.items():
            setattr(model, name, function)


def main():
    source = subprocess.check_output(
        ("git", "cat-file", "blob", EXPECTED_BLOB), cwd=ROOT
    )
    model = types.ModuleType("operator_audit_pinned_model")
    model.__file__ = f"git-blob:{EXPECTED_BLOB}"
    exec(compile(source, model.__file__, "exec"), model.__dict__)

    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()
    print(f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f}")

    current_shares, partitioned_shares = capacities(model, data)
    for label, shares in (("current", current_shares), ("partitioned", partitioned_shares)):
        share_sum = np.sum(np.stack(shares, axis=0), axis=0)
        print(
            f"SHARES {label} mean={np.mean(share_sum):.8f} "
            f"p95={np.quantile(share_sum, 0.95):.8f} "
            f"p99={np.quantile(share_sum, 0.99):.8f} "
            f"max={np.max(share_sum):.8f} "
            f"fraction_gt_1={np.mean(share_sum > 1.0):.8f}"
        )

    identity = lambda prediction, data_, p_, enabled_: prediction
    incoming = {}
    original_multipath = model._multi_pathway_opportunity_bank

    def capture_multipath(prediction, data_, p_, enabled_):
        incoming["prediction"] = np.asarray(prediction, dtype=np.float64).copy()
        return original_multipath(prediction, data_, p_, enabled_)

    run_variant(model, data, {"_multi_pathway_opportunity_bank": capture_multipath})
    incoming_prediction = incoming["prediction"]
    hazard, ready, managed_gate = readiness(
        model, data, incoming_prediction, model.PARAMS
    )
    effective_store = (
        model.PARAMS["managed_bank_store"] * current_shares[0] * managed_gate
        + model.PARAMS["crop_bank_store"] * current_shares[1]
        + model.PARAMS["woody_bank_store"] * current_shares[2]
        + model.PARAMS["background_bank_store"] * current_shares[3]
    )
    print(
        f"EFFECTIVE_STORE mean={np.mean(effective_store):.8f} "
        f"p95={np.quantile(effective_store, 0.95):.8f} "
        f"p99={np.quantile(effective_store, 0.99):.8f} "
        f"max={np.max(effective_store):.8f} "
        f"fraction_gt_1={np.mean(effective_store > 1.0):.8f}"
    )
    del incoming_prediction, hazard, ready, managed_gate, effective_store

    variants = {
        "canonical": {},
        "skip_surface_bank": {"_surface_fire_opportunity_bank": identity},
        "skip_multipath_bank": {"_multi_pathway_opportunity_bank": identity},
        "skip_fuel_recovery": {"_pathway_fuel_recovery_reservoir": identity},
        "skip_secondary_litter": {"_secondary_fuel_litter_banks": identity},
        "skip_fragment_recurrence": {"_fragmented_managed_recurrence_brake": identity},
    }
    results = {}
    for label, replacements in variants.items():
        prediction = run_variant(model, data, replacements)
        results[label] = metrics(prediction, observed, area, reference_weight, folds)
        print(label.upper() + " " + format_metrics(results[label][0]), flush=True)

    def partitioned(prediction, data_, p_, enabled_):
        if "surface_opportunity_bank" not in enabled_:
            return prediction
        shares, fixed = capacities(model, data_)
        del shares
        return multipath_with_shares(model, prediction, data_, p_, fixed)

    def no_managed(prediction, data_, p_, enabled_):
        if "surface_opportunity_bank" not in enabled_:
            return prediction
        shares, fixed = capacities(model, data_)
        del shares
        return multipath_with_shares(model, prediction, data_, p_, fixed, True)

    try:
        model._multi_pathway_opportunity_bank = partitioned
        prediction = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
        results["partitioned_multipath"] = metrics(prediction, observed, area, reference_weight, folds)
        print("PARTITIONED_MULTIPATH " + format_metrics(results["partitioned_multipath"][0]), flush=True)
        model._multi_pathway_opportunity_bank = no_managed
        prediction = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
        results["no_managed_multipath"] = metrics(prediction, observed, area, reference_weight, folds)
        print("NO_MANAGED_MULTIPATH " + format_metrics(results["no_managed_multipath"][0]), flush=True)
    finally:
        model._multi_pathway_opportunity_bank = original_multipath

    base = results["canonical"][0]
    print("DELTAS_VS_CANONICAL")
    for label, (global_metrics, fold_metrics) in results.items():
        delta = tuple(value - reference for value, reference in zip(global_metrics, base))
        incumbent_folds = results["canonical"][1]
        improving = (
            sum(candidate[0] < incumbent[0] for candidate, incumbent in zip(fold_metrics, incumbent_folds)),
            sum(candidate[1] < incumbent[1] for candidate, incumbent in zip(fold_metrics, incumbent_folds)),
            sum(candidate[2] < incumbent[2] for candidate, incumbent in zip(fold_metrics, incumbent_folds)),
            sum(candidate[3] > incumbent[3] for candidate, incumbent in zip(fold_metrics, incumbent_folds)),
        )
        print(
            f"{label} alloc={delta[0]:+.8f} annual_log={delta[1]:+.8f} "
            f"raw_cycle={delta[2]:+.8f} phase={delta[3]:+.8f} "
            f"area_ratio={delta[4]:+.8f} improving_folds="
            f"alloc:{improving[0]}/4,annual:{improving[1]}/4,"
            f"raw_cycle:{improving[2]}/4,phase:{improving[3]}/4"
        )


if __name__ == "__main__":
    main()
