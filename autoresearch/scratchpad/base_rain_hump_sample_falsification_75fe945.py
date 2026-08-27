"""Sampled falsification of a causal unimodal base rain-capacity factor.

The canonical model already contains several downstream rain-built-fuel and
humid-system terms.  This probe tests the narrower, structurally distinct
hypothesis that the *base* precipitation requirement should express both dry
fuel limitation and wet fuel-moisture limitation.  It uses three fixed physical
brackets and no learned runtime, coordinates, regions, neighbours, future state,
or full-record climatology.  Coordinates select score-dominant cells and define
held geographic blocks only; they never enter an equation.

This is a diagnostic sampled run.  It does not edit the canonical model, invoke
the official evaluator, or write the experiment ledger.
"""

from __future__ import annotations

import gc
import sys
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
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def normalized_rain_capacity(
    annual_rain: np.ndarray,
    rise_half: float,
    wet_decay: float,
) -> np.ndarray:
    """Return a unit-peak dry-rise x wet-decay capacity curve."""
    annual = np.clip(np.asarray(annual_rain, dtype=np.float64), 0.0, None)
    optimum = 0.5 * (
        np.sqrt(rise_half * rise_half + 4.0 * rise_half * wet_decay)
        - rise_half
    )
    maximum = optimum / (optimum + rise_half) * np.exp(-optimum / wet_decay)
    capacity = annual / (annual + rise_half) * np.exp(-annual / wet_decay)
    return np.clip(capacity / maximum, 0.0, 1.0)


def fire_rate_factory(model, rise_half: float, wet_decay: float, mix: float):
    """Rebuild only the canonical base aggregation with a new rain factor."""
    def fire_rate(data, p, enabled):
        factors = []
        rate = np.ones_like(data["dryness"], dtype=np.float32)
        if "dryness" in enabled:
            term = model._rising(data["dryness"], p["k1"], p["D_low"]) * model._falling(
                data["dryness"], p["k2"], p["D_high"]
            )
            factors.append(term)
            rate = rate * term
        if "precipitation" in enabled:
            annual = np.clip(data["annual_precipitation"], 0.0, None)
            monthly = np.clip(data["monthly_precipitation"], 0.0, None)
            incumbent_capacity = annual / (annual + p["P_half"] + 1e-12)
            hump_capacity = normalized_rain_capacity(annual, rise_half, wet_decay)
            capacity = (1.0 - mix) * incumbent_capacity + mix * hump_capacity
            term = capacity / (
                1.0 + monthly / (p["pre_dampen_half"] + 1e-12)
            )
            factors.append(term)
            rate = rate * term
        if "fuel" in enabled:
            term = model._hump(p["gpp_af"] * data["gpp"], p["gpp_b"], p["gpp_d"])
            factors.append(term)
            rate = rate * term
        if "temperature" in enabled:
            term = model._managed_open_temperature_gate(data, p)
            factors.append(term)
            rate = rate * term
        weight = float(np.clip(p.get("soft_w", 0.0), 0.0, 1.0))
        if "softmin" in enabled and weight > 0.0 and len(factors) > 1:
            stack = np.stack(factors, axis=0)
            sharp = float(np.clip(p.get("soft_s", 4.0), 0.5, 50.0))
            softmin = -np.log(
                np.exp(-sharp * np.clip(stack, 1e-6, None)).mean(axis=0) + 1e-12
            ) / sharp
            softmin = np.clip(softmin, 1e-6, None)
            rate = np.power(np.clip(rate, 1e-9, None), 1.0 - weight) * np.power(
                softmin, weight
            )
        rate = np.power(np.clip(rate, 0.0, None), p["fire_exp"])
        if "fuel" in enabled and "fuel_k" in p:
            capacity = data["gpp"].mean(axis=0, keepdims=True)
            capacity = capacity / (capacity + p["fuel_half"] + 1e-9)
            rate *= 1.0 + p["fuel_k"] * capacity
        elif "fire_amp" in p:
            rate *= p["fire_amp"]
        return rate

    return fire_rate


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()

    original = model._fire_rate
    official_components = tuple(name for name in model.COMPONENTS if name != "softmin")
    baselines = {}
    for label, components in (
        ("causal_product", official_components),
        ("normalized_softmin", model.COMPONENTS),
    ):
        prediction = np.asarray(
            model.predict(data, dict(model.PARAMS), components), dtype=np.float64
        )[:, 0, :]
        baselines[label] = metrics(prediction, observed, area, reference_weight, folds)
        print(f"BASE {label} " + format_metrics(baselines[label][0]), flush=True)
        del prediction
        gc.collect()

    shapes = (
        ("broad", 100.0, 6000.0),
        ("mesic", 250.0, 3000.0),
        ("wet_peak", 250.0, 6000.0),
    )
    try:
        for aggregation, components in (
            ("causal_product", official_components),
            ("normalized_softmin", model.COMPONENTS),
        ):
            baseline, baseline_folds = baselines[aggregation]
            for shape, rise_half, wet_decay in shapes:
                optimum = 0.5 * (
                    np.sqrt(rise_half * rise_half + 4.0 * rise_half * wet_decay)
                    - rise_half
                )
                for mix in (0.25, 0.5, 1.0):
                    model._fire_rate = fire_rate_factory(
                        model, rise_half, wet_decay, mix
                    )
                    prediction = np.asarray(
                        model.predict(data, dict(model.PARAMS), components),
                        dtype=np.float64,
                    )[:, 0, :]
                    current, current_folds = metrics(
                        prediction, observed, area, reference_weight, folds
                    )
                    label = f"{aggregation}:{shape}:mix={mix:g}"
                    print(
                        f"CANDIDATE {label} optimum_mm={optimum:.3f} "
                        + format_metrics(current),
                        flush=True,
                    )
                    print(
                        "DELTA "
                        + label
                        + " "
                        + " ".join(
                            f"{name}={current[index] - baseline[index]:+.8f}"
                            for index, name in enumerate(
                                (
                                    "alloc_rmse",
                                    "annual_log_rmse",
                                    "raw_cycle_rmse",
                                    "phase",
                                    "area_ratio",
                                )
                            )
                        ),
                        flush=True,
                    )
                    for fold in range(4):
                        print(
                            f"FOLD {label} fold={fold} "
                            f"alloc_delta={current_folds[fold][0] - baseline_folds[fold][0]:+.8f} "
                            f"annual_delta={current_folds[fold][1] - baseline_folds[fold][1]:+.8f} "
                            f"raw_cycle_delta={current_folds[fold][2] - baseline_folds[fold][2]:+.8f} "
                            f"phase_delta={current_folds[fold][3] - baseline_folds[fold][3]:+.8f}",
                            flush=True,
                        )
                    del prediction
                    gc.collect()
    finally:
        model._fire_rate = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
