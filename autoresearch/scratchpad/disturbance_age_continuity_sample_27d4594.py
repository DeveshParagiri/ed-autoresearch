"""Bounded sampled falsification of LUH2 disturbance-age fuel continuity.

Recent absolute changes in coupled LUH2 cover create a causal disturbance
memory that decays over two years.  Managed-open fire capacity recovers as the
land-use matrix stabilizes.  The equation contains no recurrence term; trailing
fire is used only for post-prediction strata so transition memory can be
distinguished from the incumbent recurrence level.  The candidate multiplies
finite Poisson hazard and never creates target-, region-, coordinate-, or
calendar-specific behavior.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.cold_mixed_heldblock_screen_80368d8 import ema, sigmoid  # noqa: E402
from autoresearch.scratchpad.cold_mixed_recession_falsification_80368d8 import (  # noqa: E402
    selected_observation,
)
from autoresearch.scratchpad.phenology_half_greenup_exact_80368d8 import (  # noqa: E402
    ecology_masks,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    MASK_INPUTS,
    one_degree_area,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    CACHE,
    EXPECTED_MODEL_BLOB,
    load_observation,
    select_high_weight,
    selected_inputs,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask, load_model  # noqa: E402


MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
PINNED_COMMIT = "27d4594"


def load_pinned_model():
    source = subprocess.run(
        ["git", "show", f"{PINNED_COMMIT}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED_COMMIT}")
    exec(compile(source, f"{PINNED_COMMIT}:autoresearch/model.py", "exec"), module.__dict__)
    blob = subprocess.run(
        ["git", "show", f"{PINNED_COMMIT}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    pinned_blob = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=ROOT,
        input=blob,
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    if pinned_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"pinned model mismatch: {pinned_blob}")
    return module


def field(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(data[name][:, 0, :], dtype=np.float32)


def disturbance_memory(data: Mapping[str, np.ndarray]) -> np.ndarray:
    covers = tuple(
        np.clip(field(data, name), 0.0, 1.0)
        for name in (
            "luh2_primary_fraction",
            "luh2_cropland_fraction",
            "luh2_pasture_fraction",
            "luh2_rangeland_fraction",
            "luh2_urban_fraction",
        )
    )
    shock = np.zeros_like(covers[0], dtype=np.float32)
    for cover in covers:
        shock[1:] += np.abs(cover[1:] - cover[:-1])
    decay = np.float32(np.exp(-1.0 / 24.0))
    memory = np.empty_like(shock, dtype=np.float32)
    state = np.zeros_like(shock[0], dtype=np.float32)
    for time_index in range(shock.shape[0]):
        state = decay * state + shock[time_index]
        memory[time_index] = state
    return memory


def capacity_state(
    data: Mapping[str, np.ndarray],
    use_stability: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field(data, "luh2_urban_fraction"), 0.0, 1.0)
    rain12 = ema(rain, 12.0)
    gpp12 = ema(gpp, 12.0)
    temperature24 = ema(temperature, 24.0)
    annual_rain = 12.0 * rain12
    rain_fuel = annual_rain / (annual_rain + 250.0) * np.exp(-annual_rain / 3000.0)
    productivity = gpp12 / (gpp12 + 0.35)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    managed_access = managed_open / (managed_open + 0.15)
    static_continuity = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    warm_background = sigmoid((temperature24 - 12.0) / 3.0)
    disturbance = disturbance_memory(data)
    stability = 1.0 / (1.0 + disturbance / 0.02) if use_stability else 1.0
    eligibility = (
        warm_background
        * managed_access
        * rain_fuel
        * productivity
        * static_continuity
        * stability
    )
    return np.asarray(eligibility, dtype=np.float32), disturbance


def candidate(
    baseline: np.ndarray,
    data: Mapping[str, np.ndarray],
    strength: float,
    use_stability: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligibility, disturbance = capacity_state(data, use_stability)
    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    adjusted = hazard * np.exp(np.clip(strength * eligibility, 0.0, 1.0))
    prediction = np.asarray(
        1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32
    )
    return prediction, eligibility, disturbance


def diagnostics(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | str]:
    chosen = selected > 0.0
    weight = area[chosen] * selected[chosen]
    pred_cycle = prediction[:, chosen].reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation[:, chosen].reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    obs_weight = weight * (obs_annual + 1e-8)
    pred_monthly = np.sum(pred_cycle * weight[None, :], axis=1)
    obs_monthly = np.sum(obs_cycle * weight[None, :], axis=1)
    pred_norm = pred_monthly / max(float(pred_monthly.sum()), 1e-12)
    obs_norm = obs_monthly / max(float(obs_monthly.sum()), 1e-12)
    return {
        "ratio": float(np.sum(pred_annual * weight))
        / max(float(np.sum(obs_annual * weight)), 1e-12),
        "annual_log": float(
            np.sum(obs_weight * np.abs(np.log((pred_annual + 1e-6) / (obs_annual + 1e-6))))
            / max(float(obs_weight.sum()), 1e-12)
        ),
        "seasonal_l1": 0.5 * float(np.sum(np.abs(pred_norm - obs_norm))),
        "peak": MONTHS[int(np.argmax(pred_monthly))],
        "obs_peak": MONTHS[int(np.argmax(obs_monthly))],
    }


def main() -> int:
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if not CACHE.exists():
        raise RuntimeError(f"missing pinned baseline cache {CACHE}")
    model = load_pinned_model()
    full_baseline = np.load(CACHE)
    full_observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    full_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    high_rows, high_columns, _, retained = select_high_weight(full_observation, full_area)
    high_mask = np.zeros((180, 360), dtype=bool)
    high_mask[high_rows, high_columns] = True

    mask_data = load_inputs(MASK_INPUTS)
    land = load_land_mask()
    ecological = ecology_masks(mask_data, model, land)
    union = high_mask | np.any(np.stack(tuple(ecological.values()), axis=0), axis=0)
    rows, columns = np.nonzero(union)
    audit_weights = {name: mask[rows, columns].astype(np.float32) for name, mask in ecological.items()}
    audit_weights["high_weight"] = high_mask[rows, columns].astype(np.float32)
    del mask_data, ecological, land, union, high_mask
    gc.collect()

    data = selected_inputs(model, rows, columns)
    baseline = full_baseline[:, rows, columns]
    observation = selected_observation(rows, columns)
    area = one_degree_area()[rows, columns]
    del full_baseline, full_observation, full_area
    gc.collect()
    high = audit_weights["high_weight"] > 0.0
    folds = (rows // 15 + 3 * (columns // 15)) % 4
    base_metrics = {
        name: diagnostics(baseline, observation, area, selected)
        for name, selected in audit_weights.items()
    }
    print(
        f"SAMPLE pinned_model_blob={EXPECTED_MODEL_BLOB} working_model_blob={blob} "
        f"cells={rows.size} high_cells={int(high.sum())} "
        f"retained={retained:.9f}", flush=True,
    )
    print("BASE high_weight " + " ".join(f"{k}={v}" for k, v in base_metrics["high_weight"].items()), flush=True)

    for use_stability in (False, True):
        label = "disturbance_age" if use_stability else "no_turnover_memory"
        for strength in (0.1, 0.25, 0.5):
            prediction, eligibility, disturbance = candidate(
                baseline, data, strength, use_stability
            )
            metrics = diagnostics(prediction, observation, area, audit_weights["high_weight"])
            base = base_metrics["high_weight"]
            print(
                f"CANDIDATE family={label} strength={strength:g} "
                + " ".join(f"{k}={v}" for k, v in metrics.items())
                + f" d_annual_log={float(metrics['annual_log']) - float(base['annual_log']):+.9f}"
                + f" d_seasonal_l1={float(metrics['seasonal_l1']) - float(base['seasonal_l1']):+.9f}",
                flush=True,
            )
            fold_deltas = []
            for fold in range(4):
                selected = (high & (folds == fold)).astype(np.float32)
                before = diagnostics(baseline, observation, area, selected)
                after = diagnostics(prediction, observation, area, selected)
                delta = float(after["annual_log"]) - float(before["annual_log"])
                fold_deltas.append(delta)
                print(
                    f"FOLD family={label} strength={strength:g} fold={fold} "
                    f"d_annual_log={delta:+.9f} "
                    f"d_seasonal_l1={float(after['seasonal_l1']) - float(before['seasonal_l1']):+.9f}",
                    flush=True,
                )
            if use_stability:
                mean_recurrence = ema(
                    -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7)), 12.0
                ).mean(axis=0)
                edges = np.quantile(mean_recurrence[high], np.linspace(0.0, 1.0, 5))
                for stratum, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    selected = high & (mean_recurrence >= lower) & (
                        mean_recurrence <= upper if upper == edges[-1] else mean_recurrence < upper
                    )
                    before = diagnostics(baseline, observation, area, selected.astype(np.float32))
                    after = diagnostics(prediction, observation, area, selected.astype(np.float32))
                    print(
                        f"RECURRENCE strength={strength:g} stratum={stratum} "
                        f"lo={lower:.9g} hi={upper:.9g} "
                        f"d_annual_log={float(after['annual_log']) - float(before['annual_log']):+.9f}",
                        flush=True,
                    )
                for name, selected in audit_weights.items():
                    if name == "high_weight":
                        continue
                    after = diagnostics(prediction, observation, area, selected)
                    before = base_metrics[name]
                    print(
                        f"ECOLOGY strength={strength:g} group={name} "
                        f"ratio={before['ratio']}->{after['ratio']} "
                        f"d_annual_log={float(after['annual_log']) - float(before['annual_log']):+.9f} "
                        f"d_seasonal_l1={float(after['seasonal_l1']) - float(before['seasonal_l1']):+.9f}",
                        flush=True,
                    )
            del prediction, eligibility, disturbance

    # Audit the actual disturbance-age equation rather than its no-memory
    # control. The strongest sampled bracket is sufficient for causality.
    prefix_prediction, _, _ = candidate(baseline, data, 0.5, True)
    full_prefix = prefix_prediction[:96].copy()
    for values in data.values():
        values[96:] *= np.float32(0.5)
    future_prediction, _, _ = candidate(baseline, data, 0.5, True)
    prefix_delta = float(np.max(np.abs(full_prefix - future_prediction[:96])))
    print(
        "PREFIX stability=1 strength=0.5 "
        f"future_start=96 factor=0.5 max_abs={prefix_delta:.12g}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
