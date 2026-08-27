"""Cheap target-stratified screen for the two canonical phenology stages.

This is diagnostic triage, not an official evaluation.  It selects high-fire
cells after defining three audit-only regimes, then runs the unchanged global
local-state model on those cell histories.  The target and masks never enter
the prediction path.  Exact full-grid candidates are serialized separately.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_80368d8 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    target_masks,
)
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    INPUTS_DIR,
    load_inputs,
    load_land_mask,
    load_model,
)


MASK_INPUTS = (
    "monthly_precipitation",
    "air_temperature",
    "dryness",
    "luh2_primary_fraction",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_urban_fraction",
    "natural_canopy_height",
    "leaf_area_index",
    "aboveground_biomass",
    "natural_vegetation_fraction",
)


def one_degree_area() -> np.ndarray:
    radius = 6.371e6
    bounds = np.deg2rad(np.arange(-90.0, 91.0, 1.0))
    latitude_band = radius**2 * np.diff(np.sin(bounds)) * np.deg2rad(1.0)
    return np.repeat(latitude_band[:, None], 360, axis=1)


def selected_input(
    name: str, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    source: Path | None = None
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            if name in dataset.variables:
                source = path
                break
    if source is None:
        raise KeyError(name)
    output = np.empty((192, rows.size), dtype=np.float32)
    with Dataset(source) as dataset:
        variable = dataset.variables[name]
        for row in np.unique(rows):
            positions = np.flatnonzero(rows == row)
            slab = np.ma.asarray(variable[:, int(row), :])
            if np.ma.getmaskarray(slab).any():
                raise ValueError(f"masked sampled input: {name}")
            output[:, positions] = np.asarray(
                slab[:, columns[positions]], dtype=np.float32
            )
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite sampled input: {name}")
    return output[:, None, :]


def select_cells(
    masks: dict[str, np.ndarray],
    obs_annual: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quotas = {
        "warm_seasonal_open": 384,
        "intact_tropical_closed": 192,
        "cropland": 192,
    }
    selected: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    used = np.zeros(obs_annual.shape, dtype=bool)
    weight = obs_annual * area
    for label_index, (name, quota) in enumerate(quotas.items()):
        candidates = np.flatnonzero(masks[name] & ~used)
        order = np.argsort(weight.ravel()[candidates])[::-1]
        chosen = candidates[order[:quota]]
        used.ravel()[chosen] = True
        selected.append(chosen)
        labels.append(np.full(chosen.size, label_index, dtype=np.int8))
        print(
            f"SAMPLE {name} requested={quota} selected={chosen.size} "
            f"obs_weight={float(weight.ravel()[chosen].sum()):.9e}",
            flush=True,
        )
    flat = np.concatenate(selected)
    group = np.concatenate(labels)
    return flat // 360, flat % 360, group


def diagnostics(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | str | int]:
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation.reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    cell_area = area[selected]
    obs_weight = obs_annual * cell_area
    pred_monthly = np.sum(pred_cycle * cell_area[None, :], axis=1)
    obs_monthly = np.sum(obs_cycle * cell_area[None, :], axis=1)
    pred_norm = pred_monthly / max(float(pred_monthly.sum()), 1e-12)
    obs_norm = obs_monthly / max(float(obs_monthly.sum()), 1e-12)
    pred_peak = int(np.argmax(pred_monthly))
    obs_peak = int(np.argmax(obs_monthly))
    phase = min(abs(pred_peak - obs_peak), 12 - abs(pred_peak - obs_peak))
    pred_anomaly = pred_cycle - pred_cycle.mean(axis=0, keepdims=True)
    obs_anomaly = obs_cycle - obs_cycle.mean(axis=0, keepdims=True)
    centered_rmse = np.sqrt(np.mean(np.square(pred_anomaly - obs_anomaly), axis=0))
    reference_scale = np.sqrt(np.mean(np.square(obs_anomaly), axis=0))
    return {
        "ratio": float(np.sum(pred_annual * cell_area))
        / max(float(np.sum(obs_annual * cell_area)), 1e-12),
        "annual_abs": float(np.sum(np.abs(pred_annual - obs_annual) * cell_area))
        / max(float(np.sum(obs_annual * cell_area)), 1e-12),
        "annual_log": float(
            np.sum(
                obs_weight
                * np.abs(
                    np.log((pred_annual + 1e-6) / (obs_annual + 1e-6))
                )
            )
        )
        / max(float(obs_weight.sum()), 1e-12),
        "centered_rmse": float(
            np.sum(obs_weight * centered_rmse / (reference_scale + 1e-6))
        )
        / max(float(obs_weight.sum()), 1e-12),
        "seasonal_l1": 0.5 * float(np.sum(np.abs(pred_norm - obs_norm))),
        "model_peak": pred_peak,
        "obs_peak": obs_peak,
        "phase": phase,
    }


def main() -> int:
    model = load_model()
    # The parent exact script checks the same blob before full-grid work.  A
    # sampled run still refuses a moving model by checking its declared values.
    import subprocess

    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"model blob changed: {blob}")

    mask_data = load_inputs(MASK_INPUTS)
    land = load_land_mask()
    masks = target_masks(mask_data, model, land)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = (
        reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    del reference
    area = one_degree_area()
    obs_cycle = observation.reshape(16, 12, 180, 360).mean(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    rows, columns, group = select_cells(masks, obs_annual, area)
    del obs_cycle, obs_annual, masks, land

    sampled: dict[str, np.ndarray] = {}
    for name in model.INPUTS:
        if name in mask_data:
            sampled[name] = np.asarray(
                mask_data[name][:, rows, columns][:, None, :], dtype=np.float32
            )
        else:
            sampled[name] = selected_input(name, rows, columns)
    del mask_data
    gc.collect()
    sampled_observation = observation[:, rows, columns][:, None, :]
    sampled_area = area[rows, columns]
    del observation, area
    gc.collect()

    base_fire = float(model.PARAMS["fire_season_w"])
    base_green = float(model.PARAMS["greenup_brake"])
    configurations = (
        ("incumbent", base_fire, base_green),
        ("half_greenup", base_fire, 0.5 * base_green),
        ("no_greenup", base_fire, 0.0),
        ("half_allocator", 0.5 * base_fire, base_green),
        ("no_allocator", 0.0, base_green),
        ("both_half", 0.5 * base_fire, 0.5 * base_green),
        ("both_off", 0.0, 0.0),
    )
    group_names = (
        "warm_seasonal_open",
        "intact_tropical_closed",
        "cropland",
    )
    outputs: dict[str, dict[str, dict[str, float | str | int]]] = {}
    for label, fire, green in configurations:
        params = dict(model.PARAMS)
        params["fire_season_w"] = fire
        params["greenup_brake"] = green
        prediction = np.asarray(
            model.predict(sampled, params, None), dtype=np.float32
        )
        if prediction.shape != sampled_observation.shape:
            raise ValueError(f"sample prediction shape {prediction.shape}")
        outputs[label] = {}
        for index, name in enumerate(group_names):
            chosen = group == index
            values = diagnostics(
                prediction[:, :, chosen],
                sampled_observation[:, :, chosen],
                sampled_area,
                chosen,
            )
            outputs[label][name] = values
            print(
                f"RESULT {label} fire={fire:g} green={green:g} group={name} "
                + " ".join(f"{key}={value}" for key, value in values.items()),
                flush=True,
            )
        del prediction
        gc.collect()

    incumbent = outputs["incumbent"]
    print("DELTAS", flush=True)
    for label, groups in outputs.items():
        for name, values in groups.items():
            base = incumbent[name]
            print(
                f"{label} {name} "
                + " ".join(
                    f"d_{metric}={float(values[metric]) - float(base[metric]):+.9f}"
                    for metric in (
                        "ratio",
                        "annual_abs",
                        "annual_log",
                        "centered_rmse",
                        "seasonal_l1",
                    )
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
