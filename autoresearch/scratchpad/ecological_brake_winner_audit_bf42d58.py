"""Audit the only scalar-positive ecological-brake placement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch import model  # noqa: E402
from autoresearch.scratchpad.ecological_brake_stage_placement_bf42d58 import (  # noqa: E402
    predict_at_stage,
)
from autoresearch.scratchpad.surface_capacity_stage_placement_bf42d58 import (  # noqa: E402
    prepared_inputs,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    ecological_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
)
from scripts.runtime import load_land_mask  # noqa: E402


def area_grid() -> np.ndarray:
    radius = 6.371e6
    bounds = np.deg2rad(np.arange(-90.0, 91.0, 1.0))
    latitude_area = radius**2 * np.diff(np.sin(bounds)) * np.deg2rad(1.0)
    return np.repeat(latitude_area[:, None], 360, axis=1)


def area_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray,
) -> float:
    model_annual = prediction.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    obs_annual = observation.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    return float(np.sum(model_annual[mask] * area[mask])) / max(
        float(np.sum(obs_annual[mask] * area[mask])), 1e-12
    )


def subset_inputs(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    columns: np.ndarray,
    future_scale: float,
) -> dict[str, np.ndarray]:
    selected = {
        name: np.asarray(data[name][:, rows, columns][:, None, :]).copy()
        for name in model.INPUTS
    }
    for values in selected.values():
        values[96:] *= future_scale
    rain = np.asarray(selected["monthly_precipitation"], dtype=np.float32)
    selected["annual_precipitation"] = 12.0 * model._antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    return selected


def main() -> int:
    data = prepared_inputs()
    observation = load_observation()
    area = area_grid()
    mean = {
        name: np.asarray(data[name], dtype=np.float64).mean(axis=0)
        for name in (
            "monthly_precipitation",
            "air_temperature",
            "leaf_area_index",
            "natural_canopy_height",
            "aboveground_biomass",
            "natural_vegetation_fraction",
            "luh2_primary_fraction",
            "luh2_cropland_fraction",
            "luh2_rangeland_fraction",
        )
    }
    masks = ecological_masks(mean)
    variants = {
        "incumbent": predict_at_stage(data, "transform"),
        "post_event_brake": predict_at_stage(data, "pathway_event_scaling"),
    }
    for label, prediction in variants.items():
        for name, mask in masks.items():
            print(
                f"ECOLOGY variant={label} name={name} cells={int(mask.sum())} "
                f"ratio={area_ratio(prediction,observation,area,mask):.9f}",
                flush=True,
            )

    land = load_land_mask()
    rows, columns = np.nonzero(land)
    chosen = np.linspace(0, rows.size - 1, 96, dtype=np.int64)
    original = subset_inputs(data, rows[chosen], columns[chosen], 1.0)
    perturbed = subset_inputs(data, rows[chosen], columns[chosen], 1.5)
    for insertion in ("transform", "pathway_event_scaling"):
        base = predict_at_stage(original, insertion)
        changed = predict_at_stage(perturbed, insertion)
        print(
            f"PREFIX insertion={insertion} "
            f"max_abs={float(np.max(np.abs(base[:96]-changed[:96]))):.9e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
