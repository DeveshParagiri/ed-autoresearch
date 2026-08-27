"""Audit prefix invariance and sensitivity to the frozen ED state fields."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.runtime import load_inputs, load_model, validate_prediction  # noqa: E402


ED_STATE_INPUTS = (
    "gpp",
    "aboveground_biomass",
    "soil_carbon",
    "leaf_area_index",
    "natural_canopy_height",
    "secondary_canopy_height",
    "natural_vegetation_fraction",
    "secondary_vegetation_fraction",
)
FRACTIONS = {
    "natural_vegetation_fraction",
    "secondary_vegetation_fraction",
}


def clone(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values).copy() for name, values in data.items()}


def main() -> int:
    model = load_model()
    loaded = load_inputs(model.INPUTS)
    data = {name: np.asarray(values) for name, values in loaded.items()}
    baseline = validate_prediction(model.predict(data, dict(model.PARAMS), None))

    future = clone(data)
    for values in future.values():
        values[96:] *= 0.5
    future_prediction = validate_prediction(
        model.predict(future, dict(model.PARAMS), None)
    )
    earlier_delta = np.abs(future_prediction[:96] - baseline[:96])
    print(
        "future_half prefix_normalized_l1="
        f"{earlier_delta.sum() / (np.abs(baseline[:96]).sum() + 1e-12):.9f} "
        f"prefix_max_abs={earlier_delta.max():.9f}",
        flush=True,
    )

    area = np.cos(
        np.deg2rad(-89.5 + np.arange(180, dtype=np.float64))
    )[None, :, None]
    baseline_total = float((baseline * area).sum())
    for multiplier in (0.5, 1.5):
        perturbed = clone(data)
        for name in ED_STATE_INPUTS:
            if name not in perturbed:
                continue
            perturbed[name] *= multiplier
            if name in FRACTIONS:
                np.clip(perturbed[name], 0.0, 1.0, out=perturbed[name])
        prediction = validate_prediction(
            model.predict(perturbed, dict(model.PARAMS), None)
        )
        delta = np.abs(prediction - baseline)
        weighted_l1 = float((delta * area).sum() / (baseline_total + 1e-12))
        total_ratio = float((prediction * area).sum() / (baseline_total + 1e-12))
        print(
            f"ed_state_x{multiplier:.1f} normalized_l1={weighted_l1:.6f} "
            f"burn_total_ratio={total_ratio:.6f} max_abs={delta.max():.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
