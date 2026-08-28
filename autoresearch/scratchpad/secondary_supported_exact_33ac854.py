"""Exact audit of the held-positive rain/warm-supported secondary footprint.

All equations are pinned, fixed-strength, globally shared, pointwise, and
prefix-causal. This diagnostic never edits or officially evaluates model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from residual_phase_space_33ac854 import antecedent, load_pinned  # noqa: E402
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from secondary_regrowth_footprint_33ac854 import (  # noqa: E402
    candidate,
    secondary_regrowth_states,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    state = secondary_regrowth_states(data)["supported"]
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 12.0
    )
    base_score = evaluator.score(incumbent)
    base_ecology = ecological_ratios(incumbent, prepared, observed, area, land)
    print(f"BASE overall={base_score['global']['overall_score']:.9f}")

    observed_annual = observed.mean(axis=0)
    ranking = np.argsort((area * observed_annual).ravel())[::-1][:64]
    rows, cols = ranking // 360, ranking % 360
    prefix_data = {
        key: np.asarray(values[:, rows, cols])[:, None, :]
        for key, values in data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    prefix_state = secondary_regrowth_states(prefix_data)["supported"]
    perturbed = {key: values.copy() for key, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    perturbed_incumbent = model.predict(perturbed, dict(model.PARAMS), None)
    perturbed_state = secondary_regrowth_states(perturbed)["supported"]

    for strength in (0.5, 1.0, 2.0, 4.0):
        trial = candidate(incumbent, state, strength)
        score = evaluator.score(validate_prediction(trial))
        global_score = score["global"]
        deltas = {
            region: score[region]["overall_score"] - base_score[region]["overall_score"]
            for region in score
            if region != "global"
        }
        ecology = ecological_ratios(trial, prepared, observed, area, land)
        prefix_trial = candidate(prefix_incumbent, prefix_state, strength)
        perturbed_trial = candidate(perturbed_incumbent, perturbed_state, strength)
        prefix_error = float(
            np.max(np.abs(prefix_trial[:96] - perturbed_trial[:96]))
        )
        print(
            f"EXACT strength={strength:g} overall={global_score['overall_score']:.9f} "
            f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
            f"seasonal={global_score['seasonal_cycle_score']:.9f} "
            f"spatial={global_score['spatial_distribution_score']:.9f} "
            f"regions={sum(value > 0.0 for value in deltas.values())}/{len(deltas)} "
            f"prefix={prefix_error:.12g}"
        )
        print(
            "REGIONAL "
            + ",".join(f"{region}:{value:+.6f}" for region, value in sorted(deltas.items()))
        )
        print(
            "ECOLOGY "
            + ",".join(
                f"{regime}:{base_ecology[regime]:.5f}->{ecology[regime]:.5f}"
                for regime in base_ecology
            )
        )


if __name__ == "__main__":
    main()
