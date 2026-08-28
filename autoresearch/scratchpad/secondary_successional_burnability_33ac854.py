"""Refine secondary-open footprint by continuous successional burnability.

The accepted secondary footprint treats every unit of open secondary structure
alike.  This diagnostic keeps its fixed 0.5 strength but redistributes that
capacity continuously: young regrowth is lower than nearby natural canopy and
has little woody biomass, whereas mature closed secondary approaches natural
canopy height and woody biomass.  Four parameter-free combinations reuse the
incumbent 8 m canopy and 1 kg C m-2 biomass half-saturations.

All candidate equations are globally shared, pointwise, and prefix causal.  No
target, coordinate, region, neighbour, future value, or new input enters the
runtime state.  Exact scoring is allowed only for a stable held-cell survivor
against the fixed generic secondary-footprint candidate.
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
from secondary_regrowth_footprint_33ac854 import (  # noqa: E402
    MONTH_DAYS,
    candidate,
    losses,
    secondary_regrowth_states,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402


EXPECTED_GENERIC = 0.719645835
STRENGTH = 0.5


def successional_factors(data) -> dict[str, np.ndarray]:
    natural_canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    secondary_canopy = np.clip(
        np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    # Young secondary is short relative to the locally available natural
    # canopy. The +8 m scale is already used to separate open and woody cover.
    height_factor = np.clip(
        1.0
        + (natural_canopy - secondary_canopy)
        / (natural_canopy + secondary_canopy + 8.0),
        0.5,
        1.5,
    )
    # The incumbent woody-capacity equation uses B/(B+1); its complement,
    # normalized to one at the same half-saturation, is a young-fuel factor.
    biomass_factor = np.clip(2.0 / (biomass + 1.0), 0.5, 1.5)
    geometric = np.sqrt(height_factor * biomass_factor)
    harmonic = 2.0 * height_factor * biomass_factor / (
        height_factor + biomass_factor + 1e-12
    )
    return {
        "height_contrast": height_factor,
        "biomass_youth": biomass_factor,
        "joint_geometric": np.clip(geometric, 0.5, 1.5),
        "joint_harmonic": np.clip(harmonic, 0.5, 1.5),
    }


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    structural = secondary_regrowth_states(data)["structural"]
    generic = candidate(incumbent, structural, STRENGTH)
    factors = successional_factors(data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    generic_score = evaluator.score(validate_prediction(generic))
    generic_global = generic_score["global"]
    print(
        f"GENERIC overall={generic_global['overall_score']:.9f} "
        f"bias={generic_global['bias_score']:.9f} "
        f"rmse={generic_global['rmse_score']:.9f} "
        f"seasonal={generic_global['seasonal_cycle_score']:.9f} "
        f"spatial={generic_global['spatial_distribution_score']:.9f}",
        flush=True,
    )
    if abs(generic_global["overall_score"] - EXPECTED_GENERIC) > 5e-7:
        raise RuntimeError("failed generic secondary-footprint reproduction")

    with Dataset(GFED5_PATH) as dataset:
        fine_observed = np.asarray(dataset.variables["burntArea"][:192])
    observed = (
        fine_observed.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    )
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observed_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    weight = area * observed_annual
    ranking = np.argsort(weight.ravel())[::-1]
    coverage = np.cumsum(weight.ravel()[ranking]) / weight.sum()
    cells = ranking[: int(np.searchsorted(coverage, 0.90) + 1)]
    rows, columns = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    generic_annual, generic_cycle = losses(
        generic, observed, area, cells, folds
    )
    print(
        f"DESIGN cells={cells.size} retained_fire={float(weight.ravel()[cells].sum()/weight.sum()):.9f} "
        "held_whole_cells=1 runtime_geography=0",
        flush=True,
    )

    survivors = []
    variants = {}
    for name, factor in factors.items():
        print(
            f"FACTOR name={name} p05={np.quantile(factor[:,rows,columns],0.05):.6f} "
            f"median={np.median(factor[:,rows,columns]):.6f} "
            f"p95={np.quantile(factor[:,rows,columns],0.95):.6f}",
            flush=True,
        )
        trial = candidate(incumbent, structural * factor, STRENGTH)
        variants[name] = trial
        annual, cycle = losses(trial, observed, area, cells, folds)
        annual_gain = generic_annual - annual
        cycle_gain = generic_cycle - cycle
        objective = float(annual_gain.sum() + cycle_gain.sum())
        stable = bool(
            np.count_nonzero(annual_gain > 0.0) >= 3
            and objective > 0.0
            and -cycle_gain.sum() <= 0.05 * max(annual_gain.sum(), 1e-12)
        )
        print(
            f"VARIANT name={name} stable={int(stable)} annual_gain="
            + ",".join(f"{value:+.6f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.6f}" for value in cycle_gain)
            + f" objective={objective:+.9f}",
            flush=True,
        )
        if stable:
            survivors.append((objective, name, trial))

    prefix_cells = ranking[:64]
    prefix_rows, prefix_columns = prefix_cells // 360, prefix_cells % 360
    prefix_data = {
        key: np.asarray(values[:, prefix_rows, prefix_columns])[:, None, :]
        for key, values in data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    prefix_structural = secondary_regrowth_states(prefix_data)["structural"]
    prefix_factors = successional_factors(prefix_data)
    perturbed = {key: values.copy() for key, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    perturbed_incumbent = model.predict(perturbed, dict(model.PARAMS), None)
    perturbed_structural = secondary_regrowth_states(perturbed)["structural"]
    perturbed_factors = successional_factors(perturbed)
    for name in factors:
        original = candidate(
            prefix_incumbent, prefix_structural * prefix_factors[name], STRENGTH
        )
        changed = candidate(
            perturbed_incumbent,
            perturbed_structural * perturbed_factors[name],
            STRENGTH,
        )
        print(
            f"PREFIX name={name} max_abs={np.max(np.abs(original[:96]-changed[:96])):.12g}",
            flush=True,
        )

    if not survivors:
        print("EXACT skipped: no stable held survivor", flush=True)
        return
    survivors.sort(reverse=True)
    _, name, trial = survivors[0]
    score = evaluator.score(validate_prediction(trial))
    global_score = score["global"]
    deltas = {
        region: score[region]["overall_score"]
        - generic_score[region]["overall_score"]
        for region in score
        if region != "global"
    }
    print(
        f"EXACT name={name} overall={global_score['overall_score']:.9f} "
        f"delta={global_score['overall_score']-generic_global['overall_score']:+.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f}",
        flush=True,
    )
    print(
        f"REGIONS improved={sum(value > 0.0 for value in deltas.values())}/14 "
        + ",".join(f"{region}:{value:+.6f}" for region, value in sorted(deltas.items())),
        flush=True,
    )
    prepared = dict(data)
    prepared["annual_precipitation"] = 12.0 * antecedent(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 12.0
    )
    land = load_land_mask()
    generic_ecology = ecological_ratios(generic, prepared, observed, area, land)
    candidate_ecology = ecological_ratios(trial, prepared, observed, area, land)
    print(
        "ECOLOGY "
        + ",".join(
            f"{regime}:{generic_ecology[regime]:.5f}->{candidate_ecology[regime]:.5f}"
            for regime in generic_ecology
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
