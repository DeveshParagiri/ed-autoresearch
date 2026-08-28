"""Held-block screen of pointwise land-cover interface mechanisms.

The two candidate families use only current local fractions.  They contain no
neighbourhood operation, coordinate, region, target, residual, or fitted
runtime statistic.  Natural-managed co-occurrence supplies anthropogenic
ignition access, while crop/urban interspersion independently limits spread.

The full-grid score is forbidden unless one fixed bracket lowers annual-log,
normalised-allocation, and raw-cycle loss in all four disjoint whole-cell
folds.  The exact canonical source is loaded from git object ``121c83c``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.rain_fuel_pathway_probe import ecological_ratios  # noqa: E402
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    MONTH_DAYS,
    PINNED,
    held_losses,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


# Each coefficient is a maximum log-hazard effect because the two signals are
# clipped to [0, 1].  Controls isolate the two effects; combined brackets test
# 10%, 20%, and 35% log-scale responses without tuning a free surface.
BRACKETS = (
    (0.20, 0.00),
    (0.00, 0.20),
    (0.10, 0.10),
    (0.20, 0.10),
    (0.10, 0.20),
    (0.20, 0.20),
    (0.35, 0.20),
    (0.20, 0.35),
)


def compositional_shares(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return five exhaustive pointwise shares on the current timestep."""
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    secondary = np.clip(
        np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    urban = np.clip(
        np.asarray(data["luh2_urban_fraction"], dtype=np.float64), 0.0, 1.0
    )
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    developed = np.clip(crop + urban, 0.0, 1.0)
    known = natural + secondary + managed_open + developed
    other = np.maximum(1.0 - known, 0.0)
    total = known + other + 1e-12
    return tuple(value / total for value in (natural, secondary, managed_open, developed, other))


def interface_signals(
    data: dict[str, np.ndarray], family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return separate access, spread-break, and diversity signals."""
    natural, secondary, managed_open, developed, other = compositional_shares(data)
    wild = natural + secondary
    burnable = np.clip(wild + managed_open, 0.0, 1.0)
    pairwise_access = np.clip(4.0 * wild * managed_open, 0.0, 1.0)
    pairwise_break = np.clip(4.0 * developed * burnable, 0.0, 1.0)
    diversity = np.clip(
        (1.0 - (
            np.square(natural)
            + np.square(secondary)
            + np.square(managed_open)
            + np.square(developed)
            + np.square(other)
        )) / 0.8,
        0.0,
        1.0,
    )
    if family == "pairwise":
        access = pairwise_access
        spread_break = pairwise_break
    elif family == "simpson":
        access = diversity * pairwise_access
        spread_break = diversity * developed
    else:
        raise ValueError(f"unknown interface family {family!r}")
    return access, np.clip(spread_break, 0.0, 1.0), diversity


def candidate(
    incumbent: np.ndarray,
    data: dict[str, np.ndarray],
    family: str,
    access_strength: float,
    spread_strength: float,
) -> np.ndarray:
    access, spread_break, _ = interface_signals(data, family)
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    log_factor = np.clip(
        access_strength * access - spread_strength * spread_break,
        -0.5,
        0.5,
    )
    adjusted = hazard * np.exp(log_factor)
    return np.asarray(-np.expm1(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def causal_annual_rain(model, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prepared = dict(data)
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float32)
    prepared["annual_precipitation"] = 12.0 * model._antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    return prepared


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_scores = evaluator.score(incumbent)
    incumbent_score = incumbent_scores["global"]["overall_score"]
    if abs(incumbent_score - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {incumbent_score:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual_grid = np.average(observed, axis=0, weights=MONTH_DAYS)
    land = load_land_mask()
    rows, columns = np.where(land)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4

    selected_data = {
        name: np.asarray(values[:, rows, columns], dtype=np.float64)
        for name, values in data.items()
    }
    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float64)
    selected_observed = np.asarray(observed[:, rows, columns], dtype=np.float64)
    selected_area = area_grid[rows, columns]
    selected_obs_annual = obs_annual_grid[rows, columns]
    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        selected_obs_annual,
        folds,
    )
    print(
        f"BASE pinned={PINNED} overall={incumbent_score:.9f} land_cells={rows.size} "
        f"fold_cells={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )

    survivors: list[tuple[float, str, float, float]] = []
    best: tuple[float, str, float, float] | None = None
    for family in ("pairwise", "simpson"):
        access, spread_break, diversity = interface_signals(selected_data, family)
        print(
            f"STATE family={family} access_mean={access.mean():.9f} "
            f"access_p95={np.quantile(access, .95):.9f} "
            f"break_mean={spread_break.mean():.9f} "
            f"break_p95={np.quantile(spread_break, .95):.9f} "
            f"diversity_mean={diversity.mean():.9f}",
            flush=True,
        )
        for access_strength, spread_strength in BRACKETS:
            trial = candidate(
                selected_incumbent,
                selected_data,
                family,
                access_strength,
                spread_strength,
            )
            losses = held_losses(
                trial,
                selected_observed,
                selected_area,
                selected_obs_annual,
                folds,
            )
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            all_gates = all(np.all(gain > 0.0) for gain in gains)
            aggregate = float(
                sum(np.sum(gains[index] / base_losses[index]) for index in range(3))
            )
            print(
                f"HELD family={family} access={access_strength:.2f} "
                f"spread={spread_strength:.2f} all_gates={int(all_gates)} "
                f"aggregate={aggregate:+.9f} annual_gain="
                + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain="
                + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain="
                + ",".join(f"{value:+.9f}" for value in gains[2]),
                flush=True,
            )
            item = (aggregate, family, access_strength, spread_strength)
            if best is None or aggregate > best[0]:
                best = item
            if all_gates and access_strength > 0.0 and spread_strength > 0.0:
                survivors.append(item)

    assert best is not None
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    probe_data = {
        name: values[:, None, probe].copy() for name, values in selected_data.items()
    }
    before_incumbent = model.predict(probe_data, dict(model.PARAMS), None)
    before = candidate(before_incumbent, probe_data, best[1], best[2], best[3])
    changed = {name: values.copy() for name, values in probe_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    after_incumbent = model.predict(changed, dict(model.PARAMS), None)
    after = candidate(after_incumbent, changed, best[1], best[2], best[3])
    prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
    print(
        f"PREFIX best={best[1]}:a{best[2]:.2f}:b{best[3]:.2f} "
        f"max_abs={prefix_max:.12g}",
        flush=True,
    )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix causality failed: {prefix_max}")

    if not survivors:
        print("DECISION exact=0 reject=no_combined_all_block_all_metric_survivor", flush=True)
        return 0

    survivors.sort(reverse=True)
    _, family, access_strength, spread_strength = survivors[0]
    trial = validate_prediction(
        candidate(incumbent, data, family, access_strength, spread_strength)
    )
    trial_scores = evaluator.score(trial)
    global_score = trial_scores["global"]
    print(
        f"DECISION exact=1 family={family} access={access_strength:.2f} "
        f"spread={spread_strength:.2f}",
        flush=True,
    )
    print(
        f"EXACT overall={global_score['overall_score']:.9f} "
        f"delta={global_score['overall_score']-incumbent_score:+.9f} "
        f"bias={global_score['bias_score']:.9f} rmse={global_score['rmse_score']:.9f} "
        f"seasonal={global_score['seasonal_cycle_score']:.9f} "
        f"spatial={global_score['spatial_distribution_score']:.9f}",
        flush=True,
    )
    prepared = causal_annual_rain(model, data)
    base_ecology = ecological_ratios(incumbent, prepared, observed, area_grid, land)
    trial_ecology = ecological_ratios(trial, prepared, observed, area_grid, land)
    print(
        "ECOLOGY "
        + ",".join(
            f"{name}:{base_ecology[name]:.5f}->{trial_ecology[name]:.5f}"
            for name in base_ecology
        ),
        flush=True,
    )
    print(
        "REGIONS "
        + ",".join(
            f"{name}:{trial_scores[name]['overall_score']-incumbent_scores[name]['overall_score']:+.6f}"
            for name in sorted(key for key in trial_scores if key != "global")
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
