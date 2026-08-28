"""Bounded exact translations of prognostic connectivity into event footprint.

This scratch-only test replaces the incumbent instantaneous local-footprint
equation.  Upstream hazard remains the ignition/combustion occurrence proxy;
the variable part of surface-event size is controlled only by a pointwise
connectivity stock.  The stock recovers toward cover/GPP capacity and is
depleted by the candidate's own realised surface burning.  All equations use
globally shared constants and causal local state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prognostic_connectivity_monotonicity_a8ed115 import (  # noqa: E402
    connectivity_drivers,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)
from temperature_pathway_blend import ecological_ratios  # noqa: E402


def surface_share(data, model):
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = model._antecedent(gpp, 1.0 - np.exp(-1.0 / 12.0))
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    biomass = np.clip(data["aboveground_biomass"], 0.0, None)
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    surface = (1.0 - crop) * fine_fuel * open_cover
    woody = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    residue = crop * fine_fuel
    return np.asarray(surface / (0.05 + surface + woody + residue), dtype=np.float64)


def dynamic_local_footprint(prediction, data, p, enabled, model, config):
    """Replace static ignition-weighted footprint with conserved connectivity."""
    if "pathway_hazards" not in enabled:
        return prediction
    law, half, sharpness, recovery_months, consumption = config
    capacity, _, burnability = connectivity_drivers(model, data)
    share = surface_share(data, model)
    state = np.asarray(capacity[0], dtype=np.float64).copy()
    recovery = 1.0 - np.exp(-1.0 / float(recovery_months))
    background = float(max(p.get("fire_footprint_background", 0.5), 0.0))
    strength = float(max(p.get("fire_footprint_w", 0.0), 0.0))
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    output = np.empty_like(hazard, dtype=np.float64)

    for time in range(hazard.shape[0]):
        state += recovery * (capacity[time] - state)
        np.clip(state, 0.0, 1.0, out=state)
        burnable_rho = state * burnability[time]
        if law == "saturation":
            connected = burnable_rho / (burnable_rho + half)
        elif law == "percolation":
            connected = 1.0 / (
                1.0
                + np.exp(
                    np.clip(-sharpness * (burnable_rho - half), -40.0, 40.0)
                )
            )
        else:
            raise ValueError(law)
        surface_footprint = np.clip(
            background + strength * connected, 0.25, 3.0
        )
        footprint = 1.0 + share[time] * (surface_footprint - 1.0)
        output[time] = 1.0 - np.exp(
            -np.clip(hazard[time] * footprint, 0.0, 50.0)
        )

        # Burned surface fraction removes connected patches after this month's
        # event.  Direct area subtraction gives the state an interpretable
        # cell-fraction unit and cannot influence the current or earlier month.
        state -= float(consumption) * output[time] * share[time]
        np.clip(state, 0.0, 1.0, out=state)
    return np.asarray(output, dtype=np.float32)


def score_text(scores):
    global_score = scores["global"]
    return (
        f"{global_score['overall_score']:.9f} "
        f"bias={global_score['bias_score']:.6f} "
        f"rmse={global_score['rmse_score']:.6f} "
        f"season={global_score['seasonal_cycle_score']:.6f} "
        f"spatial={global_score['spatial_distribution_score']:.6f}"
    )


def main():
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    audit_data = dict(data)
    audit_data["annual_precipitation"] = 12.0 * model._antecedent(
        np.clip(data["monthly_precipitation"], 0.0, None),
        1.0 - np.exp(-1.0 / 12.0),
    )
    evaluator = GFED5Evaluator(GFED5_PATH)
    land = load_land_mask()
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    with Dataset(GFED5_PATH) as dataset:
        raw = np.asarray(dataset.variables["burntArea"][:192])
    observation = np.asarray(
        raw.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0,
        dtype=np.float32,
    )

    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_scores = evaluator.score(incumbent)
    incumbent_ecology = ecological_ratios(
        incumbent, audit_data, observation, area, land
    )
    capacity, ignition, burnability = connectivity_drivers(model, data)
    state_probe = capacity * burnability
    print("incumbent " + score_text(incumbent_scores), flush=True)
    print(
        "instant_rho_quantiles="
        + ",".join(
            f"{value:.6f}"
            for value in np.quantile(
                state_probe[:, land], (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
            )
        ),
        flush=True,
    )
    del capacity, ignition, burnability, state_probe

    # Structural brackets only.  Both laws inherit the incumbent finite
    # footprint range; thresholds bracket the observed input-state quartiles.
    configs = (
        ("saturation", 0.010, 0.0, 6.0, 1.0),
        ("saturation", 0.025, 0.0, 12.0, 1.0),
        ("saturation", 0.050, 0.0, 24.0, 1.0),
        ("saturation", 0.075, 0.0, 24.0, 0.0),
        ("saturation", 0.075, 0.0, 24.0, 0.5),
        ("saturation", 0.075, 0.0, 24.0, 1.0),
        ("saturation", 0.075, 0.0, 24.0, 2.0),
        ("saturation", 0.100, 0.0, 24.0, 1.0),
        ("saturation", 0.200, 0.0, 24.0, 1.0),
        ("percolation", 0.015, 50.0, 12.0, 1.0),
        ("percolation", 0.030, 35.0, 12.0, 1.0),
        ("percolation", 0.050, 25.0, 24.0, 1.0),
    )
    original = model._local_fire_footprint
    results = []
    for config in configs:
        model._local_fire_footprint = (
            lambda prediction, data_, p, enabled, config_=config:
            dynamic_local_footprint(
                prediction, data_, p, enabled, model, config_
            )
        )
        try:
            prediction = validate_prediction(
                model.predict(data, dict(model.PARAMS), None)
            )
        finally:
            model._local_fire_footprint = original
        scores = evaluator.score(prediction)
        label = ":".join(str(value) for value in config)
        results.append((scores["global"]["overall_score"], label, config, prediction, scores))
        print(
            f"{label} {score_text(scores)} "
            f"delta={scores['global']['overall_score'] - incumbent_scores['global']['overall_score']:+.9f}",
            flush=True,
        )

    print("TOP_AUDIT", flush=True)
    for overall, label, config, prediction, scores in sorted(results, reverse=True)[:4]:
        ecology = ecological_ratios(
            prediction, audit_data, observation, area, land
        )
        deltas = {
            region: scores[region]["overall_score"]
            - incumbent_scores[region]["overall_score"]
            for region in scores
            if region != "global"
        }
        print(
            f"{label} overall={overall:.9f} "
            f"regions={sum(delta > 0.0 for delta in deltas.values())}/14 "
            + "regional="
            + ",".join(
                f"{name}:{delta:+.6f}" for name, delta in sorted(deltas.items())
            )
            + " ecology="
            + ",".join(
                f"{name}:{value:.4f}({value - incumbent_ecology[name]:+.4f})"
                for name, value in ecology.items()
            ),
            flush=True,
        )

    winner = max(results, key=lambda row: row[0])
    future_data = {name: np.asarray(values).copy() for name, values in data.items()}
    for values in future_data.values():
        values[96:] = values[96:] * 1.7 + 0.123
    model._local_fire_footprint = (
        lambda prediction, data_, p, enabled:
        dynamic_local_footprint(
            prediction, data_, p, enabled, model, winner[2]
        )
    )
    try:
        future_prediction = validate_prediction(
            model.predict(future_data, dict(model.PARAMS), None)
        )
    finally:
        model._local_fire_footprint = original
    difference = np.abs(future_prediction[:96] - winner[3][:96])
    print(
        f"PREFIX winner={winner[1]} max_abs={float(difference.max()):.12g} "
        f"mean_abs={float(difference.mean()):.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
