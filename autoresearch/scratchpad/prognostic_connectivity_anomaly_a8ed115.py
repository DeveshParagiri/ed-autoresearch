"""Evolve prognostic connectivity as a weak causal footprint anomaly.

The incumbent static local-footprint equation retains its mean ecological and
spatial structure.  A conserved pointwise connectivity stock only modulates
surface-event size when current burnable connectivity departs from its own
trailing causal state.  Fixed weak strengths are screened on held whole-cell
annual and cycle errors before any full model replay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prognostic_connectivity_footprint_a8ed115 import surface_share  # noqa: E402
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

MONTH_DAYS = np.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    dtype=np.float64,
)


def static_footprint_fields(model, data, p):
    """Reproduce the incumbent static surface-footprint ingredients."""
    lightning = np.clip(data["lightning_flash_rate"], 0.0, None)
    lightning_12 = model._antecedent(
        lightning, 1.0 - np.exp(-1.0 / 12.0)
    )
    natural_ignition = lightning_12 / (
        lightning_12 + max(float(p["fire_footprint_lightning_half"]), 1e-8)
    )
    rangeland = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
    pasture = np.clip(data["luh2_pasture_fraction"], 0.0, 1.0)
    crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
    managed = np.clip(rangeland + pasture + crop, 0.0, 1.0)
    managed_access = managed / (
        managed + max(float(p["fire_footprint_managed_half"]), 1e-8)
    )
    natural = np.clip(data["natural_vegetation_fraction"], 0.0, 1.0)
    canopy = np.clip(data["natural_canopy_height"], 0.0, None)
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    natural_weight = float(np.clip(p["fire_footprint_natural_w"], 0.0, 1.0))
    activity = open_cover * (
        natural_weight * natural_ignition
        + (1.0 - natural_weight) * managed_access
    )
    surface_footprint = np.clip(
        float(p["fire_footprint_background"])
        + float(p["fire_footprint_w"]) * activity,
        0.1,
        3.0,
    )
    return np.asarray(surface_footprint, dtype=np.float64), surface_share(data, model)


def anomaly_value(rho, trailing, shape):
    if shape == "log_ratio":
        return np.clip(
            np.log((rho + 0.01) / (trailing + 0.01)), -1.0, 1.0
        )
    if shape == "bounded_difference":
        return np.clip(
            (rho - trailing) / (rho + trailing + 0.02), -0.75, 0.75
        )
    raise ValueError(shape)


def centered_footprint(
    prediction,
    data,
    p,
    enabled,
    model,
    shape,
    strength,
):
    """Apply one weak rho anomaly to the incumbent surface footprint."""
    if "pathway_hazards" not in enabled:
        return prediction
    capacity, _, burnability = connectivity_drivers(model, data)
    static_surface, share = static_footprint_fields(model, data, p)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    output = np.empty_like(hazard, dtype=np.float64)
    state = np.asarray(capacity[0], dtype=np.float64).copy()
    recovery = 1.0 - np.exp(-1.0 / 24.0)
    trailing = state * burnability[0]
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)

    for time in range(hazard.shape[0]):
        state += recovery * (capacity[time] - state)
        np.clip(state, 0.0, 1.0, out=state)
        rho = state * burnability[time]
        anomaly = anomaly_value(rho, trailing, shape)
        modulated_surface = np.clip(
            static_surface[time] * np.exp(float(strength) * anomaly),
            0.25,
            3.0,
        )
        footprint = 1.0 + share[time] * (modulated_surface - 1.0)
        output[time] = 1.0 - np.exp(
            -np.clip(hazard[time] * footprint, 0.0, 50.0)
        )

        # Realized surface burning breaks the post-event connectivity state.
        state -= output[time] * share[time]
        np.clip(state, 0.0, 1.0, out=state)
        trailing += alpha_12 * (rho - trailing)
    return np.asarray(output, dtype=np.float32)


def held_losses(prediction, observation, incumbent, area, land):
    """Return held annual-map and centered-cycle log losses."""
    rows, cols = np.where(land)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    pred = prediction[:, rows, cols].reshape(16, 12, -1).mean(axis=0)
    obs = observation[:, rows, cols].reshape(16, 12, -1).mean(axis=0)
    base = incumbent[:, rows, cols].reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred.sum(axis=0)
    obs_annual = obs.sum(axis=0)
    base_annual = base.sum(axis=0)
    eps = 1e-6
    annual_weight = area[rows, cols] * (
        obs_annual + np.maximum(base_annual - obs_annual, 0.0)
    )
    pred_allocation = pred / (pred_annual[None, :] + eps)
    obs_allocation = obs / (obs_annual[None, :] + eps)
    cycle_weight = (
        annual_weight[None, :]
        * np.maximum(obs_allocation, 1e-3)
        * MONTH_DAYS[:, None]
    )
    annual_residual = np.abs(
        np.log((pred_annual + eps) / (obs_annual + eps))
    )
    cycle_residual = np.abs(
        np.log((pred_allocation + eps) / (obs_allocation + eps))
    )
    annual = np.empty(4, dtype=np.float64)
    cycle = np.empty(4, dtype=np.float64)
    for fold in range(4):
        held = folds == fold
        annual[fold] = np.average(
            annual_residual[held], weights=annual_weight[held]
        )
        held_cycle = np.broadcast_to(held[None, :], cycle_residual.shape)
        cycle[fold] = np.average(
            cycle_residual[held_cycle], weights=cycle_weight[held_cycle]
        )
    return annual, cycle


def screen_prediction(
    stage_input,
    prepared,
    p,
    model,
    shape,
    strength,
    incumbent_local,
    incumbent_final,
):
    candidate_local = centered_footprint(
        stage_input,
        prepared,
        p,
        set(model.COMPONENTS),
        model,
        shape,
        strength,
    )
    old_hazard = -np.log1p(-np.clip(incumbent_local, 0.0, 1.0 - 1e-7))
    new_hazard = -np.log1p(-np.clip(candidate_local, 0.0, 1.0 - 1e-7))
    final_hazard = -np.log1p(-np.clip(incumbent_final, 0.0, 1.0 - 1e-7))
    relative = np.divide(
        new_hazard,
        old_hazard,
        out=np.ones_like(new_hazard),
        where=old_hazard > 1e-12,
    )
    return np.asarray(
        1.0 - np.exp(-np.clip(final_hazard * relative, 0.0, 50.0)),
        dtype=np.float32,
    )


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
    p = dict(model.PARAMS)
    land = load_land_mask()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    with Dataset(GFED5_PATH) as dataset:
        raw = np.asarray(dataset.variables["burntArea"][:192])
    observation = np.asarray(
        raw.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0,
        dtype=np.float32,
    )

    original = model._local_fire_footprint
    captured = {}

    def capture(prediction, prepared, params, enabled):
        captured["input"] = np.asarray(prediction, dtype=np.float32).copy()
        captured["prepared"] = prepared
        output = original(prediction, prepared, params, enabled)
        captured["output"] = np.asarray(output, dtype=np.float32).copy()
        return output

    model._local_fire_footprint = capture
    try:
        incumbent = validate_prediction(model.predict(data, p, None))
    finally:
        model._local_fire_footprint = original
    incumbent_scores = evaluator.score(incumbent)
    base_annual, base_cycle = held_losses(
        incumbent, observation, incumbent, area, land
    )
    print("incumbent " + score_text(incumbent_scores), flush=True)
    print(
        "base_annual=" + ",".join(f"{value:.9g}" for value in base_annual)
        + " base_cycle=" + ",".join(f"{value:.9g}" for value in base_cycle),
        flush=True,
    )

    survivors = []
    screens = []
    best_screen = None
    for shape in ("log_ratio", "bounded_difference"):
        for strength in (0.05, 0.10, 0.20):
            screened = screen_prediction(
                captured["input"],
                captured["prepared"],
                p,
                model,
                shape,
                strength,
                captured["output"],
                incumbent,
            )
            annual, cycle = held_losses(
                screened, observation, incumbent, area, land
            )
            annual_delta = annual - base_annual
            cycle_delta = cycle - base_cycle
            combined_delta = (
                annual_delta / base_annual + cycle_delta / base_cycle
            )
            annual_wins = int(np.sum(annual_delta < 0.0))
            cycle_wins = int(np.sum(cycle_delta < 0.0))
            combined_wins = int(np.sum(combined_delta < 0.0))
            stable = (
                annual_wins >= 3
                and cycle_wins >= 3
                and combined_wins >= 3
            )
            label = f"{shape}:s{strength:g}"
            mean_combined = float(combined_delta.mean())
            screens.append((mean_combined, label, shape, strength))
            if best_screen is None or mean_combined < best_screen[0]:
                best_screen = (
                    mean_combined,
                    label,
                    shape,
                    strength,
                    screened,
                )
            if stable:
                survivors.append((label, shape, strength))
            print(
                f"SCREEN {label} annual_delta="
                + ",".join(f"{value:+.8g}" for value in annual_delta)
                + f" wins={annual_wins}/4 cycle_delta="
                + ",".join(f"{value:+.8g}" for value in cycle_delta)
                + f" wins={cycle_wins}/4 combined="
                + ",".join(f"{value:+.8g}" for value in combined_delta)
                + f" wins={combined_wins}/4 stable={int(stable)}",
                flush=True,
            )

    print(
        "SURVIVORS " + (",".join(row[0] for row in survivors) or "none"),
        flush=True,
    )
    if not survivors:
        assert best_screen is not None
        best = best_screen
        print(
            f"NO_EXACT best_screen={best[1]} mean_combined={best[0]:+.9g}",
            flush=True,
        )
        # This remains an approximate screen prediction, not a full model
        # replay.  Audit it only to expose ecological direction and verify that
        # the rejected causal equation itself is prefix invariant.
        screen_scores = evaluator.score(best[4])
        audit_data = dict(data)
        audit_data["annual_precipitation"] = 12.0 * model._antecedent(
            np.clip(data["monthly_precipitation"], 0.0, None),
            1.0 - np.exp(-1.0 / 12.0),
        )
        incumbent_ecology = ecological_ratios(
            incumbent, audit_data, observation, area, land
        )
        screen_ecology = ecological_ratios(
            best[4], audit_data, observation, area, land
        )
        region_delta = {
            region: screen_scores[region]["overall_score"]
            - incumbent_scores[region]["overall_score"]
            for region in screen_scores
            if region != "global"
        }
        print(
            f"SCREEN_AUDIT {best[1]} {score_text(screen_scores)} "
            f"delta={screen_scores['global']['overall_score'] - incumbent_scores['global']['overall_score']:+.9f} "
            f"regions={sum(value > 0.0 for value in region_delta.values())}/14 "
            + "regional="
            + ",".join(
                f"{name}:{value:+.7f}" for name, value in sorted(region_delta.items())
            )
            + " ecology="
            + ",".join(
                f"{name}:{value:.5f}({value - incumbent_ecology[name]:+.5f})"
                for name, value in screen_ecology.items()
            ),
            flush=True,
        )

        future_data = {
            name: np.asarray(values).copy() for name, values in data.items()
        }
        for values in future_data.values():
            values[96:] = values[96:] * 1.7 + 0.123
        future_capture = {}

        def capture_future(prediction, prepared, params, enabled):
            future_capture["input"] = np.asarray(
                prediction, dtype=np.float32
            ).copy()
            future_capture["prepared"] = prepared
            output = original(prediction, prepared, params, enabled)
            future_capture["output"] = np.asarray(
                output, dtype=np.float32
            ).copy()
            return output

        model._local_fire_footprint = capture_future
        try:
            future_incumbent = validate_prediction(
                model.predict(future_data, p, None)
            )
        finally:
            model._local_fire_footprint = original
        future_screen = screen_prediction(
            future_capture["input"],
            future_capture["prepared"],
            p,
            model,
            best[2],
            best[3],
            future_capture["output"],
            future_incumbent,
        )
        difference = np.abs(future_screen[:96] - best[4][:96])
        print(
            f"PREFIX_SCREEN winner={best[1]} "
            f"max_abs={float(difference.max()):.12g} "
            f"mean_abs={float(difference.mean()):.12g}",
            flush=True,
        )
        return 0

    exact = []
    for label, shape, strength in survivors:
        model._local_fire_footprint = (
            lambda prediction, prepared, params, enabled,
            shape_=shape, strength_=strength:
            centered_footprint(
                prediction,
                prepared,
                params,
                enabled,
                model,
                shape_,
                strength_,
            )
        )
        try:
            prediction = validate_prediction(model.predict(data, p, None))
        finally:
            model._local_fire_footprint = original
        scores = evaluator.score(prediction)
        exact.append((scores["global"]["overall_score"], label, shape, strength, prediction, scores))
        print(
            f"EXACT {label} {score_text(scores)} "
            f"delta={scores['global']['overall_score'] - incumbent_scores['global']['overall_score']:+.9f}",
            flush=True,
        )

    audit_data = dict(data)
    audit_data["annual_precipitation"] = 12.0 * model._antecedent(
        np.clip(data["monthly_precipitation"], 0.0, None),
        1.0 - np.exp(-1.0 / 12.0),
    )
    incumbent_ecology = ecological_ratios(
        incumbent, audit_data, observation, area, land
    )
    print("EXACT_AUDIT", flush=True)
    for overall, label, shape, strength, prediction, scores in sorted(exact, reverse=True):
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
            f"regions={sum(value > 0.0 for value in deltas.values())}/14 "
            + "regional="
            + ",".join(f"{name}:{value:+.7f}" for name, value in sorted(deltas.items()))
            + " ecology="
            + ",".join(
                f"{name}:{value:.5f}({value - incumbent_ecology[name]:+.5f})"
                for name, value in ecology.items()
            ),
            flush=True,
        )

    winner = max(exact, key=lambda row: row[0])
    future_data = {name: np.asarray(values).copy() for name, values in data.items()}
    for values in future_data.values():
        values[96:] = values[96:] * 1.7 + 0.123
    model._local_fire_footprint = (
        lambda prediction, prepared, params, enabled:
        centered_footprint(
            prediction,
            prepared,
            params,
            enabled,
            model,
            winner[2],
            winner[3],
        )
    )
    try:
        future_prediction = validate_prediction(model.predict(future_data, p, None))
    finally:
        model._local_fire_footprint = original
    difference = np.abs(future_prediction[:96] - winner[4][:96])
    print(
        f"PREFIX winner={winner[1]} max_abs={float(difference.max()):.12g} "
        f"mean_abs={float(difference.mean()):.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
