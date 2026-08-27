"""Held-cell diagnostic of causal LUH2 land-use transitions.

The baseline feature set contains current LUH2 cover and prefix-causal local
climate, fuel, lightning, and incumbent-fire states.  The augmented set adds
only current/lagged LUH2 changes.  Coordinates define four held spatial blocks
but are never features.  Learned surfaces are diagnostic only and never enter
the canonical model or official ledger.
"""

from __future__ import annotations

import gc
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    MONTH_DAYS,
    load_observed,
    load_selected,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


LAND_NAMES = (
    "luh2_primary_fraction",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_urban_fraction",
)


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def lag_difference(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    output[lag:] = values[lag:] - values[:-lag]
    return output


def select_cells(evaluator: GFED5Evaluator, fraction: float = 0.82):
    mean = (
        np.asarray(evaluator.reference_mean, dtype=np.float64)
        .reshape(180, 2, 360, 2)
        .mean(axis=(1, 3))
        / 100.0
    )
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    weight = mean * area
    order = np.argsort(weight.ravel())[::-1]
    cumulative = np.cumsum(weight.ravel()[order]) / weight.sum()
    count = int(np.searchsorted(cumulative, fraction) + 1)
    cells = order[:count]
    rows, cols = cells // 360, cells % 360
    return rows, cols, area[rows, cols], weight[rows, cols], float(cumulative[count - 1])


def tree_counts(regressor, names):
    counts = Counter()
    pairs = Counter()
    for stage in regressor._predictors:
        nodes = stage[0].nodes
        stack = [0]
        while stack:
            node = stack.pop()
            if nodes["is_leaf"][node]:
                continue
            name = names[int(nodes["feature_idx"][node])]
            counts[name] += 1
            for child_name in ("left", "right"):
                child = int(nodes[child_name][node])
                if not nodes["is_leaf"][child]:
                    other = names[int(nodes["feature_idx"][child])]
                    pairs[tuple(sorted((name, other)))] += 1
                stack.append(child)
    return counts, pairs


def build_monthly_features(data, prediction):
    base_names = []
    base_columns = []
    transition_names = []
    transition_columns = []

    def field(name):
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    def add_base(name, values):
        base_names.append(name)
        base_columns.append(np.asarray(values, dtype=np.float32).reshape(-1))

    def add_transition(name, values):
        transition_names.append(name)
        transition_columns.append(np.asarray(values, dtype=np.float32).reshape(-1))

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    dryness = np.clip(field("dryness"), 0.0, None)
    temperature = field("air_temperature")
    gpp = np.clip(field("gpp"), 0.0, None)
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))

    rain3, rain12 = antecedent(rain, 3.0), antecedent(rain, 12.0)
    dry3, dry12 = antecedent(dryness, 3.0), antecedent(dryness, 12.0)
    temp3, temp12 = antecedent(temperature, 3.0), antecedent(temperature, 12.0)
    gpp3, gpp12 = antecedent(gpp, 3.0), antecedent(gpp, 12.0)
    flash12 = antecedent(lightning, 12.0)
    hazard12 = antecedent(hazard, 12.0)

    add_base("log_rain", np.log1p(rain))
    add_base("log_dryness", np.log1p(dryness))
    add_base("temperature", temperature)
    add_base("log_gpp", np.log1p(gpp))
    add_base("log_lightning", np.log1p(1000.0 * lightning))
    add_base("log_rain3", np.log1p(rain3))
    add_base("log_rain12", np.log1p(rain12))
    add_base("log_dryness3", np.log1p(dry3))
    add_base("log_dryness12", np.log1p(dry12))
    add_base("temperature3", temp3)
    add_base("temperature12", temp12)
    add_base("log_gpp3", np.log1p(gpp3))
    add_base("log_gpp12", np.log1p(gpp12))
    add_base("log_lightning12", np.log1p(1000.0 * flash12))
    add_base("incumbent_hazard", hazard)
    add_base("incumbent_hazard12", hazard12)
    add_base("rain_deficit3", (rain3 - rain) / (rain3 + rain + 10.0))
    add_base("dryness_rise3", (dryness - dry3) / (dryness + dry3 + 100.0))
    add_base("warming3", temperature - temp3)
    add_base("gpp_departure3", (gpp - gpp3) / (gpp + gpp3 + 0.05))

    land = {}
    changes12 = {}
    impulses = {}
    for name in LAND_NAMES:
        values = np.clip(field(name), 0.0, 1.0)
        land[name] = values
        add_base(name, values)
        change12 = lag_difference(values, 12)
        changes12[name] = change12
        impulse = lag_difference(values, 1)
        impulses[name] = impulse
        short = antecedent(impulse, 3.0)
        long = antecedent(impulse, 12.0)
        stem = name.removeprefix("luh2_").removesuffix("_fraction")
        add_transition(f"{stem}_change12", change12)
        add_transition(f"{stem}_gain12", np.maximum(change12, 0.0))
        add_transition(f"{stem}_loss12", np.maximum(-change12, 0.0))
        add_transition(f"{stem}_impulse_ema3", short)
        add_transition(f"{stem}_impulse_ema12", long)

    primary_loss = np.maximum(-changes12["luh2_primary_fraction"], 0.0)
    crop_gain = np.maximum(changes12["luh2_cropland_fraction"], 0.0)
    pasture_gain = np.maximum(changes12["luh2_pasture_fraction"], 0.0)
    range_gain = np.maximum(changes12["luh2_rangeland_fraction"], 0.0)
    urban_gain = np.maximum(changes12["luh2_urban_fraction"], 0.0)
    managed_gain = crop_gain + pasture_gain + range_gain + urban_gain
    conversion = 2.0 * primary_loss * managed_gain / (
        primary_loss + managed_gain + 1e-5
    )
    add_transition("managed_gain12", managed_gain)
    add_transition("primary_loss_x_managed_gain", conversion)

    base = np.column_stack(base_columns).astype(np.float32, copy=False)
    transition = np.column_stack(transition_columns).astype(np.float32, copy=False)
    augmented = np.column_stack((base, transition)).astype(np.float32, copy=False)
    return (
        tuple(base_names),
        base,
        tuple(base_names + transition_names),
        augmented,
        {
            "primary_loss": primary_loss,
            "crop_gain": crop_gain,
            "pasture_gain": pasture_gain,
            "range_gain": range_gain,
            "urban_gain": urban_gain,
            "managed_gain": managed_gain,
            "conversion": conversion,
            "rain12": rain12,
            "dryness": dryness,
            "temperature12": temp12,
            "gpp12": gpp12,
            "hazard12": hazard12,
        },
    )


def targets(prediction, observed):
    ncell = prediction.shape[1]
    pred_year = prediction.reshape(16, 12, ncell)
    obs_year = observed.reshape(16, 12, ncell)
    days = MONTH_DAYS.reshape(16, 12, 1)
    pred_mass = np.sum(pred_year * days, axis=1)
    obs_mass = np.sum(obs_year * days, axis=1)
    annual = np.clip(
        np.log((obs_mass + 1e-4) / (pred_mass + 1e-4)), -4.0, 4.0
    )
    pred_alloc = pred_year * days / (pred_mass[:, None, :] + 1e-8)
    obs_alloc = obs_year * days / (obs_mass[:, None, :] + 1e-8)
    cycle = np.clip(
        np.log((obs_alloc + 1e-4) / (pred_alloc + 1e-4)), -4.0, 4.0
    )
    return annual, cycle, pred_mass, obs_mass


def fit_oof(label, names, x, target, folds, weights, transition_names):
    output = np.empty_like(target, dtype=np.float32)
    split_counts = Counter()
    pair_counts = Counter()
    fold_losses = []
    rng = np.random.default_rng(20260827)
    for fold in range(4):
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if train.size > 180000:
            probabilities = weights[train].astype(np.float64)
            probabilities /= probabilities.sum()
            train = rng.choice(train, 180000, replace=False, p=probabilities)
        learner = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=72,
            learning_rate=0.07,
            l2_regularization=2.0,
            min_samples_leaf=300,
            early_stopping=False,
            random_state=4100 + fold,
        )
        learner.fit(x[train], target[train], sample_weight=weights[train])
        output[held] = learner.predict(x[held]).astype(np.float32)
        counts, pairs = tree_counts(learner, names)
        split_counts.update(counts)
        pair_counts.update(pairs)
        loss = float(
            np.average(
                np.square(target[held] - output[held]), weights=weights[held]
            )
        )
        fold_losses.append(loss)
        top_transition = [
            (name, count)
            for name, count in counts.most_common()
            if name in transition_names
        ][:6]
        print(
            f"FIT label={label} fold={fold} held={held.size} loss={loss:.9f} "
            f"top_transition="
            + ",".join(f"{name}:{count}" for name, count in top_transition),
            flush=True,
        )
        del learner
        gc.collect()
    print(
        f"FIT_SUMMARY label={label} loss="
        f"{float(np.average(np.square(target-output), weights=weights)):.9f} "
        f"fold_losses={','.join(f'{value:.9f}' for value in fold_losses)}",
        flush=True,
    )
    if transition_names:
        top = [
            (name, count)
            for name, count in split_counts.most_common()
            if name in transition_names
        ][:12]
        print(
            f"TRANSITION_SPLITS label={label} "
            + ",".join(f"{name}:{count}" for name, count in top),
            flush=True,
        )
        relevant_pairs = [
            (pair, count)
            for pair, count in pair_counts.most_common()
            if pair[0] in transition_names or pair[1] in transition_names
        ][:12]
        print(
            f"TRANSITION_PAIRS label={label} "
            + ",".join(
                f"{left}*{right}:{count}"
                for (left, right), count in relevant_pairs
            ),
            flush=True,
        )
    return output, np.asarray(fold_losses)


def correction_metrics(
    label, prediction, observed, annual_hat, cycle_hat, area, reference_weight
):
    ncell = prediction.shape[1]
    pred = prediction.reshape(16, 12, ncell)
    obs = observed.reshape(16, 12, ncell)
    days = MONTH_DAYS.reshape(16, 12, 1)

    def summarize(values):
        pred_mass = np.sum(values * days, axis=1)
        obs_mass = np.sum(obs * days, axis=1)
        annual_weights = (
            area[None, :] * (obs_mass + 0.02 * np.mean(obs_mass, axis=0)[None, :])
        )
        annual_rmse = float(
            np.sqrt(
                np.average(
                    np.square(np.log((pred_mass + 1e-4) / (obs_mass + 1e-4))),
                    weights=annual_weights,
                )
            )
        )
        pred_alloc = values * days / (pred_mass[:, None, :] + 1e-8)
        obs_alloc = obs * days / (obs_mass[:, None, :] + 1e-8)
        cycle_rmse = float(
            np.sqrt(
                np.average(
                    np.mean(np.square(pred_alloc - obs_alloc), axis=1),
                    weights=annual_weights,
                )
            )
        )
        ratio = float(
            np.sum(np.mean(pred_mass, axis=0) * area)
            / np.sum(np.mean(obs_mass, axis=0) * area)
        )
        return annual_rmse, cycle_rmse, ratio

    print(
        f"CORRECTION {label}:baseline annual_rmse={summarize(pred)[0]:.9f} "
        f"cycle_rmse={summarize(pred)[1]:.9f} ratio={summarize(pred)[2]:.9f}",
        flush=True,
    )
    for blend in (0.5, 1.0):
        annual_candidate = pred * np.exp(
            np.clip(blend * annual_hat[:, None, :], -2.0, 2.0)
        )
        annual_values = summarize(annual_candidate)

        cycle_factor = np.exp(np.clip(blend * cycle_hat, -2.0, 2.0))
        cycle_candidate = pred * cycle_factor
        base_mass = np.sum(pred * days, axis=1, keepdims=True)
        new_mass = np.sum(cycle_candidate * days, axis=1, keepdims=True)
        cycle_candidate *= base_mass / (new_mass + 1e-12)
        cycle_values = summarize(cycle_candidate)
        print(
            f"CORRECTION {label}:annual blend={blend:g} "
            f"annual_rmse={annual_values[0]:.9f} cycle_rmse={annual_values[1]:.9f} "
            f"ratio={annual_values[2]:.9f}",
            flush=True,
        )
        print(
            f"CORRECTION {label}:cycle blend={blend:g} "
            f"annual_rmse={cycle_values[0]:.9f} cycle_rmse={cycle_values[1]:.9f} "
            f"ratio={cycle_values[2]:.9f}",
            flush=True,
        )


def binned_relationship(name, state, target, weight):
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    positive = values > 1e-10
    print(
        f"STATE {name} nonzero={float(np.average(positive, weights=weight)):.8f} "
        f"max={float(values.max()):.8g}",
        flush=True,
    )
    if np.sum(positive) < 20:
        return
    edges = np.unique(np.quantile(values[positive], (0.0, 0.25, 0.5, 0.75, 1.0)))
    for lower, upper in zip(edges[:-1], edges[1:]):
        chosen = positive & (values >= lower) & (values <= upper)
        print(
            f"STATE_BIN {name} range={lower:.8g}:{upper:.8g} n={int(chosen.sum())} "
            f"target={float(np.average(target[chosen], weights=weight[chosen])):+.8f}",
            flush=True,
        )


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, cell_weight, retained = select_cells(evaluator)
    folds_cell = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds_cell == fold))) for fold in range(4))}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    prediction = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float64
    )[:, 0, :]
    observed = np.asarray(load_observed(rows, cols), dtype=np.float64)
    (
        base_names,
        monthly_base,
        augmented_names,
        monthly_augmented,
        states,
    ) = build_monthly_features(data, prediction)
    transition_names = set(augmented_names) - set(base_names)
    annual_target, cycle_target, pred_mass, obs_mass = targets(prediction, observed)

    # Annual propensity is paired with the December causal state of each year.
    # This uses no later year and avoids pretending that an annual LUH2 value
    # provides a within-year calendar.
    ncell = rows.size
    end_index = np.arange(11, 192, 12)
    annual_base = monthly_base.reshape(192, ncell, -1)[end_index].reshape(16 * ncell, -1)
    annual_augmented = monthly_augmented.reshape(192, ncell, -1)[end_index].reshape(16 * ncell, -1)
    annual_y = annual_target.reshape(-1).astype(np.float32)
    annual_folds = np.tile(folds_cell, 16)
    year_weight = area[None, :] * (
        obs_mass + 0.02 * np.mean(obs_mass, axis=0)[None, :]
    )
    annual_weight = year_weight.reshape(-1)
    annual_weight /= annual_weight.mean()

    cycle_y = cycle_target.reshape(-1).astype(np.float32)
    cycle_folds = np.tile(folds_cell, 192)
    cycle_weight = np.repeat(year_weight[:, None, :], 12, axis=1).reshape(-1)
    cycle_weight /= cycle_weight.mean()

    print(
        f"MATRIX monthly_rows={monthly_base.shape[0]} base_cols={monthly_base.shape[1]} "
        f"augmented_cols={monthly_augmented.shape[1]} bytes="
        f"{monthly_base.nbytes + monthly_augmented.nbytes}",
        flush=True,
    )
    annual_base_hat, annual_base_losses = fit_oof(
        "annual_base", base_names, annual_base, annual_y, annual_folds,
        annual_weight, set()
    )
    annual_transition_hat, annual_transition_losses = fit_oof(
        "annual_transition", augmented_names, annual_augmented, annual_y,
        annual_folds, annual_weight, transition_names
    )
    cycle_base_hat, cycle_base_losses = fit_oof(
        "cycle_base", base_names, monthly_base, cycle_y, cycle_folds,
        cycle_weight, set()
    )
    cycle_transition_hat, cycle_transition_losses = fit_oof(
        "cycle_transition", augmented_names, monthly_augmented, cycle_y,
        cycle_folds, cycle_weight, transition_names
    )
    print(
        "INCREMENT annual_fold_loss_delta="
        + ",".join(
            f"{value:+.9f}"
            for value in annual_transition_losses - annual_base_losses
        )
        + f" stable={bool(np.all(annual_transition_losses < annual_base_losses))}",
        flush=True,
    )
    print(
        "INCREMENT cycle_fold_loss_delta="
        + ",".join(
            f"{value:+.9f}"
            for value in cycle_transition_losses - cycle_base_losses
        )
        + f" stable={bool(np.all(cycle_transition_losses < cycle_base_losses))}",
        flush=True,
    )

    correction_metrics(
        "base",
        prediction,
        observed,
        annual_base_hat.reshape(16, ncell),
        cycle_base_hat.reshape(16, 12, ncell),
        area,
        cell_weight,
    )
    correction_metrics(
        "transition",
        prediction,
        observed,
        annual_transition_hat.reshape(16, ncell),
        cycle_transition_hat.reshape(16, 12, ncell),
        area,
        cell_weight,
    )

    # Inspect raw direction at the annual resolution.  These states are
    # computed from the same year-end causal state used by the annual learner.
    annual_states = {
        name: values[end_index].reshape(-1)
        for name, values in states.items()
    }
    for name in (
        "primary_loss", "crop_gain", "pasture_gain", "range_gain",
        "urban_gain", "managed_gain", "conversion"
    ):
        binned_relationship(name, annual_states[name], annual_y, annual_weight)

    # Smallest physical candidate suggested a priori by clearing ecology.  It
    # is not fitted or scored as runtime here; the signs below decide whether it
    # is justified for a later mechanistic sampled sweep.
    combustion = annual_states["dryness"] / (annual_states["dryness"] + 250.0)
    warm = 1.0 / (1.0 + np.exp(-(annual_states["temperature12"] - 10.0) / 4.0))
    fuel = annual_states["gpp12"] / (annual_states["gpp12"] + 0.25)
    opportunity_gap = 0.01 / (annual_states["hazard12"] + 0.01)
    clearing = (
        annual_states["conversion"]
        / (annual_states["conversion"] + 0.002)
        * combustion * warm * fuel * opportunity_gap
    )
    binned_relationship("clearing_candidate", clearing, annual_y, annual_weight)
    print(
        "EQUATION clearing_source=s*2*primary_loss*managed_gain/"
        "(primary_loss+managed_gain+eps)/(conversion+.002)*"
        "dryness/(dryness+250)*sigmoid((T12-10)/4)*"
        "GPP12/(GPP12+.25)*.01/(H12+.01)",
        flush=True,
    )
    del data, monthly_base, monthly_augmented, prediction, observed
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
