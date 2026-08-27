"""One-dimensional held-block sign audit for zoom-out pathway interactions."""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    CACHE,
    EXPECTED_MODEL_BLOB,
    build_pathway_features,
    load_observation,
    select_high_weight,
    selected_inputs,
    weighted_mean,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def main() -> int:
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB or not CACHE.exists():
        raise RuntimeError(f"missing pinned cache for {blob}")
    model = load_model()
    baseline = np.load(CACHE)
    observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    rows, columns, cell_weight, retained = select_high_weight(observation, area)
    data = selected_inputs(model, rows, columns)
    selected_base = baseline[:, rows, columns]
    selected_obs = observation[:, rows, columns]
    del observation, baseline
    gc.collect()

    names, x, transition_diagnostics = build_pathway_features(data, selected_base)
    values = {name: x[:, index] for index, name in enumerate(names)}
    turnover = transition_diagnostics["turnover"].reshape(-1)
    stability = np.exp(-turnover / 0.01)
    pairs = {
        "turnover_x_warm_recurrence": values["landuse_turnover_combustion"]
        * values["warm_open_recurrence"],
        "turnover_x_humid_woody": values["landuse_turnover_combustion"]
        * values["humid_woody_shield"],
        "urban_transition_x_recurrence": values["urban_expansion_fragmentation"]
        * values["recurrent_surface_event"],
        "turnover_x_rain_drydown": values["landuse_turnover_combustion"]
        * values["rain_built_drydown"],
        "crop_transition_x_live_dead": values["crop_expansion_combustion"]
        * values["live_to_dead_transition"],
        "abandonment_x_live_dead": values["managed_abandonment_regrowth"]
        * values["live_to_dead_transition"],
        "stable_warm_recurrence": stability * values["warm_open_recurrence"],
        "stable_managed_recurrence": stability
        * values["warm_open_recurrence"]
        * values["managed_open_event"],
    }
    probes = {
        name: values[name]
        for name in (
            "urban_expansion_fragmentation",
            "landuse_turnover_combustion",
            "managed_abandonment_regrowth",
            "crop_expansion_combustion",
            "grazing_expansion_combustion",
            "primary_conversion_residue",
        )
    }
    probes.update(pairs)

    count = rows.size
    eps = np.float32(1e-6)
    base_cycle = selected_base.reshape(16, 12, count).mean(axis=0)
    obs_cycle = selected_obs.reshape(16, 12, count).mean(axis=0)
    base_annual = base_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    map_cell = np.clip(
        np.log((obs_annual + eps) / (base_annual + eps)), -4.0, 4.0
    ).astype(np.float32)
    map_target = np.tile(map_cell, 192)
    base_alloc = base_cycle / (base_annual[None, :] + eps)
    obs_alloc = obs_cycle / (obs_annual[None, :] + eps)
    cycle_target = np.tile(
        np.clip(obs_alloc - base_alloc, -0.5, 0.5), (16, 1, 1)
    ).reshape(-1).astype(np.float32)
    folds = np.tile(((rows // 15) + 3 * (columns // 15)) % 4, 192)
    weights = np.tile(cell_weight, 192).astype(np.float64)
    weights /= weights.mean()

    print(
        f"SHAPE_AUDIT model_blob={blob} cells={count} retained={retained:.9f}",
        flush=True,
    )
    for target_name, target in (("map", map_target), ("cycle", cycle_target)):
        ranking = []
        for probe_name, probe in probes.items():
            oof = np.empty_like(target, dtype=np.float32)
            fold_deltas = []
            for fold in range(4):
                train, held = folds != fold, folds == fold
                learner = HistGradientBoostingRegressor(
                    max_depth=2,
                    max_iter=45,
                    learning_rate=0.06,
                    l2_regularization=2.0,
                    min_samples_leaf=300,
                    early_stopping=False,
                    random_state=9180 + 10 * fold + (target_name == "cycle"),
                )
                learner.fit(
                    probe[train, None], target[train], sample_weight=weights[train]
                )
                oof[held] = learner.predict(probe[held, None]).astype(np.float32)
                before = weighted_mean(np.abs(target[held]), weights[held])
                after = weighted_mean(np.abs(target[held] - oof[held]), weights[held])
                fold_deltas.append(after - before)
            baseline_loss = weighted_mean(np.abs(target), weights)
            loss = weighted_mean(np.abs(target - oof), weights)
            ranking.append((loss - baseline_loss, probe_name, fold_deltas))
        print(f"ONE_D_RANK target={target_name}", flush=True)
        for delta, probe_name, fold_deltas in sorted(ranking):
            print(
                f"ONE_D target={target_name} probe={probe_name} delta={delta:+.9f} "
                + " ".join(
                    f"fold{fold}={fold_delta:+.9f}"
                    for fold, fold_delta in enumerate(fold_deltas)
                ),
                flush=True,
            )

        # Residual shape in zero plus positive quantile bins for the strongest
        # transition composites.  These target summaries are diagnostics only.
        for probe_name in (
            "stable_managed_recurrence",
            "urban_transition_x_recurrence",
            "turnover_x_warm_recurrence",
            "turnover_x_rain_drydown",
            "crop_transition_x_live_dead",
        ):
            probe = probes[probe_name]
            positive = probe[probe > 1e-12]
            print(
                f"SHAPE target={target_name} probe={probe_name} "
                f"zero={weighted_mean(target[probe <= 1e-12], weights[probe <= 1e-12]):+.9f}",
                flush=True,
            )
            if positive.size:
                edges = np.unique(np.quantile(positive, np.linspace(0.0, 1.0, 6)))
                for lower, upper in zip(edges[:-1], edges[1:], strict=True):
                    chosen = (probe > 1e-12) & (probe >= lower) & (
                        probe <= upper if upper == edges[-1] else probe < upper
                    )
                    print(
                        f"BIN target={target_name} probe={probe_name} "
                        f"lo={lower:.9g} hi={upper:.9g} n={int(chosen.sum())} "
                        f"residual={weighted_mean(target[chosen], weights[chosen]):+.9f}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
