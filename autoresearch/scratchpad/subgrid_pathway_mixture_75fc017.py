"""Test a small independently factorized fire-process tile fraction.

Most of each grid cell retains the incumbent product-limited fire process.  A
fixed subgrid fraction instead follows four independent occurrence-times-event
pathways for surface, woody, crop-residue, and unresolved background fire.  The
area mixture is physical, ``BA=(1-f) BA_product + f BA_pathway``; it is not a
target-fitted ensemble.  Pathway scales reuse fixed event magnitudes already in
the incumbent model and are never fitted here.  Coordinates assign held blocks
only.  No canonical or official artifact is written.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.additive_pathway_replacement_a8ed115 import (  # noqa: E402
    build_sources,
    ecological_ratios_selected,
    load_observation,
    metric_line,
    metrics,
    predict_from_sources,
    select_high_weight,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


EXPECTED_COMMIT = "75fc017"
EXPECTED_MODEL_BLOB = "f526cbfa0a9747b78bf71506c665e4b1fd3c8605"
EXPECTED_INCUMBENT = 0.719756369
FRACTIONS = (0.02, 0.05, 0.10, 0.20)


def pinned_model():
    source = subprocess.run(
        ("git", "show", f"{EXPECTED_COMMIT}:autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    blob = subprocess.run(
        ("git", "hash-object", "--stdin"),
        cwd=ROOT,
        input=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"unexpected pinned blob {blob}")
    module = types.ModuleType("ed_fire_pinned_75fc017_subgrid_path")
    module.__file__ = f"git:{EXPECTED_COMMIT}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def fixed_pathway_scales(model) -> np.ndarray:
    """Reuse incumbent global event magnitudes; do not fit pathway weights."""
    monthly = float(model.PARAMS["month_scale"])
    return monthly * np.asarray(
        (
            1.10,
            0.65 + 1.85,
            0.60 + 1.20,
            float(model.PARAMS["annual_scale"]),
        ),
        dtype=np.float64,
    )


def mixture(
    incumbent: np.ndarray,
    alternative: np.ndarray,
    fraction: float,
) -> np.ndarray:
    return np.asarray(
        np.clip((1.0 - fraction) * incumbent + fraction * alternative, 0.0, 1.0),
        dtype=np.float32,
    )


def main() -> int:
    model = pinned_model()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observation_grid = load_observation()
    rows, columns, cell_weight, retained = select_high_weight(
        observation_grid, area_grid
    )
    data = {
        name: selected_input(name, rows, columns) for name in model.INPUTS
    }
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    incumbent = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    scales = fixed_pathway_scales(model)
    sources = build_sources(data, "direct")
    alternative = predict_from_sources(sources, scales)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    base = metrics(incumbent, observation, cell_weight, area)
    base_ecology = ecological_ratios_selected(
        incumbent, observation, data, area
    )
    print(
        f"DESIGN cells={rows.size} retained={retained:.9f} "
        f"scales={','.join(f'{value:.9f}' for value in scales)} "
        f"base={metric_line(base)} alternative={metric_line(metrics(alternative, observation, cell_weight, area))}",
        flush=True,
    )

    survivors: list[tuple[float, float]] = []
    for fraction in FRACTIONS:
        trial = mixture(incumbent, alternative, fraction)
        trial_all = metrics(trial, observation, cell_weight, area)
        annual_wins = raw_wins = cycle_wins = 0
        for fold in range(4):
            held = folds == fold
            old = metrics(
                incumbent[:, held], observation[:, held], cell_weight[held], area[held]
            )
            new = metrics(
                trial[:, held], observation[:, held], cell_weight[held], area[held]
            )
            annual_wins += new["annual_log_rmse"] < old["annual_log_rmse"]
            cycle_wins += new["cycle_rmse"] < old["cycle_rmse"]
            raw_wins += new["raw_cycle_rmse"] < old["raw_cycle_rmse"]
        ecology = ecological_ratios_selected(trial, observation, data, area)
        max_ecology_shift = max(
            abs(ecology[name] - base_ecology[name]) for name in ecology
        )
        print(
            f"VARIANT fraction={fraction:.2f} annual_wins={annual_wins}/4 "
            f"cycle_wins={cycle_wins}/4 raw_wins={raw_wins}/4 "
            f"{metric_line(trial_all)} ecology_max_shift={max_ecology_shift:.6f}",
            flush=True,
        )
        if (
            annual_wins >= 3
            and raw_wins >= 2
            and trial_all["annual_log_rmse"] < base["annual_log_rmse"]
            and trial_all["raw_cycle_rmse"] <= 1.002 * base["raw_cycle_rmse"]
            and max_ecology_shift < 0.15
        ):
            survivors.append((trial_all["annual_log_rmse"], fraction))
    survivors.sort()

    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    probe_data = {
        name: values[:, :, probe].copy() for name, values in data.items()
    }
    before = build_sources(probe_data, "direct")
    for values in probe_data.values():
        values[96:] *= 1.5
    after = build_sources(probe_data, "direct")
    print(
        f"PREFIX source_max_abs={float(np.max(np.abs(before[:96]-after[:96]))):.12g}",
        flush=True,
    )
    if not survivors:
        print("DECISION exact=0 reject=no_bounded_held_survivor", flush=True)
        return 0

    fraction = survivors[0][1]
    print(f"DECISION exact=1 fraction={fraction:.2f}", flush=True)
    del sources, alternative, data, observation, incumbent
    gc.collect()

    land = load_land_mask()
    all_rows, all_columns = np.nonzero(land)
    incumbent_grid = np.zeros_like(observation_grid)
    alternative_grid = np.zeros_like(observation_grid)
    for start in range(0, all_rows.size, 1024):
        stop = min(start + 1024, all_rows.size)
        chunk_rows, chunk_columns = all_rows[start:stop], all_columns[start:stop]
        chunk_data = {
            name: selected_input(name, chunk_rows, chunk_columns)
            for name in model.INPUTS
        }
        incumbent_grid[:, chunk_rows, chunk_columns] = np.asarray(
            model.predict(chunk_data, dict(model.PARAMS), None), dtype=np.float32
        )[:, 0, :]
        alternative_grid[:, chunk_rows, chunk_columns] = predict_from_sources(
            build_sources(chunk_data, "direct"), scales
        )
        del chunk_data
        gc.collect()
    trial_grid = mixture(incumbent_grid, alternative_grid, fraction)
    incumbent_score = evaluator.score(incumbent_grid)
    trial_score = evaluator.score(trial_grid)
    if abs(incumbent_score["global"]["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError("failed exact incumbent reproduction")
    old = incumbent_score["global"]
    new = trial_score["global"]
    print(
        f"EXACT fraction={fraction:.2f} overall={new['overall_score']:.9f} "
        f"delta={new['overall_score']-old['overall_score']:+.9f} "
        f"bias={new['bias_score']:.9f} rmse={new['rmse_score']:.9f} "
        f"seasonal={new['seasonal_cycle_score']:.9f} spatial={new['spatial_distribution_score']:.9f}",
        flush=True,
    )
    print(
        "REGIONS improved="
        f"{sum(trial_score[name]['overall_score'] > incumbent_score[name]['overall_score'] for name in trial_score if name != 'global')}/14 "
        + ",".join(
            f"{name}:{trial_score[name]['overall_score']-incumbent_score[name]['overall_score']:+.6f}"
            for name in sorted(key for key in trial_score if key != "global")
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
