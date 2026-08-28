"""Exact audit of the held-stable ignition/combustibility arrival state.

The held-cell screen selected one conservative equation before this script was
run: a -0.25 bounded redistribution of incumbent natural surface and woody
hazard.  The negative sign represents lightning ignition preceding peak
combustibility, as in a holdover ignition.  This script pins the current model,
reproduces its exact score, and audits the candidate's annual map, centered
cycle, regions, ecological regimes, and prefix causality.  It does not edit the
canonical model or invoke official evaluation.
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.ignition_combustibility_arrival_order_a8ed115 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    arrival_order_states,
    fields,
    pinned_model,
    redistribute_hazard,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    annual_loss,
    area_ratio,
    cycle_loss,
    ecological_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    metric_text,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


EXPECTED_INCUMBENT = 0.719021686
PATHWAY = "natural_signal"
STRENGTH = -0.25
CACHE = Path(__file__).with_name(
    f"canonical_{EXPECTED_MODEL_BLOB[:8]}_chunked.npy"
)


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
        1024.0 * 1024.0
    )


def main() -> int:
    started = time.perf_counter()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"moving canonical model {current_blob}")
    model = pinned_model()
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area_grid = one_degree_area()

    if CACHE.exists():
        incumbent = np.load(CACHE, mmap_mode="r")
        if incumbent.shape != (192, 180, 360) or incumbent.dtype != np.float32:
            raise ValueError(f"invalid incumbent cache {CACHE}")
        print(f"CACHE reused={CACHE} bytes={CACHE.stat().st_size}", flush=True)
    else:
        writable = np.zeros((192, 180, 360), dtype=np.float32)
        chunk_size = 1536
        for start in range(0, rows.size, chunk_size):
            stop = min(start + chunk_size, rows.size)
            data = {
                name: selected_input(name, rows[start:stop], columns[start:stop])
                for name in model.INPUTS
            }
            writable[:, rows[start:stop], columns[start:stop]] = np.asarray(
                model.predict(data, dict(model.PARAMS), None), dtype=np.float32
            )[:, 0, :]
            print(
                f"BASE_CHUNK start={start} stop={stop} cells={rows.size} "
                f"rss_mb={rss_mb():.1f}",
                flush=True,
            )
            del data
            gc.collect()
        np.save(CACHE, writable, allow_pickle=False)
        incumbent = np.load(CACHE, mmap_mode="r")
        del writable
        print(f"CACHE created={CACHE} bytes={CACHE.stat().st_size}", flush=True)

    incumbent_scores = evaluator.score(incumbent)
    incumbent_global = incumbent_scores["global"]
    print(f"INCUMBENT {metric_text(incumbent_global)}", flush=True)
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError("failed exact incumbent reproduction")

    candidate = np.zeros_like(observation)
    ecology_totals: dict[str, list[float]] = {}
    chunk_size = 1024
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        chunk_rows, chunk_columns = rows[start:stop], columns[start:stop]
        data = {
            name: selected_input(name, chunk_rows, chunk_columns)
            for name in model.INPUTS
        }
        base_chunk = np.asarray(
            incumbent[:, chunk_rows, chunk_columns], dtype=np.float32
        )
        state = arrival_order_states(data)
        candidate_chunk = redistribute_hazard(
            base_chunk, state[PATHWAY], STRENGTH
        )
        candidate[:, chunk_rows, chunk_columns] = candidate_chunk

        chunk_observation = observation[:, chunk_rows, chunk_columns]
        chunk_area = area_grid[chunk_rows, chunk_columns]
        mean = {name: values.mean(axis=0) for name, values in fields(data).items()}
        for regime, mask in ecological_masks(mean).items():
            totals = ecology_totals.setdefault(regime, [0.0, 0.0, 0.0, 0])
            incumbent_ratio_numerator = area_ratio(
                base_chunk, chunk_observation, chunk_area, mask
            )
            candidate_ratio_numerator = area_ratio(
                candidate_chunk, chunk_observation, chunk_area, mask
            )
            obs_cycle = chunk_observation.reshape(16, 12, -1).mean(axis=0)
            obs_fire = float(
                np.sum(obs_cycle.sum(axis=0)[mask] * chunk_area[mask])
            )
            totals[0] += incumbent_ratio_numerator * obs_fire
            totals[1] += candidate_ratio_numerator * obs_fire
            totals[2] += obs_fire
            totals[3] += int(mask.sum())
        print(
            f"CANDIDATE_CHUNK start={start} stop={stop} cells={rows.size} "
            f"rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del data, base_chunk, state, candidate_chunk, chunk_observation
        gc.collect()

    candidate_scores = evaluator.score(candidate)
    candidate_global = candidate_scores["global"]
    print(
        f"CANDIDATE pathway={PATHWAY} strength={STRENGTH:+g} "
        f"{metric_text(candidate_global)} "
        f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f}",
        flush=True,
    )
    positive = 0
    for name in sorted(key for key in candidate_scores if key != "global"):
        old = incumbent_scores[name]["overall_score"]
        new = candidate_scores[name]["overall_score"]
        positive += int(new > old)
        print(
            f"REGION name={name} incumbent={old:.9f} candidate={new:.9f} "
            f"delta={new-old:+.9f}",
            flush=True,
        )
    print(f"REGIONAL_BREADTH positive={positive}/14", flush=True)

    incumbent_land = np.asarray(incumbent[:, rows, columns], dtype=np.float32)
    candidate_land = candidate[:, rows, columns]
    observation_land = observation[:, rows, columns]
    area_land = area_grid[rows, columns]
    folds = ((rows // 12) + 3 * (columns // 12)) % 4
    incumbent_annual = annual_loss(
        incumbent_land, observation_land, area_land, folds
    )
    candidate_annual = annual_loss(
        candidate_land, observation_land, area_land, folds
    )
    incumbent_cycle = cycle_loss(
        incumbent_land, observation_land, area_land, folds
    )
    candidate_cycle = cycle_loss(
        candidate_land, observation_land, area_land, folds
    )
    print(
        f"ANNUAL_MAP incumbent={incumbent_annual[0]:.9f} "
        f"candidate={candidate_annual[0]:.9f} "
        f"delta={candidate_annual[0]-incumbent_annual[0]:+.9f} "
        f"folds_improved={sum(new < old for new,old in zip(candidate_annual[1],incumbent_annual[1]))}/4",
        flush=True,
    )
    print(
        f"CENTERED_CYCLE incumbent={incumbent_cycle[0]:.9f} "
        f"candidate={candidate_cycle[0]:.9f} "
        f"delta={candidate_cycle[0]-incumbent_cycle[0]:+.9f} "
        f"folds_improved={sum(new < old for new,old in zip(candidate_cycle[1],incumbent_cycle[1]))}/4",
        flush=True,
    )
    for regime, (old_fire, new_fire, obs_fire, cells) in ecology_totals.items():
        print(
            f"ECOLOGY regime={regime} cells={int(cells)} "
            f"incumbent={old_fire/max(obs_fire,1e-12):.9f} "
            f"candidate={new_fire/max(obs_fire,1e-12):.9f}",
            flush=True,
        )

    sample_index = np.linspace(0, rows.size - 1, 96, dtype=np.int64)
    prefix_rows, prefix_columns = rows[sample_index], columns[sample_index]
    prefix_data = {
        name: selected_input(name, prefix_rows, prefix_columns)
        for name in model.INPUTS
    }
    prefix_base = np.asarray(
        model.predict(prefix_data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    prefix_state = arrival_order_states(prefix_data)
    prefix_candidate = redistribute_hazard(
        prefix_base, prefix_state[PATHWAY], STRENGTH
    )
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_base = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed_state = arrival_order_states(perturbed)
    perturbed_candidate = redistribute_hazard(
        perturbed_base, perturbed_state[PATHWAY], STRENGTH
    )
    prefix_delta = float(
        np.max(np.abs(prefix_candidate[:96] - perturbed_candidate[:96]))
    )
    print(f"PREFIX candidate_max_abs={prefix_delta:.12g}", flush=True)
    print(
        f"DONE overall={candidate_global['overall_score']:.9f} "
        f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f} peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
