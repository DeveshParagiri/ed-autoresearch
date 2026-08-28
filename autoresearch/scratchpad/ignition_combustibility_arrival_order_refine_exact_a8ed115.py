"""Exact audit of coherent-gated ignition/combustibility arrival order.

The companion held-cell screen selected the incumbent coherent-surface factor
as the only exact candidate.  This script keeps the original arrival-order
coefficient fixed at -0.25 and compares the globally shared, pointwise result
against the pinned a8ed115 incumbent.  No canonical or official artifact is
modified.
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
from autoresearch.scratchpad.ignition_combustibility_arrival_order_refine_a8ed115 import (  # noqa: E402
    STRENGTH,
    attenuation_gates,
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
CACHE = Path(__file__).with_name(
    f"canonical_{EXPECTED_MODEL_BLOB[:8]}_chunked.npy"
)
UNGATED_DELTAS = {
    "ceam": -0.001353601,
    "nhaf": +0.001503135,
    "nhsa": -0.004680006,
    "shaf": +0.000419048,
    "shsa": -0.003742157,
}


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
    if not CACHE.exists():
        raise RuntimeError(f"missing incumbent cache {CACHE}")
    model = pinned_model()
    incumbent = np.load(CACHE, mmap_mode="r")
    if incumbent.shape != (192, 180, 360) or incumbent.dtype != np.float32:
        raise ValueError(f"invalid incumbent cache {CACHE}")
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    observation = load_observation()
    area_grid = one_degree_area()
    evaluator = GFED5Evaluator(GFED5_PATH)
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
        coherent = attenuation_gates(data)["coherent"]
        candidate_chunk = redistribute_hazard(
            base_chunk, state["natural_signal"] * coherent, STRENGTH
        )
        candidate[:, chunk_rows, chunk_columns] = candidate_chunk
        chunk_observation = observation[:, chunk_rows, chunk_columns]
        chunk_area = area_grid[chunk_rows, chunk_columns]
        mean = {name: values.mean(axis=0) for name, values in fields(data).items()}
        for regime, mask in ecological_masks(mean).items():
            totals = ecology_totals.setdefault(regime, [0.0, 0.0, 0.0, 0])
            obs_cycle = chunk_observation.reshape(16, 12, -1).mean(axis=0)
            obs_fire = float(
                np.sum(obs_cycle.sum(axis=0)[mask] * chunk_area[mask])
            )
            totals[0] += area_ratio(
                base_chunk, chunk_observation, chunk_area, mask
            ) * obs_fire
            totals[1] += area_ratio(
                candidate_chunk, chunk_observation, chunk_area, mask
            ) * obs_fire
            totals[2] += obs_fire
            totals[3] += int(mask.sum())
        print(
            f"CHUNK start={start} stop={stop} cells={rows.size} rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del data, base_chunk, state, coherent, candidate_chunk, chunk_observation
        gc.collect()

    candidate_scores = evaluator.score(candidate)
    candidate_global = candidate_scores["global"]
    print(
        f"CANDIDATE gate=coherent strength={STRENGTH:+g} "
        f"{metric_text(candidate_global)} "
        f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f}",
        flush=True,
    )
    positive = 0
    for name in sorted(key for key in candidate_scores if key != "global"):
        old = incumbent_scores[name]["overall_score"]
        new = candidate_scores[name]["overall_score"]
        delta = new - old
        positive += int(delta > 0.0)
        comparison = ""
        if name in UNGATED_DELTAS:
            comparison = f" ungated_delta={UNGATED_DELTAS[name]:+.9f}"
        print(
            f"REGION name={name} incumbent={old:.9f} candidate={new:.9f} "
            f"delta={delta:+.9f}{comparison}",
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
        f"ANNUAL_MAP delta={candidate_annual[0]-incumbent_annual[0]:+.9f} "
        f"folds_improved={sum(new < old for new,old in zip(candidate_annual[1],incumbent_annual[1]))}/4",
        flush=True,
    )
    print(
        f"CENTERED_CYCLE delta={candidate_cycle[0]-incumbent_cycle[0]:+.9f} "
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
    prefix_coherent = attenuation_gates(prefix_data)["coherent"]
    prefix_candidate = redistribute_hazard(
        prefix_base, prefix_state["natural_signal"] * prefix_coherent, STRENGTH
    )
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_base = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed_state = arrival_order_states(perturbed)
    perturbed_coherent = attenuation_gates(perturbed)["coherent"]
    perturbed_candidate = redistribute_hazard(
        perturbed_base,
        perturbed_state["natural_signal"] * perturbed_coherent,
        STRENGTH,
    )
    print(
        f"PREFIX candidate_max_abs={float(np.max(np.abs(prefix_candidate[:96]-perturbed_candidate[:96]))):.12g}",
        flush=True,
    )
    print(
        f"DONE overall={candidate_global['overall_score']:.9f} "
        f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f} peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
