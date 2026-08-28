"""Exact full-grid replay of the annual-positive cross-stratum gate.

The held screen rejected this family under a strict requirement that annual
and centered-cycle losses both improve.  Official Overall weights RMSE twice
and also includes the annual spatial distribution, so this script replays the
two most informative fixed strengths against the pinned 33ac854 mechanistic
incumbent before making the final decision.  It writes no canonical or official
artifact.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.cross_stratum_propagation_gate_33ac854 import (  # noqa: E402
    apply_gate,
    cross_stratum_state,
    fields,
    pinned_model,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from autoresearch.scratchpad.surface_seasonality_persistent_dryness_gate_2127874 import (  # noqa: E402
    ecological_masks,
)
from autoresearch.scratchpad.zoomout_pathway_headroom_27d4594 import (  # noqa: E402
    load_observation,
    metric_text,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask  # noqa: E402


EXPECTED_INCUMBENT = 0.719107756
STRENGTHS = (0.125, 0.5)


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
        1024.0 * 1024.0
    )


def main() -> int:
    started = time.perf_counter()
    model = pinned_model()
    observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    incumbent = np.zeros_like(observation)
    candidates = {
        strength: np.zeros_like(observation) for strength in STRENGTHS
    }
    ecology_sums: dict[str, dict[float | str, float]] = {}

    chunk_size = 1024
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        chunk_rows, chunk_columns = rows[start:stop], columns[start:stop]
        data = {
            name: selected_input(name, chunk_rows, chunk_columns)
            for name in model.INPUTS
        }
        incumbent_chunk = np.asarray(
            model.predict(data, dict(model.PARAMS), None), dtype=np.float32
        )[:, 0, :]
        state = cross_stratum_state(data, 0.0)
        candidate_chunks = {
            strength: apply_gate(incumbent_chunk, state, strength)
            for strength in STRENGTHS
        }
        incumbent[:, chunk_rows, chunk_columns] = incumbent_chunk
        for strength, values in candidate_chunks.items():
            candidates[strength][:, chunk_rows, chunk_columns] = values

        chunk_observation = observation[:, chunk_rows, chunk_columns]
        chunk_area = area[chunk_rows, chunk_columns]
        obs_annual = (
            chunk_observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        )
        incumbent_annual = (
            incumbent_chunk.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        )
        candidate_annual = {
            strength: values.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
            for strength, values in candidate_chunks.items()
        }
        mean = {
            name: values.mean(axis=0)
            for name, values in fields(data).items()
        }
        for regime, mask in ecological_masks(mean).items():
            totals = ecology_sums.setdefault(
                regime,
                {
                    "incumbent": 0.0,
                    "observation": 0.0,
                    "cells": 0.0,
                    **{strength: 0.0 for strength in STRENGTHS},
                },
            )
            totals["incumbent"] += float(
                np.sum(incumbent_annual[mask] * chunk_area[mask])
            )
            totals["observation"] += float(
                np.sum(obs_annual[mask] * chunk_area[mask])
            )
            totals["cells"] += int(mask.sum())
            for strength in STRENGTHS:
                totals[strength] += float(
                    np.sum(candidate_annual[strength][mask] * chunk_area[mask])
                )
        print(
            f"CHUNK start={start} stop={stop} cells={rows.size} rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del data, incumbent_chunk, state, candidate_chunks
        del chunk_observation, obs_annual, incumbent_annual, candidate_annual
        gc.collect()

    incumbent_scores = evaluator.score(incumbent)
    incumbent_global = incumbent_scores["global"]
    print(f"INCUMBENT {metric_text(incumbent_global)}", flush=True)
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError("failed exact incumbent reproduction")

    for strength in STRENGTHS:
        candidate_scores = evaluator.score(candidates[strength])
        candidate_global = candidate_scores["global"]
        print(
            f"CANDIDATE strength={strength:.3f} {metric_text(candidate_global)} "
            f"delta={candidate_global['overall_score']-incumbent_global['overall_score']:+.9f}",
            flush=True,
        )
        positive = 0
        for name in sorted(key for key in candidate_scores if key != "global"):
            old = incumbent_scores[name]["overall_score"]
            new = candidate_scores[name]["overall_score"]
            positive += int(new > old)
            print(
                f"REGION strength={strength:.3f} name={name} "
                f"incumbent={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
                flush=True,
            )
        print(
            f"REGIONAL_BREADTH strength={strength:.3f} positive={positive}/14",
            flush=True,
        )

    for regime, totals in ecology_sums.items():
        observed = max(totals["observation"], 1e-12)
        values = " ".join(
            f"candidate_{strength:g}={totals[strength]/observed:.9f}"
            for strength in STRENGTHS
        )
        print(
            f"ECOLOGY regime={regime} cells={int(totals['cells'])} "
            f"incumbent={totals['incumbent']/observed:.9f} {values}",
            flush=True,
        )

    sample_index = np.linspace(0, rows.size - 1, 96, dtype=np.int64)
    prefix_rows, prefix_columns = rows[sample_index], columns[sample_index]
    prefix_data = {
        name: selected_input(name, prefix_rows, prefix_columns)
        for name in model.INPUTS
    }
    prefix_incumbent = np.asarray(
        model.predict(prefix_data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    prefix_state = cross_stratum_state(prefix_data, 0.0)
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_incumbent = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed_state = cross_stratum_state(perturbed, 0.0)
    for strength in STRENGTHS:
        prefix_candidate = apply_gate(
            prefix_incumbent, prefix_state, strength
        )
        perturbed_candidate = apply_gate(
            perturbed_incumbent, perturbed_state, strength
        )
        print(
            f"PREFIX strength={strength:.3f} max_abs="
            f"{float(np.max(np.abs(prefix_candidate[:96]-perturbed_candidate[:96]))):.12g}",
            flush=True,
        )
    print(
        f"DONE wall_seconds={time.perf_counter()-started:.3f} peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
