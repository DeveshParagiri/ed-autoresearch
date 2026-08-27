"""Exact falsification of prior-year same-month drought capacity.

The held-cell screen in ``causal_same_month_normals_bf42d58.py`` found one
stable physical translation: existing surface-fire hazard can expand during a
month that is drier than the same month in prior years.  This script applies
that equation globally at five bounded strengths.  It pins the canonical
model, computes only pointwise causal state, and invokes the local exact scorer
without changing ``model.py`` or the official experiment ledger.
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

from autoresearch.scratchpad.causal_same_month_normals_bf42d58 import (  # noqa: E402
    EXPECTED_MODEL_BLOB,
    hazard_capacity,
    pinned_model,
    states,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    one_degree_area,
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


EXPECTED_OVERALL = 0.718995365
CACHE = Path(__file__).with_name(
    f"canonical_{EXPECTED_MODEL_BLOB[:8]}_chunked.npy"
)
STRENGTHS = (0.1, 0.25, 0.5, 1.0, 2.0)


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
        1024.0 * 1024.0
    )


def main() -> int:
    started = time.perf_counter()
    signal_name = (
        "stable_fuel_gap" if "--stable-fuel-gap" in sys.argv
        else "anomalous_dry_capacity"
    )
    model = pinned_model()
    land = load_land_mask()
    rows, columns = np.nonzero(land)
    observation = load_observation()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area = one_degree_area()

    if CACHE.exists():
        incumbent = np.load(CACHE)
        if incumbent.shape != (192, 180, 360) or incumbent.dtype != np.float32:
            raise ValueError(f"invalid incumbent cache {CACHE}")
        print(f"CACHE reused={CACHE} bytes={CACHE.stat().st_size}", flush=True)
    else:
        incumbent = np.zeros((192, 180, 360), dtype=np.float32)
        chunk_size = 1536
        for start in range(0, rows.size, chunk_size):
            stop = min(start + chunk_size, rows.size)
            data = {
                name: selected_input(name, rows[start:stop], columns[start:stop])
                for name in model.INPUTS
            }
            incumbent[:, rows[start:stop], columns[start:stop]] = np.asarray(
                model.predict(data, dict(model.PARAMS), None), dtype=np.float32
            )[:, 0, :]
            print(
                f"BASE_CHUNK start={start} stop={stop} cells={rows.size} "
                f"rss_mb={rss_mb():.1f}",
                flush=True,
            )
            del data
            gc.collect()
        np.save(CACHE, incumbent, allow_pickle=False)
        print(f"CACHE created={CACHE} bytes={CACHE.stat().st_size}", flush=True)

    incumbent_score = evaluator.score(incumbent)
    incumbent_global = incumbent_score["global"]
    print(f"INCUMBENT {metric_text(incumbent_global)}", flush=True)
    if abs(incumbent_global["overall_score"] - EXPECTED_OVERALL) > 5e-7:
        raise RuntimeError("failed exact incumbent reproduction")

    candidates = {
        strength: np.zeros_like(incumbent) for strength in STRENGTHS
    }
    ecology_sums: dict[float | str, dict[str, list[float]]] = {
        "incumbent": {},
        **{strength: {} for strength in STRENGTHS},
    }
    chunk_size = 1024
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        chunk_rows, chunk_columns = rows[start:stop], columns[start:stop]
        data = {
            name: selected_input(name, chunk_rows, chunk_columns)
            for name in model.INPUTS
        }
        base_chunk = incumbent[:, chunk_rows, chunk_columns]
        state = states(data, base_chunk)
        signal = state[signal_name]
        candidate_chunks = {
            strength: hazard_capacity(base_chunk, signal, strength)
            for strength in STRENGTHS
        }
        for strength, values in candidate_chunks.items():
            candidates[strength][:, chunk_rows, chunk_columns] = values

        chunk_observation = observation[:, chunk_rows, chunk_columns]
        chunk_area = area[chunk_rows, chunk_columns]
        obs_annual = chunk_observation.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
        predictions: dict[float | str, np.ndarray] = {
            "incumbent": base_chunk,
            **candidate_chunks,
        }
        means = {
            name: np.asarray(values[:, 0, :], dtype=np.float32).mean(axis=0)
            for name, values in data.items()
        }
        masks = ecological_masks(means)
        for label, prediction in predictions.items():
            pred_annual = prediction.reshape(16, 12, -1).mean(axis=0).sum(axis=0)
            for regime, mask in masks.items():
                totals = ecology_sums[label].setdefault(regime, [0.0, 0.0, 0])
                totals[0] += float(np.sum(pred_annual[mask] * chunk_area[mask]))
                totals[1] += float(np.sum(obs_annual[mask] * chunk_area[mask]))
                totals[2] += int(mask.sum())
        print(
            f"CANDIDATE_CHUNK start={start} stop={stop} cells={rows.size} "
            f"rss_mb={rss_mb():.1f}",
            flush=True,
        )
        del data, state, signal, candidate_chunks, predictions
        gc.collect()

    scores: dict[float, dict[str, dict[str, float]]] = {}
    for strength, prediction in candidates.items():
        scores[strength] = evaluator.score(prediction)
        global_score = scores[strength]["global"]
        breadth = sum(
            scores[strength][name]["overall_score"]
            > incumbent_score[name]["overall_score"]
            for name in scores[strength]
            if name != "global"
        )
        print(
            f"CANDIDATE strength={strength:g} {metric_text(global_score)} "
            f"delta={global_score['overall_score']-incumbent_global['overall_score']:+.9f} "
            f"regional_breadth={breadth}/14",
            flush=True,
        )
        for name in sorted(key for key in scores[strength] if key != "global"):
            old = incumbent_score[name]["overall_score"]
            new = scores[strength][name]["overall_score"]
            print(
                f"REGION strength={strength:g} name={name} "
                f"incumbent={old:.9f} candidate={new:.9f} delta={new-old:+.9f}",
                flush=True,
            )

    best = max(STRENGTHS, key=lambda strength: scores[strength]["global"]["overall_score"])
    for label in ("incumbent", best):
        for regime, (predicted, observed, cells) in ecology_sums[label].items():
            print(
                f"ECOLOGY label={label} regime={regime} cells={int(cells)} "
                f"ratio={predicted/max(observed,1e-12):.9f}",
                flush=True,
            )

    prefix_rows = rows[np.linspace(0, rows.size - 1, 96, dtype=np.int64)]
    prefix_columns = columns[np.linspace(0, columns.size - 1, 96, dtype=np.int64)]
    prefix_data = {
        name: selected_input(name, prefix_rows, prefix_columns)
        for name in model.INPUTS
    }
    prefix_base = np.asarray(
        model.predict(prefix_data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    prefix_state = states(prefix_data, prefix_base)
    original = hazard_capacity(
        prefix_base, prefix_state[signal_name], best
    )
    perturbed = {name: values.copy() for name, values in prefix_data.items()}
    for values in perturbed.values():
        values[96:] *= 1.5
    perturbed_base = np.asarray(
        model.predict(perturbed, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    perturbed_state = states(perturbed, perturbed_base)
    changed = hazard_capacity(
        perturbed_base, perturbed_state[signal_name], best
    )
    print(
        f"PREFIX best={best:g} max_abs={float(np.max(np.abs(original[:96]-changed[:96]))):.12g}",
        flush=True,
    )
    print(
        f"DONE signal={signal_name} best={best:g} "
        f"overall={scores[best]['global']['overall_score']:.9f} "
        f"delta={scores[best]['global']['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"wall_seconds={time.perf_counter()-started:.3f} peak_rss_mb={rss_mb():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
