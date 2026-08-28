"""Held test of smooth clean-cover thermal-coherence capacity laws.

The strict strong annual-ranking audit found a positive LUH2-primary main
effect and a negative primary-by-trailing-temperature-variability interaction
in every fold of all three learner families.  It also found the corresponding
pasture pattern in every fold of the deeper HGB and random forest.  These two
translations do not copy a learned split or coefficient.  Each cover signal
can expand fire event capacity only when the trailing thermal season is
coherent::

    sigma_t = sqrt(EMA12(T**2)_t - EMA12(T)_t**2)
    C_t = 4 C / (4 C + sigma_t)
    M_t(k) = 1 + k * cover_t * C_t

The twelve-month memory is the physical annual horizon and 4 C is the fixed
thermal-variability saturation already used in the prior physical audit.  The
law is global, smooth, point-local, target blind, and prefix causal.  No
coordinate or region enters it; coordinates define held whole-cell folds only.

Exact full-grid scoring is allowed only if annual-log, normalized-allocation,
and raw-cycle losses improve in every held fold at one fixed declared strength.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "121c83c"
EXPECTED_MODEL_BLOB = "b82c285259f35f0f942ddc8a78663d8d14dd36b1"
EXPECTED_INCUMBENT = 0.719892388
STRENGTHS = (0.02, 0.05, 0.10, 0.20, 0.40)
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ("git", "show", f"{PINNED}:autoresearch/model.py"),
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
        raise RuntimeError(f"unexpected pinned model blob {blob}")
    module = types.ModuleType(f"model_{PINNED}_pasture_thermal_coherence")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, blob


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for step in range(values.shape[0]):
        state += alpha * (values[step] - state)
        output[step] = state
    return output


def coherent_cover(
    data: dict[str, np.ndarray],
    cover_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    mean = antecedent(temperature, 12.0)
    second = antecedent(np.square(temperature), 12.0)
    sigma = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    coherence = 4.0 / (4.0 + sigma)
    cover = np.clip(
        np.asarray(data[cover_name], dtype=np.float64),
        0.0,
        1.0,
    )
    return cover * coherence, sigma


def candidate(
    incumbent: np.ndarray,
    support: np.ndarray,
    strength: float,
) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    multiplier = 1.0 + strength * np.clip(support, 0.0, 1.0)
    return np.asarray(
        -np.expm1(-np.clip(hazard * multiplier, 0.0, 50.0)),
        dtype=np.float32,
    )


def held_losses(
    prediction: np.ndarray,
    observed: np.ndarray,
    area: np.ndarray,
    observed_annual: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    weight = area * observed_annual
    observed_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    predicted_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    observed_allocation = observed_cycle / (observed_cycle.sum(axis=0, keepdims=True) + 1e-12)
    predicted_allocation = predicted_cycle / (predicted_cycle.sum(axis=0, keepdims=True) + 1e-12)
    annual, allocation, raw_cycle = [], [], []
    for fold in range(4):
        held = folds == fold
        held_weight = weight[held]
        denominator = np.sum(held_weight) + 1e-15
        annual.append(np.sqrt(np.sum(
            held_weight * np.square(
                np.log(observed_annual[held] + 1e-5)
                - np.log(predicted_annual[held] + 1e-5)
            )
        ) / denominator))
        allocation.append(np.sqrt(np.sum(
            held_weight[None, :] * np.square(
                observed_allocation[:, held] - predicted_allocation[:, held]
            )
        ) / (12.0 * denominator)))
        raw_cycle.append(np.sqrt(np.sum(
            held_weight[None, :] * np.square(
                observed_cycle[:, held] - predicted_cycle[:, held]
            )
        ) / (12.0 * denominator)))
    return np.asarray(annual), np.asarray(allocation), np.asarray(raw_cycle)


def main() -> int:
    model, blob = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    incumbent_scores = evaluator.score(incumbent)
    incumbent_global = incumbent_scores["global"]
    if abs(incumbent_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {incumbent_global['overall_score']:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192], dtype=np.float32)
    observed_grid = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / np.float32(100.0)
    del fine
    evaluator_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    cells = np.flatnonzero(land.ravel())
    rows, columns = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    selected_incumbent = np.asarray(incumbent[:, rows, columns], dtype=np.float64)
    selected_observed = np.asarray(observed_grid[:, rows, columns], dtype=np.float64)
    selected_area = evaluator_area[rows, columns]
    observed_annual = np.average(selected_observed, axis=0, weights=MONTH_DAYS)
    selected_data = {
        name: np.asarray(values[:, rows, columns], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    families = {
        "primary": "luh2_primary_fraction",
        "pasture": "luh2_pasture_fraction",
    }
    supports: dict[str, np.ndarray] = {}
    sigma: np.ndarray | None = None
    for family, cover_name in families.items():
        family_support, family_sigma = coherent_cover(selected_data, cover_name)
        supports[family] = family_support[:, 0, :]
        sigma = family_sigma[:, 0, :]
    assert sigma is not None
    base_losses = held_losses(
        selected_incumbent,
        selected_observed,
        selected_area,
        observed_annual,
        folds,
    )
    print(
        f"IDENTITY pinned={PINNED} model_blob={blob} incumbent={incumbent_global['overall_score']:.9f} "
        f"land_cells={cells.size} fold_cells="
        + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )
    for family, support in supports.items():
        print(
            f"STATE family={family} support_mean={support.mean():.9f} "
            f"support_p95={np.quantile(support, .95):.9f} "
            f"sigma_mean={sigma.mean():.9f} sigma_p95={np.quantile(sigma, .95):.9f}",
            flush=True,
        )

    survivors: list[tuple[float, str, float]] = []
    for family, support in supports.items():
        for strength in STRENGTHS:
            trial = candidate(selected_incumbent, support, strength)
            losses = held_losses(
                trial,
                selected_observed,
                selected_area,
                observed_annual,
                folds,
            )
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            stable = bool(all(np.all(gain > 0.0) for gain in gains))
            aggregate = float(sum(gain.sum() for gain in gains))
            if stable:
                survivors.append((aggregate, family, strength))
            print(
                f"BRACKET family={family} strength={strength:.2f} stable={int(stable)} "
                "annual_gain=" + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2]),
                flush=True,
            )

    probe = np.linspace(0, cells.size - 1, 64, dtype=np.int64)
    prefix_data = {
        name: values[:, :, probe].copy() for name, values in selected_data.items()
    }
    prefix_incumbent = model.predict(prefix_data, dict(model.PARAMS), None)
    changed = {name: values.copy() for name, values in prefix_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    changed_incumbent = model.predict(changed, dict(model.PARAMS), None)
    for family, cover_name in families.items():
        prefix_support, _ = coherent_cover(prefix_data, cover_name)
        before = candidate(prefix_incumbent, prefix_support, STRENGTHS[-1])
        changed_support, _ = coherent_cover(changed, cover_name)
        after = candidate(changed_incumbent, changed_support, STRENGTHS[-1])
        print(
            f"PREFIX family={family} strength={STRENGTHS[-1]:.2f} "
            f"max_abs={np.max(np.abs(before[:96]-after[:96])):.12g}",
            flush=True,
        )

    if not survivors:
        print("DECISION exact=0 ecology=0 reject=no_all_block_all_metric_survivor", flush=True)
        return 0

    survivors.sort(reverse=True)
    _, family, strength = survivors[0]
    full_support, _ = coherent_cover(data, families[family])
    trial = validate_prediction(candidate(incumbent, full_support, strength))
    trial_scores = evaluator.score(trial)
    trial_global = trial_scores["global"]
    print(
        f"DECISION exact=1 ecology=1 family={family} strength={strength:.2f}",
        flush=True,
    )
    print(
        f"EXACT overall={trial_global['overall_score']:.9f} "
        f"delta={trial_global['overall_score']-incumbent_global['overall_score']:+.9f} "
        f"bias={trial_global['bias_score']:.9f} rmse={trial_global['rmse_score']:.9f} "
        f"seasonal={trial_global['seasonal_cycle_score']:.9f} "
        f"spatial={trial_global['spatial_distribution_score']:.9f}",
        flush=True,
    )
    print(
        "REGIONS " + ",".join(
            f"{name}:{trial_scores[name]['overall_score']-incumbent_scores[name]['overall_score']:+.6f}"
            for name in sorted(key for key in trial_scores if key != "global")
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
