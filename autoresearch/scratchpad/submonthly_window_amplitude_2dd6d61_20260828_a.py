"""Held-fold audit of a current-input submonthly burn-window amplitude law.

This scratch-only experiment pins the canonical model to ``2dd6d61``.  For
current dryness D and temperature T, it evaluates the proposed window

    Dbar_t = EMA12(D_t),                  Tbar_t = EMA12(T_t)
    W_t = sigmoid((D_t-Dbar_t)/250) * sigmoid((T_t-Tbar_t)/4)
    q_t = (W_t / (EMA12(W_t)+1e-8))**gamma

at predeclared gamma=(.15,.30,.50,.75).  It compares literal hazard scaling
``h*q`` against the exact causal log-factor decomposition

    log qslow_t = EMA12(log(q_t))
    qfast_t = q_t / qslow_t
    q_t = qslow_t*qfast_t.

The second candidate applies only ``qfast`` to incumbent hazard.  This removes
the selected slow log-factor algebraically, but neither construction is called
mean-neutral or mass-conserving: actual annual hazard and burned-area drift are
measured.  All operations are point-local and prefix causal.  Coordinates
assign disjoint whole-cell folds only; GFED enters only after predictions are
fixed.  No target, coordinate, region, or fitted statistic enters prediction.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    MONTH_DAYS,
    held_losses,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


PINNED = "2dd6d61"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
EXPECTED_OVERALL = 0.720105466
GAMMAS = (0.15, 0.30, 0.50, 0.75)
DRY_SCALE = 250.0
TEMPERATURE_SCALE = 4.0
SLOW_MONTHS = 12.0
METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)


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
    module = types.ModuleType(f"model_{PINNED}_submonthly_window")
    module.__file__ = f"git:{PINNED}:autoresearch/model.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def causal_mean(values: np.ndarray, months: float = SLOW_MONTHS) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -50.0, 50.0)))


def burn_window(
    dryness: np.ndarray,
    temperature: np.ndarray,
    dry_scale: float = DRY_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dryness = np.asarray(dryness, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    dryness_reference = causal_mean(dryness)
    temperature_reference = causal_mean(temperature)
    window = sigmoid((dryness - dryness_reference) / dry_scale) * sigmoid(
        (temperature - temperature_reference) / TEMPERATURE_SCALE
    )
    return window, dryness_reference, temperature_reference


def factors(window: np.ndarray, gamma: float) -> tuple[np.ndarray, ...]:
    window_reference = causal_mean(window)
    literal = np.power(window / (window_reference + 1e-8), gamma)
    slow = np.exp(causal_mean(np.log(literal + 1e-30)))
    fast = literal / slow
    reconstruction = float(np.max(np.abs(slow * fast - literal)))
    return literal, slow, fast, window_reference, reconstruction


def apply_factor(incumbent: np.ndarray, factor: np.ndarray) -> np.ndarray:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    return np.asarray(
        -np.expm1(-np.clip(hazard * factor, 0.0, 50.0)), dtype=np.float32
    )


def annual_mass_drift(
    incumbent: np.ndarray,
    candidate: np.ndarray,
    area: np.ndarray,
    folds: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    days = MONTH_DAYS[:, None]
    base_hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    trial_hazard = -np.log1p(-np.clip(candidate, 0.0, 1.0 - 1e-7))
    burn_drift = []
    hazard_drift = []
    for fold in range(4):
        held = folds == fold
        weight = days * area[None, held]
        burn_drift.append(
            np.sum(candidate[:, held] * weight)
            / np.sum(incumbent[:, held] * weight)
            - 1.0
        )
        hazard_drift.append(
            np.sum(trial_hazard[:, held] * weight)
            / np.sum(base_hazard[:, held] * weight)
            - 1.0
        )
    base_cell = np.sum(incumbent * days, axis=0)
    trial_cell = np.sum(candidate * days, axis=0)
    valid = base_cell > 1e-8
    local = trial_cell[valid] / base_cell[valid] - 1.0
    full_weight = days * area[None, :]
    global_drift = (
        float(np.sum(candidate * full_weight) / np.sum(incumbent * full_weight) - 1.0),
        float(
            np.sum(trial_hazard * full_weight)
            / np.sum(base_hazard * full_weight)
            - 1.0
        ),
    )
    year_days = MONTH_DAYS.reshape(16, 12)
    base_year = np.sum(
        incumbent.reshape(16, 12, -1) * year_days[:, :, None] * area[None, None, :],
        axis=(1, 2),
    )
    trial_year = np.sum(
        candidate.reshape(16, 12, -1) * year_days[:, :, None] * area[None, None, :],
        axis=(1, 2),
    )
    year_drift = trial_year / base_year - 1.0
    return (
        np.asarray(burn_drift),
        np.asarray(hazard_drift),
        global_drift,
        tuple(float(value) for value in (year_drift.min(), year_drift.mean(), year_drift.max())),
        tuple(float(value) for value in np.quantile(local, (0.05, 0.50, 0.95))),
    )


def load_observation(evaluator: GFED5Evaluator) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    return observation, area


def ecology_masks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def mean(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64).mean(axis=0)

    rain = (12.0 * causal_mean(np.asarray(data["monthly_precipitation"]))).mean(axis=0)
    temperature = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    return {
        "intact_tropical_closed": (temperature >= 20) & (rain >= 1200) & (canopy >= 20) & (lai >= 3) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temperature >= 5) & (temperature < 20) & (canopy >= 15) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temperature < 5) & (canopy >= 10) & (natural >= 0.6),
        "tropical_open": (temperature >= 20) & (rain >= 500) & (rain < 1500) & (canopy >= 5) & (canopy < 20) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250) & (rain < 1500) & (biomass >= 0.2),
        "cropland": crop >= 0.5,
        "arid_low_fuel": (rain < 250) & (biomass < 0.3) & (lai < 1),
    }


def ecology_ratios(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    output = {}
    for name, mask in masks.items():
        weight = area * mask
        output[name] = float(
            np.sum(pred_annual * weight) / max(np.sum(obs_annual * weight), 1e-12)
        )
    return output


def full_grid_candidate(
    incumbent: np.ndarray,
    data: dict[str, np.ndarray],
    mode: str,
    gamma: float,
    chunk_size: int = 4096,
) -> np.ndarray:
    time = incumbent.shape[0]
    cells = int(np.prod(incumbent.shape[1:]))
    base = np.asarray(incumbent).reshape(time, cells)
    dryness = np.asarray(data["dryness"]).reshape(time, cells)
    temperature = np.asarray(data["air_temperature"]).reshape(time, cells)
    output = np.empty_like(base, dtype=np.float32)
    for start in range(0, cells, chunk_size):
        stop = min(start + chunk_size, cells)
        window = burn_window(dryness[:, start:stop], temperature[:, start:stop])[0]
        literal, _, fast, _, _ = factors(window, gamma)
        factor = literal if mode == "literal" else fast
        output[:, start:stop] = apply_factor(base[:, start:stop], factor)
    return output.reshape(incumbent.shape)


def prefix_test(
    model,
    full_data: dict[str, np.ndarray],
    rows: np.ndarray,
    columns: np.ndarray,
) -> float:
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    before_data = {
        name: np.asarray(values[:, rows[probe], columns[probe]])[:, None, :]
        for name, values in full_data.items()
    }
    after_data = {name: values.copy() for name, values in before_data.items()}
    for values in after_data.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    before_incumbent = np.asarray(
        model.predict(before_data, dict(model.PARAMS), None)[:, 0, :], dtype=np.float64
    )
    after_incumbent = np.asarray(
        model.predict(after_data, dict(model.PARAMS), None)[:, 0, :], dtype=np.float64
    )
    maximum = float(np.max(np.abs(before_incumbent[:96] - after_incumbent[:96])))
    before_window = burn_window(
        before_data["dryness"][:, 0, :], before_data["air_temperature"][:, 0, :]
    )[0]
    after_window = burn_window(
        after_data["dryness"][:, 0, :], after_data["air_temperature"][:, 0, :]
    )[0]
    for gamma in GAMMAS:
        before_literal, _, before_fast, _, _ = factors(before_window, gamma)
        after_literal, _, after_fast, _, _ = factors(after_window, gamma)
        for before_factor, after_factor in (
            (before_literal, after_literal),
            (before_fast, after_fast),
        ):
            before_prediction = apply_factor(before_incumbent, before_factor)
            after_prediction = apply_factor(after_incumbent, after_factor)
            maximum = max(
                maximum,
                float(np.max(np.abs(before_prediction[:96] - after_prediction[:96]))),
            )
    return maximum


def main() -> int:
    model = load_pinned()
    current_blob = subprocess.run(
        ("git", "hash-object", "autoresearch/model.py"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"current model drift: {current_blob}")

    full_data = load_inputs(model.INPUTS)
    incumbent_grid = validate_prediction(model.predict(full_data, dict(model.PARAMS), None))
    land = load_land_mask()
    rows, columns = np.where(land)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    incumbent = np.asarray(incumbent_grid[:, rows, columns], dtype=np.float64)
    selected_data = {
        name: np.asarray(values[:, rows, columns]) for name, values in full_data.items()
    }
    window, dryness_reference, temperature_reference = burn_window(
        selected_data["dryness"], selected_data["air_temperature"]
    )
    candidates = {}
    factor_diagnostics = {}
    for gamma in GAMMAS:
        literal, slow, fast, window_reference, reconstruction = factors(window, gamma)
        candidates[("literal", gamma)] = apply_factor(incumbent, literal)
        candidates[("factorized", gamma)] = apply_factor(incumbent, fast)
        factor_diagnostics[gamma] = (
            literal,
            slow,
            fast,
            window_reference,
            reconstruction,
        )

    print(
        f"BASE pinned={PINNED} blob={current_blob} expected_overall={EXPECTED_OVERALL:.9f} "
        f"land_cells={rows.size} folds="
        + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(
        f"STATE dry_scale={DRY_SCALE:.1f} temperature_scale={TEMPERATURE_SCALE:.1f} "
        f"dry_departure_mean={(selected_data['dryness']-dryness_reference).mean():+.9f} "
        f"temperature_departure_mean={(selected_data['air_temperature']-temperature_reference).mean():+.9f} "
        f"window_mean={window.mean():.9f} window_p05={np.quantile(window,.05):.9f} "
        f"window_p95={np.quantile(window,.95):.9f}",
        flush=True,
    )
    for gamma in GAMMAS:
        literal, slow, fast, reference, reconstruction = factor_diagnostics[gamma]
        print(
            f"FACTOR gamma={gamma:.2f} W_over_EMA_mean={(window/(reference+1e-8)).mean():.9f} "
            f"literal_mean={literal.mean():.9f} slow_mean={slow.mean():.9f} "
            f"fast_mean={fast.mean():.9f} literal_p05={np.quantile(literal,.05):.9f} "
            f"literal_p95={np.quantile(literal,.95):.9f} fast_p05={np.quantile(fast,.05):.9f} "
            f"fast_p95={np.quantile(fast,.95):.9f} reconstruction={reconstruction:.3g}",
            flush=True,
        )

    evaluator = GFED5Evaluator(GFED5_PATH)
    observation_grid, area_grid = load_observation(evaluator)
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float64)
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    observed_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    base_losses = held_losses(incumbent, observation, area, observed_annual, folds)
    masks = ecology_masks(selected_data)
    base_ecology = ecology_ratios(incumbent, observation, area, masks)
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2]),
        flush=True,
    )

    records = []
    for mode in ("literal", "factorized"):
        for gamma in GAMMAS:
            prediction = candidates[(mode, gamma)]
            losses = held_losses(prediction, observation, area, observed_annual, folds)
            gains = tuple(base_losses[index] - losses[index] for index in range(3))
            stable = tuple(bool(np.all(gain > 0.0)) for gain in gains)
            aggregate = float(
                sum(np.sum(gains[index] / base_losses[index]) for index in range(3))
            )
            burn_drift, hazard_drift, global_drift, year_drift, local_drift = annual_mass_drift(
                incumbent, prediction, area, folds
            )
            trial_ecology = ecology_ratios(prediction, observation, area, masks)
            pathologies = [
                name
                for name in base_ecology
                if trial_ecology[name] < 0.25
                or trial_ecology[name] > 4.0
                or trial_ecology[name] / base_ecology[name] < 0.75
                or trial_ecology[name] / base_ecology[name] > 1.25
            ]
            records.append(
                (
                    aggregate,
                    mode,
                    gamma,
                    gains,
                    stable,
                    burn_drift,
                    hazard_drift,
                    global_drift,
                    year_drift,
                    local_drift,
                    trial_ecology,
                    pathologies,
                )
            )
            print(
                f"HELD mode={mode} gamma={gamma:.2f} "
                f"annual_stable={int(stable[0])} allocation_stable={int(stable[1])} "
                f"raw_stable={int(stable[2])} aggregate={aggregate:+.9f} "
                f"annual_gain=" + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2]),
                flush=True,
            )
            print(
                f"MASS mode={mode} gamma={gamma:.2f} burn_fold="
                + ",".join(f"{value:+.6%}" for value in burn_drift)
                + " hazard_fold=" + ",".join(f"{value:+.6%}" for value in hazard_drift)
                + f" global_burn={global_drift[0]:+.6%} global_hazard={global_drift[1]:+.6%}"
                + " annual_global_min_mean_max="
                + ",".join(f"{value:+.6%}" for value in year_drift)
                + " local_burn_p05_p50_p95="
                + ",".join(f"{value:+.6%}" for value in local_drift),
                flush=True,
            )
            print(
                f"ECOLOGY mode={mode} gamma={gamma:.2f} "
                + ",".join(
                    f"{name}:{base_ecology[name]:.9f}->{trial_ecology[name]:.9f}"
                    for name in base_ecology
                )
                + " severe=" + (",".join(pathologies) if pathologies else "none"),
                flush=True,
            )

    prefix_max = prefix_test(model, full_data, rows, columns)
    print(
        f"PREFIX cutoff=96 cells=64 gammas=4 modes=2 max_abs={prefix_max:.12g}",
        flush=True,
    )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix causality failed: {prefix_max}")

    survivors = [record for record in records if all(record[4]) and not record[11]]
    best = max(records, key=lambda record: record[0])
    if not survivors:
        print(
            f"DECISION exact=0 survivors=0 best={best[1]}:{best[2]:.2f} "
            f"aggregate={best[0]:+.9f} components=skipped_no_all_fold_all_metric_gate",
            flush=True,
        )
        return 0

    survivor = max(survivors, key=lambda record: record[0])
    _, mode, gamma, *_ = survivor
    base_scores = evaluator.score(incumbent_grid)
    base_global = base_scores["global"]
    if abs(base_global["overall_score"] - EXPECTED_OVERALL) > 1e-6:
        raise RuntimeError(f"exact incumbent drift {base_global['overall_score']:.9f}")
    trial_grid = validate_prediction(
        full_grid_candidate(incumbent_grid, full_data, mode, gamma)
    )
    trial_global = evaluator.score(trial_grid)["global"]
    print(
        f"EXACT mode={mode} gamma={gamma:.2f} "
        + " ".join(
            f"{label}={trial_global[key]:.9f} delta={trial_global[key]-base_global[key]:+.9f}"
            for label, key in METRICS
        ),
        flush=True,
    )
    print(
        f"DECISION exact=1 survivors={len(survivors)} selected={mode}:{gamma:.2f} "
        f"overall_delta={trial_global['overall_score']-base_global['overall_score']:+.9f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
