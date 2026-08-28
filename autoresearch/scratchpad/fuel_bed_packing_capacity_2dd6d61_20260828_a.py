"""Held-fold probe of a non-monotonic fuel-bed packing capacity law.

The source model is pinned to commit ``2dd6d61``.  The candidate contains no
target, coordinate, region, fitted coefficient, completed-year statistic, or
future normalization.  Leaf-area density is represented in its native units,

    beta_t = LAI_t / D_eff,t,
    D_eff,t = max(0.30 m,
        (n H_n + s H_s + 0.30 m * o) / (n + s + o + eps)),

where ``o`` is local crop, pasture, and rangeland surface cover.  A fixed
Rothermel-like optimum response is

    x_t = beta_t / (1.0 m^-1),
    phi_A(x_t) = x_t**A * exp(A * (1 - x_t)),
    q_t = exp(k * (phi_A(x_t) - 0.5)).

It peaks smoothly at ``beta=1 m^-1`` and declines on both sides.  The declared
shape exponents A=(0.5, 1, 2) are broad, standard, and narrow physical
sensitivity brackets; k=(0.05, 0.10, 0.20) are fixed symmetric log-capacity
amplitudes, not fitted coefficients.

The law enters only through the exact post-incumbent causal slow envelope,

    h_t = -log(1-p_t),              B_t = EMA12(h_t),
    h_raw,t = h_t q_t,              B_raw,t = EMA12(h_raw,t),
    h_new,t = B_raw,t * (h_t/B_t).

Thus the incumbent normalized timing ``h_t/B_t`` is preserved algebraically.
All EMA updates are prefix causal and initialized from the first month.
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
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask  # noqa: E402


PINNED = "2dd6d61"
EXPECTED_MODEL_BLOB = "0d05b1c75489fbdde6a1996aa993ed1e67657c71"
OFFICIAL_EXACT = 0.720099688
DEPTH_FLOOR_M = 0.30
OPTIMUM_M_INV = 1.0
WIDTHS = (0.5, 1.0, 2.0)
STRENGTHS = (0.05, 0.10, 0.20)
SLOW_MONTHS = 12.0


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
    module = types.ModuleType(f"model_{PINNED}_packing_capacity")
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


def packing_state(
    data: dict[str, np.ndarray], width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lai = np.clip(np.asarray(data["leaf_area_index"], dtype=np.float64), 0.0, None)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    secondary = np.clip(
        np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    natural_height = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    secondary_height = np.clip(
        np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    surface = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    vegetated = natural + secondary + surface
    effective_depth = (
        natural * natural_height
        + secondary * secondary_height
        + DEPTH_FLOOR_M * surface
    ) / (vegetated + 1e-12)
    effective_depth = np.maximum(effective_depth, DEPTH_FLOOR_M)
    beta = lai / effective_depth
    x = np.clip(beta / OPTIMUM_M_INV, 1e-12, 80.0)
    response = np.power(x, width) * np.exp(width * (1.0 - x))
    response = np.where(beta > 0.0, response, 0.0)
    return beta, effective_depth, np.clip(response, 0.0, 1.0)


def raw_factor(response: np.ndarray, strength: float) -> np.ndarray:
    return np.exp(strength * (response - 0.5))


def slow_capacity_projection(
    incumbent: np.ndarray, factor: np.ndarray
) -> tuple[np.ndarray, float, float]:
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    base_slow = causal_mean(hazard)
    raw_slow = causal_mean(hazard * factor)
    incumbent_timing = np.divide(
        hazard,
        base_slow,
        out=np.zeros_like(hazard),
        where=base_slow > 0.0,
    )
    projected_hazard = raw_slow * incumbent_timing
    capacity_ratio = np.divide(
        raw_slow,
        base_slow,
        out=np.ones_like(raw_slow),
        where=base_slow > 0.0,
    )
    projected_timing = np.divide(
        projected_hazard,
        raw_slow,
        out=np.zeros_like(projected_hazard),
        where=raw_slow > 0.0,
    )
    timing_error = float(np.max(np.abs(projected_timing - incumbent_timing)))
    reconstruction_error = float(
        np.max(np.abs(projected_hazard - hazard * capacity_ratio))
    )
    prediction = np.asarray(
        -np.expm1(-np.clip(projected_hazard, 0.0, 50.0)), dtype=np.float32
    )
    return prediction, timing_error, reconstruction_error


def existing_capacity_basis(
    data: dict[str, np.ndarray], incumbent: np.ndarray
) -> dict[str, np.ndarray]:
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    fine_fuel = causal_mean(gpp) / (causal_mean(gpp) + 0.35)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    secondary = np.clip(
        np.asarray(data["secondary_vegetation_fraction"], dtype=np.float64), 0.0, 1.0
    )
    natural_height = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    secondary_height = np.clip(
        np.asarray(data["secondary_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    urban = np.clip(
        np.asarray(data["luh2_urban_fraction"], dtype=np.float64), 0.0, 1.0
    )
    open_cover = np.clip(
        natural * 8.0 / (natural_height + 8.0)
        + secondary * 8.0 / (secondary_height + 8.0)
        + pasture
        + rangeland,
        0.0,
        2.0,
    )
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover * continuity
    woody_capacity = (
        natural
        * natural_height
        / (natural_height + 8.0)
        * biomass
        / (biomass + 1.0)
        + secondary
        * secondary_height
        / (secondary_height + 8.0)
        * biomass
        / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel
    secondary_open = secondary * 8.0 / (secondary_height + 8.0)
    secondary_capacity = secondary_open * fine_fuel * continuity
    total_capacity = (
        0.05 + surface_capacity + woody_capacity + crop_capacity + secondary_capacity
    )
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    canonical_slow = causal_mean(hazard)
    return {
        "canonical_slow": np.log1p(canonical_slow).mean(axis=0),
        "total_structural": np.log1p(total_capacity).mean(axis=0),
        "fine_fuel": fine_fuel.mean(axis=0),
        "surface": surface_capacity.mean(axis=0),
        "woody": woody_capacity.mean(axis=0),
        "crop": crop_capacity.mean(axis=0),
        "secondary": secondary_capacity.mean(axis=0),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def basis_r2(signal: np.ndarray, basis: dict[str, np.ndarray]) -> float:
    columns = []
    for values in basis.values():
        scale = values.std()
        if scale > 0.0:
            columns.append((values - values.mean()) / scale)
    matrix = np.column_stack((np.ones(signal.size), *columns))
    centered = signal - signal.mean()
    coefficients = np.linalg.lstsq(matrix, centered, rcond=None)[0]
    residual = centered - matrix @ coefficients
    return float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(centered)))


def half_max_width(width: float) -> tuple[float, float]:
    x = np.geomspace(1e-4, 40.0, 500000)
    response = np.power(x, width) * np.exp(width * (1.0 - x))
    accepted = x[response >= 0.5]
    return float(accepted[0]), float(accepted[-1])


def load_observation(evaluator: GFED5Evaluator) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    return observation, area


def ecology_masks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = lambda name: np.asarray(data[name], dtype=np.float64).mean(axis=0)
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


def prefix_test(
    model,
    full_data: dict[str, np.ndarray],
    rows: np.ndarray,
    columns: np.ndarray,
) -> float:
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    before_3d = {
        name: np.asarray(values[:, rows[probe], columns[probe]])[:, None, :]
        for name, values in full_data.items()
    }
    after_3d = {name: values.copy() for name, values in before_3d.items()}
    for values in after_3d.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    before_incumbent = np.asarray(
        model.predict(before_3d, dict(model.PARAMS), None)[:, 0, :], dtype=np.float64
    )
    after_incumbent = np.asarray(
        model.predict(after_3d, dict(model.PARAMS), None)[:, 0, :], dtype=np.float64
    )
    before = {name: values[:, 0, :] for name, values in before_3d.items()}
    after = {name: values[:, 0, :] for name, values in after_3d.items()}
    maximum = float(np.max(np.abs(before_incumbent[:96] - after_incumbent[:96])))
    for width in WIDTHS:
        before_response = packing_state(before, width)[2]
        after_response = packing_state(after, width)[2]
        for strength in STRENGTHS:
            before_prediction = slow_capacity_projection(
                before_incumbent, raw_factor(before_response, strength)
            )[0]
            after_prediction = slow_capacity_projection(
                after_incumbent, raw_factor(after_response, strength)
            )[0]
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
    incumbent_grid = np.asarray(model.predict(full_data, dict(model.PARAMS), None))
    land = load_land_mask()
    rows, columns = np.where(land)
    folds = ((rows // 15) + 3 * (columns // 15)) % 4
    incumbent = np.asarray(incumbent_grid[:, rows, columns], dtype=np.float64)
    data = {
        name: np.asarray(values[:, rows, columns]) for name, values in full_data.items()
    }

    basis = existing_capacity_basis(data, incumbent)
    responses = {}
    beta, depth, _ = packing_state(data, 1.0)
    print(
        f"BASE pinned={PINNED} blob={current_blob} official_exact={OFFICIAL_EXACT:.9f} "
        f"land_cells={rows.size} folds="
        + ",".join(str(int(np.sum(folds == fold))) for fold in range(4)),
        flush=True,
    )
    print(
        f"PHYSICS depth_floor_m={DEPTH_FLOOR_M:.2f} optimum_m_inv={OPTIMUM_M_INV:.2f} "
        f"beta_mean={beta.mean():.9f} beta_p05={np.quantile(beta,.05):.9f} "
        f"beta_p50={np.quantile(beta,.50):.9f} beta_p95={np.quantile(beta,.95):.9f} "
        f"depth_mean={depth.mean():.9f}",
        flush=True,
    )
    for width in WIDTHS:
        response = packing_state(data, width)[2]
        responses[width] = response
        signal = response.mean(axis=0)
        half_low, half_high = half_max_width(width)
        primary = basis["canonical_slow"]
        fold_pearson = [pearson(signal[folds == fold], primary[folds == fold]) for fold in range(4)]
        fold_spearman = [
            pearson(rank(signal[folds == fold]), rank(primary[folds == fold]))
            for fold in range(4)
        ]
        correlations = {name: pearson(signal, values) for name, values in basis.items()}
        strongest = max(correlations.items(), key=lambda item: abs(item[1]))
        print(
            f"COLLINEARITY width_A={width:.1f} halfmax_x={half_low:.6f}:{half_high:.6f} "
            f"canonical_pearson=" + ",".join(f"{value:+.6f}" for value in fold_pearson)
            + " canonical_spearman=" + ",".join(f"{value:+.6f}" for value in fold_spearman)
            + f" strongest_basis={strongest[0]}:{strongest[1]:+.6f} "
            f"basis_r2={basis_r2(signal,basis):.6f}",
            flush=True,
        )

    evaluator = GFED5Evaluator(GFED5_PATH)
    observation_grid, area_grid = load_observation(evaluator)
    observation = np.asarray(observation_grid[:, rows, columns], dtype=np.float64)
    area = np.asarray(area_grid[rows, columns], dtype=np.float64)
    observed_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    base_losses = held_losses(incumbent, observation, area, observed_annual, folds)
    masks = ecology_masks(data)
    base_ecology = ecology_ratios(incumbent, observation, area, masks)
    base_area_ratio = float(
        np.sum(np.average(incumbent, axis=0, weights=MONTH_DAYS) * area)
        / np.sum(observed_annual * area)
    )
    print(
        "BASE_HELD annual=" + ",".join(f"{value:.9f}" for value in base_losses[0])
        + " allocation=" + ",".join(f"{value:.9f}" for value in base_losses[1])
        + " raw_cycle=" + ",".join(f"{value:.9f}" for value in base_losses[2])
        + f" area_ratio={base_area_ratio:.9f}",
        flush=True,
    )

    records = []
    for width in WIDTHS:
        response = responses[width]
        for strength in STRENGTHS:
            prediction, timing_error, reconstruction_error = slow_capacity_projection(
                incumbent, raw_factor(response, strength)
            )
            losses = held_losses(prediction, observation, area, observed_annual, folds)
            gains = tuple(
                base_losses[index] - losses[index] for index in range(3)
            )
            stable = tuple(bool(np.all(gain > 0.0)) for gain in gains)
            aggregate = float(
                sum(np.sum(gains[index] / base_losses[index]) for index in range(3))
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
            area_ratio = float(
                np.sum(np.average(prediction, axis=0, weights=MONTH_DAYS) * area)
                / np.sum(observed_annual * area)
            )
            records.append(
                (
                    aggregate,
                    width,
                    strength,
                    gains,
                    stable,
                    trial_ecology,
                    area_ratio,
                    pathologies,
                )
            )
            print(
                f"HELD width_A={width:.1f} strength={strength:.2f} "
                f"annual_stable={int(stable[0])} allocation_stable={int(stable[1])} "
                f"raw_stable={int(stable[2])} aggregate={aggregate:+.9f} "
                f"annual_gain=" + ",".join(f"{value:+.9f}" for value in gains[0])
                + " allocation_gain=" + ",".join(f"{value:+.9f}" for value in gains[1])
                + " raw_cycle_gain=" + ",".join(f"{value:+.9f}" for value in gains[2])
                + f" timing_error={timing_error:.3g} reconstruction={reconstruction_error:.3g} "
                f"area_ratio={area_ratio:.9f} severe="
                + (",".join(pathologies) if pathologies else "none"),
                flush=True,
            )
            print(
                f"ECOLOGY width_A={width:.1f} strength={strength:.2f} "
                + ",".join(
                    f"{name}:{base_ecology[name]:.9f}->{trial_ecology[name]:.9f}"
                    for name in base_ecology
                ),
                flush=True,
            )

    survivors = [record for record in records if all(record[4]) and not record[7]]
    best = max(records, key=lambda record: record[0])
    print(
        f"SUMMARY all_metric_survivors={len(survivors)} "
        f"best=width{best[1]:.1f}:strength{best[2]:.2f} "
        f"aggregate={best[0]:+.9f} gates="
        + ",".join(str(int(value)) for value in best[4]),
        flush=True,
    )
    prefix_max = prefix_test(model, full_data, rows, columns)
    print(
        f"PREFIX cutoff=96 cells=64 widths=3 strengths=3 max_abs={prefix_max:.12g}",
        flush=True,
    )
    if prefix_max != 0.0:
        raise RuntimeError(f"prefix causality failed: {prefix_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
