"""Held-block falsifier for a fully additive pathway topology.

This scratch-only diagnostic replaces, rather than corrects, the canonical
product/event/bank cascade with four independently resolved hazard sources::

    BA = 1 - exp(-(s_surface H_surface + s_woody H_woody
                   + s_crop H_crop + s_background H_background)).

Each source receives a fuel-bearing capacity, ignition access, combustion
readiness, and calendar-month exposure exactly once.  The only learned values
are four non-negative global pathway scales fitted on three spatial blocks and
tested on the fourth.  Coordinates define folds only.  No fitted surface,
target, region, coordinate, neighbour, future value, or invalid forcing enters
the source equations.  This script never edits or evaluates the canonical
model or official ledger.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.optimize import least_squares, nnls


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    selected_input,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_land_mask, load_model  # noqa: E402


EXPECTED_MODEL_BLOB = "731e1ee048fd1099dffe75d11a738fd9125f8064"
EXPECTED_BASE = 0.719021686
PATHWAYS = ("surface", "woody", "crop", "background")
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0
MONTH_EXPOSURE = MONTH_DAYS / MONTH_DAYS.mean()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def ema(values: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    output = np.empty_like(values, dtype=np.float32)
    state = np.asarray(values[0], dtype=np.float32).copy()
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def field(data: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(data[name][:, 0, :], dtype=np.float32)


def load_observation() -> np.ndarray:
    output = np.empty((192, 180, 360), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for row in range(180):
            slab = np.ma.asarray(variable[:192, 2 * row : 2 * row + 2, :])
            if np.ma.getmaskarray(slab).any():
                raise ValueError("masked GFED observation")
            output[:, row, :] = np.asarray(slab, dtype=np.float32).reshape(
                192, 2, 360, 2
            ).mean(axis=(1, 3)) / 100.0
    return output


def select_high_weight(
    observation: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    annual = observation.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
    weight = area * annual
    order = np.argsort(weight.ravel())[::-1]
    cumulative = np.cumsum(weight.ravel()[order]) / max(float(weight.sum()), 1e-12)
    count = int(np.searchsorted(cumulative, 0.85) + 1)
    selected = order[:count]
    retained = float(weight.ravel()[selected].sum() / weight.sum())
    return selected // 360, selected % 360, weight.ravel()[selected], retained


def selected_inputs(model, rows: np.ndarray, columns: np.ndarray):
    return {name: selected_input(name, rows, columns) for name in model.INPUTS}


def build_sources(
    data: Mapping[str, np.ndarray],
    shape: str,
) -> np.ndarray:
    """Return four transparent monthly hazard bases with no fitted values."""
    rain = np.clip(field(data, "monthly_precipitation"), 0.0, None)
    temperature = field(data, "air_temperature")
    dryness = np.clip(field(data, "dryness"), 0.0, None)
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    lightning = np.clip(field(data, "lightning_flash_rate"), 0.0, None)

    rain3, rain6, rain12 = ema(rain, 3.0), ema(rain, 6.0), ema(rain, 12.0)
    temperature3, temperature12, temperature24 = (
        ema(temperature, 3.0),
        ema(temperature, 12.0),
        ema(temperature, 24.0),
    )
    gpp3, gpp12 = ema(gpp, 3.0), ema(gpp, 12.0)
    lightning12 = ema(lightning, 12.0)

    annual_rain = 12.0 * rain12
    rain_support = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
        -annual_rain / 3000.0
    )
    fine_fuel = gpp12 / (gpp12 + 0.35)
    rain_exclusion = 1.0 / (1.0 + rain / 70.0)
    combustion = dryness / (dryness + 350.0) * rain_exclusion
    deficit6 = np.maximum((rain6 - rain) / (rain6 + rain + 10.0), 0.0)
    deficit12 = np.maximum((rain12 - rain) / (rain12 + rain + 10.0), 0.0)
    wet_anomaly = np.maximum((rain - rain12) / (rain + rain12 + 10.0), 0.0)
    curing = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.2), 0.0)
    curing = curing / (curing + 0.05)
    thermal_open = sigmoid((temperature - 8.0) / 4.0)
    thermal_woody = sigmoid((temperature - 5.0) / 3.0)
    warm3 = sigmoid((temperature - temperature3 - 1.0) / 2.0)
    warm12 = sigmoid((temperature - temperature12 - 2.0) / 2.0)
    cold_mean = sigmoid((5.0 - temperature24) / 3.0)
    thaw = sigmoid((temperature - temperature24 - 1.0) / 2.0)

    natural = np.clip(field(data, "natural_vegetation_fraction"), 0.0, 1.0)
    secondary = np.clip(field(data, "secondary_vegetation_fraction"), 0.0, 1.0)
    canopy = np.clip(field(data, "natural_canopy_height"), 0.0, None)
    secondary_canopy = np.clip(field(data, "secondary_canopy_height"), 0.0, None)
    biomass = np.clip(field(data, "aboveground_biomass"), 0.0, None)
    leaf_area = np.clip(field(data, "leaf_area_index"), 0.0, None)
    crop = np.clip(field(data, "luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field(data, "luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field(data, "luh2_rangeland_fraction"), 0.0, 1.0)
    urban = np.clip(field(data, "luh2_urban_fraction"), 0.0, 1.0)

    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    managed_open = np.clip(pasture + rangeland, 0.0, 1.0)
    woody_cover = np.clip(
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0),
        0.0,
        1.0,
    )
    humid_closed = (
        sigmoid((annual_rain - 1200.0) / 250.0)
        * sigmoid((temperature24 - 18.0) / 4.0)
        * sigmoid((leaf_area - 2.5) / 0.5)
        * woody_cover
    )
    continuity = 1.0 / (1.0 + 2.0 * np.power(crop, 1.5) + 5.0 * urban)
    natural_surface = natural_open * np.exp(-3.0 * humid_closed)
    open_cover = np.clip(natural_surface + secondary_open + managed_open, 0.0, 1.5)

    natural_ignition = lightning12 / (lightning12 + 0.02)
    managed_ignition = managed_open / (managed_open + 0.10)
    crop_ignition = crop / (crop + 0.08)
    surface_ignition = 1.0 - (1.0 - natural_ignition) * (
        1.0 - 0.65 * managed_ignition
    )

    surface_capacity = (1.0 - crop) * fine_fuel * open_cover * continuity
    surface_readiness = (
        combustion * (0.20 + 0.80 * deficit6) * (0.25 + 0.75 * curing) * thermal_open
    )
    woody_capacity = woody_cover * biomass / (biomass + 1.0)
    warm_woody = deficit12 * combustion * warm12 * thermal_woody * (1.0 - wet_anomaly)
    cold_woody = cold_mean * thaw * combustion * natural_ignition
    woody_readiness = np.clip(warm_woody + 0.35 * cold_woody, 0.0, 1.0)
    woody_readiness *= np.exp(-3.0 * humid_closed)
    crop_capacity = crop * fine_fuel * continuity
    crop_readiness = combustion * curing * warm3 * thermal_open
    background_capacity = 0.05 * (
        0.20 + 0.80 * np.clip(fine_fuel + woody_capacity, 0.0, 1.0)
    ) * rain_support
    background_ignition = 1.0 - (1.0 - natural_ignition) * (
        1.0 - 0.50 * np.maximum(managed_ignition, crop_ignition)
    )
    background_readiness = combustion * (0.25 + 0.75 * deficit12) * thermal_open
    background_readiness *= np.exp(-2.0 * humid_closed)

    if shape == "direct":
        responses = (
            surface_readiness,
            woody_readiness,
            crop_readiness,
            background_readiness,
        )
    elif shape == "soft_ready":
        # A bounded square-root response asks whether the direct products are
        # simply too veto-prone without introducing any new coefficient.
        responses = tuple(
            np.sqrt(np.clip(readiness, 0.0, 1.0))
            for readiness in (
                surface_readiness,
                woody_readiness,
                crop_readiness,
                background_readiness,
            )
        )
    elif shape == "relative_ready":
        # Capacity controls the annual map while current readiness relative to
        # its causal annual state controls allocation.  The bounded ratio is a
        # constitutive shape, not a learned curve or completed-record normal.
        responses = tuple(
            np.clip(
                (readiness + 0.02) / (ema(readiness, 12.0) + 0.02),
                0.05,
                4.0,
            )
            for readiness in (
                surface_readiness,
                woody_readiness,
                crop_readiness,
                background_readiness,
            )
        )
    else:
        raise ValueError(shape)

    surface_response, woody_response, crop_response, background_response = responses

    exposure = MONTH_EXPOSURE[:, None]
    sources = np.stack(
        (
            surface_capacity * surface_ignition * surface_response * exposure,
            woody_capacity * natural_ignition * woody_response * exposure,
            crop_capacity * crop_ignition * crop_response * exposure,
            background_capacity * background_ignition * background_response * exposure,
        ),
        axis=-1,
    )
    return np.asarray(np.clip(sources, 0.0, 4.0), dtype=np.float32)


def predict_from_sources(sources: np.ndarray, scales: np.ndarray) -> np.ndarray:
    hazard = np.tensordot(sources, scales, axes=([-1], [0]))
    return np.asarray(-np.expm1(-np.clip(hazard, 0.0, 50.0)), dtype=np.float32)


def cycle_annual(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cycle = values.reshape(16, 12, values.shape[1]).mean(axis=0)
    return cycle, cycle.sum(axis=0)


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def metrics(
    prediction: np.ndarray,
    observation: np.ndarray,
    cell_weight: np.ndarray,
    area: np.ndarray,
) -> dict[str, float]:
    eps = 1e-6
    pred_cycle, pred_annual = cycle_annual(prediction)
    obs_cycle, obs_annual = cycle_annual(observation)
    weights = cell_weight / max(float(np.mean(cell_weight)), 1e-12)
    annual_log = np.log(pred_annual + eps) - np.log(obs_annual + eps)
    pred_allocation = pred_cycle / (pred_annual[None, :] + eps)
    obs_allocation = obs_cycle / (obs_annual[None, :] + eps)
    cycle_error = pred_allocation - obs_allocation
    raw_cycle_error = pred_cycle - obs_cycle
    return {
        "annual_log_rmse": weighted_rmse(annual_log, weights),
        "annual_log_mae": float(np.sum(weights * np.abs(annual_log)) / np.sum(weights)),
        "cycle_rmse": weighted_rmse(cycle_error, weights[None, :]),
        "raw_cycle_rmse": weighted_rmse(raw_cycle_error, weights[None, :]),
        "area_ratio": float(
            np.sum(pred_annual * area) / max(float(np.sum(obs_annual * area)), 1e-12)
        ),
    }


def fit_scales(
    sources: np.ndarray,
    observation: np.ndarray,
    incumbent: np.ndarray,
    cell_weight: np.ndarray,
) -> np.ndarray:
    """Fit four non-negative scales to balanced annual and cycle residuals."""
    eps = 1e-6
    obs_cycle, obs_annual = cycle_annual(observation)
    base_cycle, base_annual = cycle_annual(incumbent)
    weights = cell_weight / max(float(np.mean(cell_weight)), 1e-12)
    base_annual_error = np.log(base_annual + eps) - np.log(obs_annual + eps)
    base_allocation = base_cycle / (base_annual[None, :] + eps)
    obs_allocation = obs_cycle / (obs_annual[None, :] + eps)
    annual_scale = max(weighted_rmse(base_annual_error, weights), 1e-4)
    cycle_scale = max(
        weighted_rmse(base_allocation - obs_allocation, weights[None, :]),
        1e-4,
    )

    target_hazard = -np.log1p(-np.clip(observation, 0.0, 1.0 - 1e-7))
    monthly_weight = weights[None, :] * (
        target_hazard + obs_annual[None, :] / 12.0 + 1e-5
    )
    matrix = sources.reshape(-1, 4).astype(np.float64)
    target = target_hazard.reshape(-1).astype(np.float64)
    root_weight = np.sqrt(monthly_weight.reshape(-1).astype(np.float64))
    initial, _ = nnls(matrix * root_weight[:, None], target * root_weight)
    initial = np.clip(initial, 1e-5, 100.0)

    def residual(log_scales: np.ndarray) -> np.ndarray:
        prediction = predict_from_sources(sources, np.exp(log_scales))
        pred_cycle, pred_annual = cycle_annual(prediction)
        annual = (
            np.log(pred_annual + eps) - np.log(obs_annual + eps)
        ) / annual_scale
        allocation = pred_cycle / (pred_annual[None, :] + eps)
        cycle = (allocation - obs_allocation) / cycle_scale
        # Equalize the total annual-map and normalized-cycle contributions.
        annual_residual = np.sqrt(weights) * annual / np.sqrt(weights.size)
        cycle_residual = (
            np.sqrt(weights)[None, :]
            * cycle
            / np.sqrt(12.0 * weights.size)
        )
        return np.concatenate((annual_residual, cycle_residual.reshape(-1)))

    result = least_squares(
        residual,
        np.log(initial),
        bounds=(np.log(1e-6), np.log(200.0)),
        max_nfev=120,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    return np.exp(result.x)


def ecological_ratios_selected(
    prediction: np.ndarray,
    observation: np.ndarray,
    data: Mapping[str, np.ndarray],
    area: np.ndarray,
) -> dict[str, float]:
    _, pred_annual = cycle_annual(prediction)
    _, obs_annual = cycle_annual(observation)

    def mean(name: str) -> np.ndarray:
        return field(data, name).mean(axis=0)

    rain = (12.0 * ema(field(data, "monthly_precipitation"), 12.0)).mean(axis=0)
    temp = mean("air_temperature")
    lai = mean("leaf_area_index")
    canopy = mean("natural_canopy_height")
    biomass = mean("aboveground_biomass")
    natural = mean("natural_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    rangeland = mean("luh2_rangeland_fraction")
    regimes = {
        "intact_tropical_closed": (temp >= 20) & (rain >= 1200) & (canopy >= 20) & (lai >= 3) & (natural >= 0.7) & (primary >= 0.5),
        "temperate_closed": (temp >= 5) & (temp < 20) & (canopy >= 15) & (lai >= 2.5) & (natural >= 0.6),
        "boreal": (temp < 5) & (canopy >= 10) & (natural >= 0.6),
        "tropical_open": (temp >= 20) & (rain >= 500) & (rain < 1500) & (canopy >= 5) & (canopy < 20) & (natural >= 0.5),
        "productive_rangeland": (rangeland >= 0.4) & (rain >= 250) & (rain < 1500) & (biomass >= 0.2),
        "cropland": crop >= 0.5,
        "arid_low_fuel": (rain < 250) & (biomass < 0.3) & (lai < 1),
    }
    output = {}
    for name, mask in regimes.items():
        weight = area * mask
        denominator = float(np.sum(obs_annual * weight))
        output[name] = float(np.sum(pred_annual * weight) / max(denominator, 1e-12))
    return output


def metric_line(values: Mapping[str, float]) -> str:
    return " ".join(f"{name}={value:.9f}" for name, value in values.items())


def main() -> int:
    blob = subprocess.run(
        ["git", "hash-object", "autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f"model blob changed: {blob}")
    model = load_model()
    evaluator = GFED5Evaluator(GFED5_PATH)
    area_grid = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    observation_grid = load_observation()
    rows, columns, cell_weight, retained = select_high_weight(observation_grid, area_grid)
    print(
        f"SELECTION cells={rows.size} observed_fire_weight={retained:.9f} "
        "coordinates_used_only_for_folds=1",
        flush=True,
    )
    data = selected_inputs(model, rows, columns)
    observation = observation_grid[:, rows, columns]
    area = area_grid[rows, columns]
    incumbent = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float32
    )[:, 0, :]
    del observation_grid
    gc.collect()

    cell_folds = ((rows // 15) + 3 * (columns // 15)) % 4
    base_all = metrics(incumbent, observation, cell_weight, area)
    print(f"BASE model_blob={blob} exact_reference={EXPECTED_BASE:.9f} {metric_line(base_all)}", flush=True)

    for shape in ("direct", "soft_ready", "relative_ready"):
        sources = build_sources(data, shape)
        oof = np.empty_like(observation, dtype=np.float32)
        fold_scales = []
        annual_wins = cycle_wins = raw_cycle_wins = 0
        for fold in range(4):
            train = cell_folds != fold
            held = cell_folds == fold
            scales = fit_scales(
                sources[:, train, :],
                observation[:, train],
                incumbent[:, train],
                cell_weight[train],
            )
            prediction = predict_from_sources(sources[:, held, :], scales)
            oof[:, held] = prediction
            fold_scales.append(scales)
            base_fold = metrics(
                incumbent[:, held], observation[:, held], cell_weight[held], area[held]
            )
            candidate_fold = metrics(
                prediction, observation[:, held], cell_weight[held], area[held]
            )
            annual_wins += candidate_fold["annual_log_rmse"] < base_fold["annual_log_rmse"]
            cycle_wins += candidate_fold["cycle_rmse"] < base_fold["cycle_rmse"]
            raw_cycle_wins += candidate_fold["raw_cycle_rmse"] < base_fold["raw_cycle_rmse"]
            scale_text = ",".join(
                f"{name}:{value:.7g}" for name, value in zip(PATHWAYS, scales)
            )
            delta_text = " ".join(
                f"d_{name}={candidate_fold[name] - base_fold[name]:+.9f}"
                for name in ("annual_log_rmse", "cycle_rmse", "raw_cycle_rmse", "area_ratio")
            )
            print(
                f"FOLD shape={shape} fold={fold} scales={scale_text} "
                f"base_{metric_line(base_fold)} candidate_{metric_line(candidate_fold)} {delta_text}",
                flush=True,
            )

        candidate_all = metrics(oof, observation, cell_weight, area)
        incumbent_ecology = ecological_ratios_selected(incumbent, observation, data, area)
        candidate_ecology = ecological_ratios_selected(oof, observation, data, area)
        scales_array = np.stack(fold_scales)
        scale_cv = scales_array.std(axis=0) / np.maximum(scales_array.mean(axis=0), 1e-12)
        print(
            f"SUMMARY shape={shape} annual_wins={annual_wins}/4 cycle_wins={cycle_wins}/4 "
            f"raw_cycle_wins={raw_cycle_wins}/4 base={metric_line(base_all)} "
            f"candidate={metric_line(candidate_all)}",
            flush=True,
        )
        print(
            "SCALE_STABILITY shape=" + shape + " "
            + " ".join(
                f"{name}_mean={scales_array[:, index].mean():.9g} "
                f"{name}_cv={scale_cv[index]:.6f}"
                for index, name in enumerate(PATHWAYS)
            ),
            flush=True,
        )
        print(
            "ECOLOGY shape=" + shape + " "
            + " ".join(
                f"{name}={incumbent_ecology[name]:.5f}->{candidate_ecology[name]:.5f}"
                for name in incumbent_ecology
            ),
            flush=True,
        )

        # The source equations are prefix-causal by construction; verify that
        # independently on a diverse subset rather than relying on inspection.
        probe = np.linspace(0, rows.size - 1, min(48, rows.size), dtype=int)
        probe_data = {name: values[:, :, probe].copy() for name, values in data.items()}
        before = build_sources(probe_data, shape)
        for values in probe_data.values():
            values[96:] *= np.float32(0.5)
        after = build_sources(probe_data, shape)
        prefix_delta = float(np.max(np.abs(before[:96] - after[:96])))
        print(f"PREFIX shape={shape} max_abs_first96={prefix_delta:.12g}", flush=True)
        del sources, oof
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
