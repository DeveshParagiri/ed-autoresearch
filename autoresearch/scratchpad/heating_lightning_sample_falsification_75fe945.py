"""Low-memory sampled falsification of heating x drying x ignition timing.

This script never edits or officially evaluates the canonical model. It runs
the unchanged model on 768 score-dominant cells, which is valid because every
canonical equation is pointwise. Coordinates select and split cells but never
enter an equation. Candidate mechanisms use fixed broad physical parameters;
there is no benchmark fit or sweep.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator, MONTH_MIDPOINTS  # noqa: E402
from scripts.runtime import GFED5_PATH, INPUTS_DIR, load_model  # noqa: E402


MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0
CYCLE_DAYS = np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64)


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -30.0, 30.0)))


def antecedent(series: np.ndarray, months: float) -> np.ndarray:
    alpha = np.float32(1.0 - np.exp(-1.0 / months))
    state = np.asarray(series[0], dtype=np.float32).copy()
    output = np.empty_like(series, dtype=np.float32)
    for time in range(series.shape[0]):
        state += alpha * (series[time] - state)
        output[time] = state
    return output


def select_cells(evaluator: GFED5Evaluator, count: int = 768):
    reference_mean = np.asarray(evaluator.reference_mean, dtype=np.float64)
    coarse_mean = reference_mean.reshape(180, 2, 360, 2).mean(axis=(1, 3)) / 100.0
    coarse_area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    weight = coarse_area * coarse_mean
    cells = np.argsort(weight.ravel())[::-1][:count]
    rows, cols = cells // 360, cells % 360
    retained = float(weight.ravel()[cells].sum() / weight.sum())
    return rows, cols, coarse_area[rows, cols], weight[rows, cols], retained


def input_index():
    result = {}
    for path in sorted(INPUTS_DIR.glob("*.nc")):
        with Dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if variable.dimensions == ("time", "lat", "lon"):
                    result[name] = path
    return result


def load_selected(names, rows: np.ndarray, cols: np.ndarray):
    index = input_index()
    result = {}
    for name in names:
        output = np.empty((192, 1, rows.size), dtype=np.float32)
        with Dataset(index[name]) as dataset:
            variable = dataset.variables[name]
            for time in range(192):
                field = np.asarray(variable[time], dtype=np.float32)
                output[time, 0] = field[rows, cols]
        result[name] = output
        print(f"LOADED {name}", flush=True)
    return result


def load_observed(rows: np.ndarray, cols: np.ndarray):
    output = np.empty((192, rows.size), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for time in range(192):
            field = np.asarray(variable[time], dtype=np.float32)
            output[time] = 0.25 * (
                field[2 * rows, 2 * cols]
                + field[2 * rows + 1, 2 * cols]
                + field[2 * rows, 2 * cols + 1]
                + field[2 * rows + 1, 2 * cols + 1]
            ) / 100.0
    return output


def trailing_sum(hazard: np.ndarray, months: int = 12):
    output = np.empty_like(hazard)
    running = np.zeros(hazard.shape[1], dtype=np.float64)
    for time in range(hazard.shape[0]):
        running += hazard[time]
        if time >= months:
            running -= hazard[time - months]
        output[time] = running
    return output


def opportunity_bank(
    hazard: np.ndarray,
    readiness: np.ndarray,
    eligibility: np.ndarray,
    share: float,
    release_gain: float = 6.0,
) -> np.ndarray:
    output = np.empty_like(hazard)
    bank = np.zeros(hazard.shape[1], dtype=np.float64)
    for time in range(hazard.shape[0]):
        store = share * eligibility[time] * (1.0 - readiness[time]) * hazard[time]
        bank += store
        release = (1.0 - np.exp(-release_gain * readiness[time])) * bank
        bank -= release
        output[time] = np.maximum(hazard[time] - store + release, 0.0)
    return output


def metrics(prediction, observed, area, reference_weight, folds):
    prediction = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    pred_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_alloc = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-8)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-8)
    alloc_square = np.average(
        np.square(pred_alloc - obs_alloc), axis=0, weights=CYCLE_DAYS
    )
    log_square = np.square(
        np.log((pred_annual + 1e-5) / (obs_annual + 1e-5))
    )
    pred_anomaly = pred_cycle - pred_cycle.mean(axis=0, keepdims=True)
    obs_anomaly = obs_cycle - obs_cycle.mean(axis=0, keepdims=True)
    raw_cycle_square = np.average(
        np.square(pred_anomaly - obs_anomaly), axis=0, weights=CYCLE_DAYS
    )
    pred_phase = MONTH_MIDPOINTS[np.argmax(pred_cycle, axis=0)]
    obs_phase = MONTH_MIDPOINTS[np.argmax(obs_cycle, axis=0)]
    shift = np.abs(pred_phase - obs_phase)
    shift = np.minimum(shift, 365.0 - shift)

    def summarize(mask):
        weight = reference_weight[mask]
        return (
            float(np.sqrt(np.average(alloc_square[mask], weights=weight))),
            float(np.sqrt(np.average(log_square[mask], weights=weight))),
            float(np.sqrt(np.average(raw_cycle_square[mask], weights=weight))),
            float(np.average(0.5 * (1.0 + np.cos(shift[mask] / 365.0 * 2.0 * np.pi)), weights=weight)),
            float(np.sum(pred_annual[mask] * area[mask]) / np.sum(obs_annual[mask] * area[mask])),
        )

    return summarize(np.ones(area.size, dtype=bool)), tuple(
        summarize(folds == fold) for fold in range(4)
    )


def format_metrics(values):
    return (
        f"alloc_rmse={values[0]:.8f} annual_log_rmse={values[1]:.8f} "
        f"raw_cycle_rmse={values[2]:.8f} phase={values[3]:.8f} "
        f"area_ratio={values[4]:.8f}"
    )


def weighted_corr(left, right, weight):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    lmean = np.average(left, weights=weight)
    rmean = np.average(right, weights=weight)
    l = left - lmean
    r = right - rmean
    return float(
        np.average(l * r, weights=weight)
        / np.sqrt(np.average(l * l, weights=weight) * np.average(r * r, weights=weight) + 1e-30)
    )


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f} "
        f"fold_counts={','.join(str(int(np.sum(folds == fold))) for fold in range(4))}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    prediction = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float64
    )[:, 0, :]
    if prediction.shape != (192, rows.size):
        raise ValueError(f"unexpected sampled prediction shape {prediction.shape}")
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()

    baseline_metrics, baseline_folds = metrics(
        prediction, observed, area, reference_weight, folds
    )
    print("BASE " + format_metrics(baseline_metrics), flush=True)

    def field(name):
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    temperature = field("air_temperature")
    dryness = np.clip(field("dryness"), 0.0, None)
    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    lightning = np.clip(field("lightning_flash_rate"), 0.0, None)
    temp3 = antecedent(temperature, 3.0)
    temp24 = antecedent(temperature, 24.0)
    dry3 = antecedent(dryness, 3.0)
    lightning3 = antecedent(lightning, 3.0)
    lightning12 = antecedent(lightning, 12.0)
    rain12 = antecedent(rain, 12.0)

    heat_onset = logistic((temperature - temp3 - 0.5) / 1.5)
    dry_departure = (dryness - dry3) / (dryness + dry3 + 100.0)
    drying_onset = logistic((dry_departure - 0.01) / 0.04)
    combustion = np.sqrt(
        dryness / (dryness + 250.0) * 1.0 / (1.0 + rain / 35.0)
    )
    lightning_departure = np.maximum(
        (lightning - lightning3) / (lightning + lightning3 + 0.002), 0.0
    )
    lightning_arrival = logistic((lightning_departure - 0.05) / 0.10)
    lightning_access = lightning3 / (lightning3 + 0.02)
    natural_ignition = lightning_access * (0.35 + 0.65 * lightning_arrival)
    heat_dry = np.clip(heat_onset * drying_onset * combustion, 0.0, 1.0)
    natural_readiness = np.clip(heat_dry * natural_ignition, 0.0, 1.0)

    primary = np.clip(field("luh2_primary_fraction"), 0.0, 1.0)
    secondary = np.clip(field("luh2_secondary_fraction"), 0.0, 1.0)
    crop = np.clip(field("luh2_cropland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    managed = np.clip(crop + pasture + rangeland, 0.0, 1.0)
    natural_cover = np.clip(primary + secondary, 0.0, 1.0)
    natural_share = natural_cover / (natural_cover + managed + 0.1)
    managed_readiness = heat_dry
    partitioned_readiness = (
        natural_share * natural_readiness + (1.0 - natural_share) * managed_readiness
    )

    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    candidate_hazards = {
        "all_hazard_lightning_bank_s10": opportunity_bank(
            hazard, natural_readiness, np.ones_like(hazard), 0.10
        ),
        "natural_partition_bank_s10": opportunity_bank(
            hazard, partitioned_readiness, natural_share, 0.10
        ),
        "natural_partition_bank_s25": opportunity_bank(
            hazard, partitioned_readiness, natural_share, 0.25
        ),
    }

    # A distinct rare-onset formulation adds ignition only where the incumbent
    # trailing hazard leaves opportunity, rather than moving all fire through a
    # bank. Fixed scales bracket a deliberately small physical contribution.
    trailing = trailing_sum(hazard)
    opportunity_gap = 1.0 / (1.0 + trailing / 0.10)
    annualized_rain = 12.0 * rain12
    fuel_support = (
        annualized_rain / (annualized_rain + 250.0)
        * np.exp(-annualized_rain / 3000.0)
    )
    onset_source = (
        natural_share * natural_readiness * fuel_support * opportunity_gap
    )
    candidate_hazards["rare_natural_onset_s0005"] = hazard + 0.0005 * onset_source
    candidate_hazards["rare_natural_onset_s0015"] = hazard + 0.0015 * onset_source

    # Long-temperature x managed-open is tested only as a directional annual
    # diagnostic. The canonical cold-thaw closure already contains this state,
    # so a negative or fold-unstable association closes it as a duplicate.
    cold = logistic((8.0 - temp24) / 3.0)
    semi_natural_range = rangeland * (1.0 - pasture) * (1.0 - crop)
    moisture_window = logistic((annualized_rain - 180.0) / 120.0) * logistic(
        (900.0 - annualized_rain) / 180.0
    )
    cold_carrier = (
        cold * semi_natural_range * moisture_window
        * lightning12 / (lightning12 + 0.01)
    )

    base_cycle = prediction.reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observed.reshape(16, 12, -1).mean(axis=0)
    base_alloc = base_cycle / (base_cycle.sum(axis=0, keepdims=True) + 1e-8)
    obs_alloc = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-8)
    cycle_target = np.tile(obs_alloc - base_alloc, (16, 1, 1)).reshape(192, -1)
    onset_contrast = natural_readiness - antecedent(natural_readiness, 12.0)
    monthly_weight = np.broadcast_to(reference_weight[None, :], cycle_target.shape)
    print(
        f"DIRECTION cycle_corr_natural_onset={weighted_corr(onset_contrast, cycle_target, monthly_weight):+.8f}",
        flush=True,
    )

    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    annual_target = np.log((obs_annual + 1e-5) / (pred_annual + 1e-5))
    cold_state = np.average(cold_carrier, axis=0, weights=MONTH_DAYS)
    print(
        f"DIRECTION annual_corr_cold_carrier={weighted_corr(cold_state, annual_target, reference_weight):+.8f}",
        flush=True,
    )
    for fold in range(4):
        held = folds == fold
        print(
            f"DIRECTION_FOLD fold={fold} cycle_corr={weighted_corr(onset_contrast[:, held], cycle_target[:, held], monthly_weight[:, held]):+.8f} "
            f"cold_corr={weighted_corr(cold_state[held], annual_target[held], reference_weight[held]):+.8f}",
            flush=True,
        )

    for label, candidate_hazard in candidate_hazards.items():
        candidate = 1.0 - np.exp(-np.clip(candidate_hazard, 0.0, 50.0))
        current_metrics, current_folds = metrics(
            candidate, observed, area, reference_weight, folds
        )
        print("CANDIDATE " + label + " " + format_metrics(current_metrics), flush=True)
        print(
            "DELTA " + label + " "
            + " ".join(
                f"{name}={current_metrics[index] - baseline_metrics[index]:+.8f}"
                for index, name in enumerate(
                    ("alloc_rmse", "annual_log_rmse", "raw_cycle_rmse", "phase", "area_ratio")
                )
            ),
            flush=True,
        )
        for fold in range(4):
            print(
                f"FOLD {label} fold={fold} alloc_delta={current_folds[fold][0] - baseline_folds[fold][0]:+.8f} "
                f"annual_delta={current_folds[fold][1] - baseline_folds[fold][1]:+.8f} "
                f"raw_cycle_delta={current_folds[fold][2] - baseline_folds[fold][2]:+.8f} "
                f"phase_delta={current_folds[fold][3] - baseline_folds[fold][3]:+.8f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
