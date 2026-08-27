"""Fixed sampled falsification after warm-open held-block diagnosis.

Tests whether any compact causal timing transition can move the warm-open peak
without changing cell-year mass, then tests a distinct smooth annual-capacity
gap using managed-open cover and trailing hazard.  All equations are fixed
before evaluation and operate without coordinates or regime labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    load_observed,
    load_selected,
)
from autoresearch.scratchpad.warm_seasonal_open_held_block_f45c0ce import (  # noqa: E402
    MONTH_DAYS,
    MONTHS,
    antecedent,
    clean_warm_open_mask,
    conserve_cell_year_mass,
    cycle_metrics,
    select_regime_cells,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def weighted_corr(left, right, weight):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    lm, rm = np.average(left, weights=weight), np.average(right, weights=weight)
    l, r = left - lm, right - rm
    return float(
        np.average(l * r, weights=weight)
        / np.sqrt(np.average(l * l, weights=weight) * np.average(r * r, weights=weight) + 1e-30)
    )


def print_result(label, prediction, observed, area, weight, rows, cols, baseline):
    values = cycle_metrics(prediction, observed, area, weight)
    print(
        f"CANDIDATE {label} ratio={values['ratio']:.6f} "
        f"annual_log_rmse={values['annual_log_rmse']:.6f} "
        f"d_annual={values['annual_log_rmse'] - baseline['annual_log_rmse']:+.6f} "
        f"alloc_rmse={values['alloc_rmse']:.6f} "
        f"d_alloc={values['alloc_rmse'] - baseline['alloc_rmse']:+.6f} "
        f"l1={values['l1']:.6f} d_l1={values['l1'] - baseline['l1']:+.6f} "
        f"peak={values['peak']}",
        flush=True,
    )
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    for fold in range(4):
        held = folds == fold
        base_fold = cycle_metrics(
            incumbent[:, held], observed[:, held], area[held], weight[held]
        )
        new_fold = cycle_metrics(
            prediction[:, held], observed[:, held], area[held], weight[held]
        )
        print(
            f"FOLD {label} fold={fold} d_annual={new_fold['annual_log_rmse'] - base_fold['annual_log_rmse']:+.6f} "
            f"d_alloc={new_fold['alloc_rmse'] - base_fold['alloc_rmse']:+.6f} "
            f"d_l1={new_fold['l1'] - base_fold['l1']:+.6f} peak={new_fold['peak']}",
            flush=True,
        )
    return values


def normalized_cycle(values, area):
    cycle = np.asarray(values, dtype=np.float64).reshape(16, 12, -1).mean(axis=0)
    monthly = cycle @ area
    return monthly / monthly.sum()


def standing_dead_signal(rain, rain3, dryness, warming3):
    production = rain / (rain + 60.0)
    drydown = np.maximum((rain3 - rain) / (rain3 + rain + 10.0), 0.0)
    combustion = dryness / (dryness + 500.0)
    moderate_warming = np.exp(-np.square(warming3 / 3.0))
    stock = np.zeros(rain.shape[1], dtype=np.float64)
    output = np.empty_like(rain, dtype=np.float64)
    for time in range(rain.shape[0]):
        stock = 0.85 * stock + production[time]
        readiness = drydown[time] * combustion[time] * moderate_warming[time]
        release = stock / (stock + 2.0) * readiness
        stock *= np.exp(-1.5 * release)
        output[time] = release
    return output


def annual_capacity_candidate(hazard, signal, strength):
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * np.exp(strength * signal), 0.0, 50.0)),
        dtype=np.float64,
    )


def recurrence_table(prediction, observed, area, managed, hazard12):
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    m = np.mean(managed, axis=0)
    h = np.mean(hazard12, axis=0)
    me = np.unique(np.quantile(m, np.linspace(0.0, 1.0, 5)))
    he = np.unique(np.quantile(h, np.linspace(0.0, 1.0, 5)))
    if me.size != 5 or he.size != 5:
        return
    ratio = np.full((4, 4), np.nan)
    share = np.full((4, 4), np.nan)
    total = float(obs_annual @ area)
    for i in range(4):
        for j in range(4):
            selected = (
                (h >= he[i])
                & (h <= he[i + 1] if i == 3 else h < he[i + 1])
                & (m >= me[j])
                & (m <= me[j + 1] if j == 3 else m < me[j + 1])
            )
            denom = float(obs_annual[selected] @ area[selected])
            ratio[i, j] = float(pred_annual[selected] @ area[selected]) / (denom + 1e-12)
            share[i, j] = denom / total
    print(
        f"RECURRENCE_TABLE hazard_edges={np.array2string(he, precision=5)} "
        f"managed_edges={np.array2string(me, precision=5)}",
        flush=True,
    )
    print("RATIO\n" + np.array2string(ratio, precision=3), flush=True)
    print("OBS_SHARE\n" + np.array2string(share, precision=3), flush=True)


def main() -> int:
    global incumbent
    evaluator = GFED5Evaluator(GFED5_PATH)
    mask = clean_warm_open_mask()
    rows, cols, area, weight, *_ = select_regime_cells(evaluator, mask)
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    incumbent = np.asarray(model.predict(data, dict(model.PARAMS), None), dtype=np.float64)[:, 0, :]
    baseline = cycle_metrics(incumbent, observed, area, weight)
    print(
        f"BASE ratio={baseline['ratio']:.6f} annual={baseline['annual_log_rmse']:.6f} "
        f"alloc={baseline['alloc_rmse']:.6f} l1={baseline['l1']:.6f} peak={baseline['peak']}",
        flush=True,
    )

    def field(name):
        return np.asarray(data[name][:, 0, :], dtype=np.float64)

    rain = np.clip(field("monthly_precipitation"), 0.0, None)
    rain3, rain12 = antecedent(rain, 3.0), antecedent(rain, 12.0)
    dryness = np.clip(field("dryness"), 0.0, None)
    temperature = field("air_temperature")
    temperature3, temperature12 = antecedent(temperature, 3.0), antecedent(temperature, 12.0)
    warming3 = temperature - temperature3
    hazard = -np.log1p(-np.clip(incumbent, 0.0, 1.0 - 1e-7))
    hazard12 = antecedent(hazard, 12.0)

    pred_cycle = normalized_cycle(incumbent, area)
    obs_cycle = normalized_cycle(observed, area)
    print("MONTHLY month pred obs rain warming3 dryness", flush=True)
    for month in range(12):
        indices = np.arange(month, 192, 12)
        print(
            f"{MONTHS[month]} {pred_cycle[month]:.6f} {obs_cycle[month]:.6f} "
            f"{np.average(rain[indices], weights=area, axis=1).mean():.4f} "
            f"{np.average(warming3[indices], weights=area, axis=1).mean():+.4f} "
            f"{np.average(dryness[indices], weights=area, axis=1).mean():.4f}",
            flush=True,
        )

    drydown = np.maximum((rain3 - rain) / (rain3 + rain + 10.0), 0.0)
    moderate_warming = np.exp(-np.square(warming3 / 3.0))
    release = drydown * dryness / (dryness + 500.0) * moderate_warming
    standing = standing_dead_signal(rain, rain3, dryness, warming3)
    for label, signal in (("drydown", release), ("standing_dead", standing)):
        centered = signal - antecedent(signal, 12.0)
        for strength in (0.5, 1.0, 2.0):
            candidate = conserve_cell_year_mass(
                incumbent, np.exp(np.clip(strength * centered, -1.5, 1.5))
            )
            print_result(
                f"{label}:strength={strength:g}", candidate, observed, area, weight, rows, cols, baseline
            )

    for months in (2, 3, 4):
        delayed = np.zeros_like(incumbent)
        delayed[months:] = incumbent[:-months]
        delayed[:months] = incumbent[:months]
        delayed = conserve_cell_year_mass(incumbent, delayed / (incumbent + 1e-9))
        print_result(
            f"fixed_delay:{months}", delayed, observed, area, weight, rows, cols, baseline
        )

    rangeland = np.clip(field("luh2_rangeland_fraction"), 0.0, 1.0)
    pasture = np.clip(field("luh2_pasture_fraction"), 0.0, 1.0)
    managed = np.clip(rangeland + pasture, 0.0, 1.0)
    managed_access = managed / (managed + 0.15)
    gap = 0.01 / (hazard12 + 0.01)
    recurrence = hazard12 / (hazard12 + 0.01)
    annual_rain = 12.0 * rain12
    fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(-annual_rain / 3000.0)
    warm = 1.0 / (1.0 + np.exp(np.clip(-(temperature12 - 18.0) / 4.0, -30.0, 30.0)))
    gain = managed_access * gap * fuel * warm
    balanced = managed_access * (gap - recurrence) * fuel * warm
    annual_target = np.log(
        (np.average(observed, axis=0, weights=MONTH_DAYS) + 1e-5)
        / (np.average(incumbent, axis=0, weights=MONTH_DAYS) + 1e-5)
    )
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    for label, signal in (("managed_gap", gain), ("managed_balance", balanced)):
        cell_signal = np.mean(signal, axis=0)
        print(
            f"SIGNAL {label} corr={weighted_corr(cell_signal, annual_target, weight):+.6f}",
            flush=True,
        )
        for fold in range(4):
            held = folds == fold
            print(
                f"SIGNAL_FOLD {label} fold={fold} corr={weighted_corr(cell_signal[held], annual_target[held], weight[held]):+.6f}",
                flush=True,
            )
        for strength in (0.25, 0.5, 1.0):
            candidate = annual_capacity_candidate(hazard, signal, strength)
            print_result(
                f"{label}:strength={strength:g}", candidate, observed, area, weight, rows, cols, baseline
            )

    recurrence_table(incumbent, observed, area, managed, hazard12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
