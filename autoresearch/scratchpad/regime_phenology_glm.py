"""Distil nonlinear ML signals into a smooth ecological phenology GAM.

The basis contains physical hinge responses and continuous regime gates only.
It uses no coordinates, labels, cell identities, or geographic branching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, load_land_mask, load_model, validate_prediction  # noqa: E402


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-value, -40.0, 40.0)))


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> float:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} bias={score['bias_score']:.4f} "
        f"rmse={score['rmse_score']:.4f} seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}", flush=True
    )
    return float(score["overall_score"])


def main() -> int:
    model = load_model()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    incumbent_cycle = incumbent.reshape(16, 12, 180, 360).mean(axis=0)
    incumbent_annual = incumbent_cycle.sum(axis=0)
    incumbent_alloc = incumbent_cycle / (incumbent_annual[None, ...] + 1e-12)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    obs = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    obs_alloc = obs_cycle / (obs_annual[None, ...] + 1e-12)

    cells = np.flatnonzero(load_land_mask().ravel())
    cell_rows, cell_cols = cells // 360, cells % 360
    months = np.tile(np.arange(12), len(cells))
    rows = np.repeat(cell_rows, 12)
    cols = np.repeat(cell_cols, 12)
    names: list[str] = []
    columns: list[np.ndarray] = []
    feature_values: dict[str, np.ndarray] = {}

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        values = np.asarray(value, dtype=np.float32)
        columns.append(values)
        feature_values[name] = values

    def spline(name: str, value: np.ndarray, knots: tuple[float, ...]) -> None:
        add(name, value)
        for knot in knots:
            add(f"{name}_above_{knot:.6g}", np.maximum(value - knot, 0.0))

    current = incumbent_alloc[months, rows, cols]
    spline("log_incumbent", np.log(current + 1e-6), tuple(np.log(np.asarray((0.01, 0.03, 0.06, 0.10, 0.16, 0.24)))))
    angle = 2.0 * np.pi * months / 12.0
    harmonics: dict[str, np.ndarray] = {}
    for harmonic in (1, 2, 3):
        for label, trig in (("sin", np.sin), ("cos", np.cos)):
            name = f"{label}{harmonic}"
            harmonics[name] = trig(harmonic * angle)
            add(name, harmonics[name])

    cycles = {
        name: data[name].reshape(16, 12, 180, 360).mean(axis=0)
        for name in model.INPUTS
    }
    current_values = {name: cycle[months, rows, cols] for name, cycle in cycles.items()}
    previous_values = {
        name: np.roll(cycle, 1, axis=0)[months, rows, cols]
        for name, cycle in cycles.items()
    }

    for lag, values in (("current", current_values), ("previous", previous_values)):
        spline(f"temperature_{lag}", values["air_temperature"], (-10, 0, 5, 10, 15, 20, 25, 30, 35))
        spline(f"precipitation_{lag}", np.log1p(values["monthly_precipitation"]), tuple(np.log1p((1, 5, 10, 25, 50, 100, 200))))
        spline(f"dryness_{lag}", np.log1p(values["dryness"]), tuple(np.log1p((10, 50, 200, 1000, 5000, 20000))))
        spline(f"gpp_{lag}", np.log1p(values["gpp"]), tuple(np.log1p((0.05, 0.2, 0.5, 1, 2, 3.5))))
        spline(f"lightning_{lag}", np.log1p(100.0 * values["lightning_flash_rate"]), tuple(np.log1p(100.0 * np.asarray((0.0001, 0.001, 0.005, 0.02, 0.08, 0.2)))))

    local_mean: dict[str, np.ndarray] = {
        name: cycle.mean(axis=0)[cell_rows, cell_cols] for name, cycle in cycles.items()
    }
    repeated = {name: np.repeat(value, 12) for name, value in local_mean.items()}
    spline("annual_rain", np.log1p(repeated["annual_precipitation"]), tuple(np.log1p((100, 250, 500, 750, 1000, 1500, 2500, 4000))))
    spline("canopy_height", repeated["natural_canopy_height"], (2, 5, 10, 15, 20, 25, 30))
    spline("leaf_area", repeated["leaf_area_index"], (0.25, 0.5, 1, 2, 3, 4, 5))
    spline("biomass", np.log1p(repeated["aboveground_biomass"]), tuple(np.log1p((0.05, 0.2, 0.5, 1, 2, 5, 10))))
    for name in (
        "luh2_cropland_fraction", "luh2_rangeland_fraction", "luh2_pasture_fraction",
        "luh2_primary_fraction", "luh2_secondary_fraction", "luh2_urban_fraction",
        "natural_vegetation_fraction", "secondary_vegetation_fraction",
    ):
        add(name, repeated[name])

    temperature = current_values["air_temperature"]
    precipitation = current_values["monthly_precipitation"]
    previous_precip = previous_values["monthly_precipitation"]
    dryness = current_values["dryness"]
    previous_gpp = previous_values["gpp"]
    lightning = current_values["lightning_flash_rate"]
    dry_window = sigmoid((np.log1p(dryness) - np.log1p(200.0)) / 0.8) / (1.0 + precipitation / 40.0)
    warm_window = sigmoid((temperature - 10.0) / 4.0)
    antecedent_fuel = np.log1p(previous_precip / 25.0) * dry_window
    green_fuel = np.log1p(previous_gpp) * dry_window
    ignition = np.log1p(100.0 * lightning) * dry_window
    add("dry_combustion_window", dry_window)
    add("warm_dry_combustion", warm_window * dry_window)
    add("antecedent_rain_fuel", antecedent_fuel)
    add("antecedent_green_fuel", green_fuel)
    add("lightning_in_dry_fuel", ignition)

    tmean = repeated["air_temperature"]
    rain = repeated["annual_precipitation"]
    canopy = repeated["natural_canopy_height"]
    lai = repeated["leaf_area_index"]
    natural = repeated["natural_vegetation_fraction"]
    crop = repeated["luh2_cropland_fraction"]
    range_ = repeated["luh2_rangeland_fraction"]
    primary = repeated["luh2_primary_fraction"]
    wet_forest = sigmoid((tmean - 20) / 2) * sigmoid((rain - 1200) / 250) * sigmoid((canopy - 15) / 3) * sigmoid((lai - 2.5) / 0.5) * natural
    intact_forest = wet_forest * sigmoid((primary - 0.5) / 0.1)
    boreal = sigmoid((5 - tmean) / 3) * sigmoid((canopy - 8) / 3) * natural
    savanna = sigmoid((rain - 350) / 120) * sigmoid((1500 - rain) / 250) * (0.5 * natural + 0.5 * range_)
    cool_crop = crop * sigmoid((18 - tmean) / 3)
    for name, gate in (("wet_forest", wet_forest), ("intact_forest", intact_forest), ("boreal", boreal), ("savanna", savanna), ("cool_crop", cool_crop)):
        add(name, gate)
        add(f"{name}_x_dry_window", gate * dry_window)
        add(f"{name}_x_antecedent_fuel", gate * antecedent_fuel)
        add(f"{name}_x_warm_window", gate * warm_window)
        for harmonic in ("sin1", "cos1"):
            add(f"{name}_x_{harmonic}", gate * harmonics[harmonic])

    # The diagnostic tree's repeated hierarchy was fire opportunity followed
    # by absolute temperature, current/antecedent rain, dryness, fuel, and
    # lightning. Express that hierarchy without branching: smooth opportunity
    # gates continuously modulate a compact set of physical response signals.
    anomaly_signals: dict[str, np.ndarray] = {}
    for name in (
        "air_temperature", "monthly_precipitation", "dryness", "gpp",
        "lightning_flash_rate", "leaf_area_index",
    ):
        cycle = cycles[name]
        mean = cycle.mean(axis=0)
        scale = cycle.std(axis=0) + 1e-6
        anomaly = (cycle - mean[None, ...]) / scale[None, ...]
        anomaly_signals[f"{name}_anomaly"] = anomaly[months, rows, cols]
        anomaly_signals[f"{name}_previous_anomaly"] = np.roll(anomaly, 1, axis=0)[months, rows, cols]

    interaction_signals = {
        "temperature_above_0": np.maximum(temperature, 0.0) / 10.0,
        "temperature_above_15": np.maximum(temperature - 15.0, 0.0) / 10.0,
        "temperature_above_25": np.maximum(temperature - 25.0, 0.0) / 10.0,
        "previous_temperature_above_0": np.maximum(previous_values["air_temperature"], 0.0) / 10.0,
        "current_rain_log": np.log1p(precipitation / 25.0),
        "current_rain_above_25": np.maximum(np.log1p(precipitation) - np.log1p(25.0), 0.0),
        "current_rain_above_100": np.maximum(np.log1p(precipitation) - np.log1p(100.0), 0.0),
        "previous_rain_log": np.log1p(previous_precip / 25.0),
        "previous_rain_above_25": np.maximum(np.log1p(previous_precip) - np.log1p(25.0), 0.0),
        "previous_rain_above_100": np.maximum(np.log1p(previous_precip) - np.log1p(100.0), 0.0),
        "annual_rain_log": np.log1p(rain / 500.0),
        "annual_rain_above_750": np.maximum(np.log1p(rain) - np.log1p(750.0), 0.0),
        "annual_rain_above_1500": np.maximum(np.log1p(rain) - np.log1p(1500.0), 0.0),
        "dryness_log": np.log1p(dryness / 200.0),
        "previous_dryness_log": np.log1p(previous_values["dryness"] / 200.0),
        "current_gpp_log": np.log1p(current_values["gpp"]),
        "previous_gpp_log": np.log1p(previous_gpp),
        "current_lightning_log": np.log1p(100.0 * lightning),
        "previous_lightning_log": np.log1p(100.0 * previous_values["lightning_flash_rate"]),
        "crop_fraction": crop,
        "rangeland_fraction": range_,
        **anomaly_signals,
    }
    for threshold in (0.02, 0.05, 0.08, 0.13, 0.20, 0.30):
        opportunity = sigmoid((current - threshold) / 0.02)
        for name, signal in interaction_signals.items():
            add(f"opportunity_{threshold:.2f}_x_{name}", opportunity * signal)

    x = np.column_stack(columns).astype(np.float64)
    y = obs_alloc[months, rows, cols].astype(np.float64)
    weight = np.repeat(obs_annual[cell_rows, cell_cols] + float(obs_annual.mean()) * 0.01, 12)
    x_mean = np.average(x, axis=0, weights=weight)
    x_scale = np.sqrt(np.average(np.square(x - x_mean), axis=0, weights=weight)) + 1e-8
    xs = (x - x_mean) / x_scale
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)
    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)

    rng = np.random.default_rng(401)
    folds = np.repeat(rng.integers(0, 3, size=len(cells)), 12)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_coefficients: list[np.ndarray] = []
    for fold in range(3):
        train = folds != fold
        held = ~train
        reg = PoissonRegressor(alpha=0.003, max_iter=1000, tol=1e-8)
        reg.fit(xs[train], y[train], sample_weight=weight[train])
        oof[held] = reg.predict(xs[held])
        fold_coefficients.append(reg.coef_)
    correlations = np.corrcoef(np.asarray(fold_coefficients))
    print(f"fold coefficient correlation min={correlations[np.triu_indices(3, 1)].min():.4f}", flush=True)

    def candidate_from(values: np.ndarray, strength: float) -> np.ndarray:
        learned = np.zeros((12, 180, 360), dtype=np.float64)
        learned[months, rows, cols] = np.maximum(values, 1e-12)
        learned /= learned.sum(axis=0, keepdims=True) + 1e-12
        blended = np.power(incumbent_alloc + 1e-12, 1.0 - strength) * np.power(learned + 1e-12, strength)
        blended /= blended.sum(axis=0, keepdims=True) + 1e-12
        cycle = incumbent_annual[None, ...] * blended
        return np.tile(cycle, (16, 1, 1)).astype(np.float32)

    for strength in (0.5, 0.75, 1.0):
        report(evaluator, f"three-fold OOF regime strength={strength}", candidate_from(oof, strength))
    for alpha in (0.01, 0.003, 0.001):
        reg = PoissonRegressor(alpha=alpha, max_iter=1500, tol=1e-8)
        reg.fit(xs, y, sample_weight=weight)
        learned = reg.predict(xs)
        for strength in (0.5, 0.75, 1.0):
            report(evaluator, f"regime alpha={alpha} strength={strength}", candidate_from(learned, strength))

    raw = reg.coef_ / x_scale
    intercept = float(reg.intercept_ - np.dot(raw, x_mean))
    print(f"REGIME_INTERCEPT={intercept!r}", flush=True)
    print("REGIME_COEFFICIENTS=" + repr(tuple(float(v) for v in raw)), flush=True)
    print("REGIME_FEATURE_NAMES=" + repr(tuple(names)), flush=True)
    print("top regime coefficients", flush=True)
    for index in np.argsort(np.abs(reg.coef_))[::-1][:80]:
        print(f"{names[index]}\t{reg.coef_[index]:+.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
