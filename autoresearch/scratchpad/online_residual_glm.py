"""Fit the compact residual equation on actual causal monthly trajectories.

The earlier distillation averaged causal states into a 16-year monthly cycle
before fitting, then evaluated those coefficients at every online time step.
This diagnostic removes that semantic mismatch.  Every row is one actual
month at one independent land cell, folds hold out whole cells, and every
feature is constructed from current or prior local state only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)
from autoresearch.scratchpad.validate_20cr_vpd import target_grid  # noqa: E402
from autoresearch.scratchpad.validate_ncep_vpd_bridge import (  # noqa: E402
    bilinear_target,
    monthly_fields,
)


NAMES = (
    "dryness_departure_6m",
    "temperature_departure_6m_negative",
    "gpp_curing_6m",
    "gpp_curing_6m_positive",
    "lai_curing_6m",
    "lai_curing_6m_negative",
    "dryness_departure_12m_positive",
    "dryness_departure_24m",
    "dryness_departure_24m_positive",
    "gpp_curing_24m",
    "gpp_curing_24m_positive",
    "drying_x_fuel_6m",
    "drying_6m_x_humid_climate",
    "drying_x_fuel_12m",
    "drying_12m_x_temperate",
    "drying_12m_x_humid_climate",
    "drying_24m_x_rangeland",
    "opportunity_0.03",
    "opportunity_0.03_x_drying_12m",
    "opportunity_0.08",
)


def running(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[:, 0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[1]):
        state += alpha * (values[:, time] - state)
        output[:, time] = state
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-values, -40.0, 40.0)))


def report(evaluator: GFED5Evaluator, label: str, prediction: np.ndarray) -> None:
    score = evaluator.score(prediction)["global"]
    print(
        f"{label} overall={score['overall_score']:.4f} "
        f"bias={score['bias_score']:.4f} rmse={score['rmse_score']:.4f} "
        f"seasonal={score['seasonal_cycle_score']:.4f} "
        f"spatial={score['spatial_distribution_score']:.4f}",
        flush=True,
    )


def load_daily_vpd_duration() -> np.ndarray:
    """Load a continuous 2001-2016 count of days with VPD above 1 kPa."""
    fields: list[np.ndarray] = []
    for year in range(2001, 2016):
        _, duration, latitudes, longitudes = monthly_fields("20cr", year)
        fields.append(target_grid(duration, latitudes, longitudes))
    _, duration, latitudes, longitudes = monthly_fields("ncep", 2016)
    duration = bilinear_target(duration, latitudes, longitudes)
    # Fixed global bridge fitted over the independent 2001-2015 overlap.
    fields.append(np.clip(0.081331 + 0.968217 * duration, 0.0, 1.0))
    result = np.concatenate(fields).astype(np.float32)
    print(
        f"daily duration shape={result.shape} finite={np.isfinite(result).mean():.6f}",
        flush=True,
    )
    return result


def main() -> int:
    vpd_only = "--vpd-only" in sys.argv
    duration_only = "--duration-only" in sys.argv
    event_only = "--event-only" in sys.argv
    broad_tree = "--broad-tree" in sys.argv
    broad_gam = "--broad-gam" in sys.argv
    interaction_gam = "--interaction-gam" in sys.argv
    shallow_tree = "--shallow-tree" in sys.argv
    tensor_gam = "--tensor-gam" in sys.argv
    bin_top = "--bin-top" in sys.argv
    physical_tree = "--physical-tree" in sys.argv
    physical_ebm = "--physical-ebm" in sys.argv
    annual_target_tree = "--annual-target-tree" in sys.argv
    bin_physical = "--bin-physical" in sys.argv
    model = load_model()
    requested = list(model.INPUTS)
    if vpd_only:
        requested.append("vapor_pressure_deficit_mean")
    if event_only:
        requested.extend(("wet_day_fraction", "maximum_consecutive_dry_days"))
    data = load_inputs(tuple(dict.fromkeys(requested)))
    params = dict(model.PARAMS)
    if vpd_only:
        params["vpd_memory_w"] = 0.0
    incumbent = validate_prediction(model.predict(data, params, None))
    cells = np.flatnonzero(load_land_mask().ravel())
    rows, cols = cells // 360, cells % 360

    def extract(name: str) -> np.ndarray:
        return np.asarray(data[name][:, rows, cols].T, dtype=np.float64)

    rain = extract("monthly_precipitation")
    dryness = extract("dryness")
    temperature = extract("air_temperature")
    gpp = extract("gpp")
    lai = extract("leaf_area_index")
    annual_rain = extract("annual_precipitation")
    rangeland = extract("luh2_rangeland_fraction")
    cropland = extract("luh2_cropland_fraction")
    vpd = extract("vapor_pressure_deficit_mean") if vpd_only else None
    wet_days = extract("wet_day_fraction") if event_only else None
    dry_spell = extract("maximum_consecutive_dry_days") if event_only else None
    duration = (
        np.asarray(load_daily_vpd_duration()[:, rows, cols].T, dtype=np.float64)
        if duration_only
        else None
    )
    baseline = np.asarray(incumbent[:, rows, cols].T, dtype=np.float64)

    rain_memory = {months: running(rain, months) for months in (6.0, 12.0, 24.0)}
    dryness_memory = {
        months: running(dryness, months) for months in (6.0, 12.0, 24.0)
    }
    gpp_memory = {months: running(gpp, months) for months in (6.0, 12.0, 24.0)}
    lai_memory_6m = running(lai, 6.0)
    temperature_memory_6m = running(temperature, 6.0)

    rain_departure = {
        months: (state - rain) / (state + rain + 10.0)
        for months, state in rain_memory.items()
    }
    dryness_departure = {
        months: (dryness - state)
        / (np.abs(state) + np.abs(dryness) + 100.0)
        for months, state in dryness_memory.items()
    }
    gpp_curing = {
        months: (state - gpp) / (state + gpp + 0.2)
        for months, state in gpp_memory.items()
    }
    lai_curing_6m = (lai_memory_6m - lai) / (lai_memory_6m + lai + 0.5)
    temperature_departure_6m = np.clip(
        (temperature - temperature_memory_6m) / 10.0, -2.0, 2.0
    )
    fuel_bank = {
        months: state / (state + 0.5) for months, state in gpp_memory.items()
    }
    humid_climate = sigmoid((annual_rain - 1300.0) / 250.0)
    temperate = sigmoid((temperature - 2.0) / 3.0) * sigmoid(
        (20.0 - temperature) / 3.0
    )

    cumulative = np.cumsum(baseline, axis=1)
    trailing = cumulative.copy()
    trailing[:, 12:] -= cumulative[:, :-12]
    exposure = np.minimum(np.arange(1, baseline.shape[1] + 1), 12)[None, :]
    trailing *= 12.0 / exposure
    share = baseline / (trailing + 1e-12)
    opportunity_003 = sigmoid((share - 0.03) / 0.025)
    opportunity_008 = sigmoid((share - 0.08) / 0.025)

    names = NAMES
    feature_fields = (
        dryness_departure[6.0],
        np.minimum(temperature_departure_6m, 0.0),
        gpp_curing[6.0],
        np.maximum(gpp_curing[6.0], 0.0),
        lai_curing_6m,
        np.minimum(lai_curing_6m, 0.0),
        np.maximum(dryness_departure[12.0], 0.0),
        dryness_departure[24.0],
        np.maximum(dryness_departure[24.0], 0.0),
        gpp_curing[24.0],
        np.maximum(gpp_curing[24.0], 0.0),
        rain_departure[6.0] * fuel_bank[6.0],
        rain_departure[6.0] * humid_climate,
        rain_departure[12.0] * fuel_bank[12.0],
        rain_departure[12.0] * temperate,
        rain_departure[12.0] * humid_climate,
        rain_departure[24.0] * rangeland,
        opportunity_003,
        opportunity_003 * rain_departure[12.0],
        opportunity_008,
    )
    if vpd_only:
        assert vpd is not None
        vpd_3m = running(vpd, 3.0)
        vpd_24m = running(vpd, 24.0)
        departure_3m = (vpd - vpd_3m) / (vpd + vpd_3m + 0.2)
        departure_24m = (vpd - vpd_24m) / (vpd + vpd_24m + 0.2)
        pulse = np.maximum(departure_3m, 0.0)
        background = vpd_24m / (vpd_24m + 0.7)
        seasonal_climate = sigmoid((annual_rain - 400.0) / 150.0) * sigmoid(
            (1700.0 - annual_rain) / 250.0
        )
        names = (
            "vpd_departure_3m",
            "vpd_departure_3m_positive",
            "vpd_departure_24m_negative",
            "vpd_pulse_3m_x_fuel_bank",
            "vpd_pulse_3m_x_seasonal_climate",
            "vpd_pulse_3m_x_cropland",
            "vpd_pulse_3m_x_opportunity",
            "vpd_background_x_fuel_bank",
        )
        feature_fields = (
            departure_3m,
            pulse,
            np.minimum(departure_24m, 0.0),
            pulse * fuel_bank[12.0],
            pulse * seasonal_climate,
            pulse * cropland,
            pulse * sigmoid((share - 0.05) / 0.025),
            background * fuel_bank[12.0],
        )
    if duration_only:
        assert duration is not None
        duration_3m = running(duration, 3.0)
        duration_12m = running(duration, 12.0)
        departure_3m = duration - duration_3m
        departure_12m = duration - duration_12m
        persistence_3m = duration * duration_3m
        opportunity = sigmoid((share - 0.05) / 0.025)
        names = (
            "vpd_days_gt_1kpa",
            "vpd_days_gt_1kpa_3m",
            "vpd_days_gt_1kpa_departure_3m",
            "vpd_days_gt_1kpa_departure_3m_positive",
            "vpd_days_gt_1kpa_12m",
            "vpd_days_gt_1kpa_departure_12m",
            "vpd_days_gt_1kpa_persistence_3m",
            "vpd_days_gt_1kpa_x_fuel_bank",
            "vpd_days_gt_1kpa_x_humid_climate",
            "vpd_days_gt_1kpa_x_rangeland",
            "vpd_days_gt_1kpa_x_cropland",
            "vpd_days_gt_1kpa_x_opportunity",
        )
        feature_fields = (
            duration,
            duration_3m,
            departure_3m,
            np.maximum(departure_3m, 0.0),
            duration_12m,
            departure_12m,
            persistence_3m,
            duration * fuel_bank[12.0],
            duration * humid_climate,
            duration * rangeland,
            duration * cropland,
            duration * opportunity,
        )
    if event_only:
        assert wet_days is not None and dry_spell is not None
        wet_3m = running(wet_days, 3.0)
        wet_12m = running(wet_days, 12.0)
        dry_3m = running(dry_spell, 3.0)
        dry_12m = running(dry_spell, 12.0)
        wet_departure = wet_days - wet_3m
        dry_departure = dry_spell - dry_3m
        opportunity = sigmoid((share - 0.05) / 0.025)
        names = (
            "wet_day_fraction",
            "wet_day_fraction_3m",
            "wet_day_fraction_departure_3m",
            "wet_day_fraction_12m",
            "dry_spell",
            "dry_spell_3m",
            "dry_spell_departure_3m",
            "dry_spell_12m",
            "dry_spell_x_fuel_bank",
            "dry_spell_x_humid_climate",
            "dry_spell_x_rangeland",
            "dry_spell_x_opportunity",
        )
        normalized_dry_spell = dry_spell / 31.0
        feature_fields = (
            wet_days,
            wet_3m,
            wet_departure,
            wet_12m,
            normalized_dry_spell,
            dry_3m / 31.0,
            dry_departure / 31.0,
            dry_12m / 31.0,
            normalized_dry_spell * fuel_bank[12.0],
            normalized_dry_spell * humid_climate,
            normalized_dry_spell * rangeland,
            normalized_dry_spell * opportunity,
        )
    if (
        broad_tree
        or broad_gam
        or interaction_gam
        or shallow_tree
        or tensor_gam
        or bin_top
        or physical_tree
        or physical_ebm
        or annual_target_tree
        or bin_physical
    ):
        names_list: list[str] = ["incumbent", "trailing_annual", "fire_share"]
        fields_list: list[np.ndarray] = [baseline, trailing, share]
        raw_fields = {name: extract(name) for name in model.INPUTS}
        for name in model.INPUTS:
            names_list.append(f"{name}:current")
            fields_list.append(raw_fields[name])
        for name in (
            "monthly_precipitation",
            "dryness",
            "air_temperature",
            "gpp",
            "leaf_area_index",
            "lightning_flash_rate",
        ):
            raw = raw_fields[name]
            for months in (3.0, 12.0, 24.0):
                memory = running(raw, months)
                names_list.extend(
                    (f"{name}:memory_{months:g}m", f"{name}:departure_{months:g}m")
                )
                fields_list.extend(
                    (memory, (raw - memory) / (np.abs(raw) + np.abs(memory) + 1e-3))
                )
        names = tuple(names_list)
        feature_fields = tuple(fields_list)
    x = np.column_stack([field.reshape(-1) for field in feature_fields]).astype(
        np.float32
    )
    if physical_tree or physical_ebm:
        x = x[:, 3:]
        names = names[3:]
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observed = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    if annual_target_tree:
        observed_annual = observed.reshape(16, 12, 180, 360).mean(axis=0).sum(axis=0)
        target = np.repeat(observed_annual[rows, cols], baseline.shape[1])
        offset = trailing.reshape(-1) + 1e-4
    else:
        target = np.asarray(observed[:, rows, cols].T, dtype=np.float64).reshape(-1)
        offset = baseline.reshape(-1) + 1e-4
    y = target / offset
    weights = offset + float(offset.mean()) * 0.02
    x_mean = np.average(x, axis=0, weights=weights)
    x_scale = np.sqrt(
        np.average(np.square(x - x_mean), axis=0, weights=weights)
    ) + 1e-8
    standardized = (x - x_mean) / x_scale
    print(f"rows={x.shape[0]} features={x.shape[1]}", flush=True)

    rng = np.random.default_rng(487)
    cell_folds = rng.integers(0, 3, size=cells.size)
    folds = np.repeat(cell_folds, baseline.shape[1])
    if bin_top or bin_physical:
        name_to_index = {name: index for index, name in enumerate(names)}
        sample = rng.choice(x.shape[0], size=min(600_000, x.shape[0]), replace=False)

        def ratio_matrix(left_name: str, right_name: str) -> None:
            left = x[sample, name_to_index[left_name]]
            right = x[sample, name_to_index[right_name]]
            quantiles = np.linspace(0.0, 1.0, 7)
            left_edges = np.unique(np.quantile(left, quantiles))
            right_edges = np.unique(np.quantile(right, quantiles))
            left_bins = np.clip(
                np.searchsorted(left_edges, left, side="right") - 1,
                0,
                left_edges.size - 2,
            )
            right_bins = np.clip(
                np.searchsorted(right_edges, right, side="right") - 1,
                0,
                right_edges.size - 2,
            )
            matrix = np.full(
                (left_edges.size - 1, right_edges.size - 1), np.nan
            )
            for left_bin in range(matrix.shape[0]):
                for right_bin in range(matrix.shape[1]):
                    selected_rows = (
                        (left_bins == left_bin) & (right_bins == right_bin)
                    )
                    if selected_rows.sum() < 100:
                        continue
                    selected_sample = sample[selected_rows]
                    matrix[left_bin, right_bin] = np.average(
                        y[selected_sample], weights=weights[selected_sample]
                    )
            print(f"ratio matrix {left_name} x {right_name}", flush=True)
            print("left_edges=" + np.array2string(left_edges, precision=6), flush=True)
            print("right_edges=" + np.array2string(right_edges, precision=6), flush=True)
            print(np.array2string(matrix, precision=3), flush=True)

        pairs = (
            (
                ("gpp:memory_24m", "dryness:memory_12m"),
                ("gpp:memory_24m", "monthly_precipitation:memory_12m"),
                ("gpp:memory_24m", "aboveground_biomass:current"),
                (
                    "luh2_cropland_fraction:current",
                    "natural_vegetation_fraction:current",
                ),
                ("annual_precipitation:current", "aboveground_biomass:current"),
                (
                    "aboveground_biomass:current",
                    "lightning_flash_rate:memory_12m",
                ),
            )
            if bin_physical
            else (
                ("trailing_annual", "dryness:current"),
                ("trailing_annual", "lightning_flash_rate:memory_24m"),
                ("incumbent", "luh2_cropland_fraction:current"),
                ("incumbent", "annual_precipitation:current"),
                ("trailing_annual", "natural_vegetation_fraction:current"),
            )
        )
        for left_name, right_name in pairs:
            ratio_matrix(left_name, right_name)
        return 0
    if tensor_gam:
        selected_names = (
            "incumbent",
            "trailing_annual",
            "fire_share",
            "annual_precipitation:current",
            "luh2_cropland_fraction:current",
            "natural_vegetation_fraction:current",
            "dryness:current",
            "dryness:departure_3m",
            "gpp:departure_3m",
            "air_temperature:departure_3m",
            "aboveground_biomass:current",
            "lightning_flash_rate:memory_24m",
            "lightning_flash_rate:memory_12m",
            "monthly_precipitation:departure_24m",
            "monthly_precipitation:memory_12m",
            "monthly_precipitation:current",
            "luh2_rangeland_fraction:current",
        )
        tensor_names = (
            ("trailing_annual", "dryness:current"),
            ("trailing_annual", "lightning_flash_rate:memory_24m"),
            ("trailing_annual", "lightning_flash_rate:memory_12m"),
            (
                "monthly_precipitation:departure_24m",
                "lightning_flash_rate:memory_12m",
            ),
            ("trailing_annual", "luh2_rangeland_fraction:current"),
            ("trailing_annual", "monthly_precipitation:departure_24m"),
            ("incumbent", "monthly_precipitation:current"),
            ("incumbent", "luh2_cropland_fraction:current"),
            ("incumbent", "dryness:departure_3m"),
            ("incumbent", "annual_precipitation:current"),
            (
                "luh2_cropland_fraction:current",
                "air_temperature:departure_3m",
            ),
            ("trailing_annual", "natural_vegetation_fraction:current"),
        )
        name_to_index = {name: index for index, name in enumerate(names)}
        selected = np.asarray([name_to_index[name] for name in selected_names])
        selected_index = {name: index for index, name in enumerate(selected_names)}
        tensor_indices = tuple(
            (selected_index[left], selected_index[right])
            for left, right in tensor_names
        )

        def tensor_design(basis: np.ndarray, width: int) -> np.ndarray:
            columns = [basis]
            for left, right in tensor_indices:
                left_basis = basis[:, left * width : (left + 1) * width]
                right_basis = basis[:, right * width : (right + 1) * width]
                columns.append(
                    (left_basis[:, :, None] * right_basis[:, None, :]).reshape(
                        basis.shape[0], -1
                    )
                )
            return np.column_stack(columns)

        out_of_fold = np.zeros_like(y)
        fold_coefficients: list[np.ndarray] = []
        for fold in range(3):
            train = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            if train.size > 200_000:
                train = rng.choice(train, size=200_000, replace=False)
            spline = SplineTransformer(
                n_knots=4,
                degree=2,
                knots="quantile",
                include_bias=False,
                extrapolation="linear",
            )
            train_basis = spline.fit_transform(x[train][:, selected])
            width = spline.n_features_out_ // len(selected_names)
            train_design = tensor_design(train_basis, width)
            scaler = StandardScaler()
            train_design = scaler.fit_transform(train_design)
            regressor = PoissonRegressor(alpha=0.01, max_iter=800, tol=1e-8)
            regressor.fit(train_design, y[train], sample_weight=weights[train])
            del train_basis, train_design
            for start in range(0, held.size, 100_000):
                batch = held[start : start + 100_000]
                basis = spline.transform(x[batch][:, selected])
                design = scaler.transform(tensor_design(basis, width))
                out_of_fold[batch] = regressor.predict(design)
            fold_coefficients.append(regressor.coef_)
            print(
                f"completed tensor GAM fold={fold} train={train.size} "
                f"held={held.size} features={regressor.coef_.size}",
                flush=True,
            )
        correlations = np.corrcoef(np.asarray(fold_coefficients))
        print(
            "tensor GAM coefficient correlation min="
            f"{correlations[np.triu_indices(3, 1)].min():.4f}",
            flush=True,
        )
        evaluator = GFED5Evaluator(GFED5_PATH)
        report(evaluator, "incumbent", incumbent)
        for strength in (0.10, 0.25, 0.50, 0.75, 1.0):
            corrected = baseline * np.power(
                np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6),
                strength,
            )
            candidate = incumbent.copy()
            candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
            report(evaluator, f"tensor GAM OOF strength={strength}", candidate)
        return 0
    if physical_ebm:
        from interpret.glassbox import ExplainableBoostingRegressor

        out_of_fold = np.zeros_like(y)
        trained = None
        log_ratio = np.log(np.clip(y, 1e-4, 1e4))
        for fold in range(3):
            train = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            if train.size > 200_000:
                train = rng.choice(train, size=200_000, replace=False)
            learner = ExplainableBoostingRegressor(
                feature_names=list(names),
                max_bins=64,
                max_interaction_bins=24,
                interactions=24,
                validation_size=0.15,
                outer_bags=2,
                inner_bags=0,
                learning_rate=0.04,
                smoothing_rounds=100,
                interaction_smoothing_rounds=50,
                max_rounds=1200,
                early_stopping_rounds=60,
                min_samples_leaf=100,
                max_leaves=3,
                objective="rmse",
                n_jobs=-2,
                random_state=811 + fold,
            )
            learner.fit(x[train], log_ratio[train], sample_weight=weights[train])
            for start in range(0, held.size, 200_000):
                batch = held[start : start + 200_000]
                out_of_fold[batch] = np.exp(
                    np.clip(learner.predict(x[batch]), -8.0, 8.0)
                )
            trained = learner
            print(
                f"completed physical EBM fold={fold} train={train.size} "
                f"held={held.size} terms={len(learner.term_names_)}",
                flush=True,
            )
        evaluator = GFED5Evaluator(GFED5_PATH)
        report(evaluator, "incumbent", incumbent)
        for strength in (0.10, 0.25, 0.50, 0.75, 1.0):
            corrected = baseline * np.power(
                np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6),
                strength,
            )
            candidate = incumbent.copy()
            candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
            report(evaluator, f"physical EBM OOF strength={strength}", candidate)
        assert trained is not None
        importances = trained.term_importances()
        print("physical EBM ranked terms", flush=True)
        for index in np.argsort(importances)[::-1][:40]:
            print(
                f"{trained.term_names_[index]}\t{importances[index]:.10f}",
                flush=True,
            )
        return 0
    if broad_gam or interaction_gam:
        selected_names = (
            "incumbent",
            "trailing_annual",
            "fire_share",
            "annual_precipitation:current",
            "luh2_cropland_fraction:current",
            "air_temperature:memory_24m",
            "natural_vegetation_fraction:current",
            "dryness:current",
            "gpp:departure_3m",
            "air_temperature:departure_3m",
            "aboveground_biomass:current",
            "lightning_flash_rate:memory_24m",
            "monthly_precipitation:departure_24m",
            "monthly_precipitation:memory_3m",
            "luh2_rangeland_fraction:current",
        )
        name_to_index = {name: index for index, name in enumerate(names)}
        selected = np.asarray([name_to_index[name] for name in selected_names])
        out_of_fold = np.zeros_like(y)
        fold_coefficients: list[np.ndarray] = []
        for fold in range(3):
            train = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            if train.size > 500_000:
                train = rng.choice(train, size=500_000, replace=False)
            if interaction_gam:
                regressor = make_pipeline(
                    RobustScaler(quantile_range=(25.0, 75.0)),
                    FeatureUnion(
                        (
                            (
                                "splines",
                                SplineTransformer(
                                    n_knots=5,
                                    degree=3,
                                    knots="quantile",
                                    include_bias=False,
                                    extrapolation="linear",
                                ),
                            ),
                            (
                                "pairs",
                                PolynomialFeatures(
                                    degree=2,
                                    interaction_only=True,
                                    include_bias=False,
                                ),
                            ),
                        )
                    ),
                    StandardScaler(),
                    PoissonRegressor(alpha=0.01, max_iter=800, tol=1e-8),
                )
            else:
                regressor = make_pipeline(
                    SplineTransformer(
                        n_knots=5,
                        degree=3,
                        knots="quantile",
                        include_bias=False,
                        extrapolation="linear",
                    ),
                    StandardScaler(),
                    PoissonRegressor(alpha=0.003, max_iter=800, tol=1e-8),
                )
            regressor.fit(
                x[train][:, selected],
                y[train],
                poissonregressor__sample_weight=weights[train],
            )
            for start in range(0, held.size, 200_000):
                batch = held[start : start + 200_000]
                out_of_fold[batch] = regressor.predict(x[batch][:, selected])
            fold_coefficients.append(regressor.named_steps["poissonregressor"].coef_)
            print(
                f"completed {'interaction' if interaction_gam else 'broad'} GAM "
                f"fold={fold} train={train.size} held={held.size}",
                flush=True,
            )
        correlations = np.corrcoef(np.asarray(fold_coefficients))
        print(
            f"{'interaction' if interaction_gam else 'broad'} GAM "
            "coefficient correlation min="
            f"{correlations[np.triu_indices(3, 1)].min():.4f}",
            flush=True,
        )
        evaluator = GFED5Evaluator(GFED5_PATH)
        report(evaluator, "incumbent", incumbent)
        for strength in (0.10, 0.25, 0.50, 0.75, 1.0):
            corrected = baseline * np.power(
                np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6),
                strength,
            )
            candidate = incumbent.copy()
            candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
            label = "interaction GAM" if interaction_gam else "broad GAM"
            report(evaluator, f"{label} OOF strength={strength}", candidate)
        return 0
    if broad_tree or shallow_tree or physical_tree or annual_target_tree:
        out_of_fold = np.zeros_like(y)
        trained: HistGradientBoostingRegressor | None = None
        held_for_importance: np.ndarray | None = None
        for fold in range(3):
            train = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            if train.size > 500_000:
                train = rng.choice(train, size=500_000, replace=False)
            regressor = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.08,
                max_iter=250 if shallow_tree else 100,
                max_leaf_nodes=4 if shallow_tree else 31,
                min_samples_leaf=500,
                l2_regularization=2.5,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=701 + fold,
            )
            regressor.fit(x[train], y[train], sample_weight=weights[train])
            for start in range(0, held.size, 250_000):
                batch = held[start : start + 250_000]
                out_of_fold[batch] = regressor.predict(x[batch])
            trained = regressor
            held_for_importance = held
            print(
                f"completed {'shallow' if shallow_tree else 'broad'} HGB "
                f"fold={fold} train={train.size} held={held.size} "
                f"iterations={regressor.n_iter_}",
                flush=True,
            )
        evaluator = GFED5Evaluator(GFED5_PATH)
        report(evaluator, "incumbent", incumbent)
        for strength in (0.10, 0.25, 0.50, 0.75, 1.0):
            corrected = baseline * np.power(
                np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6),
                strength,
            )
            candidate = incumbent.copy()
            candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
            label = (
                "annual-target HGB"
                if annual_target_tree
                else ("shallow HGB" if shallow_tree else "broad HGB")
            )
            report(evaluator, f"{label} OOF strength={strength}", candidate)
        assert trained is not None and held_for_importance is not None
        importance_rows = rng.choice(
            held_for_importance,
            size=min(100_000, held_for_importance.size),
            replace=False,
        )
        importance = permutation_importance(
            trained,
            x[importance_rows],
            y[importance_rows],
            scoring="neg_mean_poisson_deviance",
            n_repeats=2,
            random_state=719,
            sample_weight=weights[importance_rows],
        )
        order = np.argsort(importance.importances_mean)[::-1]
        print("broad HGB permutation importance", flush=True)
        for index in order[:30]:
            print(
                f"{names[index]}\t{importance.importances_mean[index]:+.9f}",
                flush=True,
            )
        pair_gain: dict[tuple[int, int], float] = defaultdict(float)
        for stage in trained._predictors:
            nodes = stage[0].nodes
            splits = nodes[nodes["is_leaf"] == 0]
            features = sorted(set(int(index) for index in splits["feature_idx"]))
            gain = float(np.maximum(splits["gain"], 0.0).sum())
            for left, right in combinations(features, 2):
                pair_gain[(left, right)] += gain
        print("HGB recurring feature pairs", flush=True)
        for (left, right), gain in sorted(
            pair_gain.items(), key=lambda item: item[1], reverse=True
        )[:30]:
            print(f"{names[left]} x {names[right]}\t{gain:.6f}", flush=True)
        return 0
    if "--tree" in sys.argv:
        out_of_fold = np.zeros_like(y)
        for fold in range(3):
            train = folds != fold
            held = ~train
            tree = DecisionTreeRegressor(
                criterion="poisson",
                max_leaf_nodes=64,
                min_samples_leaf=500,
                random_state=503 + fold,
            )
            tree.fit(x[train], y[train], sample_weight=weights[train])
            out_of_fold[held] = tree.predict(x[held])
            print(f"completed tree fold={fold}", flush=True)
        evaluator = GFED5Evaluator(GFED5_PATH)
        report(evaluator, "incumbent", incumbent)
        for strength in (0.25, 0.50, 0.75, 1.0):
            corrected = baseline * np.power(
                np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6),
                strength,
            )
            candidate = incumbent.copy()
            candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
            report(evaluator, f"online tree OOF strength={strength}", candidate)
        tree = DecisionTreeRegressor(
            criterion="poisson",
            max_leaf_nodes=64,
            min_samples_leaf=500,
            random_state=509,
        )
        tree.fit(x, y, sample_weight=weights)
        print(export_text(tree, feature_names=list(names), decimals=5), flush=True)
        return 0

    out_of_fold = np.zeros_like(y)
    coefficients: list[np.ndarray] = []
    for fold in range(3):
        train = folds != fold
        held = ~train
        regressor = PoissonRegressor(alpha=0.001, max_iter=1000, tol=1e-8)
        regressor.fit(
            standardized[train], y[train], sample_weight=weights[train]
        )
        out_of_fold[held] = regressor.predict(standardized[held])
        coefficients.append(regressor.coef_)
        print(f"completed fold={fold}", flush=True)
    correlations = np.corrcoef(np.asarray(coefficients))
    print(
        "fold coefficient correlation min="
        f"{correlations[np.triu_indices(3, 1)].min():.4f}",
        flush=True,
    )

    evaluator = GFED5Evaluator(GFED5_PATH)
    report(evaluator, "incumbent", incumbent)
    for strength in (0.25, 0.50, 0.75, 1.0):
        corrected = baseline * np.power(np.clip(out_of_fold.reshape(baseline.shape), 1e-6, 1e6), strength)
        candidate = incumbent.copy()
        candidate[:, rows, cols] = np.clip(corrected.T, 0.0, 1.0)
        report(evaluator, f"online OOF strength={strength}", candidate)

    regressor = PoissonRegressor(alpha=0.001, max_iter=1500, tol=1e-8)
    regressor.fit(standardized, y, sample_weight=weights)
    print(f"ONLINE_GLM_NAMES={names!r}", flush=True)
    print(f"ONLINE_GLM_INTERCEPT={regressor.intercept_!r}", flush=True)
    print(f"ONLINE_GLM_COEFFICIENTS={tuple(regressor.coef_)!r}", flush=True)
    print(f"ONLINE_GLM_CENTER={tuple(x_mean)!r}", flush=True)
    print(f"ONLINE_GLM_SCALE={tuple(x_scale)!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
