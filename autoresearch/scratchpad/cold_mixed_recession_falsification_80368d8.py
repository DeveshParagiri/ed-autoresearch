"""Held-block and guardrail falsification for the cold-mixed complement.

This scratch-only diagnostic follows Entry 142.  It first repeats the annual
residual screen after excluding every feature derived from the incumbent fire
prediction.  It then tests a globally shared, smooth, causal timing mechanism:
a multiplicative Poisson-hazard reallocation toward a brief partial-snow-
recession window.  The mechanism creates no fuel and uses no region, country,
coordinate, calendar, target, future value, or completed-year statistic.

Ecological and country masks are post-prediction audits only.  The script never
edits ``model.py`` or the official experiment ledger.
"""

from __future__ import annotations

import gc
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.cold_mixed_heldblock_screen_80368d8 import (  # noqa: E402
    ema,
    fit_held_blocks,
    monthly_features,
    sigmoid,
    weighted_corr,
    weighted_mean,
)
from autoresearch.scratchpad.ecological_geography_audit import (  # noqa: E402
    DEFAULT_SHP,
    country_masks,
)
from autoresearch.scratchpad.phenology_half_greenup_exact_80368d8 import (  # noqa: E402
    ecology_masks,
)
from autoresearch.scratchpad.phenology_stage_split_80368d8 import (  # noqa: E402
    EXPECTED_MODEL_BLOB as PRE_PRUNING_MODEL_BLOB,
    causal_mean_states,
)
from autoresearch.scratchpad.phenology_stage_split_sampled_80368d8 import (  # noqa: E402
    MASK_INPUTS,
    one_degree_area,
    selected_input,
)
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
)


MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
EXPECTED_MODEL_BLOB = "39ee93ebf1155af9ae9d70e05847b9c3f086887d"
COUNTRIES = ("Canada", "Mongolia", "Ukraine")
CURRENT_DERIVED = {
    "incumbent",
    "incumbent12",
    "opportunity_gap",
    "thaw_opportunity",
    "post_melt_exposure",
}
DYNAMIC_STD = {
    "temperature",
    "warming3",
    "rain",
    "rain_deficit",
    "dryness",
    "drying3",
    "gpp",
    "gpp_decline",
    "lightning",
    "lightning_arrival",
    "combustion",
    "snow_cover",
    "melt_release",
    "warm_dry_open_fuel",
}


def selected_observation(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Read only selected one-degree GFED cells without loading the full cube."""
    output = np.empty((192, rows.size), dtype=np.float32)
    with Dataset(GFED5_PATH) as dataset:
        variable = dataset.variables["burntArea"]
        for row in np.unique(rows):
            positions = np.flatnonzero(rows == row)
            slab = np.ma.asarray(variable[:192, 2 * int(row): 2 * int(row) + 2, :])
            if np.ma.getmaskarray(slab).any():
                raise ValueError("masked GFED5 selected observation")
            one_degree = np.asarray(slab, dtype=np.float32).reshape(
                192, 2, 360, 2
            ).mean(axis=(1, 3))
            output[:, positions] = one_degree[:, columns[positions]] / 100.0
    return output


def selected_model_inputs(
    model,
    mask_data: Mapping[str, np.ndarray],
    rows: np.ndarray,
    columns: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in model.INPUTS:
        if name in mask_data:
            output[name] = np.asarray(
                mask_data[name][:, rows, columns][:, None, :], dtype=np.float32
            )
        else:
            output[name] = selected_input(name, rows, columns)
    return output


def annual_input_only_screen(
    features: Mapping[str, np.ndarray],
    current: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    cold: np.ndarray,
) -> None:
    chosen = np.flatnonzero(cold)
    pred_cycle = current[:, 0, chosen].reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation[:, chosen].reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    positive = obs_annual[obs_annual > 1e-8]
    floor = 0.02 * float(np.median(positive)) if positive.size else 1e-6
    weights = area[chosen] * (obs_annual + floor)
    target = np.clip(
        np.log((obs_annual + 1e-6) / (pred_annual + 1e-6)), -4.0, 4.0
    )

    columns_out: list[np.ndarray] = []
    names: list[str] = []
    for name, values in features.items():
        if name in CURRENT_DERIVED:
            continue
        values = values[:, chosen]
        columns_out.append(values.mean(axis=0))
        names.append(f"mean:{name}")
        if name in DYNAMIC_STD:
            columns_out.append(values.std(axis=0))
            names.append(f"std:{name}")
    x = np.column_stack(columns_out).astype(np.float32)
    folds = (rows[chosen] // 15 + 2 * (columns[chosen] // 15)) % 4
    oof, importance = fit_held_blocks(
        x, target, weights, folds, tuple(names), "annual_input_only"
    )
    baseline_loss = weighted_mean(np.abs(target), weights)
    print(
        f"ANNUAL_INPUT_ONLY corr={weighted_corr(oof, target, weights):+.9f} "
        f"baseline_mae={baseline_loss:.9f}",
        flush=True,
    )
    for blend in (0.25, 0.5, 1.0):
        adjustment = blend * np.clip(oof, -3.0, 3.0)
        loss = weighted_mean(np.abs(target - adjustment), weights)
        corrected = pred_annual * np.exp(adjustment)
        ratio = float(np.sum(corrected * area[chosen])) / max(
            float(np.sum(obs_annual * area[chosen])), 1e-12
        )
        print(
            f"ANNUAL_INPUT_ONLY_BLEND blend={blend:g} loss={loss:.9f} "
            f"delta={loss - baseline_loss:+.9f} ratio={ratio:.9f}",
            flush=True,
        )
    rank = np.argsort(importance.mean(axis=0))[::-1]
    for index in rank[:12]:
        top_count = int(sum(index in np.argsort(row)[-8:] for row in importance))
        print(
            f"ANNUAL_INPUT_ONLY_RANK {names[index]} "
            f"mean={importance[:, index].mean():.9f} "
            f"std={importance[:, index].std():.9f} top8_folds={top_count}",
            flush=True,
        )


def recession_windows(
    features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return smooth, bounded candidate windows; none uses audit masks."""
    temperature = features["temperature"]
    rain = features["rain"]
    snow = features["snow_cover"]
    melt = features["melt_release"]
    natural = np.clip(features["natural"], 0.0, 1.0)
    canopy = np.clip(features["canopy"], 0.0, None)

    # Held-block ML identifies a hump: no correction in fully snow-free or
    # deeply snowbound months, with the positive residual concentrated after
    # appreciable melt at 0--9 C and outside heavy rainfall.
    thermal = sigmoid((temperature + 1.0) / 1.5) * sigmoid(
        (10.0 - temperature) / 2.5
    )
    partial_snow = sigmoid((snow - 0.02) / 0.04) * sigmoid(
        (0.65 - snow) / 0.10
    )
    melt_gate = melt / (melt + 0.05)
    rain_gate = 1.0 / (1.0 + np.square(rain / 65.0))
    cold_background = sigmoid((5.0 - features["temperature24"]) / 3.0)
    climate = np.clip(
        cold_background * thermal * partial_snow * melt_gate * rain_gate,
        0.0,
        1.0,
    )
    # This optional smooth shield is the smallest local distinction between
    # the target mixed/open surface and established closed-canopy boreal fire.
    surface_access = np.clip(1.0 - natural * canopy / (canopy + 10.0), 0.0, 1.0)
    return {
        "climate": np.asarray(climate, dtype=np.float32),
        "surface": np.asarray(climate * surface_access, dtype=np.float32),
    }


def apply_recession_allocator(
    current: np.ndarray,
    window: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Causally reallocate finite Poisson hazard toward recession exposure."""
    background = ema(window, 12.0)
    anomaly = (window - background) / (window + background + 0.05)
    log_factor = strength * np.tanh(anomaly)
    hazard = -np.log1p(-np.clip(current[:, 0, :], 0.0, 1.0 - 1e-7))
    adjusted = hazard * np.exp(log_factor)
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def apply_recession_bank(
    current: np.ndarray,
    features: Mapping[str, np.ndarray],
    strength: float,
) -> np.ndarray:
    """Store finite hazard outside recession and release it during exposure.

    This form preserves the missing annual propensity as far as possible.  A
    smooth cold, non-closed surface eligibility controls capture; a separate
    partial-recession event controls release.  A slow background return keeps
    the store finite at cold sites without a pronounced snow season.  The t0
    store is initialized at the equilibrium implied by the first local state.
    """
    temperature = features["temperature"]
    rain = features["rain"]
    snow = features["snow_cover"]
    melt = features["melt_release"]
    natural = np.clip(features["natural"], 0.0, 1.0)
    canopy = np.clip(features["canopy"], 0.0, None)
    cold = sigmoid((5.0 - features["temperature24"]) / 3.0)
    surface = np.clip(1.0 - natural * canopy / (canopy + 10.0), 0.0, 1.0)
    eligibility = cold * surface
    thermal = sigmoid((temperature + 1.0) / 1.5) * sigmoid(
        (10.0 - temperature) / 2.5
    )
    partial_snow = sigmoid((snow - 0.02) / 0.04) * sigmoid(
        (0.65 - snow) / 0.10
    )
    melt_gate = melt / (melt + 0.05)
    rain_gate = 1.0 / (1.0 + np.square(rain / 65.0))
    event = np.clip(thermal * partial_snow * melt_gate * rain_gate, 0.0, 1.0)

    hazard = -np.log1p(-np.clip(current[:, 0, :], 0.0, 1.0 - 1e-7))
    adjusted = np.empty_like(hazard, dtype=np.float32)
    capture_fraction = np.clip(strength * eligibility * (1.0 - event), 0.0, 0.8)
    return_fraction = 1.0 - np.exp(-(1.0 / 12.0 + 2.0 * event))
    first_capture = hazard[0] * capture_fraction[0]
    bank = first_capture * (1.0 - return_fraction[0]) / np.maximum(
        return_fraction[0], 1e-4
    )
    for time in range(hazard.shape[0]):
        captured = hazard[time] * capture_fraction[time]
        available = bank + captured
        released = available * return_fraction[time]
        adjusted[time] = hazard[time] - captured + released
        bank = available - released
    return np.asarray(1.0 - np.exp(-np.clip(adjusted, 0.0, 50.0)), dtype=np.float32)


def apply_cold_mosaic_onset(
    current: np.ndarray,
    data: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    scale: float,
    carrier_name: str,
) -> np.ndarray:
    """Add a bounded rare onset source for cold fuel-bearing mosaics.

    The timing deliberately reuses the already-supported warming, drying and
    ignition-arrival logic of the canonical rare-natural onset.  Only the
    carrier is changed, so this tests the input-only annual screen's central
    claim: the unresolved complement is low-natural or managed-open land, not
    another cold-forest, soil-carbon, or generic snow-fuel regime.
    """
    def field(name: str) -> np.ndarray:
        return np.asarray(data[name][:, 0, :], dtype=np.float32)

    natural = np.clip(features["natural"], 0.0, 1.0)
    biomass = np.clip(features["biomass"], 0.0, None)
    canopy = np.clip(features["canopy"], 0.0, None)
    surface = np.clip(1.0 - natural * canopy / (canopy + 10.0), 0.0, 1.0)
    if carrier_name == "low_natural":
        carrier = (1.0 - natural) * biomass / (biomass + 0.3) * surface
    elif carrier_name == "managed_open":
        managed_open = np.clip(
            field("luh2_rangeland_fraction") + field("luh2_pasture_fraction"),
            0.0,
            1.0,
        )
        carrier = managed_open / (managed_open + 0.1) * biomass / (biomass + 0.3)
    else:
        raise ValueError(carrier_name)

    annual_rain = np.clip(features["annual_rain"], 0.0, None)
    fuel = np.square(annual_rain / (annual_rain + 250.0)) * np.exp(
        -annual_rain / 3000.0
    )
    cold = sigmoid((5.0 - features["temperature24"]) / 3.0)
    warming = sigmoid((features["warming3"] - 0.5) / 1.5)
    drying = sigmoid((features["drying3"] - 0.01) / 0.04)
    ignition = features["lightning12"] / (features["lightning12"] + 0.02)
    ignition *= 0.35 + 0.65 * sigmoid(
        (features["lightning_arrival"] - 0.05) / 0.10
    )
    source = (
        scale
        * cold
        * carrier
        * fuel
        * warming
        * drying
        * features["combustion"]
        * ignition
        * features["opportunity_gap"]
    )
    return np.asarray(
        1.0 - (1.0 - current[:, 0, :]) * np.exp(-np.clip(source, 0.0, 50.0)),
        dtype=np.float32,
    )


def diagnostics(
    prediction: np.ndarray,
    observation: np.ndarray,
    cell_area: np.ndarray,
    group_weight: np.ndarray,
) -> dict[str, float | str]:
    chosen = group_weight > 0.0
    pred_cycle = prediction[:, chosen].reshape(16, 12, -1).mean(axis=0)
    obs_cycle = observation[:, chosen].reshape(16, 12, -1).mean(axis=0)
    pred_annual = pred_cycle.sum(axis=0)
    obs_annual = obs_cycle.sum(axis=0)
    weights = cell_area[chosen] * group_weight[chosen]
    obs_weight = weights * (obs_annual + 1e-8)
    pred_monthly = np.sum(pred_cycle * weights[None, :], axis=1)
    obs_monthly = np.sum(obs_cycle * weights[None, :], axis=1)
    pred_norm = pred_monthly / max(float(pred_monthly.sum()), 1e-12)
    obs_norm = obs_monthly / max(float(obs_monthly.sum()), 1e-12)
    cell_pred_norm = pred_cycle / np.maximum(pred_annual[None, :], 1e-8)
    cell_obs_norm = obs_cycle / np.maximum(obs_annual[None, :], 1e-8)
    cell_cycle_mae = float(
        np.sum(np.abs(cell_pred_norm - cell_obs_norm) * obs_weight[None, :])
        / max(float(12.0 * obs_weight.sum()), 1e-12)
    )
    return {
        "ratio": float(np.sum(pred_annual * weights))
        / max(float(np.sum(obs_annual * weights)), 1e-12),
        "seasonal_l1": 0.5 * float(np.sum(np.abs(pred_norm - obs_norm))),
        "cell_cycle_mae": cell_cycle_mae,
        "annual_log": float(
            np.sum(
                obs_weight
                * np.abs(np.log((pred_annual + 1e-6) / (obs_annual + 1e-6)))
            )
            / max(float(obs_weight.sum()), 1e-12)
        ),
        "peak": MONTHS[int(np.argmax(pred_monthly))],
        "obs_peak": MONTHS[int(np.argmax(obs_monthly))],
        "apr_jul": float(pred_monthly[3] / max(float(pred_monthly[6]), 1e-12)),
        "obs_apr_jul": float(obs_monthly[3] / max(float(obs_monthly[6]), 1e-12)),
    }


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

    mask_data = load_inputs(MASK_INPUTS)
    land = load_land_mask()
    states = causal_mean_states(mask_data, model)
    ecological = ecology_masks(mask_data, model, land)
    established_union = np.any(np.stack(tuple(ecological.values()), axis=0), axis=0)
    cold_mixed = land & ~established_union & (states["temperature"] < 5.0)

    audits: dict[str, np.ndarray] = {
        "cold_mixed": cold_mixed.astype(np.float32),
        "boreal": ecological["boreal"].astype(np.float32),
        "temperate_closed": ecological["temperate_closed"].astype(np.float32),
        "cropland": ecological["cropland"].astype(np.float32),
        "arid_low_fuel": ecological["arid_low_fuel"].astype(np.float32),
    }
    if DEFAULT_SHP.exists():
        countries, _ = country_masks(DEFAULT_SHP)
        for country in COUNTRIES:
            matches = [mask for (name, _), mask in countries.items() if name == country]
            if matches:
                audits[country.lower()] = np.asarray(matches[0] * land, dtype=np.float32)
        del countries

    selected_mask = np.any(
        np.stack(tuple(values > 0.0 for values in audits.values()), axis=0), axis=0
    )
    rows, columns = np.nonzero(selected_mask)
    print(
        f"SELECTION model_blob={blob} cells={rows.size} "
        f"pre_pruning_model_blob={PRE_PRUNING_MODEL_BLOB} "
        f"cold_mixed_cells={int(cold_mixed.sum())} audits={','.join(audits)}",
        flush=True,
    )
    sampled = selected_model_inputs(model, mask_data, rows, columns)
    audit_weights = {name: values[rows, columns] for name, values in audits.items()}
    del audits, selected_mask, ecological, established_union, land, states, mask_data
    gc.collect()

    current = np.asarray(model.predict(sampled, dict(model.PARAMS), None), dtype=np.float32)
    features = monthly_features(sampled, current)
    observation = selected_observation(rows, columns)
    cell_area = one_degree_area()[rows, columns]
    cold_selected = audit_weights["cold_mixed"] > 0.0

    annual_input_only_screen(
        features,
        current,
        observation,
        cell_area,
        rows,
        columns,
        cold_selected,
    )

    baseline = current[:, 0, :]
    base_diagnostics = {
        name: diagnostics(baseline, observation, cell_area, weight)
        for name, weight in audit_weights.items()
    }
    for name, values in base_diagnostics.items():
        print(
            f"BASE group={name} "
            + " ".join(f"{key}={value}" for key, value in values.items()),
            flush=True,
        )

    windows = recession_windows(features)
    for window_name, window in windows.items():
        cold_window = window[:, cold_selected].reshape(16, 12, -1).mean(axis=0)
        monthly_window = np.average(
            cold_window,
            axis=1,
            weights=cell_area[cold_selected],
        )
        print(
            f"WINDOW name={window_name} peak={MONTHS[int(np.argmax(monthly_window))]} "
            f"min={float(monthly_window.min()):.9f} "
            f"max={float(monthly_window.max()):.9f}",
            flush=True,
        )
        for strength in (0.125, 0.25, 0.5):
            candidate = apply_recession_allocator(current, window, strength)
            print(f"CANDIDATE window={window_name} strength={strength:g}", flush=True)
            for name, weight in audit_weights.items():
                values = diagnostics(candidate, observation, cell_area, weight)
                base = base_diagnostics[name]
                print(
                    f"AUDIT group={name} "
                    + " ".join(f"{key}={value}" for key, value in values.items())
                    + f" d_ratio={float(values['ratio']) - float(base['ratio']):+.9f}"
                    + f" d_seasonal_l1={float(values['seasonal_l1']) - float(base['seasonal_l1']):+.9f}"
                    + f" d_cell_cycle_mae={float(values['cell_cycle_mae']) - float(base['cell_cycle_mae']):+.9f}",
                    flush=True,
                )
            for fold in range(4):
                fold_mask = (
                    cold_selected
                    & ((rows // 15 + 2 * (columns // 15)) % 4 == fold)
                ).astype(np.float32)
                before = diagnostics(baseline, observation, cell_area, fold_mask)
                after = diagnostics(candidate, observation, cell_area, fold_mask)
                print(
                    f"HELD window={window_name} strength={strength:g} fold={fold} "
                    f"d_ratio={float(after['ratio']) - float(before['ratio']):+.9f} "
                    f"d_seasonal_l1={float(after['seasonal_l1']) - float(before['seasonal_l1']):+.9f} "
                    f"d_cell_cycle_mae={float(after['cell_cycle_mae']) - float(before['cell_cycle_mae']):+.9f} "
                    f"peak={after['peak']}/{after['obs_peak']}",
                    flush=True,
                )
            del candidate

    for strength in (0.1, 0.25, 0.5):
        candidate = apply_recession_bank(current, features, strength)
        print(f"BANK_CANDIDATE strength={strength:g}", flush=True)
        for name, weight in audit_weights.items():
            values = diagnostics(candidate, observation, cell_area, weight)
            base = base_diagnostics[name]
            print(
                f"BANK_AUDIT group={name} "
                + " ".join(f"{key}={value}" for key, value in values.items())
                + f" d_ratio={float(values['ratio']) - float(base['ratio']):+.9f}"
                + f" d_seasonal_l1={float(values['seasonal_l1']) - float(base['seasonal_l1']):+.9f}"
                + f" d_cell_cycle_mae={float(values['cell_cycle_mae']) - float(base['cell_cycle_mae']):+.9f}",
                flush=True,
            )
        for fold in range(4):
            fold_mask = (
                cold_selected
                & ((rows // 15 + 2 * (columns // 15)) % 4 == fold)
            ).astype(np.float32)
            before = diagnostics(baseline, observation, cell_area, fold_mask)
            after = diagnostics(candidate, observation, cell_area, fold_mask)
            print(
                f"BANK_HELD strength={strength:g} fold={fold} "
                f"d_ratio={float(after['ratio']) - float(before['ratio']):+.9f} "
                f"d_seasonal_l1={float(after['seasonal_l1']) - float(before['seasonal_l1']):+.9f} "
                f"d_cell_cycle_mae={float(after['cell_cycle_mae']) - float(before['cell_cycle_mae']):+.9f} "
                f"peak={after['peak']}/{after['obs_peak']}",
                flush=True,
            )
        del candidate

    for carrier_name in ("low_natural", "managed_open"):
        for scale in (0.001, 0.003, 0.01):
            candidate = apply_cold_mosaic_onset(
                current, sampled, features, scale, carrier_name
            )
            print(
                f"ONSET_CANDIDATE carrier={carrier_name} scale={scale:g}",
                flush=True,
            )
            for name, weight in audit_weights.items():
                values = diagnostics(candidate, observation, cell_area, weight)
                base = base_diagnostics[name]
                print(
                    f"ONSET_AUDIT group={name} "
                    + " ".join(f"{key}={value}" for key, value in values.items())
                    + f" d_ratio={float(values['ratio']) - float(base['ratio']):+.9f}"
                    + f" d_annual_log={float(values['annual_log']) - float(base['annual_log']):+.9f}"
                    + f" d_seasonal_l1={float(values['seasonal_l1']) - float(base['seasonal_l1']):+.9f}",
                    flush=True,
                )
            for fold in range(4):
                fold_mask = (
                    cold_selected
                    & ((rows // 15 + 2 * (columns // 15)) % 4 == fold)
                ).astype(np.float32)
                before = diagnostics(baseline, observation, cell_area, fold_mask)
                after = diagnostics(candidate, observation, cell_area, fold_mask)
                print(
                    f"ONSET_HELD carrier={carrier_name} scale={scale:g} fold={fold} "
                    f"d_ratio={float(after['ratio']) - float(before['ratio']):+.9f} "
                    f"d_annual_log={float(after['annual_log']) - float(before['annual_log']):+.9f} "
                    f"d_seasonal_l1={float(after['seasonal_l1']) - float(before['seasonal_l1']):+.9f} "
                    f"peak={after['peak']}/{after['obs_peak']}",
                    flush=True,
                )
            del candidate

    # Perturbing every supplied field after month 96 must not affect the
    # candidate prefix.  This also covers the new internally carried bank.
    full_bank_prefix = apply_recession_bank(current, features, 0.25)[:96].copy()
    for values in sampled.values():
        values[96:] *= np.float32(0.5)
    future_current = np.asarray(
        model.predict(sampled, dict(model.PARAMS), None), dtype=np.float32
    )
    future_features = monthly_features(sampled, future_current)
    future_bank = apply_recession_bank(future_current, future_features, 0.25)
    prefix_delta = float(np.max(np.abs(full_bank_prefix - future_bank[:96])))
    print(
        f"PREFIX family=bank strength=0.25 future_start=96 "
        f"factor=0.5 max_abs={prefix_delta:.12g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
