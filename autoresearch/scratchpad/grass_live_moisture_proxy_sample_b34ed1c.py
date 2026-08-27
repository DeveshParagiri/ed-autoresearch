"""Sample whether ED phenology can replace VPD in a live/dead grass split.

LPJmL-SPITFIRE1.9 separates live and dead grass.  A phenological live-fuel
moisture estimate transfers a smooth fraction of grass load into the cured
dead class, whose combustion is then governed by fine-dead-fuel moisture.
This scratch diagnostic implements that architecture rather than treating a
dry-season delay itself as the mechanism.

ED GPP and LAI provide coupled, site-local phenological status.  Monthly rain
and the certified atmospheric dryness field provide a distinct dead-fuel
moisture response.  A causal root-zone water bucket is retained only as an
independent proxy check and physical control.  Coordinates, observations, and
the warm-seasonal-open mask are used after prediction for sampling and
diagnostics only.  No tracked file, official evaluation, or ledger is changed.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.grass_standing_dead_release_sample_b34ed1c import (  # noqa: E402
    GrassState,
    carrier_mask,
    common_drivers,
    field,
    finite_release_bank,
    print_delta,
)
from autoresearch.scratchpad.heating_lightning_sample_falsification_75fe945 import (  # noqa: E402
    antecedent,
    load_observed,
    load_selected,
    logistic,
    select_cells,
    weighted_corr,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_model  # noqa: E402


def live_dead_proxy(
    data: Mapping[str, np.ndarray],
    drivers: Mapping[str, np.ndarray],
    source: str,
) -> tuple[GrassState, dict[str, np.ndarray]]:
    """Return a SPITFIRE-like live/dead partition and its diagnostics."""
    rain = drivers["rain"]
    combustion = drivers["combustion"]
    temperature = drivers["temperature"]
    if source == "ed_phenology":
        gpp = np.clip(field(data, "gpp"), 0.0, None)
        lai = np.clip(field(data, "leaf_area_index"), 0.0, None)
        gpp12 = antecedent(gpp, 12.0)
        lai12 = antecedent(lai, 12.0)
        # Equal current and annual-background states map to full activity.
        # Either falling GPP or falling LAI can limit live phenology.
        gpp_activity = 2.0 * gpp / (gpp + gpp12 + 0.20)
        lai_activity = 2.0 * lai / (lai + lai12 + 0.50)
        activity = np.clip(np.sqrt(gpp_activity * lai_activity), 0.0, 1.0)
        production = 0.32 * drivers["pathway_share"] * gpp / (gpp + 0.35)
    elif source == "root_water":
        thermal = logistic((temperature - 10.0) / 4.0)
        activity = np.clip(drivers["water_fraction"] * thermal, 0.0, 1.0)
        production = drivers["production"]
    else:
        raise ValueError(source)

    # SPITFIRE bracket: 250% water at fully active, 120% starts curing,
    # and 30% is fully cured.  Units are water / oven-dry grass mass.
    live_moisture = 0.30 + 2.20 * activity
    cured_fraction = np.clip((1.20 - live_moisture) / 0.90, 0.0, 1.0)

    # Dead and live fractions retain distinct moisture/extinction responses.
    dead_combustibility = np.sqrt(
        combustion * 1.0 / (1.0 + rain / 35.0)
    )
    live_combustibility = logistic((1.20 - live_moisture) / 0.18)
    readiness = np.clip(
        cured_fraction * dead_combustibility
        + (1.0 - cured_fraction) * live_combustibility,
        0.0,
        1.0,
    )
    grass_load = np.clip(antecedent(production, 6.0), 0.0, None)
    state = GrassState(
        drivers["pathway_share"],
        readiness,
        production,
        grass_load * (1.0 - cured_fraction),
        grass_load * cured_fraction,
    )
    return state, {
        "activity": activity,
        "live_moisture": live_moisture,
        "cured_fraction": cured_fraction,
        "dead_combustibility": dead_combustibility,
        "live_combustibility": live_combustibility,
        "readiness": readiness,
    }


def monthly_cycle(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(16, 12, -1).mean(axis=0)


def cycle_target(prediction: np.ndarray, observed: np.ndarray) -> np.ndarray:
    pred_cycle = monthly_cycle(prediction)
    obs_cycle = monthly_cycle(observed)
    pred_allocation = pred_cycle / (pred_cycle.sum(axis=0, keepdims=True) + 1e-8)
    obs_allocation = obs_cycle / (obs_cycle.sum(axis=0, keepdims=True) + 1e-8)
    return obs_allocation - pred_allocation


def correlation_audit(
    data: Mapping[str, np.ndarray],
    drivers: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, dict[str, np.ndarray]],
    prediction: np.ndarray,
    observed: np.ndarray,
    reference_weight: np.ndarray,
    folds: np.ndarray,
    carrier: np.ndarray,
) -> None:
    gpp = np.clip(field(data, "gpp"), 0.0, None)
    lai = np.clip(field(data, "leaf_area_index"), 0.0, None)
    gpp3 = antecedent(gpp, 3.0)
    lai3 = antecedent(lai, 3.0)
    gpp_decline = np.maximum((gpp3 - gpp) / (gpp3 + gpp + 0.20), 0.0)
    lai_decline = np.maximum((lai3 - lai) / (lai3 + lai + 0.50), 0.0)
    phenology_decline = np.sqrt(gpp_decline * lai_decline)
    root_drawdown = drivers["drawdown"]
    root_curing = diagnostics["root_water"]["cured_fraction"]
    ed_curing = diagnostics["ed_phenology"]["cured_fraction"]
    target = cycle_target(prediction, observed)
    target_full = np.tile(target, (16, 1, 1)).reshape(192, -1)
    weight_full = np.broadcast_to(reference_weight[None, :], target_full.shape)
    pred_annual = np.average(prediction, axis=0, weights=np.tile(
        np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
        16,
    ))
    obs_annual = np.average(observed, axis=0, weights=np.tile(
        np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
        16,
    ))
    annual_target = np.log((obs_annual + 1e-5) / (pred_annual + 1e-5))

    def corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
        return weighted_corr(
            left[:, mask], right[:, mask], weight_full[:, mask]
        )

    print(
        "PROXY "
        f"gpp_decline_vs_root_drawdown={corr(gpp_decline, root_drawdown, carrier):+.8f} "
        f"lai_decline_vs_root_drawdown={corr(lai_decline, root_drawdown, carrier):+.8f} "
        f"joint_decline_vs_root_drawdown={corr(phenology_decline, root_drawdown, carrier):+.8f} "
        f"ed_curing_vs_root_curing={corr(ed_curing, root_curing, carrier):+.8f}",
        flush=True,
    )
    for name in ("ed_phenology", "root_water"):
        diag = diagnostics[name]
        state = live_dead_proxy(data, drivers, name)[0]
        annual_states = {
            "mean_curing": diag["cured_fraction"].mean(axis=0),
            "mean_readiness": diag["readiness"].mean(axis=0),
            "burnable_dead_load": (
                state.dead_stock * diag["dead_combustibility"]
            ).mean(axis=0),
            "live_load": state.live_stock.mean(axis=0),
        }
        print(
            f"DIRECTION {name} "
            f"curing_corr={corr(diag['cured_fraction'], target_full, carrier):+.8f} "
            f"dead_combustibility_corr={corr(diag['dead_combustibility'], target_full, carrier):+.8f} "
            f"separate_readiness_corr={corr(diag['readiness'], target_full, carrier):+.8f} "
            f"fully_green={np.average(diag['cured_fraction'][:, carrier] <= 1e-6, weights=weight_full[:, carrier]):.6f} "
            f"fully_cured={np.average(diag['cured_fraction'][:, carrier] >= 1.0 - 1e-6, weights=weight_full[:, carrier]):.6f}",
            flush=True,
        )
        print(
            f"ANNUAL_DIRECTION {name} "
            + " ".join(
                f"{state_name}={weighted_corr(values[carrier], annual_target[carrier], reference_weight[carrier]):+.8f}"
                for state_name, values in annual_states.items()
            ),
            flush=True,
        )
        signal_cycle = monthly_cycle(diag["readiness"])
        for lag in range(-3, 4):
            correlation = weighted_corr(
                np.roll(signal_cycle[:, carrier], lag, axis=0),
                target[:, carrier],
                np.broadcast_to(reference_weight[None, carrier], target[:, carrier].shape),
            )
            print(f"LAG {name} months={lag:+d} target_corr={correlation:+.8f}", flush=True)
        for fold in range(4):
            selected = carrier & (folds == fold)
            print(
                f"PROXY_FOLD {name} fold={fold} "
                f"curing_root_corr={corr(diag['cured_fraction'], root_curing, selected):+.8f} "
                f"readiness_target_corr={corr(diag['readiness'], target_full, selected):+.8f} "
                f"annual_curing_corr={weighted_corr(annual_states['mean_curing'][selected], annual_target[selected], reference_weight[selected]):+.8f} "
                f"annual_readiness_corr={weighted_corr(annual_states['mean_readiness'][selected], annual_target[selected], reference_weight[selected]):+.8f} "
                f"annual_dead_load_corr={weighted_corr(annual_states['burnable_dead_load'][selected], annual_target[selected], reference_weight[selected]):+.8f}",
                flush=True,
            )


def prefix_audit(
    data: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    source: str,
    split: int = 96,
) -> float:
    changed = {name: np.asarray(values).copy() for name, values in data.items()}
    rng = np.random.default_rng(240902)
    for name in (
        "monthly_precipitation",
        "air_temperature",
        "dryness",
        "gpp",
        "leaf_area_index",
        "luh2_primary_fraction",
        "luh2_cropland_fraction",
        "luh2_pasture_fraction",
        "luh2_rangeland_fraction",
        "luh2_urban_fraction",
    ):
        values = changed[name]
        scale = np.maximum(np.abs(values[split:]), 1.0)
        values[split:] += rng.normal(0.0, 2.0, values[split:].shape) * scale
    original_drivers = common_drivers(data)
    changed_drivers = common_drivers(changed)
    original_state, _ = live_dead_proxy(data, original_drivers, source)
    changed_state, _ = live_dead_proxy(changed, changed_drivers, source)
    original, _, _ = finite_release_bank(prediction, original_state)
    changed_prediction = prediction.copy()
    changed_prediction[split:] = np.clip(1.0 - prediction[split:], 0.0, 1.0 - 1e-7)
    future_changed, _, _ = finite_release_bank(changed_prediction, changed_state)
    return float(np.max(np.abs(original[:split] - future_changed[:split])))


def main() -> int:
    evaluator = GFED5Evaluator(GFED5_PATH)
    rows, cols, area, reference_weight, retained = select_cells(evaluator, count=1536)
    folds = ((rows // 15) + 3 * (cols // 15)) % 4
    print(
        f"DESIGN cells={rows.size} retained_reference_weight={retained:.8f}",
        flush=True,
    )
    model = load_model()
    data = load_selected(model.INPUTS, rows, cols)
    observed = load_observed(rows, cols)
    del evaluator
    gc.collect()
    baseline = np.asarray(
        model.predict(data, dict(model.PARAMS), None), dtype=np.float64
    )[:, 0, :]
    drivers = common_drivers(data)
    carrier = carrier_mask(data, drivers)
    states: dict[str, GrassState] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for source in ("ed_phenology", "root_water"):
        states[source], diagnostics[source] = live_dead_proxy(data, drivers, source)
    correlation_audit(
        data, drivers, diagnostics, baseline, observed, reference_weight, folds, carrier
    )

    # The held-block annual/cycle audit found that the Aug-to-Dec aggregate
    # mismatch is primarily a between-cell annual-propensity mixture, not a
    # within-cell timing error.  Stop at proxy validity: do not score a second
    # timing transform unless the signals show a strong, stable annual map
    # association in a separately designed follow-up.
    for source in states:
        print(
            f"PREFIX {source} max={prefix_audit(data, baseline, source):.12e}",
            flush=True,
        )
    print("DECISION proxy_only_no_timing_candidate", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
