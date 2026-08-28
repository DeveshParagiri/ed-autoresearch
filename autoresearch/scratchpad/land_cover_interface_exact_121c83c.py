"""Exact full-grid proxy for pointwise land-interface access only.

This reopens the physically supported access signal under an Overall-first
tradeoff rule.  It evaluates fixed access strengths for the pairwise and
Simpson-weighted formulations at the exact ``121c83c`` incumbent.  Geography,
ecological masks, countries, and GFED enter only after prediction for audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch.scratchpad.current_component_loo_causal_b34ed1c import (  # noqa: E402
    ecological_statistics,
    regime_masks,
)
from autoresearch.scratchpad.ecological_geography_audit import (  # noqa: E402
    DEFAULT_SHP,
    area_statistics,
    country_masks,
    cycle_and_annual,
)
from autoresearch.scratchpad.land_cover_interface_mechanism_121c83c import (  # noqa: E402
    candidate,
)
from autoresearch.scratchpad.rothermel_event_closure_121c83c import (  # noqa: E402
    EXPECTED_INCUMBENT,
    MONTH_DAYS,
    PINNED,
    load_pinned,
)
from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    validate_prediction,
)


STRENGTHS = (0.05, 0.10, 0.20)
METRICS = (
    ("overall", "overall_score"),
    ("bias", "bias_score"),
    ("rmse", "rmse_score"),
    ("seasonal", "seasonal_cycle_score"),
    ("spatial", "spatial_distribution_score"),
)
REGIONS = (
    "bona", "tena", "ceam", "nhsa", "shsa", "euro", "mide",
    "nhaf", "shaf", "boas", "ceas", "seas", "eqas", "aust",
)


def global_area_ratio(
    prediction: np.ndarray,
    observation: np.ndarray,
    area: np.ndarray,
    land: np.ndarray,
) -> float:
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    obs_annual = np.average(observation, axis=0, weights=MONTH_DAYS)
    weights = area * land
    return float(np.sum(pred_annual * weights) / np.sum(obs_annual * weights))


def audit_masks(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks = dict(regime_masks(data))
    countries, _ = country_masks(DEFAULT_SHP)
    by_iso = {iso: mask for (_name, iso), mask in countries.items()}
    if "COD" not in by_iso or "COG" not in by_iso:
        raise RuntimeError("Congo country masks unavailable")
    masks["democratic_republic_congo"] = by_iso["COD"] > 0.0
    masks["republic_congo"] = by_iso["COG"] > 0.0
    masks["congo_combined"] = (by_iso["COD"] + by_iso["COG"]) > 0.0
    fractional = {
        "democratic_republic_congo": by_iso["COD"],
        "republic_congo": by_iso["COG"],
        "congo_combined": np.clip(by_iso["COD"] + by_iso["COG"], 0.0, 1.0),
    }
    return masks, fractional


def ecology(
    prediction: np.ndarray,
    masks: dict[str, np.ndarray],
    fractional_countries: dict[str, np.ndarray],
    observation: np.ndarray,
    area: np.ndarray,
    land: np.ndarray,
) -> dict[str, dict[str, float | int | str]]:
    regimes = ecological_statistics(prediction, masks, observation, area, land)
    # Keep country masks fractional for the exact country audit.  The standard
    # ecological helper above is deliberately boolean, so recompute the three
    # Congo entries with their fractional cell coverage.
    model_cycle, model_annual = cycle_and_annual(prediction)
    obs_cycle, obs_annual = cycle_and_annual(observation)
    for name, mask in fractional_countries.items():
        regimes[name] = area_statistics(
            mask * land,
            model_cycle,
            model_annual,
            obs_cycle,
            obs_annual,
            area,
        )
    return regimes


def severe_pathology(
    baseline: dict[str, dict[str, float | int | str]],
    trial: dict[str, dict[str, float | int | str]],
) -> list[str]:
    """Flag only large new ecological failures under the tradeoff rule."""
    failures: list[str] = []
    for name in baseline:
        old = float(baseline[name]["ratio"])
        new = float(trial[name]["ratio"])
        relative = new / old if old > 0.0 else float("inf")
        if not np.isfinite(new) or new < 0.25 or new > 4.0 or relative < 0.75 or relative > 1.25:
            failures.append(name)
    return failures


def main() -> int:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    base_scores = evaluator.score(incumbent)
    base_global = base_scores["global"]
    if abs(base_global["overall_score"] - EXPECTED_INCUMBENT) > 5e-7:
        raise RuntimeError(f"incumbent drift {base_global['overall_score']:.9f}")

    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observation = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    masks, fractional_countries = audit_masks(data)
    base_ecology = ecology(
        incumbent, masks, fractional_countries, observation, area, land
    )
    base_area_ratio = global_area_ratio(incumbent, observation, area, land)

    print(
        f"BASE pinned={PINNED} "
        + " ".join(f"{label}={base_global[key]:.9f}" for label, key in METRICS)
        + f" area_ratio={base_area_ratio:.9f}",
        flush=True,
    )
    print(
        "BASE_ECOLOGY "
        + ",".join(
            f"{name}:{float(values['ratio']):.9f}" for name, values in base_ecology.items()
        ),
        flush=True,
    )

    # One future-counterfactual pair is enough for all six pointwise candidates.
    rows, columns = np.where(land)
    probe = np.linspace(0, rows.size - 1, 64, dtype=np.int64)
    probe_data = {
        name: np.asarray(values[:, rows[probe], columns[probe]], dtype=np.float64)[:, None, :]
        for name, values in data.items()
    }
    before_incumbent = model.predict(probe_data, dict(model.PARAMS), None)
    changed = {name: values.copy() for name, values in probe_data.items()}
    for values in changed.values():
        values[96:] = values[96:][::-1] * 1.37 + 0.123
    after_incumbent = model.predict(changed, dict(model.PARAMS), None)

    winners: list[tuple[float, str, float, list[str]]] = []
    for family in ("pairwise", "simpson"):
        for strength in STRENGTHS:
            label = f"{family}_a{strength:.2f}"
            trial = validate_prediction(candidate(incumbent, data, family, strength, 0.0))
            scores = evaluator.score(trial)
            global_scores = scores["global"]
            ratio = global_area_ratio(trial, observation, area, land)
            trial_ecology = ecology(
                trial, masks, fractional_countries, observation, area, land
            )
            pathologies = severe_pathology(base_ecology, trial_ecology)

            before = candidate(before_incumbent, probe_data, family, strength, 0.0)
            after = candidate(after_incumbent, changed, family, strength, 0.0)
            prefix_max = float(np.max(np.abs(before[:96] - after[:96])))
            if prefix_max != 0.0:
                raise RuntimeError(f"prefix causality failed for {label}: {prefix_max}")

            print(
                f"EXACT label={label} "
                + " ".join(f"{metric}={global_scores[key]:.9f}" for metric, key in METRICS)
                + " deltas="
                + ",".join(
                    f"{metric}:{global_scores[key]-base_global[key]:+.9f}"
                    for metric, key in METRICS
                )
                + f" area_ratio={ratio:.9f} area_delta={ratio-base_area_ratio:+.9f} "
                + f"prefix_max={prefix_max:.12g}",
                flush=True,
            )
            print(
                f"REGIONS label={label} "
                + ",".join(
                    f"{region}:{scores[region]['overall_score']-base_scores[region]['overall_score']:+.9f}"
                    for region in REGIONS
                ),
                flush=True,
            )
            print(
                f"ECOLOGY label={label} "
                + ",".join(
                    f"{name}:{float(base_ecology[name]['ratio']):.9f}->{float(values['ratio']):.9f}"
                    for name, values in trial_ecology.items()
                )
                + " severe=" + (",".join(pathologies) if pathologies else "none"),
                flush=True,
            )
            overall_delta = global_scores["overall_score"] - base_global["overall_score"]
            if overall_delta > 0.0 and not pathologies:
                winners.append((overall_delta, family, strength, pathologies))

    if not winners:
        print("DECISION accept=0 reason=no_overall_safe_candidate", flush=True)
        return 0
    winners.sort(reverse=True)
    delta, family, strength, _ = winners[0]
    print(
        f"DECISION accept=1 family={family} strength={strength:.2f} "
        f"overall_delta={delta:+.9f} rule=overall_first_no_severe_ecology",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
