"""Starting ED-Fire model adapted from the retained dev ILAMB winner.

The GPP driver now matches the coupled-ED dump used by the coupled GFED5
models. This complete scaffold remains unevaluated until its first baseline.
"""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = (
    "dryness",
    "annual_precipitation",
    "monthly_precipitation",
    "air_temperature",
    "gpp",
    "luh2_cropland_fraction",
    "luh2_pasture_fraction",
    "luh2_rangeland_fraction",
    "luh2_primary_fraction",
    "luh2_secondary_fraction",
)

# Retained parameters from dev's modelc-luh2-anomaly winner. The six LUH2
# values were its final Optuna trial; the remaining Model C values were frozen.
PARAMS = {
    "k1": 0.03635503353478365,
    "D_low": 70.18267183720735,
    "k2": 0.012758211164590085,
    "D_high": 2940.51756322311,
    "P_half": 12.808863354047988,
    "pre_dampen_half": 107.40052367919465,
    "gpp_af": 0.1476584299248268,
    "gpp_b": 0.0005752434266899163,
    "gpp_d": 660.9129108295722,
    "ign_k": 8.708806168038567,
    "ign_c": 20.03995359361212,
    "fire_exp": 1.165368636520435,
    "lu_k": 6.0,
    "lu_c": 0.2,
    "lw_k": 6.0,
    "lw_c": 0.8,
    "hb": 0.34585063246240166,
    "hd": 0.8184090602983825,
    "hc": 0.13570862977741782,
    "lu_managed_affinity_w": -1.1137815300640124,
    "lu_wild_suppression_w": -0.627385219371479,
    "lu_managed_hump_w": 1.4336812786549538,
}

# Start with the search that produced the retained winner. The researcher can
# widen or replace this space when testing a different structural hypothesis.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "hb": {"type": "float", "low": 0.05, "high": 0.4},
    "hc": {"type": "float", "low": 0.05, "high": 0.4},
    "hd": {"type": "float", "low": 0.2, "high": 0.9},
    "lu_managed_affinity_w": {"type": "float", "low": -2.0, "high": 2.0},
    "lu_wild_suppression_w": {"type": "float", "low": -2.0, "high": 2.0},
    "lu_managed_hump_w": {"type": "float", "low": -2.0, "high": 2.0},
}

# These are physical process groups rather than every algebraic factor. Five
# groups require 32 subsets for exact Shapley attribution.
COMPONENTS = ("dryness", "precipitation", "fuel", "temperature", "land_use")


def _rising(array: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-k * (array - center), -50.0, 50.0)))


def _falling(array: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(k * (array - center), -50.0, 50.0)))


def _hump(array: np.ndarray, rise: float, decay: float) -> np.ndarray:
    rise = max(float(rise), 1e-9)
    decay = max(float(decay), 1e-9)
    return (1.0 - np.exp(-np.clip(array / rise, 0.0, 500.0))) * np.exp(
        -np.clip(array / decay, 0.0, 500.0)
    )


def predict(
    data: Mapping[str, np.ndarray],
    params: Mapping[str, float] | None = None,
    components: Collection[str] | None = None,
) -> np.ndarray:
    """Return monthly burned-area fractions with shape (192, 180, 360)."""
    p = dict(PARAMS)
    if params is not None:
        p.update(params)

    enabled = set(COMPONENTS if components is None else components)
    unknown = enabled - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown model components: {sorted(unknown)}")

    rate = np.ones_like(data["dryness"], dtype=np.float32)

    if "dryness" in enabled:
        dryness = data["dryness"]
        rate *= _rising(dryness, p["k1"], p["D_low"])
        rate *= _falling(dryness, p["k2"], p["D_high"])

    if "precipitation" in enabled:
        annual = data["annual_precipitation"]
        monthly = data["monthly_precipitation"]
        rate *= annual / (annual + p["P_half"] + 1e-12)
        rate *= 1.0 / (1.0 + monthly / (p["pre_dampen_half"] + 1e-12))

    if "fuel" in enabled:
        rate *= _hump(p["gpp_af"] * data["gpp"], p["gpp_b"], p["gpp_d"])

    if "temperature" in enabled:
        rate *= _rising(data["air_temperature"], p["ign_k"], p["ign_c"])

    if "land_use" in enabled:
        managed = (
            data["luh2_cropland_fraction"]
            + data["luh2_pasture_fraction"]
            + data["luh2_rangeland_fraction"]
        )
        wild = data["luh2_primary_fraction"] + data["luh2_secondary_fraction"]
        affinity = _rising(managed, p["lu_k"], p["lu_c"])
        wild_suppression = _falling(wild, p["lw_k"], p["lw_c"])
        managed_hump = _hump(managed, p["hb"], p["hd"])
        hump_reference = float(
            _hump(np.asarray(p["hc"], dtype=np.float32), p["hb"], p["hd"])
        )
        managed_hump /= hump_reference + 1e-9
        anomaly = (
            p["lu_managed_affinity_w"] * (affinity - affinity.mean())
            + p["lu_wild_suppression_w"]
            * (wild_suppression - wild_suppression.mean())
            + p["lu_managed_hump_w"] * (managed_hump - managed_hump.mean())
        )
        rate *= np.clip(1.0 + anomaly, 0.0, None)

    rate = np.power(np.clip(rate, 0.0, None), p["fire_exp"])
    rate = np.minimum(rate, 5.0)
    prediction = (1.0 - np.exp(-rate)) / 12.0
    return np.asarray(prediction, dtype=np.float32)
