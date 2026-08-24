"""Historical GFED5 Model H checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = ('dryness',
 'annual_precipitation',
 'monthly_precipitation',
 'air_temperature',
 'gpp',
 'gdp_per_capita')
SEARCH_SPACE: dict[str, dict[str, Any]] = {}
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'human')

PARAMS = {'D_high': 2970.7920514735997,
 'D_low': 148.10800730363493,
 'P_half': 35.828811741825064,
 'fire_exp': 1.0881360604103363,
 'gdp_gamma': 1.8068697350278546,
 'gpp_af': 0.09145768948268033,
 'gpp_b': 0.0030742893625766146,
 'gpp_d': 25.184822598549903,
 'ign_c': 11.77190025324203,
 'ign_k': 2.7728241828010627,
 'k1': 0.025648117673977434,
 'k2': 0.0007081721844034209,
 'pre_dampen_half': 8.904067307938409}

GDP_LOG10_PIVOT = 3.965778107089856


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


def _fire_rate(
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    rate = np.ones_like(data["dryness"], dtype=np.float32)
    if "dryness" in enabled:
        rate *= _rising(data["dryness"], p["k1"], p["D_low"])
        rate *= _falling(data["dryness"], p["k2"], p["D_high"])
    if "precipitation" in enabled:
        annual = data["annual_precipitation"]
        monthly = data["monthly_precipitation"]
        rate *= annual / (annual + p["P_half"] + 1e-12)
        rate *= 1.0 / (1.0 + monthly / (p["pre_dampen_half"] + 1e-12))
    if "fuel" in enabled:
        rate *= _hump(p["gpp_af"] * data["gpp"], p["gpp_b"], p["gpp_d"])
    if "temperature" in enabled:
        rate *= _rising(data["air_temperature"], p["ign_k"], p["ign_c"])
    if "vegetation" in enabled and "trop_agb_crit" in p:
        lat = -89.5 + np.arange(180, dtype=np.float32)
        tropical = (np.abs(lat) < 23.5).astype(np.float32)[None, :, None]
        ratio = np.clip(
            data["aboveground_biomass"] / (p["trop_agb_crit"] + 1e-12),
            0.0,
            None,
        )
        canopy = 1.0 / (1.0 + np.power(ratio, p["trop_k_veg"]))
        rate *= tropical * canopy + (1.0 - tropical)
    rate = np.power(np.clip(rate, 0.0, None), p["fire_exp"])
    if "fuel" in enabled and "fuel_k" in p:
        capacity = data["gpp"].mean(axis=0, keepdims=True)
        capacity = capacity / (capacity + p["fuel_half"] + 1e-9)
        rate *= 1.0 + p["fuel_k"] * capacity
    elif "fire_amp" in p:
        rate *= p["fire_amp"]
    return rate


def predict(
    data: Mapping[str, np.ndarray],
    params: Mapping[str, float] | None = None,
    components: Collection[str] | None = None,
) -> np.ndarray:
    p = dict(PARAMS)
    if params is not None:
        p.update(params)
    enabled = set(COMPONENTS if components is None else components)
    unknown = enabled - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown model components: {sorted(unknown)}")
    rate = _fire_rate(data, p, enabled)
    if "human" in enabled:
        gdp = data["gdp_per_capita"]
        log_gdp = np.log10(np.clip(gdp, 50.0, None))
        multiplier = np.power(10.0, p["gdp_gamma"] * (GDP_LOG10_PIVOT - log_gdp))
        multiplier = np.where(np.isfinite(gdp) & (gdp > 0.0), multiplier, 1.0)
        rate *= np.clip(multiplier, 0.15, 6.0)
    rate = np.minimum(rate, 5.0)
    return np.asarray((1.0 - np.exp(-rate)) / 12.0, dtype=np.float32)
