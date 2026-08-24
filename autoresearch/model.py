"""Historical GFED5 Model F checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = ('dryness',
 'annual_precipitation',
 'monthly_precipitation',
 'air_temperature',
 'gpp',
 'aboveground_biomass',
 'gdp_per_capita')
SEARCH_SPACE: dict[str, dict[str, Any]] = {}
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'vegetation', 'human')

PARAMS = {'D_high': 2213405.8827401847,
 'D_low': 119.80284169042964,
 'P_half': 259.7047545874044,
 'fire_exp': 2.1451212576162297,
 'fuel_half': 64.24391915315266,
 'fuel_k': 0.02956523954197129,
 'gdp_gamma': 1.3618461221651414,
 'gpp_af': 0.03654858845478425,
 'gpp_b': 3.061482587064216e-05,
 'gpp_d': 463.53757730983125,
 'ign_c': 19.360027178029885,
 'ign_k': 0.003451466589750744,
 'k1': 0.0021057085956952444,
 'k2': 0.03417430132236012,
 'pre_dampen_half': 17.927186863760173,
 'trop_agb_crit': 11.23526917747892,
 'trop_k_veg': 0.834794461563624}

GDP_REGION_GAMMA = {'Africa': 1.6,
 'Australia': 0.0,
 'Boreal': 0.5,
 'Europe': 0.7000000000000001,
 'N.America': 0.6000000000000001,
 'S.America': 0.30000000000000004,
 'SEAsia': 0.1,
 'fb': 0.0}

GDP_LOG10_PIVOT = 3.965778107089856

GDP_SIGMA = 4.0

GLOBAL_SCALE = 1.0658593131508673

REGION_BOXES = {'Africa': (-20.0, 52.0, -36.0, 18.0),
 'Australia': (112.0, 154.0, -44.0, -10.0),
 'Boreal': (40.0, 180.0, 48.0, 78.0),
 'Europe': (-12.0, 40.0, 36.0, 72.0),
 'N.America': (-168.0, -52.0, 14.0, 74.0),
 'S.America': (-82.0, -34.0, -56.0, 14.0),
 'SEAsia': (60.0, 150.0, -11.0, 30.0)}


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


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def _gamma_field() -> np.ndarray:
    lat = -89.5 + np.arange(180, dtype=np.float64)
    lon = -179.5 + np.arange(360, dtype=np.float64)
    longitude, latitude = np.meshgrid(lon, lat)
    field = np.full((180, 360), GDP_REGION_GAMMA["fb"], dtype=np.float64)
    assigned = np.zeros((180, 360), dtype=bool)
    for region in REGION_BOXES:
        west, east, south, north = REGION_BOXES[region]
        mask = (
            (longitude >= west)
            & (longitude <= east)
            & (latitude >= south)
            & (latitude <= north)
            & ~assigned
        )
        field[mask] = GDP_REGION_GAMMA[region]
        assigned |= mask
    kernel = _gaussian_kernel(GDP_SIGMA)
    return _smooth_axis(_smooth_axis(field, kernel, 0), kernel, 1)


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
        multiplier = np.power(10.0, _gamma_field()[None] * (GDP_LOG10_PIVOT - log_gdp))
        multiplier = np.where(np.isfinite(gdp) & (gdp > 0.0), multiplier, 1.0)
        rate *= np.clip(multiplier, 0.15, 6.0)
    rate = np.minimum(GLOBAL_SCALE * rate, 5.0)
    return np.asarray(1.0 - np.exp(-rate / 12.0), dtype=np.float32)
