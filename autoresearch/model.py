"""Historical GFED5 Model E checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = ('dryness',
 'annual_precipitation',
 'monthly_precipitation',
 'air_temperature',
 'gpp',
 'aboveground_biomass')
SEARCH_SPACE: dict[str, dict[str, Any]] = {}
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'vegetation')

PARAMS = {'D_high': 2427.054417940131,
 'D_low': 16.00237013173313,
 'P_half': 157.80309954142965,
 'fire_amp': 5.44779717154213,
 'fire_exp': 1.5191397119156032,
 'gpp_af': 0.03522602097143089,
 'gpp_b': 0.00012118589392553798,
 'gpp_d': 12.338826503601998,
 'ign_c': 22.126836566118193,
 'ign_k': 3.965503094323014,
 'k1': 0.028401063201780267,
 'k2': 0.00033271381912662167,
 'pre_dampen_half': 8.259599205585761,
 'trop_agb_crit': 5.526463318565575,
 'trop_k_veg': 2.724170387400844}

REGION_PARAMS = {'Africa': {'D_high': 2289.1315306837046,
            'D_low': 5.351452349843333,
            'P_half': 221.2683656611821,
            'fire_exp': 1.4126958348191876,
            'fuel_half': 11.781063202742125,
            'fuel_k': 4.710433409437072,
            'gpp_af': 0.3002317246575309,
            'gpp_b': 0.0009153552948098625,
            'gpp_d': 229.41380323183625,
            'ign_c': 22.126836566118193,
            'ign_k': 0.8599189552684969,
            'k1': 0.012840669372724958,
            'k2': 0.002884223471687844,
            'pre_dampen_half': 565.0116730717873,
            'trop_agb_crit': 10.215415653610627,
            'trop_k_veg': 6.45312300188076},
 'Boreal': {'D_high': 7333.200379637292,
            'D_low': 236.79352040223893,
            'P_half': 1.4551749476072258,
            'fire_amp': 2.7806853670293616,
            'fire_exp': 1.5191397119156032,
            'gpp_af': 0.060029341913968644,
            'gpp_b': 5.0875243087016374e-05,
            'gpp_d': 1.4397852866783676,
            'ign_c': 22.126836566118193,
            'ign_k': 0.0032383308053152635,
            'k1': 0.03648456083047798,
            'k2': 0.019383858782150494,
            'pre_dampen_half': 8.259599205585761,
            'trop_agb_crit': 1.8023542850487386,
            'trop_k_veg': 2.724170387400844},
 'Europe': {'D_high': 16780.50857913099,
            'D_low': 408.3047491580097,
            'P_half': 9.580682294110435,
            'fire_amp': 1.8357623383234334,
            'fire_exp': 1.7505406131449859,
            'gpp_af': 0.10098504976210308,
            'gpp_b': 0.002416023630607169,
            'gpp_d': 0.12027930570908299,
            'ign_c': 2.226297338331262,
            'ign_k': 0.002357279455676982,
            'k1': 0.08006837661379566,
            'k2': 4.69955602109909e-05,
            'pre_dampen_half': 189.88136301569827,
            'trop_agb_crit': 12.708819454040446,
            'trop_k_veg': 0.828441460847982},
 'S.America': {'D_high': 2298.2155660045737,
               'D_low': 1121.5915868662764,
               'P_half': 1265.0580856641704,
               'fire_amp': 3.1543496524370136,
               'fire_exp': 1.4848387260494937,
               'gpp_af': 1.9706963379095486,
               'gpp_b': 0.00012118589392553798,
               'gpp_d': 96.93415502195788,
               'ign_c': 0.4177928311636542,
               'ign_k': 0.012480180965117606,
               'k1': 3.609310926766545e-05,
               'k2': 0.003767022142229395,
               'pre_dampen_half': 170.71098124594695,
               'trop_agb_crit': 5.635210829694053,
               'trop_k_veg': 4.989748894578724},
 'SEAsia': {'D_high': 262.11304189124235,
            'D_low': 12.105387404469584,
            'P_half': 61.6097529474144,
            'fire_amp': 2.042193818779288,
            'fire_exp': 1.7220686611084326,
            'gpp_af': 0.016306190741524337,
            'gpp_b': 0.0012611058927054877,
            'gpp_d': 545.0253444559129,
            'ign_c': 1.4091459170104739,
            'ign_k': 0.8861249804463233,
            'k1': 0.012840669372724958,
            'k2': 0.02065142557895925,
            'pre_dampen_half': 16.009350908593582,
            'trop_agb_crit': 29.831859755984098,
            'trop_k_veg': 6.168921645828263}}

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


def _transform(rate: np.ndarray) -> np.ndarray:
    rate = np.minimum(rate, 5.0)
    return 1.0 - np.exp(-rate / 12.0)


def predict(
    data: Mapping[str, np.ndarray],
    params: Mapping[str, float] | None = None,
    components: Collection[str] | None = None,
) -> np.ndarray:
    fallback = dict(PARAMS)
    if params is not None:
        fallback.update(params)
    enabled = set(COMPONENTS if components is None else components)
    unknown = enabled - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown model components: {sorted(unknown)}")
    prediction = _transform(_fire_rate(data, fallback, enabled))
    lat = -89.5 + np.arange(180, dtype=np.float32)
    lon = -179.5 + np.arange(360, dtype=np.float32)
    longitude, latitude = np.meshgrid(lon, lat)
    assigned = np.zeros((180, 360), dtype=bool)
    for region, region_params in REGION_PARAMS.items():
        west, east, south, north = REGION_BOXES[region]
        mask = (
            (longitude >= west)
            & (longitude <= east)
            & (latitude >= south)
            & (latitude <= north)
            & ~assigned
        )
        regional = _transform(_fire_rate(data, region_params, enabled))
        prediction[:, mask] = regional[:, mask]
        assigned |= mask
    return np.asarray(prediction, dtype=np.float32)
