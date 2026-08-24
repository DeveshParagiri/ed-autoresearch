"""Historical GFED5 Model G checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = ('dryness', 'annual_precipitation', 'monthly_precipitation', 'air_temperature', 'gpp')
SEARCH_SPACE: dict[str, dict[str, Any]] = {}
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature')

PARAMS = {'D_high': 2940.51756322311,
 'D_low': 70.18267183720735,
 'P_half': 12.808863354047988,
 'fire_exp': 1.165368636520435,
 'gpp_af': 0.1476584299248268,
 'gpp_b': 0.0005752434266899163,
 'gpp_d': 660.9129108295722,
 'ign_c': 20.03995359361212,
 'ign_k': 8.708806168038567,
 'k1': 0.03635503353478365,
 'k2': 0.012758211164590085,
 'pre_dampen_half': 107.40052367919465}

REGION_PARAMS = {'Africa': {'D_high': 2941.5751391396584,
            'D_low': 8.1043507389441,
            'P_half': 13.019246714361572,
            'fire_exp': 1.010722707674549,
            'gpp_af': 0.0055518823149351484,
            'gpp_b': 7.402343469704428e-05,
            'gpp_d': 5.878566318396007,
            'ign_c': 8.671454322584175,
            'ign_k': 2.2292793902120662,
            'k1': 0.02641449645424223,
            'k2': 0.08894151386481904,
            'pre_dampen_half': 304.7488318185732},
 'Australia': {'D_high': 2969.087094484682,
               'D_low': 2.458603276328005,
               'P_half': 215.55950571043874,
               'fire_exp': 1.518582349269638,
               'gpp_af': 5.620854611217178,
               'gpp_b': 0.00031478066835753096,
               'gpp_d': 4.5768578731086835,
               'ign_c': 9.34330265731847,
               'ign_k': 0.6454459531371669,
               'k1': 0.0035051730090894812,
               'k2': 0.0022957398214549472,
               'pre_dampen_half': 724.4804125761929},
 'Boreal': {'D_high': 54669.374787660905,
            'D_low': 4.051452710153359,
            'P_half': 12.808863354047988,
            'fire_exp': 1.5201277403788451,
            'gpp_af': 35.20481045526041,
            'gpp_b': 0.10123867001105968,
            'gpp_d': 34.37107943937136,
            'ign_c': 6.160975294741436,
            'ign_k': 0.4702627143647234,
            'k1': 0.03232155805617897,
            'k2': 0.00038791280730284417,
            'pre_dampen_half': 53.73093479067421},
 'Europe': {'D_high': 3121.3520322039926,
            'D_low': 918.4900818511542,
            'P_half': 14.94057723907974,
            'fire_exp': 1.1670151053031461,
            'gpp_af': 5.511863475708366,
            'gpp_b': 0.011750724729953272,
            'gpp_d': 5.085491038878331,
            'ign_c': 0.6235377135673155,
            'ign_k': 0.11101191524846071,
            'k1': 0.002971796631657383,
            'k2': 4.3214816246375446e-05,
            'pre_dampen_half': 107.40052367919465},
 'N.America': {'D_high': 2969.087094484682,
               'D_low': 51.9967392387832,
               'P_half': 958.8659498594012,
               'fire_exp': 1.010722707674549,
               'gpp_af': 0.004339536161007625,
               'gpp_b': 0.000832333943844274,
               'gpp_d': 300.3420512048734,
               'ign_c': 19.19168994439752,
               'ign_k': 0.6454459531371669,
               'k1': 0.025383218374507292,
               'k2': 0.006138509947844854,
               'pre_dampen_half': 14.689898077648792},
 'S.America': {'D_high': 11069.082134755383,
               'D_low': 2.1234210963705626,
               'P_half': 958.8659498594012,
               'fire_exp': 1.151006986743773,
               'gpp_af': 5.511863475708366,
               'gpp_b': 8.539677611488478e-05,
               'gpp_d': 13.076473382928533,
               'ign_c': 19.19168994439752,
               'ign_k': 0.09071933427331372,
               'k1': 0.002971796631657383,
               'k2': 0.006138509947844854,
               'pre_dampen_half': 111.37963167994629},
 'SEAsia': {'D_high': 48249.14281180252,
            'D_low': 57.22950094567826,
            'P_half': 12.808863354047988,
            'fire_exp': 1.324458134009935,
            'gpp_af': 0.0025511964576171086,
            'gpp_b': 0.00012797528816608022,
            'gpp_d': 29.749683811098393,
            'ign_c': 23.872908899163228,
            'ign_k': 0.3818468247973969,
            'k1': 0.01783192664937594,
            'k2': 0.0003607301614542842,
            'pre_dampen_half': 107.40052367919465}}

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
    return (1.0 - np.exp(-rate)) / 12.0


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
