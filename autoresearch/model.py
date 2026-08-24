"""Historical GFED5 Model I checkpoint with inline coefficients."""

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

REGION_PARAMS = {'Africa': {'D_high': 1588.2035726402378,
            'D_low': 5.59983195169599,
            'P_half': 12.808863354047988,
            'fire_exp': 1.0629462657398439,
            'gpp_af': 0.0689916853868204,
            'gpp_b': 0.0005752434266899163,
            'gpp_d': 431.69073113052116,
            'ign_c': 20.03995359361212,
            'ign_k': 8.708806168038567,
            'k1': 0.01889771394135163,
            'k2': 0.032845299795769424,
            'pre_dampen_half': 552.4163637315892,
            'trop_agb_crit': 7.500118950416987,
            'trop_k_veg': 6.974666381674929},
 'Australia': {'D_high': 1578.0309954142967,
               'D_low': 34.08427910923046,
               'P_half': 4.213183643253957,
               'fire_exp': 1.423480109671474,
               'gpp_af': 91.367149130616,
               'gpp_b': 1.3289448722869181e-05,
               'gpp_d': 84.31013932082456,
               'ign_c': 20.03995359361212,
               'ign_k': 2.545070581573529,
               'k1': 0.01889771394135163,
               'k2': 0.03538758864779238,
               'pre_dampen_half': 107.40052367919465,
               'trop_agb_crit': 8.997007571451054,
               'trop_k_veg': 6.9782110227926815},
 'Boreal': {'D_high': 11832.967057468773,
            'D_low': 355.00125258511594,
            'P_half': 5.076307782729897,
            'fire_exp': 1.1263716762638152,
            'gpp_af': 4.4416173741744,
            'gpp_b': 0.0325181766809307,
            'gpp_d': 2.944272359149677,
            'ign_c': 5.13055176058983,
            'ign_k': 0.21006485619596416,
            'k1': 0.008912611835817005,
            'k2': 0.03538758864779238,
            'pre_dampen_half': 63.6685916079943,
            'trop_agb_crit': 1.0903028125370984,
            'trop_k_veg': 3.1473758868965476},
 'Europe': {'D_high': 52941.32201721563,
            'D_low': 1683.621997588799,
            'P_half': 4.213183643253957,
            'fire_exp': 1.0601964743503993,
            'gpp_af': 56.20900787064881,
            'gpp_b': 2.0194763897972194,
            'gpp_d': 40.950628944766926,
            'ign_c': 1.9580595319750387,
            'ign_k': 0.11619873314872933,
            'k1': 0.004708159793250189,
            'k2': 0.0007047307383115598,
            'pre_dampen_half': 38.00678588560607,
            'trop_agb_crit': 1.0693376406863295,
            'trop_k_veg': 1.9799513595694562},
 'N.America': {'D_high': 3346.220103048382,
               'D_low': 60.42560675552672,
               'P_half': 475.11752699165913,
               'fire_exp': 1.423480109671474,
               'gpp_af': 0.6403036652671171,
               'gpp_b': 0.14577755664212386,
               'gpp_d': 229.41380323183625,
               'ign_c': 18.219343937475102,
               'ign_k': 0.36014695835220345,
               'k1': 0.002155595057104387,
               'k2': 0.003419108473554454,
               'pre_dampen_half': 63.6685916079943,
               'trop_agb_crit': 14.948114979829949,
               'trop_k_veg': 4.983458095432897},
 'S.America': {'D_high': 1236.6924205139144,
               'D_low': 38.82837025280678,
               'P_half': 712.8188058401365,
               'fire_exp': 1.3170256885255098,
               'gpp_af': 1.9706963379095486,
               'gpp_b': 2.176805640408474e-05,
               'gpp_d': 19.600139992927435,
               'ign_c': 1.9580595319750387,
               'ign_k': 0.12329098365270515,
               'k1': 0.002155595057104387,
               'k2': 0.019894080167856886,
               'pre_dampen_half': 63.6685916079943,
               'trop_agb_crit': 10.691477813613638,
               'trop_k_veg': 5.580504276549027},
 'SEAsia': {'D_high': 2556.7821406588646,
            'D_low': 97.43142535650219,
            'P_half': 1.605843632310088,
            'fire_exp': 1.1263716762638152,
            'gpp_af': 1.9706963379095486,
            'gpp_b': 0.14577755664212386,
            'gpp_d': 84.31013932082456,
            'ign_c': 7.4063809263110905,
            'ign_k': 0.11619873314872933,
            'k1': 0.05705198955850472,
            'k2': 0.03538758864779238,
            'pre_dampen_half': 38.00678588560607,
            'trop_agb_crit': 24.199164431661277,
            'trop_k_veg': 3.1473758868965476}}

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
