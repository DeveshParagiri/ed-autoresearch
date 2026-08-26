"""Historical GFED5 Model G checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


INPUTS = ('dryness', 'annual_precipitation', 'monthly_precipitation', 'air_temperature', 'gpp')
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'curing', 'spread')

# Only the new memory coefficients are searched. The seven fitted regional
# parameter sets stay frozen so this experiment tests one added mechanism
# instead of becoming a refit of the whole model.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'cure_alpha': {'type': 'float', 'low': 0.05, 'high': 1.0},
    'cure_half': {'type': 'float', 'low': 1.0, 'high': 300.0, 'log': True},
    'cure_n': {'type': 'float', 'low': 0.3, 'high': 6.0},
    'cure_cap': {'type': 'float', 'low': 1.2, 'high': 12.0},
    'spread_crit': {'type': 'float', 'low': 0.05, 'high': 20.0, 'log': True},
    'spread_k': {'type': 'float', 'low': 1.0, 'high': 12.0},
    'spread_gain': {'type': 'float', 'low': 0.2, 'high': 30.0, 'log': True},
}

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
 'pre_dampen_half': 107.40052367919465,
 'cure_alpha': 0.9022958255196324,
 'cure_half': 101.69797202430976,
 'cure_n': 0.7938582223133299,
 'cure_cap': 4.041069635757127,
 'spread_crit': 1.3740943282898994,
 'spread_k': 7.040273184439577,
 'spread_gain': 3.8622513492967787}

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


def _spread(rate: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """Percolation-style spread multiplier on the instantaneous fire rate.

    Burned area is ignition times how far each fire runs, and spread is not
    linear in flammability. Below a connectivity threshold a fire dies in the
    unburnable gaps between fuel patches; above it neighbouring patches carry
    the front and one ignition clears a whole landscape. That threshold makes
    the response to conditions sharply nonlinear, which is what separates a
    savanna burning a sixth of its area from a marginal cell that only
    smoulders. The model without it computes a smooth product of favourability
    and so paints every mediocre cell with some fire while capping the good
    ones, compressing the observed dynamic range at both ends.
    """
    ratio = np.clip(rate / (p["spread_crit"] + 1e-12), 0.0, None)
    powered = np.power(ratio, p["spread_k"])
    connected = powered / (1.0 + powered)
    factor = 1.0 + p["spread_gain"] * connected
    # Normalise by each cell's own time-mean so the term redistributes burned
    # area between good and marginal months instead of inflating the total.
    return factor / (factor.mean(axis=0, keepdims=True) + 1e-12)


def _antecedent(series: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average of a monthly field over preceding months.

    Fuel moisture and grass curing respond to rainfall integrated over the
    preceding weeks, not to the instantaneous month. ``alpha`` near one means
    almost no memory; small ``alpha`` means a long antecedent window. The
    accumulator is initialised with each cell's own climatology so the result
    stays a pure function of the inputs with no spin-up transient.
    """
    alpha = float(np.clip(alpha, 1e-3, 1.0))
    state = series.mean(axis=0)
    output = np.empty_like(series)
    for step in range(series.shape[0]):
        state = state + alpha * (series[step] - state)
        output[step] = state
    return output


def _curing(
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
) -> np.ndarray:
    """Relative flammability of fuel given antecedent wetness.

    Returns a field whose per-cell time mean is one, so this term redistributes
    burning across the season rather than rescaling the annual total. A cell
    that has been wet for the preceding months is below one, a cell that has
    been drying is above one, which sharpens season onset into a threshold
    instead of the gentle ramp an instantaneous predictor produces.
    """
    wetness = _antecedent(data["monthly_precipitation"], p["cure_alpha"])
    ratio = np.clip(wetness / (p["cure_half"] + 1e-12), 0.0, None)
    flammable = 1.0 / (1.0 + np.power(ratio, p["cure_n"]))
    mean = flammable.mean(axis=0, keepdims=True)
    return np.clip(flammable / (mean + 1e-12), 0.0, p["cure_cap"])


# A fuel-depletion state was tested here and removed. Optuna drove its
# consumption coefficient to the floor of its range and its regrowth rate
# toward one, i.e. it switched the mechanism off: African savannas burn about a
# sixth of their area every year, so depletion strong enough to truncate a fire
# season also silences the annually-reburning grasslands that carry the spatial
# score. Pruning it cost nothing (0.6977 with it, 0.6980 without).


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

    # Build the instantaneous fire rate, splicing the fitted regional parameter
    # sets over the global fallback. The memory terms below are deliberately
    # shared by every region so a single mechanism has to earn its place
    # everywhere rather than being refitted per region.
    rate = _fire_rate(data, fallback, enabled)
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
        regional = _fire_rate(data, region_params, enabled)
        rate[:, mask] = regional[:, mask]
        assigned |= mask

    if "curing" in enabled:
        rate = rate * _curing(data, fallback)
    if "spread" in enabled:
        rate = rate * _spread(rate, fallback)
    prediction = _transform(rate)
    return np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
