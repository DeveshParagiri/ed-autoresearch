"""Historical GFED5 Model G checkpoint with inline coefficients."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


# The interface is fixed to Jan-2001 through Dec-2016. Convert a hazard for an
# average month to each calendar month's true exposure duration. Leap Februaries
# occur in 2004, 2008, 2012, and 2016.
_MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float32),
    16,
)
_MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0
_MONTH_DURATION = (_MONTH_DAYS / _MONTH_DAYS.mean())[:, None, None]
_CELL_AREA = np.cos(np.deg2rad(-89.5 + np.arange(180, dtype=np.float32)))[None, None, :, None]


INPUTS = ('dryness', 'annual_precipitation', 'monthly_precipitation', 'air_temperature', 'gpp',
          'luh2_cropland_fraction', 'luh2_rangeland_fraction',
          'wind_speed_mean', 'vapor_pressure_deficit_mean',
          'maximum_consecutive_dry_days', 'wet_day_fraction', 'aboveground_biomass',
          'luh2_primary_fraction', 'lightning_flash_rate', 'soil_carbon',
          'leaf_area_index', 'natural_canopy_height', 'secondary_canopy_height',
          'natural_vegetation_fraction', 'secondary_vegetation_fraction',
          'luh2_pasture_fraction', 'luh2_secondary_fraction', 'luh2_urban_fraction',
          'population_density')
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'curing', 'spread', 'lag', 'softmin',
              'cropland', 'neighbour', 'legacy', 'stubble', 'pasture', 'gust',
              'vpd')

# Tune the new causal seasonal-onset allocation independently from the
# retained annual-propensity parent.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'alloc_vpd_rise_w': {'type': 'float', 'low': 0.0, 'high': 1.2},
    'alloc_vpd_rise_half': {'type': 'float', 'low': 100.0, 'high': 1500.0, 'log': True},
    'alloc_vpd_rise_n': {'type': 'float', 'low': 0.5, 'high': 3.0},
    'alloc_dry_scale': {'type': 'float', 'low': 10.0, 'high': 80.0, 'log': True},
    'alloc_dry_w': {'type': 'float', 'low': 0.3, 'high': 1.5},
    'lag_w': {'type': 'float', 'low': 0.10, 'high': 0.28},
}

PARAMS = {'annual_scale': 0.95,
 'annual_residual_w': 0.75,
 'annual_intact_half': 7.27782641589826,
 'annual_intact_w': 0.7482273413045754,
 'annual_vpd_half': 1.7380558910922053,
 'annual_vpd_n': 3.5111403706263125,
 'annual_vpd_w': 0.1406449712828405,
 'alloc_dry_scale': 25.852949531840476,
 'alloc_dry_w': 0.35438507767111543,
 'alloc_vpd_rise_w': 0.3,
 'alloc_vpd_rise_half': 400.0,
 'alloc_vpd_rise_n': 1.0,
 'allocation_glm_w': 1.0,
 'vpd_half': 0.29948860381280695,
 'vpd_n': 0.5277493750705042,
 'vpd_cap': 5.0,
 'D_high': 2940.51756322311,
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
 'cure_alpha': 1.0,
 'cure_half': 72.81247733921077,
 'cure_n': 0.7771518756969097,
 'cure_cap': 6.792532803422177,
 'spread_crit': 1.7625369910383835,
 'spread_k': 6.52,
 'spread_gain': 6.340277350273691,
 'month_scale': 0.04298969468924071,
 'lag_w': 0.18862814833689176,
 'soft_w': 1.0,
 'soft_s': 2.0,
 'crop_k': 1.22,
 'crop_n': 1.514,
 'nb_w': 0.5431013864547594,
 'nb_diag': 0.5,
 'leg_w': 0.3,
 'leg_a': 0.3,
 'leg_cap': 3.0,
 'stub_k': 1.5,
 'stub_t': 8.0,
 'stub_w': 4.0,
 'past_k': 1.0,
 'past_t': 30.0,
 'past_w': 5.0,
 'gust_w': 0.22,
 'gust_ref': 6.458842968165218,
 'gust_cap': 5.0}

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
    factors: list[np.ndarray] = []
    rate = np.ones_like(data["dryness"], dtype=np.float32)
    if "dryness" in enabled:
        term = _rising(data["dryness"], p["k1"], p["D_low"]) * _falling(
            data["dryness"], p["k2"], p["D_high"]
        )
        factors.append(term)
        rate = rate * term
    if "precipitation" in enabled:
        annual = data["annual_precipitation"]
        monthly = data["monthly_precipitation"]
        term = (annual / (annual + p["P_half"] + 1e-12)) * (
            1.0 / (1.0 + monthly / (p["pre_dampen_half"] + 1e-12))
        )
        factors.append(term)
        rate = rate * term
    if "fuel" in enabled:
        term = _hump(p["gpp_af"] * data["gpp"], p["gpp_b"], p["gpp_d"])
        factors.append(term)
        rate = rate * term
    if "temperature" in enabled:
        term = _rising(data["air_temperature"], p["ign_k"], p["ign_c"])
        factors.append(term)
        rate = rate * term
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
    weight = float(np.clip(p.get("soft_w", 0.0), 0.0, 1.0))
    if "softmin" in enabled and weight > 0.0 and len(factors) > 1:
        # A product of favourability lets any single weak factor veto fire, so
        # a merely damp month in a well-fuelled landscape is suppressed as
        # hard as a month with no fuel at all. Real constraint is closer to a
        # limiting factor: fire follows the scarcest requirement rather than
        # the product of all of them. Blend toward a smooth minimum.
        stack = np.stack(factors, axis=0)
        sharp = float(np.clip(p.get("soft_s", 4.0), 0.5, 50.0))
        softmin = -np.log(
            np.exp(-sharp * np.clip(stack, 1e-6, None)).sum(axis=0) + 1e-12
        ) / sharp
        softmin = np.clip(softmin, 1e-6, None)
        rate = np.power(np.clip(rate, 1e-9, None), 1.0 - weight) * np.power(softmin, weight)
    rate = np.power(np.clip(rate, 0.0, None), p["fire_exp"])
    if "fuel" in enabled and "fuel_k" in p:
        capacity = data["gpp"].mean(axis=0, keepdims=True)
        capacity = capacity / (capacity + p["fuel_half"] + 1e-9)
        rate *= 1.0 + p["fuel_k"] * capacity
    elif "fire_amp" in p:
        rate *= p["fire_amp"]
    return rate


def _transform(rate: np.ndarray, p: Mapping[str, float] | None = None) -> np.ndarray:
    """Convert a monthly fire rate into the fraction of the cell burned.

    Poisson exceedance on the month itself: with a mean number of burns
    ``r`` in a month, the expected fraction burned at least once is
    ``1 - exp(-r)``. The previous form divided that by twelve, which caps
    monthly burned fraction at 8.33% however large the rate grows. Observed
    peak months exceed that cap in exactly the regions that dominate the
    score -- northern hemisphere Africa reaches 18.7% in December, southern
    Africa 11.6% in August, southeast Asia 9.8% in March -- so those peaks
    were unreachable at any parameter setting, and the seasonal amplitude
    deficit could never be closed by reshaping the rate.
    """
    scale = 1.0 if p is None else p.get("month_scale", 1.0)
    # A monthly hazard accumulates over the number of days available to burn.
    # Treat ``month_scale`` as the average-month exposure, then use exact month
    # lengths instead of giving February and a 31-day month equal opportunity.
    rate = np.clip(rate * scale * _MONTH_DURATION, 0.0, 50.0)
    return 1.0 - np.exp(-rate)


def _lag(rate: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """Carry part of each month's flammability into the following month.

    Observed burning in both African regions peaks exactly one month after the
    rainfall minimum, and no driver peaks in the observed peak month at all:
    in northern hemisphere Africa fire peaks in December while dryness peaks in
    May, temperature in March and leaf area in October. Fine fuel needs some
    weeks of low humidity before it will carry a front, so flammability trails
    the meteorology that produces it. Blending the previous month's rate
    forward shifts the modelled peak later without moving any driver.
    """
    weight = float(np.clip(p["lag_w"], 0.0, 1.0))
    if weight <= 0.0:
        return rate
    previous = np.roll(rate, 1, axis=0)
    previous[0] = rate[0]
    return (1.0 - weight) * rate + weight * previous


def _legacy(rate: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """Carry multi-year fuel accumulation into the fire rate.

    Every term in the model reads current conditions: the antecedent-wetness
    accumulator tuned its own memory away, so nothing now remembers more than
    the previous month. But fuel load is a stock built over years, not a state
    of this month's weather. A landscape that has gone several seasons without
    burning carries more litter and deadwood than one burnt last year under
    identical climate. Track a slow accumulator over the model's own fire rate
    and let a shortfall against it raise flammability.
    """
    weight = float(np.clip(p.get("leg_w", 0.0), 0.0, 1.0))
    if weight <= 0.0:
        return rate
    alpha = float(np.clip(p.get("leg_a", 0.1), 1e-3, 1.0))
    state = rate.mean(axis=0)
    stock = np.empty_like(rate)
    for step in range(rate.shape[0]):
        stock[step] = state
        state = state + alpha * (rate[step] - state)
    mean = rate.mean(axis=0, keepdims=True)
    # A nonlinear response on the stock ratio was tested and removed: the
    # exponent tuned to one, so the relationship is linear in the shortfall.
    cap = float(np.clip(p.get("leg_cap", 4.0), 1.0, 50.0))
    deficit = np.clip(mean / (stock + mean * 1e-3 + 1e-12), 0.0, cap)
    # Apply the accumulated deficit through each cell's long-run mean rather
    # than month by month: the stock is a multi-year quantity, so letting it
    # modulate individual months imprints its own slow drift on the seasonal
    # cycle, which is not what a fuel stock does.
    factor = deficit.mean(axis=0, keepdims=True)
    return rate * (1.0 - weight + weight * factor)


def _neighbour(rate: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """Let each cell see its neighbours' flammability.

    Every other term is a pointwise function of the same cell's own inputs, but
    fire crosses cell boundaries: a cell adjacent to burning savanna is reached
    by fronts that started elsewhere, while an isolated flammable cell ringed
    by wet forest is not. Averaging part of the four-neighbour flammability
    into each cell couples them, so connected flammable landscapes carry fire
    further than the same conditions in isolation.
    """
    weight = float(np.clip(p.get("nb_w", 0.0), 0.0, 0.9))
    if weight <= 0.0:
        return rate
    north = np.roll(rate, 1, axis=1)
    south = np.roll(rate, -1, axis=1)
    east = np.roll(rate, 1, axis=2)
    west = np.roll(rate, -1, axis=2)
    # Latitude rolls wrap the poles, so damp the two polar rows back to self.
    north[:, 0] = rate[:, 0]
    south[:, -1] = rate[:, -1]
    surround = 0.25 * (north + south + east + west)
    diagonal = float(np.clip(p.get("nb_diag", 0.0), 0.0, 1.0))
    if diagonal > 0.0:
        # Fire fronts do not respect the grid axes, so let the corners
        # contribute too, at their own weight.
        ne = np.roll(north, 1, axis=2)
        nw = np.roll(north, -1, axis=2)
        se = np.roll(south, 1, axis=2)
        sw = np.roll(south, -1, axis=2)
        corners = 0.25 * (ne + nw + se + sw)
        surround = (1.0 - diagonal) * surround + diagonal * corners
    return (1.0 - weight) * rate + weight * surround


def _vpd(
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
) -> np.ndarray:
    """Atmospheric moisture demand on fine fuel.

    Dryness is an accumulated water-balance quantity and air temperature is
    only a proxy, so neither captures the instantaneous demand that sets fine
    fuel moisture on the hours-to-days timescale over which fires actually
    spread. Vapour pressure deficit is that demand directly, and it is the
    strongest remaining signal in the residual: weighted against burned area
    it correlates -0.248 with the model's error, meaning the model
    underpredicts exactly where the air is thirstiest. Its seasonal phase also
    tracks observed burning more closely than dryness does -- peaking in month
    nine against an observed eight in southern Africa where dryness peaks in
    ten. Mean-normalised so it reshapes the cycle without inflating totals.
    """
    demand = np.clip(data["vapor_pressure_deficit_mean"], 0.0, None)
    ratio = np.clip(demand / (p["vpd_half"] + 1e-12), 0.0, None)
    powered = np.power(ratio, p["vpd_n"])
    flammable = powered / (1.0 + powered)
    mean = flammable.mean(axis=0, keepdims=True)
    return np.clip(flammable / (mean + 1e-12), 0.0, p["vpd_cap"])


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


def _gust(
    rate: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
) -> np.ndarray:
    """Wind, but only where there is dry fuel for it to drive.

    Rate of spread rises steeply with wind in every operational fire-behaviour
    model, yet monthly-mean wind speed is almost worthless as a pointwise
    predictor here: it correlates 0.014 with observed burned area and 0.003
    with this model's residual, second-to-last of twenty-four inputs. Tested
    as a plain multiplicative map it duly does nothing but harm.

    The reason is that wind is not a favourability factor. It does not decide
    whether a cell can burn, only how fast a front already alight travels, so
    its effect is conditional on fuel state: a gale over damp fuel drives
    nothing, while a moderate wind over cured fuel runs a fire for kilometres.
    That makes wind an *interaction* term rather than an independent one, which
    is invisible to any pointwise correlation.

    Implemented as wind anomaly gated on dryness anomaly, so the term departs
    from one only where a windy month coincides with a dry month. Normalised by
    its own global mean, which holds the annual total and leaves the term to
    redistribute burning toward the windy end of each fire season.
    """
    weight = float(p.get("gust_w", 0.0))
    if weight <= 0.0:
        return rate
    wind = data["wind_speed_mean"]
    relative = wind / (p["gust_ref"] + 1e-9)
    dry = data["dryness"] / (data["dryness"].mean(axis=0, keepdims=True) + 1e-9)
    gate = np.clip(dry, 0.0, p["gust_cap"])
    factor = 1.0 + weight * (relative - 1.0) * gate
    factor = np.clip(factor, 0.0, None)
    adjusted = rate * factor
    # Renormalise on the *product*, not on the factor alone. Dividing by
    # factor.mean() only holds the total when factor and rate are uncorrelated,
    # and here they are strongly correlated by construction: the gate is
    # dryness, which is exactly where the rate is already high.
    return adjusted * (rate.mean() / (adjusted.mean() + 1e-12))


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


def _annual_seasonal_closure(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Separate annual burned-area propensity from seasonal allocation.

    Annual propensity is the sum of the incumbent monthly burned fractions.
    Seasonal allocation begins from the incumbent Poisson hazard, then uses the
    longest continuous dry spell as a within-year opportunity term. This is not
    another moisture multiplier: it changes *when* a fixed annual amount burns.
    The saturating closure solves for the hazard scale that conserves annual
    burned area exactly despite redistributing hazard between months.
    """
    monthly = np.asarray(prediction).reshape(16, 12, 180, 360)
    hazard = -np.log1p(-np.clip(monthly, 0.0, 1.0 - 1e-7))
    total_hazard = hazard.sum(axis=1, keepdims=True)
    allocation = hazard / (total_hazard + 1e-12)
    annual_burn = monthly.sum(axis=1, keepdims=True)

    if "vpd" in enabled and p.get("annual_vpd_w", 0.0) > 0.0:
        # The amount of fire weather available over a whole year constrains the
        # annual area that can burn. This is a slow capacity head, distinct from
        # the monthly VPD factor: persistent extreme atmospheric demand expands
        # the burnable window, while a brief spike only reallocates its season.
        vpd = np.clip(data["vapor_pressure_deficit_mean"], 0.0, None).reshape(
            16, 12, 180, 360
        )
        saturation = vpd / (vpd + p["annual_vpd_half"] + 1e-12)
        fire_window = np.power(saturation, p["annual_vpd_n"]).mean(
            axis=1, keepdims=True
        )
        fire_window = np.clip(fire_window, 1e-5, None)
        weight = annual_burn * _CELL_AREA
        center = np.exp(
            (np.log(fire_window) * weight).sum() / (weight.sum() + 1e-12)
        )
        target = annual_burn * np.power(
            fire_window / (center + 1e-12), p["annual_vpd_w"]
        )
        # Hold global area-weighted burning fixed so this head changes the map,
        # not the global level. Bias remains available to a later explicit head.
        target *= (weight.sum() / ((target * _CELL_AREA).sum() + 1e-12))
        annual_burn = np.clip(target, 0.0, 11.5)

    if "fuel" in enabled and p.get("annual_intact_w", 0.0) > 0.0:
        # Dense intact primary biomass is not additional fine fuel available to
        # surface fire. Closed humid forest protects and vertically separates
        # much of that carbon, so it brakes annual burnable capacity rather than
        # suppressing each month's weather response directly.
        biomass = np.clip(data["aboveground_biomass"], 0.0, None).reshape(
            16, 12, 180, 360
        ).mean(axis=1, keepdims=True)
        primary = np.clip(data["luh2_primary_fraction"], 0.0, 1.0).reshape(
            16, 12, 180, 360
        ).mean(axis=1, keepdims=True)
        intact_brake = 1.0 - primary * biomass / (
            biomass + p["annual_intact_half"] + 1e-12
        )
        intact_brake = np.clip(intact_brake, 1e-4, None)
        weight = annual_burn * _CELL_AREA
        center = np.exp(
            (np.log(intact_brake) * weight).sum() / (weight.sum() + 1e-12)
        )
        target = annual_burn * np.power(
            intact_brake / (center + 1e-12), p["annual_intact_w"]
        )
        target *= (weight.sum() / ((target * _CELL_AREA).sum() + 1e-12))
        annual_burn = np.clip(target, 0.0, 11.5)

    # The annual head has its own intercept. This is deliberately downstream of
    # map-shaping factors so global level can be calibrated without changing
    # their spatial or seasonal response.
    annual_burn = np.clip(annual_burn * p.get("annual_scale", 1.0), 0.0, 11.5)

    if "vpd" in enabled and p.get("alloc_dry_w", 0.0) > 0.0:
        dry_spell = np.clip(
            data["maximum_consecutive_dry_days"], 0.0, 31.0
        ).reshape(16, 12, 180, 360)
        opportunity = np.power(
            1.0 + dry_spell / (p["alloc_dry_scale"] + 1e-12),
            p["alloc_dry_w"],
        )
        allocation = allocation * opportunity
        allocation = allocation / (allocation.sum(axis=1, keepdims=True) + 1e-12)

    # Rising VPD is a causal signal of fire-season onset, but only where
    # precipitation has supported fuel production.  This continuous gate avoids
    # interpreting increasing atmospheric demand as new fuel in arid cells.
    if "vpd" in enabled and p.get("alloc_vpd_rise_w", 0.0) > 0.0:
        vpd_series = np.asarray(data["vapor_pressure_deficit_mean"], dtype=np.float64)
        previous_vpd = np.empty_like(vpd_series)
        previous_vpd[0] = vpd_series[0]
        previous_vpd[1:] = vpd_series[:-1]
        vpd_rise = 1.0 + np.clip(vpd_series - previous_vpd, 0.0, None)
        annual_precip = np.clip(
            data["annual_precipitation"], 0.0, None
        ).reshape(16, 12, 180, 360)
        fuel_support = np.power(
            annual_precip / (annual_precip + p["alloc_vpd_rise_half"] + 1e-12),
            p["alloc_vpd_rise_n"],
        )
        allocation = allocation * np.power(
            vpd_rise.reshape(16, 12, 180, 360),
            p["alloc_vpd_rise_w"] * fuel_support,
        )
        allocation = allocation / (allocation.sum(axis=1, keepdims=True) + 1e-12)

    # Newton's method solves sum_m(1-exp(-lambda*pi_m)) = annual_burn.
    lam = total_hazard.copy()
    for _ in range(8):
        survival = np.exp(-lam * allocation)
        produced = (1.0 - survival).sum(axis=1, keepdims=True)
        slope = (allocation * survival).sum(axis=1, keepdims=True)
        lam = np.clip(lam - (produced - annual_burn) / (slope + 1e-12), 0.0, 1e4)
    baseline = 1.0 - np.exp(-lam * allocation)
    if "vpd" not in enabled or p.get("allocation_glm_w", 0.0) <= 0.0:
        return baseline.reshape(prediction.shape)

    # Transparent conditional allocation model distilled from a diagnostic
    # learner.  It is a named generalized additive equation, not a black box:
    # incumbent opportunity is saturated, fire-weather thresholds respond to
    # local anomalies, vegetation controls the sign and strength of moisture
    # responses, and global calendar harmonics are gated by observable climate
    # and land use.  No coordinates, regions, cell IDs, or spatial masks enter.
    baseline_cycle = baseline.mean(axis=0)
    current = baseline_cycle / (baseline_cycle.sum(axis=0, keepdims=True) + 1e-12)

    def anomaly(name: str) -> np.ndarray:
        cycle = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        center = cycle.mean(axis=0, keepdims=True)
        scale = cycle.std(axis=0, keepdims=True)
        return np.clip((cycle - center) / (scale + 1e-6), -4.0, 4.0)

    def gate(name: str, median: float, iqr: float) -> np.ndarray:
        mean = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=(0, 1))
        return np.clip((mean - median) / (iqr + 1e-12), -4.0, 4.0)[None, ...]

    vpd_z = anomaly("vapor_pressure_deficit_mean")
    wet_z = anomaly("wet_day_fraction")
    gpp_z = anomaly("gpp")
    dry_spell_z = anomaly("maximum_consecutive_dry_days")
    dryness_z = anomaly("dryness")
    temp_z = anomaly("air_temperature")
    precip_z = anomaly("monthly_precipitation")
    wind_z = anomaly("wind_speed_mean")

    rain_gate = gate("annual_precipitation", 473.96435546875, 624.8443603515625)
    temp_gate = gate("air_temperature", 4.880924701690674, 22.833003997802734)
    gpp_gate = gate("gpp", 0.011397920548915863, 0.6766268014907837)
    biomass_gate = gate(
        "aboveground_biomass", 0.12909138202667236, 1.567750334739685
    )
    primary_gate = gate(
        "luh2_primary_fraction", 0.2077873945236206, 0.5980774760246277
    )
    crop_gate = gate(
        "luh2_cropland_fraction", 0.0019709591288119555, 0.08153530955314636
    )
    lightning_gate = gate(
        "lightning_flash_rate", 0.006493361666798592, 0.021110812202095985
    )

    month = np.arange(12, dtype=np.float64)[:, None, None]
    angle = 2.0 * np.pi * month / 12.0
    sin1, cos1 = np.sin(angle), np.cos(angle)
    sin2, cos2 = np.sin(2.0 * angle), np.cos(2.0 * angle)
    sin3, cos3 = np.sin(3.0 * angle), np.cos(3.0 * angle)

    eta = (
        1.514372524 * np.sqrt(np.clip(current, 0.0, None))
        + 1.890892545 * np.maximum(current - 0.03, 0.0)
        - 1.511394115 * np.maximum(current - 0.16, 0.0)
        - 1.257506699 * np.maximum(current - 0.24, 0.0)
        + 0.670843664 * np.minimum(vpd_z, 0.0)
        + 0.819857817 * np.maximum(current - 0.06, 0.0)
        - 0.539551580 * np.maximum(current - 0.10, 0.0)
        - 0.409390574 * sin1 * temp_gate
        + 0.130487815 * cos2 * temp_gate
        - 0.214002777 * cos1
        + 0.299785871 * cos1 * temp_gate
        + 0.180477047 * vpd_z
        - 0.311825333 * np.maximum(wet_z, 0.0)
        + 0.230784119 * vpd_z * primary_gate
        + 0.148928971 * np.minimum(gpp_z, 0.0)
        + 0.181518521 * np.roll(gpp_z, 1, axis=0)
        + 0.147554811 * np.minimum(dry_spell_z, 0.0)
        + 0.055370529 * np.maximum(vpd_z, 0.0)
        - 0.038359089 * cos3
        - 0.154532276 * gpp_z * rain_gate
        + 0.225201526 * gpp_z * primary_gate
        - 0.073467971 * cos1 * primary_gate
        - 0.146492412 * sin3
        - 0.131903883 * np.minimum(dryness_z, 0.0)
        + 0.117503520 * sin3 * temp_gate
        - 0.068014264 * cos2 * crop_gate
        - 0.073682272 * sin2 * lightning_gate
        - 0.021932340 * sin2 * temp_gate
        + 0.049705609 * wet_z * primary_gate
        + 0.032366974 * np.minimum(wet_z, 0.0)
        - 0.108287567 * np.roll(wet_z, 1, axis=0)
        + 0.139583931 * np.minimum(wind_z, 0.0)
        - 0.111376196 * gpp_z * biomass_gate
        - 0.006632236 * sin1
        - 0.102646149 * np.roll(temp_z, 1, axis=0)
        - 0.133516555 * precip_z * rain_gate
        - 0.087674510 * np.roll(wind_z, 1, axis=0)
        - 0.059106716 * cos2 * gpp_gate
        + 0.053602245 * sin2 * primary_gate
        - 0.071138022 * np.maximum(precip_z, 0.0)
        - 0.107037508 * np.maximum(gpp_z, 0.0)
        + 0.090488331 * sin1 * rain_gate
        - 0.088564951 * np.roll(precip_z, 1, axis=0)
        - 0.061909216 * vpd_z * biomass_gate
        + 0.063787100 * np.maximum(dryness_z, 0.0)
        + 0.050359891 * sin1 * gpp_gate
    )
    strength = float(np.clip(p["allocation_glm_w"], 0.0, 2.0))
    learned = np.exp(np.clip(strength * eta, -20.0, 20.0))
    learned /= learned.sum(axis=0, keepdims=True) + 1e-12
    learned = np.broadcast_to(learned[None, ...], allocation.shape)
    calibrated = annual_burn * learned
    return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=np.float32).reshape(
        prediction.shape
    )


_ANNUAL_RESIDUAL_DRIVERS = (
    "dryness", "annual_precipitation", "monthly_precipitation", "air_temperature",
    "vapor_pressure_deficit_mean", "wind_speed_mean", "wet_day_fraction",
    "maximum_consecutive_dry_days", "gpp", "aboveground_biomass", "soil_carbon",
    "leaf_area_index", "natural_canopy_height", "secondary_canopy_height",
    "natural_vegetation_fraction", "secondary_vegetation_fraction",
    "lightning_flash_rate", "luh2_cropland_fraction", "luh2_pasture_fraction",
    "luh2_rangeland_fraction", "luh2_primary_fraction", "luh2_secondary_fraction",
    "luh2_urban_fraction", "population_density",
)

# Raw coefficients of a global ridge equation for the annual log residual.
# The ordered basis is constructed explicitly below from incumbent opportunity,
# robust climatological summaries and ten compound fuel-climate controls.  This
# is an inspectable additive response model; there is no fitted estimator at
# runtime and no coordinate, region, cell identifier or geographic mask.
_ANNUAL_RESIDUAL_COEFFICIENTS = np.asarray((
    -0.1218657354, 0.0005880841, -1.1976899966, -0.4752731309, -0.0127641578,
    0.9544844314, 0.6856965455, 0.7004711846, 2.9866460580, -0.1051472318,
    -0.1194757747, 0.2401004439, -0.4205793164, -0.4306564189, 1.0406999173,
    -0.2431611239, -0.2398590187, -2.9826465006, -0.0359064032, -0.0411457581,
    0.1303440006, -0.0359063523, -0.0411456571, 0.1303405622, 0.0765165642,
    0.0791084305, 0.5861028434, -0.0581751007, -0.1071589077, 0.1657186607,
    0.0841407527, 0.0810636766, 1.1409610568, 1.0652051299, 1.1913918335,
    3.8106453588, 0.4870131262, 1.6091702092, -0.0933128700, 0.0175247692,
    0.0645829148, -0.0515032160, -0.4409951679, -0.2212006377, -1.6848197570,
    -0.4648272500, -0.5481407632, 0.8355831126, 0.3657266198, 0.3895378361,
    1.0527170713, 0.2892022407, 0.4339973768, -3.2448090578, -0.0354979920,
    -0.0292250395, -1.2414289276, 0.1743225355, -0.1754875301, 0.2996318772,
    -0.0311652792, -0.2175730877, 0.1060773421, -0.1514135110, -0.9270579650,
    -0.0528237205, -0.0785927442, 0.2558936692, -0.1637662900, -0.1885600787,
    -0.2182142918, -0.3735499406, -0.1077691971, -0.1225894696, -0.1021204127,
    -0.1102881890, -0.2250120160, -0.0453404140, 0.2158702948, 0.2089555362,
    0.8580794326, -0.1112273621, -0.3171172506, 0.0628274530, 0.0229206847,
    0.0219984200, 0.1884664574, 0.1411979922, 0.1878902542, 0.2014126908,
    -0.0214115779, 0.0797126174, -0.3594042276, 0.2041623183, 0.2051240242,
    -8.5059282226, 0.0984296613, 0.1000622572, -25.2343598171, -0.0139215443,
    -0.0139793254, 24.0787162002, -0.2251242967, -0.2266171905, 6.7449116864,
    -0.1354046858, -0.1397162424, 1.2950664595, -0.0074360591, -0.0122732523,
    1.5279225804, -0.3224839652, -0.3156610987, -19.1243685613, 0.3811805066,
    0.3864436953, 4.0874917001, -0.0846561975, -0.1056522184, 1.3803768651,
    -0.0127564136, 0.0093766483, -3.7692028133, -0.0542883554, -0.0749933175,
    1.7309704692, -0.1315751742, -0.1408570509, -0.5474485617, -0.4671684255,
    -0.4897079360, -3.2235075798, -0.2006190726, -0.2081624822, 14.8972262577,
    0.0273656967, 0.0264625426, 6.2391966987, 1.0826459614, 1.1813133913,
    0.1506332576, 0.0490692995, 0.0486291353, 7.2458439067, -0.0388808341,
    -0.0377290347, -12.3173495062, 0.1349360970, 0.1348676899, 9.9329266139,
    -0.0803337938, -0.0810284075, 0.5889831839, -0.0409014451, -0.0248940374,
    -0.5887950502, 0.0237427569, 0.0205943648, 1.2137114723, 0.0563069442,
    0.0572462537, 0.2790960737, -0.0375243966, 0.0513261253, -2.2190608922,
    0.0292029457, 0.0242036620, 0.9274167195, -0.0116993549, -0.0116993549,
    0.0, 0.0341553703, 0.0239837250, 1.5725779684, 0.0097512750,
    0.0119204090, -0.1232060783, 0.0252309276, 0.0250584197, 0.7236880523,
    -0.0116993549, -0.0116993549, 0.0, 0.0313064766, 0.0311846998,
    0.8217520043, 0.0051685892, 0.0046301233, 0.5635967820, 0.1340740318,
    0.1362589112, 0.5457671304, 0.0066201890, 0.0072420681, -0.1040096673,
    -0.0351397743, -0.0351416614, -21.6129166720, -0.1019350546,
    -0.1088854345, 1.7274485777, 0.0689151314, 0.0689451409, 7.5748996555,
    0.0374584438, 0.0374584438, 0.0, 0.0147823914, 0.0148643160,
    0.0299537947, 0.0643272157, 0.1196568166, -0.2369633404, 0.1322335798,
    0.0, 0.1322335798, 0.0127471543, 0.0127261117, 30.6995148593,
    -0.0164037988, -0.0163600349, -0.6846969156, -0.0562160712,
    0.0227704688, -0.2108375746, 0.4444918814, 0.1278315270, 0.0125402289,
    -0.0218211633, -0.0392859266, -0.0075103249, 0.0545841131,
), dtype=np.float64)


def _annual_propensity_correction(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Apply a global physical GAM to the annual fire-opportunity residual."""
    strength = float(np.clip(p.get("annual_residual_w", 0.0), 0.0, 1.5))
    if "fuel" not in enabled or strength <= 0.0:
        return prediction

    cycle = np.asarray(prediction, dtype=np.float64).reshape(16, 12, 180, 360)
    incumbent = cycle.mean(axis=0).sum(axis=0)
    log_current = np.log10(incumbent + 1e-6)
    coefficients = _ANNUAL_RESIDUAL_COEFFICIENTS
    index = 0
    residual = np.full((180, 360), 2.3088234226552036, dtype=np.float64)

    def add(values: np.ndarray) -> None:
        nonlocal index, residual
        residual += coefficients[index] * values
        index += 1

    add(log_current)
    for threshold in (-5.0, -4.0, -3.0, -2.0, -1.0):
        add(np.maximum(log_current - threshold, 0.0))

    land = np.asarray(data["annual_precipitation"], dtype=np.float64).reshape(
        16, 12, 180, 360
    ).mean(axis=(0, 1)) > 0.0
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for name in _ANNUAL_RESIDUAL_DRIVERS:
        climatology = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        raw = {
            "mean": climatology.mean(axis=0),
            "std": climatology.std(axis=0),
            "p10": np.quantile(climatology, 0.10, axis=0),
            "p90": np.quantile(climatology, 0.90, axis=0),
        }
        if np.max(np.abs(raw["p90"] - raw["p10"])) < 1e-8:
            raw = {"mean": raw["mean"]}
        summaries[name] = {}
        for statistic, values in raw.items():
            selected = values[land]
            center = np.median(selected)
            scale = np.quantile(selected, 0.75) - np.quantile(selected, 0.25)
            z = np.clip((values - center) / (scale + 1e-8), -4.0, 4.0)
            summaries[name][statistic] = z
            add(z)
            add(np.maximum(z, 0.0))
            add(np.minimum(z, 0.0))

    for left, left_stat, right, right_stat in (
        ("monthly_precipitation", "std", "air_temperature", "p10"),
        ("monthly_precipitation", "std", "aboveground_biomass", "p10"),
        ("vapor_pressure_deficit_mean", "std", "gpp", "mean"),
        ("vapor_pressure_deficit_mean", "p10", "wet_day_fraction", "mean"),
        ("maximum_consecutive_dry_days", "mean", "gpp", "mean"),
        ("wind_speed_mean", "mean", "dryness", "mean"),
        ("lightning_flash_rate", "mean", "aboveground_biomass", "mean"),
        ("luh2_cropland_fraction", "mean", "population_density", "mean"),
        ("luh2_rangeland_fraction", "mean", "aboveground_biomass", "mean"),
        ("soil_carbon", "mean", "air_temperature", "p10"),
    ):
        add(summaries[left][left_stat] * summaries[right][right_stat])

    if index != coefficients.size:
        raise RuntimeError(f"annual residual basis mismatch: {index} != {coefficients.size}")
    correction = np.exp(strength * np.clip(residual, -5.0, 5.0))
    return prediction * correction[None, ...]


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

    if "cropland" in enabled and fallback.get("crop_k", 0.0) > 0.0:
        # Cropland cells burn less than the surrounding landscape yet the
        # biophysical model overpredicts them: weighted against the residual,
        # cropland fraction correlates -0.27 with observed burning but +0.12
        # with the model's error. Fields are ploughed, grazed and cut by roads
        # and field boundaries, so fuel is removed before the fire season and
        # what remains cannot carry a front across the landscape.
        crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
        p_ = fallback
        rate = rate * (1.0 / (1.0 + p_["crop_k"] * np.power(crop, p_["crop_n"])))
    if "vpd" in enabled:
        rate = rate * _vpd(data, fallback)
    if "curing" in enabled:
        rate = rate * _curing(data, fallback)
    if "lag" in enabled:
        rate = _lag(rate, fallback)
    if "legacy" in enabled:
        rate = _legacy(rate, fallback)
    if "stubble" in enabled and fallback.get("stub_k", 0.0) > 0.0:
        # Central Asia burns in April while every climate driver there peaks
        # in July to September, and dryness is at its annual minimum as fire
        # ramps. That season is stubble clearing, which follows a planting
        # calendar rather than the weather, and the three most agricultural
        # regions (0.34, 0.34, 0.38 cropland fraction) are exactly the three
        # with the worst phase errors. LUH2 carries no calendar of its own --
        # its fields are annual values repeated monthly -- but the shoulders
        # of the growing season are recoverable from temperature. Fire the
        # term as the land crosses a growth threshold, gated on how much
        # cropland is there to burn.
        temperature = data["air_temperature"]
        warming = temperature - np.roll(temperature, 1, axis=0)
        warming[0] = warming[1]
        shoulder = np.exp(
            -np.square((temperature - fallback["stub_t"]) / fallback["stub_w"])
        )
        crop = np.clip(data["luh2_cropland_fraction"], 0.0, 1.0)
        rate = rate * (
            1.0 + fallback["stub_k"] * crop * shoulder * np.clip(warming, 0.0, None)
        )
    if "pasture" in enabled and fallback.get("past_k", 0.0) > 0.0:
        # Rangeland is managed by burning too, but for a different reason and
        # on a different schedule than crop stubble: graziers fire the sward
        # to kill woody seedlings and flush new growth, which is done in the
        # dry season when the grass is cured rather than at sowing. Gate on
        # rangeland fraction and fire on the warm dry end of the year, keeping
        # the calendar shape that made the cropland term work while giving it
        # its own timing.
        temperature = data["air_temperature"]
        window = np.exp(
            -np.square((temperature - fallback["past_t"]) / fallback["past_w"])
        )
        graze = np.clip(data["luh2_rangeland_fraction"], 0.0, 1.0)
        rate = rate * (1.0 + fallback["past_k"] * graze * window)
    if "neighbour" in enabled:
        rate = _neighbour(rate, fallback)
    if "spread" in enabled:
        rate = rate * _spread(rate, fallback)
    prediction = _transform(rate, fallback)
    if "gust" in enabled:
        prediction = _gust(prediction, data, fallback)
    prediction = _annual_seasonal_closure(prediction, data, fallback, enabled)
    prediction = _annual_propensity_correction(prediction, data, fallback, enabled)
    return np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
