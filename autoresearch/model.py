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
          'aboveground_biomass',
          'luh2_primary_fraction', 'lightning_flash_rate', 'soil_carbon',
          'leaf_area_index', 'natural_canopy_height', 'secondary_canopy_height',
          'natural_vegetation_fraction', 'secondary_vegetation_fraction',
          'luh2_pasture_fraction', 'luh2_secondary_fraction', 'luh2_urban_fraction')
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'curing', 'spread', 'lag', 'softmin',
              'cropland', 'legacy', 'stubble', 'pasture', 'phenology')

# Focus tuning on the independently validated global annual and seasonal heads.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'annual_scale': {'type': 'float', 'low': 0.75, 'high': 1.15},
    'annual_residual_w': {'type': 'float', 'low': 0.35, 'high': 1.20},
    'allocation_glm_w': {'type': 'float', 'low': 0.60, 'high': 1.40},
}

PARAMS = {'annual_scale': 0.9956870515976485,
 'annual_residual_w': 1.0,
 'seasonal_residual_w': 0.0,
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
    learned = _seasonal_residual_allocation(learned, data, p)
    learned = np.broadcast_to(learned[None, ...], allocation.shape)
    calibrated = annual_burn * learned
    return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=np.float32).reshape(
        prediction.shape
    )


_SEASONAL_RESIDUAL_DYNAMIC = (
    "vapor_pressure_deficit_mean", "maximum_consecutive_dry_days",
    "wet_day_fraction", "monthly_precipitation", "dryness", "air_temperature",
    "gpp", "wind_speed_mean", "leaf_area_index", "secondary_vegetation_fraction",
    "secondary_canopy_height", "soil_carbon", "lightning_flash_rate",
)
_SEASONAL_RESIDUAL_STATIC = (
    "annual_precipitation", "vapor_pressure_deficit_mean", "air_temperature", "gpp",
    "aboveground_biomass", "soil_carbon", "leaf_area_index", "luh2_primary_fraction",
    "luh2_secondary_fraction", "luh2_cropland_fraction", "luh2_rangeland_fraction",
    "luh2_pasture_fraction", "population_density", "lightning_flash_rate",
)

_SEASONAL_RESIDUAL_COEFFICIENTS = np.asarray((-0.07990203336556635, 3.242134604073216, 2.8634933430723533, 0.0658262981856942, -2.27635875182494, -1.7169700001514054, -0.13129596071342198, -0.12609098141151243, -0.31358684102740036, -0.06931303876895328, -0.14160871144247725, -0.060296378264695694, -0.1576898038854396, 0.0591661140643654, 0.12648391253407137, -0.02691313896931406, 0.10499344655315448, 0.36320042931654745, 0.15398513163935848, 0.16659168063378985, -0.022825526560874075, -0.0028932268039934396, -0.002259320039426404, 0.010481407435602602, 0.006819746886715434, 0.03298010079686247, -0.004765173960433107, 0.030560670705968327, 0.034317052194804076, -0.16162428131965, -0.1694446876538449, -0.2492647948999544, -0.026245539409162424, -0.0775554436400425, -0.14494173129354332, 0.08423008580629528, 0.017906132230892293, -0.027118485927547793, -0.26987386457023244, 0.15511249301733368, -0.0018372061283945238, -0.022817050046086102, -0.015957448346523954, -0.06372045752072698, 0.005845318451423089, 0.004936120082513647, 0.034038623724069156, -0.19893390062373187, 0.4656832431928163, 1.2922541479628608e-06, -0.02295671236014339, -0.007335735171424197, -0.042079405246242825, -0.028953297867196616, -0.015941208289504214, -0.010086861557110225, 0.2367324872089016, 0.3899915277907384, 0.016540396558443023, -0.02756775078867793, -0.055579981836332434, -0.1694624507379732, 0.08321236031599531, 0.02945974960004772, 0.06213162636790734, 0.2889239327144256, -0.03546332305329366, 0.035254911707503264, 0.018441474505912926, 0.09300494850495689, 0.0016045220544457898, 0.04910703327193489, -0.044169155842000506, -0.0008963376834475447, -0.16307657820254584, 0.957266347230276, 0.014817301528924173, 0.00524437943393447, -0.05416517766192844, -0.004713293332839655, 0.022483169310526904, 0.0012046382883108225, 0.13926372638510412, -0.025025056154903063, -0.014878150786524961, -0.10030496721640642, 0.023602779459218095, 0.00578506647666401, -0.035197692401245345, 0.07840418936410634, 0.05702120369641965, 0.09146698856213405, 0.02263077156327468, 0.40639626200159035, -0.06182016279468036, -0.03758361960797609, -0.22841531211976307, 0.004532053744997846, -0.0923852689462323, -0.039956065668973244, 0.0065971437968424295, 0.46937452636771243, -0.5410858189558192, 0.00274138907678509, 0.011129809249712746, -0.013564862688349036, -0.010425907127702985, 0.07283218567247143, 0.02057269287577306, 0.07496853045501364, -0.14285994690981255, 0.06231313145442922, -0.0015788815103283751, 0.016193369186175244, -0.029894735877002133, -0.0032311647961880937, 0.06191591864397209, 0.021307410118805797, 0.0667598314237617, -0.08542822419778512, 0.3033368805768588, 1.1358065777259574, -0.03787189628965699, 0.0058034411859025515, -0.09494849697008681, 0.004240373942319168, -0.049282688182930745, 0.015265360664598245, -0.03135629953106814, 0.11594118228159192, -0.0492986120555585, -0.11987009251127238, -0.12841468842861958, 0.029278956036937698, 0.09759333460328093, -0.027152385937585642, 0.1116516958610138, -0.08179982375335716, 0.026124462221304906, -0.0015673098138545658, 0.07947146988470923, -0.002660045820500752, 0.027021135638798408, -0.014164251137721821, -0.07405542342689019, -0.03762251832545476, -0.05799193629526454, 0.11658711909947911, 0.014281590134053439, -0.07272660508235049, -0.016885179939301234, 0.031820119777186864, 0.05422352983353572, 0.09156308028922716, -0.03082481826611359, 0.02434900931447252, 0.004205984424908482, -0.06649668230345952, 0.06719912615914352, -0.022958390486213905, -0.0310135620765809, -0.009416063166151999, -0.010498925434066235, 0.06508152542425687, -0.05547312072312401, 0.015661035303793282, -0.015979766760617957, 0.010180958712058683, -0.020834366485256724, 0.15188970392900858, -0.033859066596305616, 0.0020296252699428074, -0.03765884963687213, -0.05262040203463438, -0.03346636638371041, -0.0007725750219351816, -0.0033620880552838643, 0.013742503541428934, 0.014926271159915367, 0.05395802364482446, -0.05437379018623678, 0.013220997511970303, 0.010134634992929461, 0.03028116411134306, 0.06606519043189338, 0.013273080883745856, -0.03932888860233748, -0.07654906757118127, 0.030765526026399755, 0.008081823567163724, 0.02674959582654055, -0.10849653945083088, -0.05549937372978713, 0.01243665094405662, -0.0014830150075095389, 0.09410577754716247, -0.08355934865332468, 0.08635494558822704, 0.04785674316352225, -0.0858377989896738, -0.032078879819519346, -0.023039063000373547, 0.013974930822802156, -0.0038989994832402943, -0.014349786811624812, -0.024228421963097685, -0.04559452151454896, 0.03347125536399747, 0.11119118657238737, 0.05498887524539253, 0.07915309626950202, 0.014000875238097205, 0.01901715125305082, -0.032341230361982075, -0.005431314308453436, -0.012813514646080675, -0.01831261368841179, 0.004629964651600632, 0.00792896460494741, 0.015882805613684103, -0.0480480720069134, -0.01862847536216578, -0.017284469557298652, -0.036262709745521766, -0.04429255518557102, -0.012612234613617986, -0.0266710131426381, -0.015142118321967615, 0.009792300986020339, 0.03304505993391562, -0.09289609157511054, 0.02953367641634228, 0.17410971231239053, -0.16704510864157193, -0.03676455534029922, 0.19257916997129654, -0.07699292471991752, 0.101537686613984, -0.022709129778772975, 0.29784344101981197, 0.3333299927243075, 0.0, -0.03856572739189626, 0.023813337725763477, -0.021124078250365597, 0.00873386414508597, 0.03994915405378336, 0.027493796228019885, -0.004008769309917285, 0.0003206544121000743, -0.05367599404540866, -0.02487060960926591, 0.0, 0.021040920540838857, 0.028570732037251346, -0.023781788359075167, -0.0042175296806926076, -0.039001316158772045, 0.004683469210630478, 0.010141870197484328, -0.014459342417654477, -0.009411488729602032, -0.0006869045417706184, 0.051343337208535234, 0.010224048555469262, 0.06544263643185541, -0.010716529645558477, -0.01985550073997772, 0.06209932402320035, -0.021978780450425648, 0.006077874702508515, 0.002740319903111828, -0.0002868205552222184, 0.03021008427944778, 0.01428646483655127, -0.03319997781587155, 0.03592189435676902, -0.10746954208614534, 0.02298412548076448, -0.004146690384332585, 0.007638502235731568, 0.050157936254798925, 0.010319238430125999, 0.012356904380812826, 0.02334868023045976, 0.03053755333940084, 0.1829558649642977, 0.04932908751104841, 0.028775243731274363, -0.0950761223442097, -0.05203761349141761, 0.019948111720301514, 0.026840224729904726, -0.017508682632458774, 0.0669877608040182, -0.09518807505216262, 0.04279144215775588, -0.02633421005929589, 0.13799088700433487, 0.029725162879946407, 0.012523648471210609, 0.046341709821576024, -0.04122387336608527, 0.012460871270707235, 0.3146294597421509, -0.05371773376375975, 0.03263969589990894, 0.009210704945118437, -0.1072492161090357, -0.03354307436830811, 0.0655523531620051, -0.023805444651357383, -0.010096858142227912, 0.13055294689741753, -0.030617917285871356, -0.011160068117300375, 0.044221266596027944, 0.012200697985611662, -0.00774804632303949, 0.01989145304058308, -0.011879889816703071, 0.03703806941142193, 0.006284855864696169, 0.024820574241411732, -0.04415521569257073, 0.05773464337700747, 0.003203299080807687, -0.00039079072734040277, -0.008956178997431343, 0.0374502414931037, -0.8119670430487131, 0.00329085471785647, 0.5690967422363516, 0.9351383816586746, 0.6167445991705881, -0.908541870540449, 0.015139930494979215, 0.20177876724982405, 0.007643300706172216, -0.16134185445337434, 0.3441300429273325, 0.9901762098095569, 0.016224815707417538, 0.03410334362872034, 0.5448104527738851, 0.3002305331954461, -0.019775883696908924, 0.7269303495259452, -0.540953708968764, -0.3512683360598867, 0.23813268497836856, 0.16307392299334184, 0.7803955581815327, 0.17862308002769167, 0.6003785434261949, 0.26952259019459435, -0.5176859437576029, -0.9285975147209073, -0.62218579475962, 1.3611763917666, -0.9904784899481022, -0.3122724266002435, 0.3873813831387104, 1.0241560933543972, -0.369073737471447, 0.2623478039082246, -0.927465188026166, -0.5899742517342936, 0.18630021967289354, 0.8637040282544185, -0.6392715785611921, 2.2444731068830643, -0.48022912378276844, -0.007710909443854245, 0.015545524898513335, 0.4385585287730025, -0.7685813993986749, -0.38280721460720923, -0.47719270050561424, -0.2349259275385387, -0.04717433381780067, 0.3650584880067144, 0.5387458183439758, -0.42514828510420055, -1.098676778358413, 1.3848684849020174, 3.876252250246477, -3.45048003484173, -1.9661270586712696, 2.727944661916726, -0.18742502385787793, -0.3403638838028526, -0.06988924010199096, 0.6048146410430564, -0.2438884422080445, -0.8523541872415383, -0.7747773964651221, -0.38831329615464344, 0.4111293874913153, 1.249789268748799, -1.484647074916671, 2.339917182092465, 0.6301347226272794, 0.3114017109072404, 0.08512755366097811, -1.118094870002004, -0.5211644804442268, 0.14902665092625197, 0.03127035321482119, 0.06425800680575565, 0.10318831403250496, 0.009187467147152932, 0.09102506553386654, -0.10139159368183599, 0.12374832576901416, 0.02033061113260102, -0.03674560165204078, -0.003966554429158267, -0.25177651371652515, 0.014822023870694714), dtype=np.float64)


def _seasonal_residual_allocation(
    current: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
) -> np.ndarray:
    """Apply a second global physical GAM to the twelve-month fire window."""
    strength = float(np.clip(p.get("seasonal_residual_w", 0.0), 0.0, 2.0))
    if strength <= 0.0:
        return current

    coefficients = _SEASONAL_RESIDUAL_COEFFICIENTS
    index = 0
    eta = np.zeros_like(current, dtype=np.float64)

    def add(values: np.ndarray) -> None:
        nonlocal index, eta
        eta += coefficients[index] * values
        index += 1

    add(np.log(current + 1e-6))
    add(np.sqrt(current))
    for threshold in (0.03, 0.06, 0.10, 0.16, 0.24):
        add(np.maximum(current - threshold, 0.0))

    month = np.arange(12, dtype=np.float64)[:, None, None]
    angle = 2.0 * np.pi * month / 12.0
    harmonics = {
        "sin1": np.sin(angle), "cos1": np.cos(angle),
        "sin2": np.sin(2.0 * angle), "cos2": np.cos(2.0 * angle),
        "sin3": np.sin(3.0 * angle), "cos3": np.cos(3.0 * angle),
    }
    for wave in harmonics.values():
        add(wave)

    cycles: dict[str, np.ndarray] = {}
    anomalies: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(_SEASONAL_RESIDUAL_DYNAMIC + _SEASONAL_RESIDUAL_STATIC):
        cycle = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        mean = cycle.mean(axis=0)
        cycles[name] = cycle
        means[name] = mean
        anomalies[name] = np.clip(
            (cycle - mean[None, ...]) / (cycle.std(axis=0)[None, ...] + 1e-6),
            -4.0,
            4.0,
        )

    for name in _SEASONAL_RESIDUAL_DYNAMIC:
        value = anomalies[name]
        add(cycles[name])
        add(value)
        add(np.roll(value, 1, axis=0))
        add(np.maximum(value, 0.0))
        add(np.minimum(value, 0.0))
        for threshold in (-1.5, -0.75, 0.75, 1.5):
            add(np.maximum(value - threshold, 0.0))

    land = means["annual_precipitation"] > 0.0
    gates: dict[str, np.ndarray] = {}
    for name in _SEASONAL_RESIDUAL_STATIC:
        selected = means[name][land]
        center = np.median(selected)
        scale = np.quantile(selected, 0.75) - np.quantile(selected, 0.25)
        gate = np.clip((means[name] - center) / (scale + 1e-6), -4.0, 4.0)[None, ...]
        gates[name] = gate
        add(gate)

    interaction_drivers = (
        "vapor_pressure_deficit_mean", "maximum_consecutive_dry_days",
        "wet_day_fraction", "monthly_precipitation", "air_temperature", "gpp",
        "wind_speed_mean", "leaf_area_index", "secondary_vegetation_fraction",
        "secondary_canopy_height", "soil_carbon", "lightning_flash_rate",
    )
    interaction_gates = (
        "annual_precipitation", "gpp", "aboveground_biomass", "luh2_primary_fraction",
        "luh2_cropland_fraction", "luh2_rangeland_fraction", "population_density",
        "soil_carbon", "leaf_area_index", "luh2_secondary_fraction",
        "lightning_flash_rate",
    )
    for driver in interaction_drivers:
        for gate in interaction_gates:
            add(anomalies[driver] * gates[gate])

    calendar_gates = (
        "annual_precipitation", "air_temperature", "gpp", "aboveground_biomass",
        "luh2_primary_fraction", "luh2_cropland_fraction", "luh2_rangeland_fraction",
        "population_density", "lightning_flash_rate",
    )
    for wave in harmonics.values():
        for gate in calendar_gates:
            add(wave * gates[gate])

    for driver in _SEASONAL_RESIDUAL_DYNAMIC:
        for threshold in (0.02, 0.05, 0.08, 0.13, 0.20, 0.30):
            add(anomalies[driver] * np.maximum(current - threshold, 0.0))

    for left, right in (
        ("vapor_pressure_deficit_mean", "gpp"),
        ("vapor_pressure_deficit_mean", "wet_day_fraction"),
        ("vapor_pressure_deficit_mean", "monthly_precipitation"),
        ("vapor_pressure_deficit_mean", "leaf_area_index"),
        ("maximum_consecutive_dry_days", "gpp"),
        ("maximum_consecutive_dry_days", "wet_day_fraction"),
        ("dryness", "gpp"),
        ("air_temperature", "wet_day_fraction"),
        ("wind_speed_mean", "dryness"),
        ("lightning_flash_rate", "dryness"),
        ("secondary_vegetation_fraction", "dryness"),
        ("secondary_canopy_height", "dryness"),
    ):
        add(anomalies[left] * anomalies[right])

    if index != coefficients.size:
        raise RuntimeError(f"seasonal residual basis mismatch: {index} != {coefficients.size}")
    learned = np.exp(np.clip(strength * eta, -30.0, 30.0))
    return learned / (learned.sum(axis=0, keepdims=True) + 1e-12)


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
_ANNUAL_STATIC_DRIVERS = {
    "annual_precipitation", "luh2_cropland_fraction", "luh2_pasture_fraction",
    "luh2_rangeland_fraction", "luh2_primary_fraction",
    "luh2_secondary_fraction", "luh2_urban_fraction", "population_density",
}

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
        if name in _ANNUAL_STATIC_DRIVERS:
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



_COUPLED_ANNUAL_DRIVERS = (
    "dryness", "annual_precipitation", "monthly_precipitation", "air_temperature",
    "gpp", "aboveground_biomass", "soil_carbon", "leaf_area_index",
    "natural_canopy_height", "secondary_canopy_height",
    "natural_vegetation_fraction", "secondary_vegetation_fraction",
    "lightning_flash_rate", "luh2_cropland_fraction", "luh2_pasture_fraction",
    "luh2_rangeland_fraction", "luh2_primary_fraction",
    "luh2_secondary_fraction", "luh2_urban_fraction",
)
_COUPLED_ANNUAL_STATIC = {
    "annual_precipitation", "luh2_cropland_fraction", "luh2_pasture_fraction",
    "luh2_rangeland_fraction", "luh2_primary_fraction",
    "luh2_secondary_fraction", "luh2_urban_fraction",
}
_COUPLED_ANNUAL_COEFFICIENTS = np.asarray((
    0.246466443790, -0.476229779858, -1.12735543323, -0.614900384368, 0.236850964152,
    0.705664252899, 0.551356794819, 0.562623480246, 2.70541633389, -0.0356181519258,
    -0.0515023670421, 0.798927073386, -0.382637461780, -0.393110168624, 3.11034646116,
    -0.125740208603, -0.114968084928, -3.49366140604, -0.0347577139745, -0.0431263755626,
    0.378344059089, -0.0347591372986, -0.0431278413778, 0.378336606268, 0.0853733882942,
    0.0875034972308, 0.795458101888, -0.0823549173879, -0.118235571840, -0.0239173560126,
    0.0415858811461, 0.0422111627846, 0.336989299334, 0.340683916759, 0.263744335258,
    2.66531125649, 0.786680752966, 3.00449675940, -0.653198813315, 0.453462353061,
    0.402325754141, 1.53481715048, -0.357793617062, -0.168984134716, -1.39897169135,
    0.321028913289, 0.321879167262, 18.0655306923, 0.0584571169751, 0.0596786792172,
    -20.9261659070, -0.0230997114738, -0.0232028338185, 44.0431542047, -0.296041297718,
    -0.296199548082, -39.5270847350, -0.184376453806, -0.189530437298, 0.986669017964,
    -0.0218742314155, -0.0311291678392, 2.76294177176, -0.291948041706, -0.283968500227,
    -19.9225463389, 0.416428264461, 0.421393924434, 5.10792918990, -0.103133580626,
    -0.120768119142, 0.766308079452, -0.00902423794335, 0.00556162225860, -2.49312772501,
    -0.0694296655280, -0.0884576800587, 1.35072183008, -0.137172961897, -0.148602019280,
    -0.368407982247, -0.522221148313, -0.555745406986, -1.99178493305, -0.251948485494,
    -0.259959861804, 13.7518049266, -0.00653254520243, -0.00762227761722, 5.10569628694,
    1.22805481084, 1.34869982621, -1.45294709037, 0.0378929722022, 0.0370908078669,
    9.68556395402, -0.0454671524383, -0.0441131716953, -14.4605972287, 0.160079276284,
    0.159981417558, 11.9352201230, -0.104027476346, -0.105099027013, 2.25808075995,
    -0.0841966483469, -0.116461857297, 0.301309262132, 0.0147884780591, 0.0240642653776,
    -2.25697496000, 0.306683299865, 0.335157208092, 0.965283275497, -0.175562201380,
    -0.177950025367, -0.871819711935, 0.0234026741133, 0.0197991564994, 0.697765640448,
    0.00225685369651, 0.00225685369652, 0.00000000000, 0.0354013905251, 0.0276541059422,
    1.31456059049, -0.00982120299406, -0.00976582448346, -0.128392089138, 0.0269095386431,
    0.0272848583300, 0.310139693491, 0.00225685367546, 0.00225685367546, 0.00000000000,
    0.0245951732305, 0.0246227168835, 0.543880139949, 0.0122721827052, 0.0127246942982,
    -0.0908107212718, 0.193994347579, 0.197297790885, 0.637951581846, -0.0180364355387,
    -0.0201241312395, 0.397421406060, -0.0191884493423, -0.0191771587093, -17.2621802489,
    -0.126414153802, -0.133768245038, 1.54695839014, 0.0187483727572, 0.0184009064023,
    16.8029280366, 0.0415971101238, 0.0415971101237, 0.00000000000, 0.0122019025583,
    0.0123387344856, -1.37553105753, 0.0979919919393, 0.148406343956, -0.0217526464619,
    0.199880689685, 0.00000000000, 0.199880689685, -0.0273768893976, -0.0274324466305,
    33.2317022141, 0.0238076718413, 0.0202931721517, 0.0139503628502, 0.237521714860,
    0.343244654934, 0.0260771729186, -0.0271540192792, -0.233544393025, -0.0324490496535,
    0.0526063848952,
), dtype=np.float64)

# Refit after removing the legacy continent-specific parameter dispatch. The
# basis remains one globally shared additive equation; these values are stable
# across five held-out spatial folds (coefficient r=.970-.977).
_COUPLED_ANNUAL_COEFFICIENTS = np.asarray((
    -0.31932671242450217, 0.08952530803475109, -0.413278377228983, -1.586237274460084, 0.6922107772094496,
    0.261480299449787, 0.8219929825628127, 0.8460730321647055, 0.4247267216432345, -0.02424929886019653,
    -0.04256486585999995, 1.0319266740601862, -0.5923265321030894, -0.606344605469859, 1.1769721247608047,
    -0.13770203631750577, -0.13359114117957896, -2.171468727664154, -0.056332312030407165, -0.07205626280522434,
    0.7784553383520372, -0.05633319257302562, -0.07205704504069206, 0.7784411260988507, 0.070923391835924,
    0.06953258165207675, 1.2479269958972778, -0.08227160526112202, -0.10249415969212988, -0.14457888161876598,
    0.0583274247717149, 0.07331208445572696, -1.019041431285576, 0.5392050157524981, 0.5180162906119617,
    2.977981955654735, 0.5326584433945579, 2.4019872886435927, -0.8982154974735937, 0.17929011736008318,
    0.23289612777003732, 0.4399920374121005, -0.5573868655463768, -0.8057109721946366, -0.5223097665535738,
    0.008226273718053904, 0.008492503192947665, -11.147207082390647, 0.00838372833871936, 0.007716095714853196,
    16.866458860935168, -0.015695137889600145, -0.015854090304440377, 80.08458048940068, 0.02418173110620863,
    0.02550649125644104, -31.94641849170439, -0.23202177646085168, -0.24295961465589433, 6.065296584122872,
    -0.04259556712489366, -0.06100013760623689, 5.51341024395075, -0.27428233666037494, -0.2617644066750515,
    -25.984171697724726, 0.3916040212529985, 0.3931499566247459, 7.36252693600513, -0.0672077166614764,
    -0.06882897104623337, -0.6379362606783928, 0.009436093236177194, 0.021067835991159495, -1.7406127919708434,
    -0.10780320273079332, -0.11545355816375096, -0.4384659629799827, -0.13485850351567386, -0.14659189584827115,
    -0.30478499308135487, -0.6958999010381689, -0.7292471867465462, -4.845865951211454, -0.3437567478802338,
    -0.3530660314181344, 13.265723939148554, 0.013723292434012869, 0.012626965279868606, 6.379631580820515,
    1.5057429092215417, 1.6353603810739106, 1.625519116787394, 0.0252018893233003, 0.024185210651695724,
    10.716861226270458, -0.0349822589492155, -0.032384583750134664, -23.609533659426027, 0.15107674833894594,
    0.15080643807980737, 12.877358078189202, -0.10011314958799784, -0.10173440309477329, 7.300803190849755,
    -0.03824380806663466, -0.03931307818998975, -0.17840817135796677, 0.03594475645554805, 0.04169152173235356,
    -0.9814594966408959, 0.268272307723344, 0.2673792017122005, 1.4572659577414664, -0.16258809349025716,
    -0.12504939684572228, -1.7115986470168651, 0.03483276528673992, 0.03534778847676907, 0.3753017658829033,
    -0.034963950820113346, -0.03496395082006064, 0.0, 0.03848772816388148, 0.03544889813491797,
    0.8217458414238888, 0.013848007732058607, 0.015723078796229042, -0.03910907297233093, 0.017619694975114703,
    0.017304922270406434, 0.6657725702372687, -0.0349639505480945, -0.03496395054809449, 0.0,
    0.019960649029052983, 0.01947387278520729, 0.861664103831092, 0.010906876073055181, 0.010771609140678064,
    0.3629560053671122, 0.24745853300005047, 0.2537173312990147, -1.3697685423457289, -0.009586404664831034,
    -0.013697236797064605, 1.0812933707333041, -0.02876040420937545, -0.02876794409076001, -15.032501354299626,
    -0.13674638921872695, -0.14376995457397307, 1.2351045569063257, -0.010874360358267658, -0.011141458122628864,
    9.680609620436552, 0.06450995462322037, 0.06450995462322168, 0.0, -0.020800438757601583,
    -0.02083604438276001, -1.6540659239984306, -0.009829929205182877, -0.024329289966291415, 0.09674535593555834,
    0.2303874821697769, 0.0, 0.23038748216986035, -0.04016677004403571, -0.040210764410735844,
    11.829105668036306, -0.08118413683732965, 0.035796436558910356, 0.01214100686326596, 0.2718443992685613,
    -0.020420866598933343, 0.043054403240902504, 0.0020420848865441744, -0.27548179041635795, -0.089329776090473,
    -0.059447801872449996,
), dtype=np.float64)


def _coupled_land(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Derive the local land domain without coordinates or neighbour state."""
    return (
        (np.asarray(data["annual_precipitation"]).max(axis=0) > 0.0)
        | (np.asarray(data["air_temperature"]).max(axis=0) != 0.0)
        | (np.asarray(data["natural_vegetation_fraction"]).max(axis=0) > 0.0)
        | (np.asarray(data["secondary_vegetation_fraction"]).max(axis=0) > 0.0)
    )


def _coupled_annual_correction(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Calibrate annual fire opportunity using only 1850-compatible local state."""
    strength = float(np.clip(p.get("annual_residual_w", 0.0), 0.0, 1.5))
    if "fuel" not in enabled or strength <= 0.0:
        return prediction

    cycle = np.asarray(prediction, dtype=np.float64).reshape(16, 12, 180, 360)
    incumbent = cycle.mean(axis=0).sum(axis=0)
    log_current = np.log10(incumbent + 1e-6)
    coefficients = _COUPLED_ANNUAL_COEFFICIENTS
    index = 0
    residual = np.full((180, 360), 2.509046582748338, dtype=np.float64)

    def add(values: np.ndarray) -> None:
        nonlocal index, residual
        residual += coefficients[index] * values
        index += 1

    add(log_current)
    for threshold in (-5.0, -4.0, -3.0, -2.0, -1.0):
        add(np.maximum(log_current - threshold, 0.0))

    land = _coupled_land(data)
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for name in _COUPLED_ANNUAL_DRIVERS:
        climatology = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        raw = {
            "mean": climatology.mean(axis=0),
            "std": climatology.std(axis=0),
            "p10": np.quantile(climatology, 0.10, axis=0),
            "p90": np.quantile(climatology, 0.90, axis=0),
        }
        if name in _COUPLED_ANNUAL_STATIC:
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
        ("monthly_precipitation", "mean", "gpp", "mean"),
        ("monthly_precipitation", "p10", "dryness", "mean"),
        ("air_temperature", "std", "gpp", "mean"),
        ("leaf_area_index", "std", "dryness", "mean"),
        ("lightning_flash_rate", "mean", "aboveground_biomass", "mean"),
        ("luh2_cropland_fraction", "mean", "luh2_secondary_fraction", "mean"),
        ("luh2_rangeland_fraction", "mean", "aboveground_biomass", "mean"),
        ("soil_carbon", "mean", "air_temperature", "p10"),
    ):
        add(summaries[left][left_stat] * summaries[right][right_stat])

    if index != coefficients.size:
        raise RuntimeError(
            f"coupled annual basis mismatch: {index} != {coefficients.size}"
        )
    correction = np.exp(strength * np.clip(residual, -5.0, 5.0))
    return prediction * correction[None, ...]


_COUPLED_SEASONAL_DYNAMIC = (
    "monthly_precipitation", "dryness", "air_temperature", "gpp",
    "leaf_area_index", "secondary_vegetation_fraction",
    "secondary_canopy_height", "soil_carbon", "lightning_flash_rate",
    "aboveground_biomass", "natural_canopy_height",
)
_COUPLED_SEASONAL_STATIC = (
    "annual_precipitation", "air_temperature", "gpp", "aboveground_biomass",
    "soil_carbon", "leaf_area_index", "luh2_primary_fraction",
    "luh2_secondary_fraction", "luh2_cropland_fraction",
    "luh2_rangeland_fraction", "luh2_pasture_fraction", "lightning_flash_rate",
)
_COUPLED_SEASONAL_COEFFICIENTS = np.asarray((
    -0.0428385121232, 1.38365053008, 1.96953995191, 1.40257814507, 0.446958437798,
    -1.79688575897, -1.37961053551, -0.181419433822, -0.433213290461, 0.0114823906180,
    -0.111372506157, -0.139240814516, -0.207467236723, -0.00378806040935, -0.0786747355629,
    -0.179496338312, -0.154570993215, -0.0861336037050, -0.0497796661818, -0.0414686077139,
    -0.0861325412265, 0.296531440349, 0.00000642089218347, -0.0220108193530, -0.0811222475667,
    -0.0119477469485, -0.0721455869581, 0.0204351730741, 0.0575388839364, 0.153449308279,
    0.211395740692, 0.0337912576093, 0.100278802223, -0.143780428593, -0.0291023654181,
    0.339832334890, 0.146534324808, 0.0906370296756, 0.350646236331, -0.197676745798,
    0.0324675120239, -0.0652066445585, 0.148851355365, -0.211539300383, -0.0251872499669,
    -0.0727355607054, 0.0291098259675, -0.127327724455, 1.03784134877, -0.0493529130641,
    0.0128201728136, 0.0270013369633, -0.0784370678921, 0.0784202123882, 0.00587718339113,
    0.0652234014936, 0.0189749246124, 0.465850942064, -0.121359511086, -0.000102625502488,
    -0.166657705545, -0.105958878584, 0.125095735020, -0.0392765891141, 0.366922677330,
    0.502925985272, -0.345905532766, -0.00755165830078, -0.0129985363756, 0.0275317638276,
    -0.0297676508073, 0.0125510967164, -0.0128118096881, 0.0734103755367, -0.168413872362,
    0.0499196410292, -0.00492161568662, 0.00391728223845, -0.0464025194773, -0.00848175994431,
    0.0251070017791, 0.0301762724190, 0.0814080418567, -0.00965768262580, 0.177851891619,
    1.23028686940, -0.0350769059969, -0.0445929378951, -0.0706129917293, -0.0286101850477,
    -0.0401837440932, 0.0807900308191, -0.160848047187, 0.123251802003, 0.0248598844482,
    -0.00760530284562, 0.150989898585, -0.0695769073883, 0.0339439994149, -0.00816275988934,
    0.132308225955, 0.0293995814963, -0.100794292522, 0.0136652442212, 0.0127833526174,
    0.0179808293669, -0.0142772431037, 0.0530972457424, 0.0116424799269, -0.0566134135117,
    -0.0450368635098, 0.0945491803170, -0.0757612101236, -0.586981182461, 0.0455568021763,
    0.0455860806688, -0.0552577296680, 0.110676289981, -0.168463324674, 0.0684003637253,
    0.00742059389494, 0.0205137934297, -0.00816717249145, -0.0566733395647, 0.00459979813359,
    -0.0157736073924, 0.0670109946139, 0.0522430801635, 0.0137846134557, 0.0436782223507,
    0.0124992482976, 0.00504298334222, -0.00402886986335, -0.0899633830278, 0.000988361587326,
    -0.0380417783682, 0.00299874692816, 0.0541477670063, -0.0217561574868, -0.0215955970895,
    0.160938430268, -0.0354282758631, 0.143618259091, -0.00157203842784, -0.0902052260174,
    -0.0375754655851, 0.0212559991948, -0.0672029434904, -0.0334118061835, 0.0231986309269,
    0.0137954186482, 0.0165381911307, 0.100971436879, 0.0493340072561, -0.0180063356376,
    -0.0179948051907, -0.0357251161809, 0.000976117071339, -0.00643956929433, -0.0253421811903,
    0.0296288816874, -0.000215408448896, -0.0716673723744, 0.0333610568263, 0.226897408720,
    -0.237296195996, -0.157855255410, 0.213197934205, -0.166893805452, 0.0563228480357,
    0.185590457949, 0.509468539335, 0.00000000000, -0.0230634858376, 0.0159070326951,
    -0.0160026182243, 0.0348763505911, 0.0282949155983, 0.0250314845406, -0.00456055082362,
    -0.0427448858095, -0.0256485191511, 0.00000000000, 0.00944565815409, 0.0122854069459,
    -0.0243881889774, -0.0356282486620, -0.0103644931816, -0.00557022209055, 0.00826784663674,
    0.0158072381890, 0.0417354425773, 0.128576092710, -0.0112113269774, 0.0310163252269,
    -0.0136761859344, -0.0219203823935, 0.0859390314445, -0.0365206115369, -0.000558218219707,
    0.0199782599970, -0.00946650546907, 0.0153466741821, -0.0186472034329, 0.101793785896,
    -0.350199093642, 0.0454136240907, 0.00480977630119, -0.0127732479738, 0.0656087195286,
    0.0299836658093, 0.0342720884838, 0.0128955217170, 0.403630559123, 0.0892338813901,
    0.0286601194026, -0.179465299287, -0.0547899995888, -0.00849080983221, -0.00900831533066,
    0.0507731573816, -0.00910829371446, 0.0431202022446, 0.00471133122216, 0.0865627294285,
    0.0331886615967, -0.00718709439391, -0.101367535607, -0.00438709705185, 0.375393699936,
    -0.0722012977652, 0.0566304025608, 0.0303401220518, -0.0783281486041, -0.0598449833478,
    -0.0407406002338, -0.0120925862197, 0.177529298677, -0.0141443714162, -0.0126776802509,
    0.0836513157120, 0.0267552802878, -0.00881799630622, -0.0218124859650, 0.0272081569253,
    0.0233487087115, 0.0237867605969, -0.0256214408953, 0.0568874124138, 0.0206756546093,
    0.00518186152927, 0.0424116269840, 0.207563980742, -0.0151226494927, -0.273203532456,
    0.919668651157, 0.551583920893, -1.19556601681, 0.253603754867, 0.259498944071,
    -0.358374276447, -0.362983175855, 0.792516504467, 0.0869026858848, -1.33892755715,
    -1.26827682142, -0.913884464340, 1.77006333936, 1.76839584104, 1.94974763410,
    -0.863343769206, -0.621981701677, 0.703208084422, 0.616151845264, 0.211498734938,
    0.405695441396, -0.195115744344, -0.198728675362, 0.167651680242, -0.207855693880,
    0.0214519339552, -0.524964989221, -0.442125440794, 3.13214888925, -2.86423817185,
    -0.944816971198, -0.978556725232, 1.79797105718, 0.120094387033, 0.171894311343,
    0.285724385719, -0.162297537716, -0.214562998602, -0.852066460949, 0.296431205801,
    0.220824981057, -0.308277656958, -0.862639332011, 0.836871460849, -1.71955074732,
    1.07917242601, 0.329123134314, -0.679364932887, 0.0173213459765, -0.979420314205,
    -0.476530559980, 0.152313845607, 0.0946467725901, -0.209373235801, -0.0235071974527,
    0.0231779497793, -0.250689237026, 0.0102830404498, 0.461793869502, 0.0258733519551,
    -0.177824567613, 0.00170981992713, -0.428259701768, -0.0223402996142, -0.142219939891,
    -0.0165316144346, 0.411009848328, -0.0139053075142, 0.0527764248981, -0.0135747956284,
    0.600486480518, -0.0000541489830733, 0.142176870314, -0.000129112778248, -0.954583652603,
    -0.00000326677226337, -0.379163054527, 0.000198379421249, -0.109286324039, -0.000199648427434,
    0.0538890805901, -0.000194344716621, 0.0236630174506, 0.0103263349199, 0.256136691473,
    0.0159239393502, 0.0161855129533, 0.0167794318013, 1.05762772189, -0.0509386539288,
    -1.41142120791, -0.00666283574313, -0.738229032478, -0.0212242022879, -0.0894042979686,
    -0.300962668704, -0.306676758849, -0.364285548133, 1.17336566541, 0.0310062760238,
    0.0321250605813, 0.511142560099, -1.23491785798, 0.0682606668933, -0.577942729197,
    0.789314779973, -0.878149243887, -0.0752831497594, -0.626334407101, -0.0588297647056,
    0.550267978766, 0.0642383234327, 1.05149722852, 0.141080739683, 1.34969392457,
    0.157971672900, -0.469810671326, 0.126880108565, -0.150653590679, -0.157821115527,
    -0.424925771635, -0.174583218714, -0.0190405955842, -0.120801816721, -0.306048963701,
    0.305796575116, 0.155649735132, 0.295012635714, 0.808644915463, -0.0949902896040,
    0.266845004430, 0.0597324461202, -0.0806861774101, 0.0000496949315639, -0.656013552737,
    -0.000663678419393, 0.0569704131430, 0.0204176047378, 1.34172213843, -0.266256780479,
    1.06029359801, 0.0855425815932, 0.205327743419, 0.0234603762187, -0.186435531108,
    0.0218150875085, -0.494702283381, 0.0592718724927, 0.0197124245364, 0.0317731569204,
    0.656168422313, -0.154337702356, -0.230296035844, -0.107501986491, -0.130510988832,
    -0.0362871778135, -0.0334518465335, -0.0525638846913, 0.380246569236, -0.0285616398989,
    0.264863083235, 0.0541921850371, -0.965228215258, 0.0924217519647, 1.41049489475,
    -0.0160387551250, -0.0929217802727, 0.0294088955021, -0.0114312950518, -0.0902388660695,
    0.0106069442857, -0.0713267626927, -0.0141272819021, -0.000688155072119, -0.0116204665441,
    -0.0875053131479, 0.143743152316, 0.0423444439493, -0.126289103280, 0.110554575080,
    -0.0278300179775, 0.0290449314162, -0.00266586338876, 0.0945082197645, -0.0278366989439,
    0.0577262723149, -0.00544270180655, -0.00495350153766, 0.00104901970560, -0.0598117624333,
    -0.0150366014437, 0.00885076670540, -0.0125339362175, 0.000338936083677, 0.0499956652058,
    0.0992502630785, -0.00752990639337, -0.0395793706572, -0.0189622305570, -0.0535576104386,
    0.0123123520276, -0.0364548919974, -0.00233765803371,
), dtype=np.float64)


def _coupled_seasonal_allocation(
    current: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Allocate fire seasonally from pointwise weather and vegetation phenology."""
    strength = float(np.clip(p.get("allocation_glm_w", 0.0), 0.0, 2.0))
    if "phenology" not in enabled or strength <= 0.0:
        return current

    coefficients = _COUPLED_SEASONAL_COEFFICIENTS
    index = 0
    eta = np.zeros_like(current, dtype=np.float64)

    def add(values: np.ndarray) -> None:
        nonlocal index, eta
        eta += coefficients[index] * values
        index += 1

    add(np.log(current + 1e-6))
    add(np.sqrt(current))
    for threshold in (0.03, 0.06, 0.10, 0.16, 0.24):
        add(np.maximum(current - threshold, 0.0))

    month = np.arange(12, dtype=np.float64)[:, None, None]
    angle = 2.0 * np.pi * month / 12.0
    harmonics = {
        "sin1": np.sin(angle), "cos1": np.cos(angle),
        "sin2": np.sin(2.0 * angle), "cos2": np.cos(2.0 * angle),
        "sin3": np.sin(3.0 * angle), "cos3": np.cos(3.0 * angle),
    }
    for wave in harmonics.values():
        add(wave)

    cycles: dict[str, np.ndarray] = {}
    anomalies: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(
        _COUPLED_SEASONAL_DYNAMIC + _COUPLED_SEASONAL_STATIC
    ):
        cycle = np.asarray(data[name], dtype=np.float64).reshape(
            16, 12, 180, 360
        ).mean(axis=0)
        mean = cycle.mean(axis=0)
        cycles[name] = cycle
        means[name] = mean
        anomalies[name] = np.clip(
            (cycle - mean[None, ...]) / (cycle.std(axis=0)[None, ...] + 1e-6),
            -4.0,
            4.0,
        )

    previous: dict[str, np.ndarray] = {}
    for name in _COUPLED_SEASONAL_DYNAMIC:
        value = anomalies[name]
        previous[name] = np.roll(value, 1, axis=0)
        add(cycles[name])
        add(value)
        add(previous[name])
        add(np.maximum(value, 0.0))
        add(np.minimum(value, 0.0))
        for threshold in (-1.5, -0.75, 0.75, 1.5):
            add(np.maximum(value - threshold, 0.0))

    land = _coupled_land(data)
    gates: dict[str, np.ndarray] = {}
    for name in _COUPLED_SEASONAL_STATIC:
        selected = means[name][land]
        center = np.median(selected)
        scale = np.quantile(selected, 0.75) - np.quantile(selected, 0.25)
        gate = np.clip((means[name] - center) / (scale + 1e-6), -4.0, 4.0)[
            None, ...
        ]
        gates[name] = gate
        add(gate)

    for driver in (
        "monthly_precipitation", "air_temperature", "gpp", "leaf_area_index",
        "secondary_vegetation_fraction", "secondary_canopy_height",
        "soil_carbon", "lightning_flash_rate",
    ):
        for gate in (
            "annual_precipitation", "gpp", "aboveground_biomass",
            "luh2_primary_fraction", "luh2_cropland_fraction",
            "luh2_rangeland_fraction", "soil_carbon", "leaf_area_index",
            "luh2_secondary_fraction", "lightning_flash_rate",
        ):
            add(anomalies[driver] * gates[gate])

    for wave in harmonics.values():
        for gate in (
            "annual_precipitation", "air_temperature", "gpp",
            "aboveground_biomass", "luh2_primary_fraction",
            "luh2_cropland_fraction", "luh2_rangeland_fraction",
            "lightning_flash_rate",
        ):
            add(wave * gates[gate])

    for driver in (
        "monthly_precipitation", "dryness", "air_temperature", "gpp",
        "leaf_area_index", "secondary_vegetation_fraction",
        "secondary_canopy_height", "soil_carbon", "lightning_flash_rate",
    ):
        for threshold in (0.02, 0.05, 0.08, 0.13, 0.20, 0.30):
            add(anomalies[driver] * np.maximum(current - threshold, 0.0))

    for left, right in (
        ("dryness", "gpp"),
        ("air_temperature", "monthly_precipitation"),
        ("monthly_precipitation", "gpp"),
        ("leaf_area_index", "dryness"),
        ("lightning_flash_rate", "dryness"),
        ("secondary_vegetation_fraction", "dryness"),
        ("secondary_canopy_height", "dryness"),
    ):
        add(anomalies[left] * anomalies[right])

    for driver in (
        "monthly_precipitation", "dryness", "air_temperature", "gpp",
        "leaf_area_index", "aboveground_biomass", "soil_carbon",
        "secondary_canopy_height", "natural_canopy_height",
    ):
        for threshold in (0.02, 0.05, 0.08, 0.13, 0.20, 0.30):
            opportunity = np.maximum(current - threshold, 0.0)
            add(previous[driver] * opportunity)
            add(cycles[driver] * opportunity)

    for driver in (
        "monthly_precipitation", "gpp", "leaf_area_index", "aboveground_biomass",
    ):
        for gate in (
            "annual_precipitation", "gpp", "aboveground_biomass",
            "luh2_primary_fraction", "luh2_cropland_fraction", "soil_carbon",
            "leaf_area_index",
        ):
            add(previous[driver] * gates[gate])

    for antecedent, current_driver in (
        ("gpp", "monthly_precipitation"),
        ("air_temperature", "gpp"),
        ("monthly_precipitation", "dryness"),
        ("leaf_area_index", "dryness"),
        ("aboveground_biomass", "dryness"),
        ("soil_carbon", "dryness"),
        ("secondary_canopy_height", "dryness"),
        ("natural_canopy_height", "dryness"),
        ("lightning_flash_rate", "dryness"),
    ):
        add(previous[antecedent] * anomalies[current_driver])

    if index != coefficients.size:
        raise RuntimeError(
            f"coupled seasonal basis mismatch: {index} != {coefficients.size}"
        )
    learned = np.exp(np.clip(strength * eta, -30.0, 30.0))
    return learned / (learned.sum(axis=0, keepdims=True) + 1e-12)


def _coupled_valid_closure(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Separate annual capacity and seasonal timing without cross-cell state."""
    monthly = np.asarray(prediction).reshape(16, 12, 180, 360)
    hazard = -np.log1p(-np.clip(monthly, 0.0, 1.0 - 1e-7))
    total_hazard = hazard.sum(axis=1, keepdims=True)
    allocation = hazard / (total_hazard + 1e-12)
    annual_burn = monthly.sum(axis=1, keepdims=True)

    if "fuel" in enabled and p.get("annual_intact_w", 0.0) > 0.0:
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
        target *= weight.sum() / ((target * _CELL_AREA).sum() + 1e-12)
        annual_burn = np.clip(target, 0.0, 11.5)

    annual_burn = np.clip(
        annual_burn * p.get("annual_scale", 1.0), 0.0, 11.5
    )
    lam = total_hazard.copy()
    for _ in range(8):
        survival = np.exp(-lam * allocation)
        produced = (1.0 - survival).sum(axis=1, keepdims=True)
        slope = (allocation * survival).sum(axis=1, keepdims=True)
        lam = np.clip(
            lam - (produced - annual_burn) / (slope + 1e-12), 0.0, 1e4
        )
    baseline = 1.0 - np.exp(-lam * allocation)
    baseline_cycle = baseline.mean(axis=0)
    current = baseline_cycle / (
        baseline_cycle.sum(axis=0, keepdims=True) + 1e-12
    )
    learned = _coupled_seasonal_allocation(current, data, p, enabled)
    calibrated = annual_burn * learned[None, ...]
    return np.asarray(np.clip(calibrated, 0.0, 1.0), dtype=np.float32).reshape(
        prediction.shape
    )



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

    # Build one globally shared pointwise fire-rate equation. Geographic boxes
    # and region-specific parameter sets are deliberately excluded: an ED site
    # must receive the same response for the same local ecological state.
    rate = _fire_rate(data, fallback, enabled)

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
    # First set the long-term fire potential from local fuel, climate and
    # ignition state; then distribute that potential through the seasonal
    # phenology equation. This factorisation keeps magnitude and timing as
    # distinct ecological processes.
    prediction = _coupled_annual_correction(prediction, data, fallback, enabled)
    prediction = _coupled_valid_closure(prediction, data, fallback, enabled)
    return np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
