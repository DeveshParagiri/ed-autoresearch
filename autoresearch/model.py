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
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'curing', 'lag',
              'softmin', 'cropland', 'phenology', 'regime_capacity',
              'rare_ignition')

# Focus tuning on the independently validated global annual and seasonal heads.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'annual_scale': {'type': 'float', 'low': 0.75, 'high': 1.15},
    'annual_residual_w': {'type': 'float', 'low': 0.35, 'high': 1.20},
    'allocation_glm_w': {'type': 'float', 'low': 0.60, 'high': 1.40},
    'memory_gam_w': {'type': 'float', 'low': 0.50, 'high': 1.00},
    'causal_glm_w': {'type': 'float', 'low': 0.25, 'high': 0.75},
}

PARAMS = {'annual_scale': 1.73,
 'annual_residual_w': 1.0,
 'seasonal_residual_w': 0.0,
 'annual_intact_half': 7.27782641589826,
 'annual_intact_w': 0.0,
 'annual_vpd_half': 1.7380558910922053,
 'annual_vpd_n': 3.5111403706263125,
 'annual_vpd_w': 0.1406449712828405,
 'alloc_dry_scale': 25.852949531840476,
 'alloc_dry_w': 0.35438507767111543,
 'alloc_vpd_rise_w': 0.3,
 'alloc_vpd_rise_half': 400.0,
 'alloc_vpd_rise_n': 1.0,
 'allocation_glm_w': 1.0,
 'memory_gam_w': 1.00,
 'memory_norm_months': 12.0,
 'causal_glm_w': 0.35,
 'absolute_glm_w': 0.50,
 'cool_crop_brake': 4.5,
 'wet_forest_brake': 1.0,
 'cold_forest_capacity': 3.0,
 'arid_fine_fuel_capacity': 2.0,
 'productive_range_brake': 2.5,
 'seasonal_rain_capacity': 0.4,
 'fire_season_w': 0.3,
 'fire_season_half': 0.04,
 'fire_season_dry_half': 500.0,
 'greenup_brake': 2.0,
 'rare_ignition_scale': 0.02,
 'crown_fire_event_scale': 0.08,
 'rain_pulse_ignition_scale': 0.24,
 'rain_pulse_opportunity_half': 0.02,
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
 'past_k': 0.0,
 'past_t': 30.0,
 'past_w': 5.0,
 'gust_w': 0.22,
 'gust_ref': 6.458842968165218,
 'gust_cap': 5.0}

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
    # A causal running reference keeps the multiplier relative to each site's
    # recent spread regime without reading future months.
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    state = np.asarray(factor[0], dtype=np.float64).copy()
    relative = np.empty_like(factor)
    for step in range(factor.shape[0]):
        state += alpha * (factor[step] - state)
        relative[step] = factor[step] / (state + 1e-12)
    return relative


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
    accumulator is initialised from the first available month so no future
    state enters an earlier coupled timestep.
    """
    alpha = float(np.clip(alpha, 1e-3, 1.0))
    state = np.asarray(series[0], dtype=np.float64).copy()
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
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    state = np.asarray(flammable[0], dtype=np.float64).copy()
    relative = np.empty_like(flammable)
    for step in range(flammable.shape[0]):
        state += alpha * (flammable[step] - state)
        relative[step] = flammable[step] / (state + 1e-12)
    return np.clip(relative, 0.0, p["cure_cap"])


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

# Frozen calibration scales make the fitted response identical at an isolated
# ED site and on the global evaluation grid. They replace runtime cross-cell
# medians and interquartile ranges; no observed fire data enter at runtime.
_COUPLED_ANNUAL_SCALING = {'dryness:mean': (254.17414801319438, 1971.3810015618799), 'dryness:std': (85.97174789533939, 149.58398729755356), 'dryness:p10': (83.17164009809494, 1824.9420560121537), 'dryness:p90': (446.5054379940033, 2126.5401218593124), 'annual_precipitation:mean': (473.96431255340576, 624.844419002533), 'monthly_precipitation:mean': (39.49702660987775, 52.07036862310149), 'monthly_precipitation:std': (17.347330650114202, 29.968269245074), 'monthly_precipitation:p10': (12.877287410013379, 25.813495210674592), 'monthly_precipitation:p90': (66.67687625288963, 90.72785850688817), 'air_temperature:mean': (4.880924940109253, 22.83300479253133), 'air_temperature:std': (6.925248453602162, 10.688747996500341), 'air_temperature:p10': (0.0, 31.054818558692933), 'air_temperature:p90': (17.647926807403564, 18.939352655410765), 'gpp:mean': (0.01139792062295027, 0.6766268447275555), 'gpp:std': (0.006728906226352785, 0.31336151838311216), 'gpp:p10': (0.0005870594619409532, 0.07957847461220809), 'gpp:p90': (0.020962763520947194, 1.2486256144940853), 'aboveground_biomass:mean': (0.1290913743238586, 1.567750244323785), 'aboveground_biomass:std': (0.004862058834260686, 0.0591332058518185), 'aboveground_biomass:p10': (0.1091192161431536, 1.5048849115148186), 'aboveground_biomass:p90': (0.15219358878675848, 1.6381184724159539), 'soil_carbon:mean': (0.8941057085370023, 7.106800893321633), 'soil_carbon:std': (0.005418998447000007, 0.03934285550205921), 'soil_carbon:p10': (0.8856353640556336, 7.055086908489466), 'soil_carbon:p90': (0.9003640593960882, 7.162637011706829), 'leaf_area_index:mean': (0.2811436417317357, 2.044282547865199), 'leaf_area_index:std': (0.028205708623091907, 0.7970191458828323), 'leaf_area_index:p10': (0.04091785420314409, 0.9186302833259106), 'leaf_area_index:p90': (0.3657135684974492, 3.5941932614892727), 'natural_canopy_height:mean': (0.2512683341046795, 8.779670105005303), 'natural_canopy_height:std': (0.0007836342969695514, 0.029906620178760127), 'natural_canopy_height:p10': (0.24773173551075162, 8.735707534104586), 'natural_canopy_height:p90': (0.25393680185079576, 8.830603800714016), 'secondary_canopy_height:mean': (3.8477236827214556, 11.363713107382258), 'secondary_canopy_height:std': (0.015787455894605486, 0.081727195095272), 'secondary_canopy_height:p10': (3.7903481356799604, 11.259638002514839), 'secondary_canopy_height:p90': (3.890541841834784, 11.44188721179962), 'natural_vegetation_fraction:mean': (0.07618884393014014, 0.6095968093723059), 'natural_vegetation_fraction:std': (0.0, 0.0), 'natural_vegetation_fraction:p10': (0.07618884393014014, 0.6095968093723059), 'natural_vegetation_fraction:p90': (0.07622363069094718, 0.6095968093723059), 'secondary_vegetation_fraction:mean': (0.016329333186149597, 0.1567498755757697), 'secondary_vegetation_fraction:std': (0.0, 0.0), 'secondary_vegetation_fraction:p10': (0.016329333186149597, 0.1567498755757697), 'secondary_vegetation_fraction:p90': (0.016329333186149597, 0.1567498755757697), 'lightning_flash_rate:mean': (0.006493361624052341, 0.021110813027569727), 'lightning_flash_rate:std': (0.009110128810931394, 0.02008475526869543), 'lightning_flash_rate:p10': (5.311021368470394e-06, 0.0007419758316245862), 'lightning_flash_rate:p90': (0.018469922151416542, 0.050555077742319564), 'luh2_cropland_fraction:mean': (0.0019709593179868534, 0.08153530675917864), 'luh2_pasture_fraction:mean': (0.0, 0.022412070218706504), 'luh2_rangeland_fraction:mean': (0.0032429337292114724, 0.1241085137007758), 'luh2_primary_fraction:mean': (0.20778736798092723, 0.5980775392381474), 'luh2_secondary_fraction:mean': (1.0, 0.0), 'luh2_urban_fraction:mean': (4.257736868851225e-06, 0.0012029065328533761)}
_COUPLED_SEASONAL_SCALING = {'annual_precipitation': (473.96431255340576, 624.844419002533), 'air_temperature': (4.880924940109253, 22.83300479253133), 'gpp': (0.01139792062295027, 0.6766268447275555), 'aboveground_biomass': (0.1290913743238586, 1.567750244323785), 'soil_carbon': (0.8941057085370023, 7.106800893321633), 'leaf_area_index': (0.2811436417317357, 2.044282547865199), 'luh2_primary_fraction': (0.20778736798092723, 0.5980775392381474), 'luh2_secondary_fraction': (1.0, 0.0), 'luh2_cropland_fraction': (0.0019709593179868534, 0.08153530675917864), 'luh2_rangeland_fraction': (0.0032429337292114724, 0.1241085137007758), 'luh2_pasture_fraction': (0.0, 0.022412070218706504), 'lightning_flash_rate': (0.006493361624052341, 0.021110813027569727)}


_CAUSAL_GLM_INTERCEPT = -0.17672283181887508
_CAUSAL_GLM_COEFFICIENTS = np.asarray((
    1.1720913556203967, -1.3379179058937585, -0.19371143747936312,
    0.9968883611170639, 0.5089140738572806, 2.8366198300565757,
    -0.6037782236578247, 0.2869439844607758, -2.5467036878441736,
    -1.070605826917424, 0.09926927078891144, -1.4539936882815296,
    0.9450375687851635, 1.0953735452305169, 0.25700489856583125,
    1.3166029646288766, 0.029554813579048402, -0.2388159094092974,
    -1.6156782014816227, 0.8584403445170153,
), dtype=np.float64)
_CAUSAL_GLM_CENTER = np.asarray((
    0.3916442575391283, 0.4288210202384565, 0.09686156339831838,
    0.1705279946814881, -0.07647216042930158, -0.04674825266340497,
    0.12547701312307588, 0.20030554873970863, -0.05349689427775958,
    0.2352590487460641, 0.34845472347325374, 0.18503648747040385,
    0.24107016127621883, 0.3750942281444727, 0.09107843887692285,
    0.19306248664678977, 0.07993985060925964, 0.3638756207993924,
    0.9071575991463425, 0.755808005846947,
), dtype=np.float64)
_CAUSAL_GLM_SCALE = np.asarray((
    0.39588863369575256, 0.3432808320001309, 0.4080285083204379,
    0.34049763544669215, 0.15899743835028432, 0.07834682201576872,
    0.4764371982639777, 0.41625774256166453, 0.08652399838242839,
    0.26725326642862157, 0.34840991282145783, 0.22062412465468026,
    0.2682956097093172, 0.34125670565424615, 0.13854010219644214,
    0.22372801428305444, 0.1348716094551787, 0.355407774047465,
    0.16638475522597768, 0.30417540903614154,
), dtype=np.float64)


_ABSOLUTE_GLM_INTERCEPT = -0.23829299398540782
_ABSOLUTE_GLM_COEFFICIENTS = np.asarray((
    0.9874767038843745, 0.3292980602872779, 0.7558797466426699,
    -0.08908569061042301, 0.05260767157181614, -0.20229696210568338,
    -2.0951525817103622, -0.838992672661851, 1.933019379053836,
    -0.3527965820968789, -0.2386897538372698, -1.1208561099389671,
    -1.079104688851721, 1.0915606170229224, 0.3145314598828101,
    1.2467431225702696, 0.19885750454881318, -1.0638703360072708,
    -0.109091902357476, 0.6890416895995078,
), dtype=np.float64)
_ABSOLUTE_GLM_CENTER = np.asarray((
    0.06425482467876094, -0.08027443243548144, 0.06882304628500184,
    0.097271616416084, 0.044378792032445144, -0.028965817101006947,
    0.11728742902344069, 0.08900610205497456, 0.13655142645783225,
    0.054529051003046225, 0.09687601746506583, 0.20544953722160683,
    0.16663063268404182, 0.21201602503306674, 0.08188222801109686,
    0.17503594399706413, 0.06375179690906478, 0.8852724145615278,
    0.4444390238170585, 0.7311104971288134,
), dtype=np.float64)
_ABSOLUTE_GLM_SCALE = np.asarray((
    0.18666894520265886, 0.18539410224217615, 0.19328014010058311,
    0.16856163342311367, 0.17315807693098909, 0.056033963437448514,
    0.12515686932268003, 0.21182049900719455, 0.13407934725645151,
    0.21349811269977828, 0.17817278638449568, 0.2539013189499832,
    0.2202778576140103, 0.25592792907275286, 0.12672906483889607,
    0.2246417363792384, 0.13439064384063137, 0.1983619423025041,
    0.3808844040580293, 0.3290780975813928,
), dtype=np.float64)


_MEMORY_GAM_GROUPS = ('incumbent', 'monthly_precipitation:current', 'monthly_precipitation:previous', 'monthly_precipitation:memory_3m', 'monthly_precipitation:departure_3m', 'monthly_precipitation:memory_6m', 'monthly_precipitation:departure_6m', 'monthly_precipitation:memory_12m', 'monthly_precipitation:departure_12m', 'monthly_precipitation:memory_24m', 'monthly_precipitation:departure_24m', 'dryness:current', 'dryness:previous', 'dryness:memory_3m', 'dryness:departure_3m', 'dryness:memory_6m', 'dryness:departure_6m', 'dryness:memory_12m', 'dryness:departure_12m', 'dryness:memory_24m', 'dryness:departure_24m', 'air_temperature:current', 'air_temperature:previous', 'air_temperature:memory_3m', 'air_temperature:departure_3m', 'air_temperature:memory_6m', 'air_temperature:departure_6m', 'air_temperature:memory_12m', 'air_temperature:departure_12m', 'air_temperature:memory_24m', 'air_temperature:departure_24m', 'gpp:current', 'gpp:previous', 'gpp:memory_3m', 'gpp:departure_3m', 'gpp:memory_6m', 'gpp:departure_6m', 'gpp:memory_12m', 'gpp:departure_12m', 'gpp:memory_24m', 'gpp:departure_24m', 'leaf_area_index:current', 'leaf_area_index:previous', 'leaf_area_index:memory_3m', 'leaf_area_index:departure_3m', 'leaf_area_index:memory_6m', 'leaf_area_index:departure_6m', 'leaf_area_index:memory_12m', 'leaf_area_index:departure_12m', 'leaf_area_index:memory_24m', 'leaf_area_index:departure_24m', 'lightning_flash_rate:current', 'lightning_flash_rate:previous', 'lightning_flash_rate:memory_3m', 'lightning_flash_rate:departure_3m', 'lightning_flash_rate:memory_6m', 'lightning_flash_rate:departure_6m', 'lightning_flash_rate:memory_12m', 'lightning_flash_rate:departure_12m', 'lightning_flash_rate:memory_24m', 'lightning_flash_rate:departure_24m', 'aboveground_biomass:current', 'natural_canopy_height:current', 'secondary_canopy_height:current', 'secondary_vegetation_fraction:current', 'natural_vegetation_fraction:current', 'soil_carbon:current', 'annual_precipitation:current', 'luh2_cropland_fraction:current', 'luh2_pasture_fraction:current', 'luh2_rangeland_fraction:current', 'luh2_primary_fraction:current')
_MEMORY_GAM_KNOTS = ((5.722944741137326e-05, 0.00061323611298576, 0.002279391512274742, 0.008629525639116764, 0.07290570795536028), (0.5838513171672821, 17.135194396972658, 51.66904640197754, 130.86759033203128, 320.407691040039), (0.5832705199718475, 17.126293182373047, 51.661638259887695, 130.88356018066406, 320.11888122558594), (8.155061149597168, 30.19585418701172, 58.12714767456055, 124.18148345947269, 262.8564648437498), (-102.36146057128906, -24.596670150756832, -2.7701950073242188, 23.78659667968751, 112.05698364257807), (9.372389640808105, 33.5717155456543, 59.739219665527344, 122.9782714843751, 243.29553771972644), (-118.14089965820312, -32.80142440795898, -3.377333402633667, 29.0597221374512, 137.30388610839836), (9.631167144775391, 34.19834594726563, 60.400657653808594, 121.81786956787111, 235.7193560791015), (-125.78455047607422, -35.824304199218744, -3.3031387329101562, 30.92751846313477, 149.55076171874967), (9.42444694519043, 33.44393692016602, 60.029083251953125, 121.48294372558594, 238.52226989746077), (-133.5793536376953, -36.854191589355466, -3.0127830505371094, 31.899298095703152, 155.43387329101557), (1.947782793045044, 66.80178985595704, 449.3795471191406, 2418.488476562501, 11904.3512109375), (1.9474595689773562, 66.88660888671875, 449.47645568847656, 2418.488476562501, 11901.3723046875), (5.630140228271484, 64.7614028930664, 342.9612731933594, 2155.775244140625, 11287.070078124943), (-283.75476806640626, -34.727116394042966, 15.791259765625, 117.33891601562503, 405.29854492187457), (7.274534854888916, 71.27728118896485, 333.1058044433594, 2096.2078613281274, 10944.666328124982), (-313.28034545898436, -41.345584106445294, 31.06622314453125, 188.69849853515638, 729.1114062499992), (7.579896240234375, 72.75033721923829, 320.4468994140625, 1978.8014160156263, 10293.935468749998), (-304.16709228515623, -38.49921798706054, 51.94573974609375, 298.63505859375016, 1323.5792968749997), (7.219553031921387, 69.62374572753907, 297.7402648925781, 1770.414404296876, 9129.94523437496), (-268.7394763183594, -30.828273773193356, 80.90971374511719, 492.0797851562501, 2439.056953125), (-18.488712768554688, 7.039702606201174, 20.948469161987305, 26.525222778320312, 31.09358612060547), (-18.489360046386718, 7.039702606201174, 20.95397186279297, 26.52352867126465, 31.0915177154541), (-10.78477569580078, 7.140351390838624, 19.759958267211914, 26.180843353271484, 29.718618011474607), (-12.911234893798827, -3.871165466308593, 0.091033935546875, 4.125566101074224, 12.509940834045382), (-7.762863483428955, 6.411308002471925, 19.32906723022461, 26.085891723632812, 29.3803636932373), (-15.853828048706054, -4.154496574401855, 0.0759267807006836, 4.793968200683597, 15.734222221374504), (-7.149583129882813, 5.710514354705812, 19.06891632080078, 26.003564834594727, 29.227028503417966), (-16.461292572021485, -4.4061010360717745, 0.1347637176513672, 5.593230628967289, 17.78421356201172), (-8.943678512573243, 4.726441764831544, 18.90632438659668, 25.916253280639648, 29.034553451538084), (-15.44516429901123, -4.284684753417968, 0.34577369689941406, 6.270446586608887, 19.83818603515623), (2.4300368932017587e-05, 0.009400387294590473, 0.34873533248901367, 1.7416204929351806, 3.3017554569244383), (2.403596005024155e-05, 0.009400387294590473, 0.34877023100852966, 1.7419650077819813, 3.309914083480835), (0.00043501897831447426, 0.018666199594736112, 0.4755718410015106, 1.5894290447235109, 3.169941921234131), (-0.8468037962913513, -0.14875523447990416, 5.531718488782644e-05, 0.1498470544815064, 0.8820381546020504), (0.0004800950456410647, 0.019126808643341073, 0.5243726372718811, 1.5227712869644165, 3.1252011013031002), (-1.0131849908828736, -0.1872878074645996, 0.0002997852861881256, 0.19889090061187747, 1.066452331542966), (0.0004965388984419406, 0.01877810992300511, 0.5349269509315491, 1.4739083528518688, 3.1095545196533196), (-1.0451232862472535, -0.2025283277034759, 0.0005450256867334247, 0.22574191093444826, 1.170845122337341), (0.0004830585839226842, 0.017839654907584202, 0.5231345891952515, 1.442946004867554, 3.1034887886047358), (-1.0162274599075318, -0.19916444420814514, 0.0009219353087246418, 0.24272978305816692, 1.2563125610351498), (0.017268125452101234, 0.4105978488922119, 1.761967658996582, 4.280908679962158, 5.795237693786619), (0.017268125452101234, 0.41061094403266907, 1.7618508338928223, 4.281027317047119, 5.796199417114255), (0.04038426697254181, 0.6158014178276062, 1.9543899893760681, 3.837101221084595, 5.51653823852539), (-1.8954846572875976, -0.3255667209625244, 0.009406045079231262, 0.35576596260070814, 1.7477129960060103), (0.04042404443025589, 0.6601634383201599, 2.011794686317444, 3.69984450340271, 5.4534112739562985), (-2.1448377990722656, -0.45985286235809325, 0.01336696743965149, 0.47414946556091314, 2.144767279624938), (0.03970672905445099, 0.6687217831611634, 2.012597680091858, 3.643108081817627, 5.432701053619384), (-2.1811579656600952, -0.5011904716491699, 0.01509210467338562, 0.5343557357788088, 2.3762852954864497), (0.03818944469094278, 0.6600845098495484, 1.9533694982528687, 3.60336332321167, 5.433879947662353), (-2.112450795173645, -0.4928324699401855, 0.01802229881286621, 0.5714660644531251, 2.588387546539306), (3.7241018844724747e-06, 0.0008958652033470571, 0.015547999180853367, 0.0622829832136631, 0.16515522748231887), (3.7207258628768614e-06, 0.0008983887149952355, 0.015554905869066715, 0.062289214879274406, 0.1651552724838257), (0.0010127336578443647, 0.0053781112655997285, 0.019194984808564186, 0.05274634659290314, 0.12707382500171618), (-0.0473081111907959, -0.014021154120564459, -0.0022760892752557993, 0.013622885942459109, 0.06236321553587911), (0.0019069749722257256, 0.006691351253539324, 0.020829960703849792, 0.05107524767518045, 0.11815153181552873), (-0.05462319299578667, -0.017343337088823317, -0.003563536796718836, 0.0175665743649006, 0.0744773161411283), (0.0023310004733502867, 0.007041631545871497, 0.021314149722456932, 0.05060321912169457, 0.1150294536352157), (-0.0574200239777565, -0.018540530651807784, -0.004022669047117233, 0.019677963852882396, 0.08002183675765982), (0.0023835205473005773, 0.006779992580413819, 0.020922964438796043, 0.04976634085178376, 0.11322153002023685), (-0.05803610950708389, -0.018446582555770873, -0.003932027146220207, 0.020808530598878865, 0.08377347856760022), (0.017926213111495604, 0.27904961444437504, 1.220943059772253, 4.095860648155213, 16.18823192119598), (0.5216378450393677, 3.9287313133478166, 12.175714492797852, 23.81570932865143, 30.770841317176817), (2.8976705837249757, 6.168750250339508, 11.262545943260193, 17.65151364803314, 26.546214866638177), (0.0028240463816473493, 0.046159059507772326, 0.12683009169995785, 0.41793832555413246, 0.911588579416275), (0.0007694081868976355, 0.018879686249420047, 0.17814459465444088, 0.6195220053195953, 0.9944213666021824), (0.04433104631258175, 1.6575211271643644, 5.653914675116539, 10.23792616128922, 16.885367391109465), (120.63125467300415, 425.4503917694092, 734.2440147399902, 1454.2104034423828, 2737.599349975586), (2.2642696786423504e-06, 0.008998778503155336, 0.10790526494383812, 0.36640989035367966, 0.7060345970094204), (6.31885841357871e-05, 0.01120044221170247, 0.0635934742167592, 0.19662146735936403, 0.5211805254220963), (4.562315924355742e-10, 0.0010795981436331203, 0.03672426799312234, 0.3741936534643173, 0.8604808375239372), (3.0631827109550702e-09, 0.029655282385647297, 0.19547863956540823, 0.5857276618480682, 0.9853674322366714))
_MEMORY_GAM_INTERCEPT = -2.8885310310704027
_MEMORY_GAM_COEFFICIENTS = np.asarray((-0.04871472471951529, -0.0379853009166668, -0.15614760155110097, -0.1328575511550417, -0.03651746186053937, 0.13586039576278278, 0.01906382288345939, 0.0711173246478745, -0.05939176453063372, -0.05120645502887195, -0.0013118858130002284, 0.050458079254842934, -0.006701561611449813, 0.03730889635488782, -0.004467931686356559, -0.024781747659182413, -0.029302993047382122, 0.023450171537187273, -0.019979178505027277, 0.03499673584726304, -0.010549052405891827, 0.041785309651307626, -0.03876169346192224, -0.09045364105738116, 0.03529176629447362, -0.0025963235537474227, -0.018660195438535736, -0.018525844661436933, 0.03974887180932775, 0.04288362341134249, -0.09168269673360849, 0.011114905571354491, 0.003145921365228266, 0.025727067174568803, -0.00837488391013467, -0.023728104153278813, -0.06495907597106285, 0.020842086102968768, 0.07018590317279286, -0.00708041964160441, -0.09199246349277529, -0.030798738257410623, -0.016151618660680215, -0.07923328200878516, 0.02702915441428293, 0.009435434850341784, 0.013192985428782316, 0.025383387854641976, -0.03872038309493967, 0.12760289905609684, -0.022354068383640534, -0.00529971226050076, -0.11625529719753952, -0.05810799051083805, 0.02805724955538821, -0.005460834818904188, -0.040869726791369576, 0.08134206184386424, -0.005710993890678786, -0.027526812702667346, -0.05056313470376848, 0.08279491297250781, -0.054625136666977575, 0.012613608632429569, -0.027070754362154125, -0.06300070901929247, 0.022853505372350025, -0.020804796266853644, 0.029989426055545672, -0.020676322653458028, -0.04881372755842816, 0.017665310944383308, 0.0323327577947526, -0.03820050160326188, 0.022872914578639655, -0.03769914913466342, -0.04008241646700932, 0.11555660171781745, -0.08735687169111409, 0.02628905303388716, -0.019179134672417823, -0.03881071049695955, 0.05429823139901342, -0.014999429993240377, -0.00565183128399651, 0.07698953362702012, 0.03770000799753674, -0.05681329909948107, 0.053249819366416255, -0.037745488714430855, -0.04859250710383896, 0.011313842939957906, -0.04600778608281398, 0.018018511341202564, 0.03433506829865192, -0.02484844255283641, -0.023372459080151276, 0.016380849939448355, -0.012042622079808636, 0.05408784587542152, -0.062398818003791226, 0.024030817801830032, -0.016815611681433813, 0.006334248780473825, -0.04740437772604133, 0.03745359255146732, 0.010481903696901801, -0.03192576228824347, -0.09175323218418109, -0.007969924183521613, 8.283588594267103e-05, 0.024362189951479762, -0.02730492269529995, 0.012875913770019296, -0.014601206756434535, -0.0033312596026135468, -0.021801647981248683, 0.018209629415418916, 0.005378580908011307, -0.028227459792068463, 0.008037814100470003, 0.051493070785381956, 0.02149596107300724, -0.06136993758737842, 0.011590144466572434, -0.016591728606272872, -0.022212362189476986, -0.1442576024280779, 0.1413321285631107, 0.04254669922151718, 0.03951796682257466, -0.056111081993874296, 0.04833555562965224, 0.12813114616979607, 0.049445044274925035, 0.016348399957388418, -0.04000602591669748, 0.1248113758943008, 0.018059832136998015, 0.08195406188687016, -0.06851088755129585, 0.17128919805097614, -0.019212899898786637, 0.10122677077861611, 0.018247196772732704, 0.11498288145197456, 0.01391522122598955, -0.11027631253910794, 0.03683395621312161, 0.062240001763177705, -0.01784510443802013, 0.04904134607852842, -0.009749529467142415, 0.06973840013647246, 0.055269971063994686, 0.11271814321576588, -0.004533868308445238, -0.007367579371614954, -0.04041360941842522, 0.04841927342040315, -0.020599730717735645, 0.023851712862226614, -0.05336450163847716, 0.0562816269628923, 0.037916823578790025, -0.053338309407240624, 0.133104533666508, 0.041447702463546496, -0.034152882163323614, -0.05927704931572271, -0.030024279691923218, 0.056691854215562915, -0.011874740208199834, 0.030159959246046884, -0.09871830545155166, 0.10092630993148669, 0.041600015345266764, -0.04255521394280659, 0.13661774200936586, -0.03289559009369597, -0.05706356426126128, -0.10694268309857026, -0.014153299213016418, -0.022656067640138867, 0.05298331394548908, 0.08154157420117153, 0.024815818310138016, 0.015093782670639058, -0.033768515741597505, 0.08625177795977942, -0.16576798630599376, 0.04733805302654629, 0.04470778882701168, -0.0011082952904922616, -0.0031871673656702847, -0.06995136422637609, 0.07181889231797756, 0.029239163792393528, 0.01862919200831819, -0.03744939172078684, 0.030009391079439642, 0.003571717808195581, 0.007295316981863317, 0.025104227930795905, -0.003556649583484585, 0.02528172597874923, 0.01652408146197704, 0.009194135543080247, -0.008567866100591752, -0.1283017581537944, -0.008015471294314468, -0.028453452580183874, 0.0038997175055279607, 0.015176947258396087, 0.02560699712765296, -0.003496918231648794, -0.07345397660290964, 0.028207321772053123, 0.013044755977078009, 0.02508606792055945, -0.06878337619366938, -0.004490482309806881, -0.013229549026994, -0.013037639680796884, -0.004512976840869193, 0.005602023488645683, 0.016417599214710882, -0.0031992009115251402, -0.030729083130948742, 0.01546659137696261, 0.028136275293673796, -0.007737410720057591, -0.040890219802464385, 0.04606304287389458, -0.007545598484098585, 0.010870004370046557, -0.0033148106431898352, -0.02598047096719148, 0.0013753982372496022, 0.011919967757123439, 0.011478109608770423, 0.05305263345895545, -0.08634116822970056, 0.008493775131359954, 0.039291393466479474, 0.010284605631136277, -0.010989816523762836, 0.04931457589665497, -0.0019482123300106425, 0.019295254281383584, -0.03782746515485526, -0.07993028921722567, -0.05885010792430736, -0.01695445573536729, 0.02330647353378076, -0.014310631635082068, 0.02947994409919393, 0.0032215360453990965, 0.008573037090824446, -0.022473834379333757, -0.05179349752989438, 0.049540765339683095, 0.007403905297346484, 0.01708191324673308, -0.09021957409464737, 0.06231294157601802, 0.05993734819825058, -0.06093682620736234, 0.026613957374680836, -0.007282348002574904, 0.004570372406643523, -0.04574642975037467, 0.0004464817943052404, -0.03636492728197097, 0.07863082794541697, 0.00045151451693131056, 0.04974939439498735, -0.009166363480573815, 0.028511471606897723, 0.02516953115786408, -0.02898718322397092, -0.02367779503756782, 0.02681392139010929, -0.04734272155568532, 0.01058603846838075, -0.008504943583670346, 0.022210781942254183, 0.017462511456947837, 0.06212423025366701, -0.03174378461237855, 0.014343156522150563, 0.014111964102513795, -3.925001656128638e-05, -0.004700287949046063, 0.058648960545759445, -0.0311524697704565, 0.05695736185636862, -0.0028187160331457155, -0.06866802833601295, 0.03274554849686306, 0.03317756301208884, -0.03718209161460261, -0.0642354546727384, -0.024854826258440675, 0.09536674644934202, 0.08481959510781749, 0.009151671882202391, -0.08007409272453608, -0.0001867368361768458, 0.04366440860619294, 0.020415898405730278, 0.08437202395257166, 0.0004404158326941758, -0.02086125049658498, -0.010980634692247417, 0.023247666607520423, -0.015390965154007027, 0.028671338086253894, 0.11215796062281141, 0.02750329493016325, -0.01819537163555975, -0.054000416770497, 0.0188283065661584, 0.010097062187560239, 0.035503362391709724, -0.004511578476389004, 0.025958929154446583, -0.04576287802795886, 0.04630048011320586, 0.03366531082552233, 0.02477384857474, -0.001973031956140998, 0.04751552251224604, -0.030359275475960276, -0.028536743358337995, 0.0008428398947690665, 0.03233349662820425, 0.026852979216042266, -0.014003134543361277, 0.023361235626328095, -0.01867227489478527, -0.031020305131021435, -0.07511326492049485, 0.02190801148907234, -0.03213365150287003, 0.023534321424598572, -0.025121907040023577, -0.017525613688203764, -0.09650376941105648, 0.05554030256204132, 0.029778514662803724, -0.010822525313833646, -0.020336633628407753, -0.04062843881825593, -0.1104168043093305, 0.05991965903601453, -0.06410384772428682, 0.0193613575898583, 0.014181826489115494, -0.047001596802305784, 0.03398908515588513, 0.057851055465615024, -0.03500691590803321, -0.029020170350553473, 0.004135018966674265, 0.0028323796940140982, 0.16356695684767808, 0.07205893720051575, -0.02236441249146564, -0.07229239152406272, -0.01366266619372277, -0.06400481554683983, -0.057967627983330924, -0.014741409286125438, -0.006202514022388827, -0.010133470156231632, 0.005844022683444962, 0.031023160735030093, 0.023879888708541724, 0.04762299432508551, -0.03235966834813032, -0.02868019991206942, -0.028255931991221814, -0.016215666344597316, -0.09394633266685838, -0.0032306460120288365, -0.02094320273903475, 0.026677639706530462, 0.022910350893874366, 0.02789651390520058, -0.031162537968476314, -0.01460428624349005, -0.00606302953670005, 0.01839724846935682, -0.026567555019613498, 0.006416384016719195, 0.023714388101937506, 0.031047412203806548, 0.034176748429614316, 0.005084155923364429, -0.06349396062626668, -0.057797158836889365, -0.010872262486329413, -0.13051132765949966, -0.030174526906354384, -0.030845294295698017, 0.03330099214186291, 0.07902668949274942, -0.022053097097210633, -0.0003176827105763654, -0.011940833363835706, -0.013291697675826055, -0.051739522031150814, 0.010825113611223548, 0.013373834369967056, -0.017616471482767706, -0.01659141353925385, -0.025130862612872108, 0.05378927056460919, 0.03796880732269936, -0.008817515687134807, 0.011292758456038934, -0.0037714632604927286, -0.008082345045721624, -0.025370370414668185, 0.04969499341615346, -0.0504790780631223, 0.0024621416201303066, -0.007234520684729239, -0.015466913455169545, 0.012504550280774171, 0.01335942866604316, -0.047541526981530594, 0.037914050499300696, -0.09801000931415725, 0.043349703219145684), dtype=np.float64)
_MEMORY_GAM_CENTER = np.asarray((0.019466901044474932, 0.09380004917611018, 0.051243151603755116, 0.2119372266862117, 0.27305282229242117, 0.28471153057881343, 0.052350056725535776, 0.38280545713505165, 0.33549004847180636, 0.17059959705249375, 0.04938496456111503, 0.008817068149519118, 0.048557557115803306, 0.36967644736066235, 0.3383854936374624, 0.1740373771042071, 0.05773690251424671, 0.01100706887530195, 0.004883274958178073, 0.10293827851113031, 0.3761230542875687, 0.3839332395113555, 0.1168880010157773, 0.014589024373631866, 0.019973162604744653, 0.16631507845001098, 0.35440308469519166, 0.3658860492633922, 0.07946158953969942, 0.012888116163281347, 0.00290143824724616, 0.06004596114754721, 0.27460345439357076, 0.4417412528122874, 0.19267723933404052, 0.026975936951910064, 0.024093393532733456, 0.21359647005476526, 0.3834131345195442, 0.304222330745123, 0.0650837818651344, 0.008908093905878165, 0.002440286174447675, 0.04990494348418069, 0.22592422243003915, 0.4407611388083336, 0.24041309123065407, 0.0388469103168533, 0.025984613172059457, 0.23579216594726407, 0.3908696853283567, 0.28163577175160953, 0.05785703255016673, 0.007315780034897338, 0.0022400380759142412, 0.04630516886110765, 0.21246593553572526, 0.430335161977374, 0.2607817711006266, 0.045728706721451254, 0.025795254230024715, 0.23384026749470085, 0.396257866686444, 0.2816976914019353, 0.0551892493664922, 0.0067279451335304045, 0.007274298307636168, 0.12124338473126312, 0.46117499801450595, 0.3063999212011357, 0.07838541178935554, 0.022546782922417637, 0.010435754737520477, 0.1806572670483338, 0.4528634154724586, 0.2574908345912662, 0.07353255561469088, 0.022070699693659348, 0.005996092014878604, 0.1326331366076165, 0.4735123955869384, 0.28230239676634156, 0.07884319646319739, 0.023581777609885696, 0.0021338119500852507, 0.016191552553020603, 0.09954790938229414, 0.48173769580019743, 0.3111213511177603, 0.0819230895049645, 0.006074350516583227, 0.13003600669083573, 0.47092951428295354, 0.28603225743977795, 0.07994407723358968, 0.023829130146108945, 0.005058645700334428, 0.03309881723768722, 0.16088446108324295, 0.5244886200860726, 0.2232587123246627, 0.0487183358908403, 0.00616540853404662, 0.12499043277358257, 0.4644199252341679, 0.2954189345915256, 0.08166959797510477, 0.02415234848286381, 0.006971947928733507, 0.04670235968779785, 0.22928801374497873, 0.5286827688442068, 0.1531956467572518, 0.03176236049566781, 0.006172132798181669, 0.1204044377837457, 0.45614473986909876, 0.3056486127025593, 0.08374277038153509, 0.02465561151043648, 0.007667380891422966, 0.05575272741553096, 0.3136259068877029, 0.4860780129082961, 0.10848677287748429, 0.025276052669428655, 0.0007540513179677652, 0.018393452814492287, 0.08818012519102278, 0.2468606741293684, 0.44856347903941585, 0.18374559665866858, 0.0019321322862312174, 0.031002990776030047, 0.09167565004809886, 0.26135330946003854, 0.44570470203380624, 0.15922140097934218, 0.003655011443374915, 0.047967998387321784, 0.09554079540196052, 0.19822962492817606, 0.4223286710007107, 0.21830901292963145, 0.0005780652747003072, 0.009621015958891425, 0.12608494525482866, 0.5830411352804344, 0.21374038128780415, 0.05854553351980505, 0.00543939159934808, 0.06087255035578928, 0.09415649887934052, 0.16353762743588693, 0.41507008746918495, 0.24390410783021674, 0.00041430599941149373, 0.007476573662548213, 0.13441501747656648, 0.6061755010377794, 0.19175196135629097, 0.05264131741527989, 0.005997464679135352, 0.06563885591878325, 0.09428829255768516, 0.14243592337321748, 0.4111434675909628, 0.26152979179433106, 0.000357186344625425, 0.007590052167370161, 0.1526922348367423, 0.6100114841295842, 0.17418311252751523, 0.048658618223464095, 0.005389658662527336, 0.06393323018309004, 0.09644814135692356, 0.13031481356145447, 0.4066542615732508, 0.2764816347337363, 0.00036088802648317006, 0.009213835322416814, 0.17967372588723401, 0.6015592416452428, 0.15579524532752656, 0.04713332380821486, 0.01824049853205441, 0.260054287077165, 0.2495602100772134, 0.24084656651633093, 0.1696057017396455, 0.058233043925676714, 0.018970072923263914, 0.2551426367766303, 0.20440529941261934, 0.2295867119099526, 0.20196011995103588, 0.08389294326131617, 0.02052665312343949, 0.24601894201928884, 0.19586047195128128, 0.23831307627877213, 0.21149443373713053, 0.08161072383709998, 0.011917944550902173, 0.08652125120725104, 0.24468719085381702, 0.572396214349169, 0.07370479261367109, 0.009761105189253725, 0.020655899777036704, 0.2491237519111183, 0.1857753816279365, 0.24212403465980517, 0.21611122021498735, 0.08022681656244686, 0.01106901125653564, 0.07853459683675394, 0.22430299284052854, 0.5785681607743113, 0.0931003428913702, 0.01316237930314125, 0.0207626394034499, 0.2516801199204894, 0.1811461206460342, 0.24694304877038484, 0.21728874739905985, 0.0767778426636498, 0.011314487175671161, 0.07674707276725404, 0.21365921332794768, 0.57299533703124, 0.10731616165279552, 0.01645243750296214, 0.020810887036348827, 0.25278878662711474, 0.17997432749143105, 0.24851357731655274, 0.2173816415102063, 0.0753983778300402, 0.012245766240977698, 0.07897258318435703, 0.2107508785813506, 0.5683384509365661, 0.11005087207147712, 0.018017809520391115, 0.02723410353844892, 0.23692741401520884, 0.2883404105466584, 0.22707995051495541, 0.16334074123643205, 0.0548745348417481, 0.02859489598976881, 0.231086759303915, 0.24695435437838975, 0.22276455734250863, 0.19375963314504782, 0.07373467317123929, 0.02988118699973984, 0.2155542417533004, 0.23579445884027941, 0.2539381922324191, 0.19854147714209813, 0.06282493352485215, 0.01127384810082652, 0.07299800253130195, 0.18762457701137875, 0.5837883189082559, 0.1181476528437671, 0.02333394356043238, 0.03041621723566639, 0.2145056749606189, 0.2316248134680599, 0.2579604704617841, 0.2012018612438815, 0.060744463863156115, 0.01192240970197819, 0.07363583046890018, 0.17952719693774463, 0.5628452539545239, 0.14473942448802712, 0.02482410361447657, 0.03068453247688087, 0.21433570507472197, 0.2330944674281, 0.26211294387141926, 0.19750262876486754, 0.058770460286858305, 0.012536505297616763, 0.07488240118599772, 0.175981431983872, 0.5519406199772567, 0.15535351262917557, 0.02676267400728989, 0.030814170887128844, 0.21337316902788503, 0.23636698087410932, 0.262712941415285, 0.19420391455489752, 0.05899524738257955, 0.013481639677581337, 0.07728654129267962, 0.17586792071730262, 0.5470445340777234, 0.15594713762361634, 0.02780926386290309, 0.023237683473136955, 0.35152119444931396, 0.2801658728904928, 0.20445070265344045, 0.10342390277208247, 0.03376341464612569, 0.02877832816019202, 0.36910416224980946, 0.25977689864971276, 0.1985348738743394, 0.10453505572420428, 0.03559722353485068, 0.008928867872667357, 0.10415138964048658, 0.25112878740782124, 0.3498951008159837, 0.2157022103138106, 0.06460141927936035, 0.03272581351856368, 0.1799475972205593, 0.2804484898415086, 0.34131839185537804, 0.11880405970738574, 0.04154421761145073, 0.0069727794676439315, 0.09095132200789761, 0.1837756027608096, 0.34229286902806555, 0.27020531486335286, 0.09633963474035331, 0.039706951837089555, 0.21758060315335545, 0.3004130281958637, 0.2973983490827486, 0.10515745582675574, 0.035516560732760244, 0.005830109181619877, 0.08923541939272744, 0.15145845776427913, 0.32627562551513273, 0.30323840234640925, 0.11240988918554738, 0.04351687456703379, 0.23949172697480592, 0.3070474597429475, 0.2769661595270456, 0.09718613909537145, 0.03209068869473185, 0.005492235421139769, 0.08969002319598204, 0.13333553514970875, 0.3157876210281288, 0.3252689118862058, 0.11817985581723633, 0.04503380318748649, 0.2517678212549665, 0.31093992127405135, 0.26635168283586175, 0.09261975260396238, 0.02992009724898366, 0.041053948338018786, 0.37636721280189994, 0.3145682320231734, 0.19226837599998464, 0.06076313923813506, 0.013858230240097016, 0.0800211044562437, 0.45659546109951, 0.1794573125254189, 0.11086659133462645, 0.10016024245155376, 0.06537821682068538, 0.024841159713539424, 0.1832139764646238, 0.2851952834858485, 0.26213414815701774, 0.17582751129390908, 0.06251238661553245, 0.028289367578818905, 0.19956429139011844, 0.35198626305754904, 0.27014770173141395, 0.11259505814987256, 0.034524578461022026, 0.018665970534910727, 0.2023837493106899, 0.22469111245451376, 0.20072400729343967, 0.2131205660064439, 0.1293553724270294, 0.03642172981121551, 0.2682257100542909, 0.2819779424388874, 0.25805090679963877, 0.12315370306784526, 0.029470512112692266, 0.002338288279665771, 0.04455076017108772, 0.20626155657032352, 0.41941766901654337, 0.26686829561225156, 0.05673673148213034, 0.020633286510857666, 0.3189999208328214, 0.2984394606465641, 0.1814163748531651, 0.11889752872349854, 0.0560248627132259, 0.03416115183192462, 0.2512108907915319, 0.20438434217239895, 0.21635847188055177, 0.1839360358624288, 0.0966848848751384, 0.024520537819826518, 0.27079866308099104, 0.30444694660944893, 0.19377125251508753, 0.12745991552734792, 0.07079801360196522, 0.014181652723607649, 0.17667201894246265, 0.21824637036460548, 0.21296431683200343, 0.22865681533678395, 0.13777523089593285, -0.07459001432297926, -0.027810039361940677, 0.02791536938954325, 0.0389983891118301), dtype=np.float64)
_MEMORY_GAM_SCALE = np.asarray((0.05323296567150037, 0.24051999682717234, 0.14063982155463658, 0.29584389853821025, 0.18872988189063317, 0.2638975323067503, 0.0676691147176715, 0.30974062606512753, 0.23403136166547703, 0.23591969387103667, 0.13090422980491273, 0.04983053907663237, 0.06590031105770207, 0.3080052472119433, 0.23824823755737268, 0.23638635229682667, 0.14499467811279612, 0.053827173047435915, 0.019904068142572826, 0.18008953220175566, 0.2597004429671253, 0.2514755725914054, 0.17240764228834024, 0.05667134521896224, 0.04062098715886029, 0.2039802653627965, 0.2341178866292887, 0.30038990090843304, 0.15811298123214812, 0.06240807722139506, 0.015755195079385286, 0.1414537117307555, 0.26236744501379067, 0.23630115879665975, 0.20263911909858023, 0.07420484812493651, 0.041687374122250605, 0.21373169282137036, 0.22797458409695534, 0.2976998896939037, 0.14268039676743013, 0.0509180060233024, 0.014931002114007245, 0.13006545659900973, 0.25785281491851975, 0.23016093077951688, 0.2135285349701306, 0.09149904814641965, 0.04129434752084606, 0.21550872234056828, 0.22103866931175423, 0.29864373914118664, 0.134653521459546, 0.045370046619779184, 0.014563341640904314, 0.12581484687075442, 0.2565327673613998, 0.22606179248677613, 0.22212968680468492, 0.09914048335827298, 0.041556386900569696, 0.21294855439854238, 0.2212472161422066, 0.29710180851636214, 0.13174115293952254, 0.04308561009679469, 0.03157020633352429, 0.22859132893036863, 0.30060125354438266, 0.2532975925410217, 0.15537656186734936, 0.08969669430465858, 0.03725926342059296, 0.26326881839433447, 0.30096425665402554, 0.25775414435681426, 0.15364303768763587, 0.08927379601600503, 0.028374971942650477, 0.2255796288976688, 0.30726776244852594, 0.2613078071868208, 0.15723745952000875, 0.09166851255097981, 0.016401690122653013, 0.07680066606049299, 0.17051495789211965, 0.25756347512644895, 0.22917589195725704, 0.14345597825224318, 0.028540635289982418, 0.2191333515269955, 0.3077709722465091, 0.26611100315630887, 0.15744918978607697, 0.09211376612367395, 0.025711766826405517, 0.11519852160621068, 0.20755575981408897, 0.25337863877288286, 0.21547232893864385, 0.11538152570002093, 0.02864382144720606, 0.2150375180155995, 0.30835476947561474, 0.2697571730064888, 0.15785808436657078, 0.0926740719871156, 0.030305190926065848, 0.1356392079264193, 0.23630742236639826, 0.26120078967787363, 0.1874028137164711, 0.0983375964322519, 0.02863620196654976, 0.2120820469331072, 0.3078661809950926, 0.27214804507994983, 0.1585164905471032, 0.0936255838521098, 0.03180304881332352, 0.1462725073910679, 0.26130591432980993, 0.26802451678732225, 0.16545919102286305, 0.09228409024414023, 0.00849590105063802, 0.06639716073364745, 0.1806876689297011, 0.22708985948121094, 0.2554826586760373, 0.22468199078772885, 0.012788579556851174, 0.09948485130304931, 0.18193258018115557, 0.23408817497017945, 0.249439880842248, 0.21116417658442144, 0.0179440370238648, 0.13476248921366574, 0.18709289119254946, 0.20804945769184002, 0.2510967138132786, 0.24833730889777025, 0.007724920268053805, 0.05107927286035418, 0.14821083007922572, 0.2446201049765023, 0.1770274364539951, 0.15577350914229435, 0.022998849117636836, 0.16073507364713577, 0.18499300641963248, 0.18325441368493228, 0.25843219407360235, 0.25909965978766025, 0.0061361114839981485, 0.0424106424395526, 0.14957378637167554, 0.23890238033318908, 0.17710067858997758, 0.1465435347284228, 0.02534696177644019, 0.167594037247894, 0.18945135015321493, 0.16620924383047378, 0.2627480642382112, 0.2642361037558244, 0.00563894457892001, 0.04086072209470298, 0.1540776677199009, 0.2318682063812225, 0.1723383565160439, 0.14221218614655448, 0.02426421217954281, 0.1618277798312563, 0.19578803232365138, 0.15682001629262085, 0.2665494153652253, 0.26983031468088287, 0.005690116509300759, 0.04300389349000732, 0.16712764479578557, 0.23168421759502722, 0.16806399973484693, 0.1401338263729167, 0.04837977474251375, 0.36382277518324774, 0.26903254798859705, 0.23366906986398917, 0.21688033475747256, 0.13611765058016967, 0.04912957139734941, 0.3670787724759412, 0.25339087075541744, 0.2324818443163081, 0.23036081212154852, 0.16412768370969316, 0.05131056779181685, 0.35651229135305973, 0.24403925582681368, 0.23598888175625218, 0.23806605063718514, 0.15983836458276285, 0.035751206508754435, 0.16836043294694442, 0.2134584067514165, 0.31170292411059025, 0.13216838063237724, 0.055816792952179596, 0.05150273667937655, 0.3586464281343003, 0.23511589900053215, 0.2390159807978828, 0.24111356750134313, 0.15697124642034274, 0.035281637193279876, 0.1635398641107378, 0.2134842018599902, 0.30631835556401965, 0.14649028232765207, 0.06434135038489722, 0.0516607549111422, 0.35997890687836237, 0.23159273906502467, 0.2428066257808244, 0.2413597741554721, 0.15179077845397373, 0.03605870867113774, 0.16578847173195568, 0.2079534887741508, 0.3037189783336214, 0.16145103488201343, 0.07055985948689111, 0.05175337503509962, 0.36076681283297585, 0.2316134065297451, 0.2443570340954415, 0.24129107626471386, 0.1491679146846056, 0.03785651519536656, 0.17100023516051302, 0.20340704474956947, 0.30613108073292505, 0.1668113807583115, 0.07306481335749974, 0.05776040771926363, 0.3062074569829721, 0.2488337829914453, 0.21653107854557863, 0.21987193061221902, 0.1366458311327413, 0.05898879203091008, 0.31122307013187944, 0.24060383027868304, 0.21835742230957872, 0.23282299059934933, 0.15734340672944794, 0.059502660348350896, 0.29843611948861876, 0.22757087537143783, 0.2374273820548065, 0.2359978147425538, 0.14401282232094156, 0.03589806328434843, 0.1647094125967588, 0.20015365774999544, 0.3091546312936081, 0.16074385426774732, 0.09303812993843301, 0.05979191112569933, 0.2964114803201716, 0.226513118349555, 0.2406092591266036, 0.23908779344041187, 0.14053331086360782, 0.037970291319176175, 0.17041524503064362, 0.19544712135936546, 0.30085822421319064, 0.1808403819094727, 0.08996938124827149, 0.059937356588422835, 0.29608828504084855, 0.2255479678056641, 0.24283887046055644, 0.23716163151884093, 0.13877526032166695, 0.03939310718435899, 0.1743231257943989, 0.19292617086885483, 0.30057372066805116, 0.1907188612759668, 0.09043410786917258, 0.0600333760324745, 0.2952701892409166, 0.22718335593070663, 0.2434212505818407, 0.2350854550088805, 0.1395491633516862, 0.04119352041285707, 0.1790552835697478, 0.190322279481501, 0.30236335504728906, 0.19413011162696425, 0.09075176193107445, 0.05093459143814901, 0.3714268745081754, 0.2674141488687742, 0.24356418047575104, 0.18455714312227375, 0.10953826051377921, 0.056720210896951555, 0.3765385505371929, 0.2614223332606984, 0.24492007160493395, 0.18634585718010196, 0.11270161314323715, 0.03252701511727126, 0.2222700463694583, 0.26571745920314194, 0.25180776971491675, 0.22782707113477743, 0.138021688598819, 0.060255869208448005, 0.24330137089530557, 0.22906405593273294, 0.30642999399034343, 0.1942700644104102, 0.12431095507277902, 0.02758675740739174, 0.21630879782423365, 0.2341621475756373, 0.2546411961171804, 0.23407896934509068, 0.16784200038975583, 0.06412341617893606, 0.25475429461776333, 0.22965174557489026, 0.2992110571871804, 0.18681042447299875, 0.11506252121245203, 0.026133269546958635, 0.21219698005536908, 0.2168932783659063, 0.25545849032745993, 0.23501100372796072, 0.1776607654124222, 0.06529358624403075, 0.2589620321602964, 0.22577914784110742, 0.29772607641900584, 0.18170148728789876, 0.10925224560502886, 0.025990454253595804, 0.21230133192924586, 0.21000077564691505, 0.2510576648596567, 0.2338248439474098, 0.1787539893879318, 0.06502626544698181, 0.25899526195892075, 0.22184365427929315, 0.2979779231738272, 0.17911638145137296, 0.10488430259393078, 0.0641713325036627, 0.3290080969901031, 0.25085446004112155, 0.26320217580214333, 0.1434209080034849, 0.06456200207460203, 0.06828869776084427, 0.34042852927388634, 0.16924885487343727, 0.19202894373915613, 0.18829570068090526, 0.17749768931105114, 0.057883693221822075, 0.253488627227176, 0.2587567792090941, 0.24314122924559947, 0.2294465236964846, 0.1511873732708519, 0.05907325105918372, 0.28055398021615113, 0.2733978091699158, 0.2508050949626033, 0.18178835242069014, 0.11081664008233717, 0.049187561577818555, 0.31620148265652603, 0.2639241025408138, 0.2129805508344566, 0.23090875637633973, 0.21505787463434367, 0.06342659648256244, 0.3098789751676332, 0.22673615922005216, 0.2517214840307229, 0.19356940663985198, 0.1047460240491908, 0.014924366357223221, 0.12343793975586918, 0.25491304574909107, 0.22976969809399558, 0.2224155797455605, 0.12078384742373982, 0.050850532027169125, 0.3411441951686434, 0.26096595381998444, 0.21990096880562496, 0.19550598281777887, 0.14978168906351066, 0.06486106653189852, 0.32929575283799, 0.24714601079648238, 0.25379470506198465, 0.2279383804560103, 0.1848860509429824, 0.057067839438483474, 0.3642590747184851, 0.3163670798988193, 0.22558872610475741, 0.19486850725709093, 0.16799746196053747, 0.041695494129885054, 0.2911596617117273, 0.251267748864994, 0.22603415099901786, 0.2325457758534963, 0.21600519068007076, 0.689156833122512, 0.720226215514328, 0.697825005824643, 0.7146608494845486), dtype=np.float64)


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
    """Calibrate fire opportunity from a causal trailing annual window."""
    strength = float(np.clip(p.get("annual_residual_w", 0.0), 0.0, 1.5))
    if "fuel" not in enabled or strength <= 0.0:
        return prediction

    coefficients = _COUPLED_ANNUAL_COEFFICIENTS
    corrected = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        window = slice(start, time + 1)
        window_months = time - start + 1
        incumbent = np.asarray(prediction[window], dtype=np.float64).sum(axis=0)
        incumbent *= 12.0 / window_months
        log_current = np.log10(incumbent + 1e-6)
        index = 0
        residual = np.full((180, 360), 2.509046582748338, dtype=np.float64)

        def add(values: np.ndarray) -> None:
            nonlocal index, residual
            residual += coefficients[index] * values
            index += 1

        add(log_current)
        for threshold in (-5.0, -4.0, -3.0, -2.0, -1.0):
            add(np.maximum(log_current - threshold, 0.0))

        summaries: dict[str, dict[str, np.ndarray]] = {}
        for name in _COUPLED_ANNUAL_DRIVERS:
            trailing = np.asarray(data[name][window], dtype=np.float64)
            raw = {
                "mean": trailing.mean(axis=0),
                "std": trailing.std(axis=0),
                "p10": np.quantile(trailing, 0.10, axis=0),
                "p90": np.quantile(trailing, 0.90, axis=0),
            }
            if name in _COUPLED_ANNUAL_STATIC:
                raw = {"mean": raw["mean"]}
            summaries[name] = {}
            for statistic, values in raw.items():
                center, scale = _COUPLED_ANNUAL_SCALING[f"{name}:{statistic}"]
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
        corrected[time] = prediction[time] * correction
    return corrected


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

    gates: dict[str, np.ndarray] = {}
    for name in _COUPLED_SEASONAL_STATIC:
        center, scale = _COUPLED_SEASONAL_SCALING[name]
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


def _memory_spline_effect(
    values: np.ndarray,
    group: int,
) -> np.ndarray:
    """Evaluate one frozen cubic response curve without runtime dependencies."""
    base = np.asarray(_MEMORY_GAM_KNOTS[group], dtype=np.float64)
    left_step = base[1] - base[0]
    right_step = base[-1] - base[-2]
    knots = np.concatenate(
        (
            base[0] - left_step * np.arange(3, 0, -1),
            base,
            base[-1] + right_step * np.arange(1, 4),
        )
    )
    x = np.clip(np.asarray(values, dtype=np.float64), base[0], base[-1])
    x = np.minimum(x, np.nextafter(base[-1], -np.inf))
    basis = [
        ((x >= knots[index]) & (x < knots[index + 1])).astype(np.float64)
        for index in range(knots.size - 1)
    ]
    for degree in range(1, 4):
        refined: list[np.ndarray] = []
        for index in range(len(basis) - 1):
            left_denominator = knots[index + degree] - knots[index]
            right_denominator = knots[index + degree + 1] - knots[index + 1]
            left = 0.0
            right = 0.0
            if left_denominator > 0.0:
                left = (x - knots[index]) / left_denominator * basis[index]
            if right_denominator > 0.0:
                right = (
                    (knots[index + degree + 1] - x)
                    / right_denominator
                    * basis[index + 1]
                )
            refined.append(left + right)
        basis = refined
    start = group * 6
    coefficients = _MEMORY_GAM_COEFFICIENTS[start:start + 6]
    center = _MEMORY_GAM_CENTER[start:start + 6]
    scale = _MEMORY_GAM_SCALE[start:start + 6]
    effect = np.zeros_like(x, dtype=np.float64)
    for index in range(6):
        effect += coefficients[index] * (basis[index] - center[index]) / scale[index]
    return effect


def _causal_memory_gam(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Blend a smooth site-local moisture, fuel, and ignition memory closure."""
    strength = float(np.clip(p.get("memory_gam_w", 0.0), 0.0, 1.0))
    if "phenology" not in enabled or strength <= 0.0:
        return prediction

    group_index = {name: index for index, name in enumerate(_MEMORY_GAM_GROUPS)}
    eta = np.full(prediction.shape, _MEMORY_GAM_INTERCEPT, dtype=np.float64)

    def add_curve(name: str, values: np.ndarray) -> None:
        group = group_index[name]
        for time in range(values.shape[0]):
            eta[time] += _memory_spline_effect(values[time], group)

    add_curve("incumbent", prediction)
    memory_inputs = (
        "monthly_precipitation", "dryness", "air_temperature", "gpp",
        "leaf_area_index", "lightning_flash_rate",
    )
    for name in memory_inputs:
        raw = np.asarray(data[name], dtype=np.float64)
        add_curve(f"{name}:current", raw)
        previous = np.empty_like(raw)
        previous[0] = raw[0]
        previous[1:] = raw[:-1]
        add_curve(f"{name}:previous", previous)
        for timescale in (3, 6, 12, 24):
            alpha = 1.0 - np.exp(-1.0 / timescale)
            state = raw[0].copy()
            memory_group = group_index[f"{name}:memory_{timescale}m"]
            departure_group = group_index[f"{name}:departure_{timescale}m"]
            for time in range(raw.shape[0]):
                state += alpha * (raw[time] - state)
                eta[time] += _memory_spline_effect(state, memory_group)
                eta[time] += _memory_spline_effect(
                    raw[time] - state, departure_group
                )

    for name in (
        "aboveground_biomass", "natural_canopy_height",
        "secondary_canopy_height", "secondary_vegetation_fraction",
        "natural_vegetation_fraction", "soil_carbon",
        "annual_precipitation", "luh2_cropland_fraction",
        "luh2_pasture_fraction", "luh2_rangeland_fraction",
        "luh2_primary_fraction",
    ):
        add_curve(f"{name}:current", np.asarray(data[name], dtype=np.float64))

    calendar_start = 72 * 6
    month = np.arange(prediction.shape[0], dtype=np.float64) % 12
    angle = 2.0 * np.pi * month / 12.0
    for offset, wave in enumerate(
        (np.sin(angle), np.cos(angle), np.sin(2.0 * angle), np.cos(2.0 * angle))
    ):
        feature = (wave - _MEMORY_GAM_CENTER[calendar_start + offset]) / (
            _MEMORY_GAM_SCALE[calendar_start + offset] + 1e-12
        )
        eta += (
            _MEMORY_GAM_COEFFICIENTS[calendar_start + offset]
            * feature[:, None, None]
        )

    learned = np.exp(np.clip(eta, -30.0, 0.0))
    # The learned response diagnoses timing; the annual closure above owns the
    # amount of fuel that can burn. Rescale the learned response by causal
    # 12-month running means at each independent site so memory redistributes
    # the available fire potential without creating a second annual source.
    normalization_months = max(float(p.get("memory_norm_months", 12.0)), 1.0)
    alpha = 1.0 - np.exp(-1.0 / normalization_months)
    baseline_state = np.asarray(prediction[0], dtype=np.float64).copy()
    learned_state = learned[0].copy()
    allocated = np.empty_like(learned)
    for time in range(prediction.shape[0]):
        baseline_state += alpha * (prediction[time] - baseline_state)
        learned_state += alpha * (learned[time] - learned_state)
        allocated[time] = learned[time] * baseline_state / (
            learned_state + 1e-12
        )
    blended = (1.0 - strength) * prediction + strength * allocated
    return np.asarray(np.clip(blended, 0.0, 1.0), dtype=np.float32)


def _causal_mechanistic_glm(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Apply a compact causal fire-opportunity and reservoir anomaly."""
    strength = float(np.clip(p.get("causal_glm_w", 0.0), 0.0, 1.0))
    if "phenology" not in enabled or strength <= 0.0:
        return prediction

    def running(values: np.ndarray, months: float) -> np.ndarray:
        alpha = 1.0 - np.exp(-1.0 / months)
        state = np.asarray(values[0], dtype=np.float64).copy()
        output = np.empty_like(values, dtype=np.float64)
        for time in range(values.shape[0]):
            state += alpha * (values[time] - state)
            output[time] = state
        return output

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.clip(-values, -40.0, 40.0)))

    rain = np.asarray(data["monthly_precipitation"], dtype=np.float64)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    gpp = np.asarray(data["gpp"], dtype=np.float64)
    lightning = np.asarray(data["lightning_flash_rate"], dtype=np.float64)
    memories = {
        name: {
            months: running(values, months)
            for months in (3.0, 6.0, 12.0, 24.0)
        }
        for name, values in (
            ("rain", rain), ("temperature", temperature),
            ("gpp", gpp), ("lightning", lightning),
        )
    }

    rain_departure = {
        months: (memories["rain"][months] - rain)
        / (memories["rain"][months] + rain + 10.0)
        for months in (3.0, 6.0, 12.0, 24.0)
    }
    temperature_departure = {
        months: np.clip(
            (temperature - memories["temperature"][months]) / 10.0,
            -2.0,
            2.0,
        )
        for months in (6.0, 12.0, 24.0)
    }
    gpp_curing = {
        months: (memories["gpp"][months] - gpp)
        / (memories["gpp"][months] + gpp + 0.2)
        for months in (12.0, 24.0)
    }
    fuel_bank_6m = memories["gpp"][6.0] / (memories["gpp"][6.0] + 0.5)
    fuel_bank_12m = memories["gpp"][12.0] / (memories["gpp"][12.0] + 0.5)
    ignition_12m = memories["lightning"][12.0] / (
        memories["lightning"][12.0] + 0.01
    )

    annual_rain = np.asarray(data["annual_precipitation"], dtype=np.float64)
    seasonal_climate = sigmoid((annual_rain - 400.0) / 150.0) * sigmoid(
        (1700.0 - annual_rain) / 250.0
    )
    humid_climate = sigmoid((annual_rain - 1300.0) / 250.0)
    temperate = sigmoid((temperature - 2.0) / 3.0) * sigmoid(
        (20.0 - temperature) / 3.0
    )
    pasture = np.asarray(data["luh2_pasture_fraction"], dtype=np.float64)

    share = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        annual = np.asarray(prediction[start:time + 1], dtype=np.float64).sum(
            axis=0
        )
        annual *= 12.0 / (time - start + 1)
        share[time] = prediction[time] / (annual + 1e-12)

    features = (
        rain_departure[3.0],
        np.maximum(rain_departure[3.0], 0.0),
        temperature_departure[6.0],
        np.maximum(temperature_departure[6.0], 0.0),
        np.minimum(temperature_departure[12.0], 0.0),
        np.minimum(gpp_curing[12.0], 0.0),
        temperature_departure[24.0],
        np.maximum(temperature_departure[24.0], 0.0),
        np.minimum(gpp_curing[24.0], 0.0),
        rain_departure[6.0] * fuel_bank_6m,
        rain_departure[6.0] * seasonal_climate,
        rain_departure[6.0] * humid_climate,
        rain_departure[12.0] * fuel_bank_12m,
        rain_departure[12.0] * ignition_12m,
        rain_departure[12.0] * temperate,
        rain_departure[12.0] * humid_climate,
        rain_departure[24.0] * pasture,
        rain_departure[24.0] * seasonal_climate,
        sigmoid((share - 0.03) / 0.025),
        sigmoid((share - 0.08) / 0.025),
    )
    eta = np.full(prediction.shape, _CAUSAL_GLM_INTERCEPT, dtype=np.float64)
    for index, values in enumerate(features):
        eta += _CAUSAL_GLM_COEFFICIENTS[index] * (
            values - _CAUSAL_GLM_CENTER[index]
        ) / (_CAUSAL_GLM_SCALE[index] + 1e-12)

    factor = np.exp(np.clip(strength * eta, -8.0, 8.0))
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    state = np.asarray(factor[0], dtype=np.float64).copy()
    relative = np.empty_like(factor)
    for time in range(factor.shape[0]):
        state += alpha * (factor[time] - state)
        relative[time] = factor[time] / (state + 1e-12)
    return np.asarray(
        np.clip(prediction * relative, 0.0, 1.0), dtype=np.float32
    )


def _absolute_causal_glm(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Correct fire amount with a compact local fuel-drying response.

    A whole-cell held-out residual model identified the same response in each
    fold: burning rises or falls jointly with accumulated hydroclimate,
    available fuel, vegetation curing, climate state, and the amount of fire
    already supported by the base process model.  This is the reduced smooth
    mathematical surface, evaluated from current and past state only.
    """
    strength = float(np.clip(p.get("absolute_glm_w", 0.0), 0.0, 1.0))
    if "phenology" not in enabled or strength <= 0.0:
        return prediction

    def running(values: np.ndarray, months: float) -> np.ndarray:
        alpha = 1.0 - np.exp(-1.0 / months)
        state = np.asarray(values[0], dtype=np.float64).copy()
        output = np.empty_like(values, dtype=np.float64)
        for time in range(values.shape[0]):
            state += alpha * (values[time] - state)
            output[time] = state
        return output

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.clip(-values, -40.0, 40.0)))

    rain = np.asarray(data["monthly_precipitation"], dtype=np.float64)
    dryness = np.asarray(data["dryness"], dtype=np.float64)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    gpp = np.asarray(data["gpp"], dtype=np.float64)
    lai = np.asarray(data["leaf_area_index"], dtype=np.float64)

    rain_memory = {months: running(rain, months) for months in (6.0, 12.0, 24.0)}
    dryness_memory = {
        months: running(dryness, months) for months in (6.0, 12.0, 24.0)
    }
    gpp_memory = {months: running(gpp, months) for months in (6.0, 24.0)}
    lai_memory_6m = running(lai, 6.0)
    temperature_memory_6m = running(temperature, 6.0)

    rain_departure = {
        months: (state - rain) / (state + rain + 10.0)
        for months, state in rain_memory.items()
    }
    dryness_departure = {
        months: (dryness - state)
        / (np.abs(state) + np.abs(dryness) + 100.0)
        for months, state in dryness_memory.items()
    }
    gpp_curing = {
        months: (state - gpp) / (state + gpp + 0.2)
        for months, state in gpp_memory.items()
    }
    lai_curing_6m = (lai_memory_6m - lai) / (lai_memory_6m + lai + 0.5)
    temperature_departure_6m = np.clip(
        (temperature - temperature_memory_6m) / 10.0, -2.0, 2.0
    )
    fuel_bank_6m = gpp_memory[6.0] / (gpp_memory[6.0] + 0.5)
    fuel_bank_12m = running(gpp, 12.0)
    fuel_bank_12m = fuel_bank_12m / (fuel_bank_12m + 0.5)

    annual_rain = np.asarray(data["annual_precipitation"], dtype=np.float64)
    humid_climate = sigmoid((annual_rain - 1300.0) / 250.0)
    temperate = sigmoid((temperature - 2.0) / 3.0) * sigmoid(
        (20.0 - temperature) / 3.0
    )
    rangeland = np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64)

    share = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        annual = np.asarray(prediction[start:time + 1], dtype=np.float64).sum(
            axis=0
        )
        annual *= 12.0 / (time - start + 1)
        share[time] = prediction[time] / (annual + 1e-12)
    opportunity_003 = sigmoid((share - 0.03) / 0.025)
    opportunity_008 = sigmoid((share - 0.08) / 0.025)

    features = (
        dryness_departure[6.0],
        np.minimum(temperature_departure_6m, 0.0),
        gpp_curing[6.0],
        np.maximum(gpp_curing[6.0], 0.0),
        lai_curing_6m,
        np.minimum(lai_curing_6m, 0.0),
        np.maximum(dryness_departure[12.0], 0.0),
        dryness_departure[24.0],
        np.maximum(dryness_departure[24.0], 0.0),
        gpp_curing[24.0],
        np.maximum(gpp_curing[24.0], 0.0),
        rain_departure[6.0] * fuel_bank_6m,
        rain_departure[6.0] * humid_climate,
        rain_departure[12.0] * fuel_bank_12m,
        rain_departure[12.0] * temperate,
        rain_departure[12.0] * humid_climate,
        rain_departure[24.0] * rangeland,
        opportunity_003,
        opportunity_003 * rain_departure[12.0],
        opportunity_008,
    )
    eta = np.full(prediction.shape, _ABSOLUTE_GLM_INTERCEPT, dtype=np.float64)
    for index, values in enumerate(features):
        eta += _ABSOLUTE_GLM_COEFFICIENTS[index] * (
            values - _ABSOLUTE_GLM_CENTER[index]
        ) / (_ABSOLUTE_GLM_SCALE[index] + 1e-12)
    factor = np.exp(np.clip(strength * eta, -8.0, 8.0))
    return np.asarray(
        np.clip(prediction * factor, 0.0, 1.0), dtype=np.float32
    )


def _ecological_regime_brakes(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Smooth local suppression in two physically distinct fire regimes.

    Cool cultivated landscapes are fragmented and repeatedly harvested, while
    humid tall closed canopy keeps fine fuel shaded and moist. Both gates are
    continuous functions of observable local state with global coefficients;
    no coordinates, region labels, or geographic branches enter.
    """
    def current_state(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64)

    temperature = current_state("air_temperature")
    log_brake = np.zeros_like(prediction, dtype=np.float64)
    if "cropland" in enabled:
        cropland = current_state("luh2_cropland_fraction")
        cool_cultivation = cropland * _falling(temperature, 1.0 / 3.0, 18.0)
        log_brake += p.get("cool_crop_brake", 0.0) * cool_cultivation
    if "fuel" in enabled:
        annual_rain = current_state("annual_precipitation")
        canopy = current_state("natural_canopy_height")
        leaf_area = current_state("leaf_area_index")
        natural = current_state("natural_vegetation_fraction")
        humid_closed_canopy = (
            _rising(temperature, 0.5, 20.0)
            * _rising(annual_rain, 1.0 / 250.0, 1200.0)
            * _rising(canopy, 1.0 / 3.0, 15.0)
            * _rising(leaf_area, 2.0, 2.5)
            * natural
        )
        log_brake += p.get("wet_forest_brake", 0.0) * humid_closed_canopy
    return prediction * np.exp(-log_brake)


def _ecological_fire_capacity(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Resolve fire-size regimes from local vegetation and climate state.

    Cold natural forest can carry rare crown fires even below the warm-season
    ignition threshold. Warm arid open land can retain a small antecedent grass
    bank that burns after curing. Conversely, productive managed rangeland is
    fragmented and grazed, limiting the contiguous fuel available to a front.
    These are smooth local-state mechanisms with one coefficient set globally.
    """
    if "regime_capacity" not in enabled:
        return prediction

    temperature = _antecedent(
        np.asarray(data["air_temperature"], dtype=np.float64),
        1.0 - np.exp(-1.0 / 24.0),
    )
    gpp = _antecedent(
        np.asarray(data["gpp"], dtype=np.float64),
        1.0 - np.exp(-1.0 / 12.0),
    )
    biomass = np.asarray(data["aboveground_biomass"], dtype=np.float64)
    natural = np.asarray(data["natural_vegetation_fraction"], dtype=np.float64)
    canopy = np.asarray(data["natural_canopy_height"], dtype=np.float64)
    rangeland = np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64)
    annual_rain = np.asarray(data["annual_precipitation"], dtype=np.float64)

    cold_forest = (
        _falling(temperature, 1.0 / 3.0, 8.0)
        * biomass / (biomass + 2.0)
        * canopy / (canopy + 8.0)
        * natural
    )
    open_land = np.clip(
        natural * 10.0 / (canopy + 10.0) + rangeland, 0.0, 1.0
    )
    arid_fine_fuel = (
        _rising(temperature, 1.0 / 3.0, 10.0)
        * _falling(annual_rain, 1.0 / 150.0, 600.0)
        * gpp / (gpp + 0.2)
        * open_land
    )
    productive_range = (
        rangeland
        * _rising(annual_rain, 1.0 / 120.0, 250.0)
        * _falling(annual_rain, 1.0 / 250.0, 1500.0)
        * biomass / (biomass + 0.2)
    )
    log_capacity = (
        p.get("cold_forest_capacity", 0.0) * cold_forest
        + p.get("arid_fine_fuel_capacity", 0.0) * arid_fine_fuel
        - p.get("productive_range_brake", 0.0) * productive_range
    )
    return np.asarray(
        np.clip(prediction * np.exp(np.clip(log_capacity, -5.0, 5.0)), 0.0, 1.0),
        dtype=np.float32,
    )


def _rare_lightning_ignition(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Supply rare natural ignitions missed by the continuous fire hazard.

    A lightning flash does not imply burned area: it must coincide with warm,
    rain-free fuel, and that fuel must be continuous enough to carry a front.
    The term combines lightning with antecedent productivity and woody biomass,
    then fades smoothly where the existing trailing fire opportunity is already
    large. This lets rare crown or open-land events occur without amplifying
    active savannas or creating fire in fuel-free deserts.
    """
    if "rare_ignition" not in enabled:
        return prediction
    scale = float(max(p.get("rare_ignition_scale", 0.0), 0.0))
    if scale <= 0.0:
        return prediction

    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_chance = lightning / (lightning + 0.02)
    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_window = 1.0 / (1.0 + rain / 25.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    thermal_window = _rising(temperature, 1.0 / 3.0, 5.0)

    gpp = _antecedent(
        np.asarray(data["gpp"], dtype=np.float64),
        1.0 - np.exp(-1.0 / 12.0),
    )
    fine_fuel = gpp / (gpp + 0.35)
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    woody_fuel = biomass / (biomass + 1.0)
    fuel_continuity = 1.0 - (1.0 - fine_fuel) * (1.0 - woody_fuel)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    leaf_area = np.clip(
        np.asarray(data["leaf_area_index"], dtype=np.float64), 0.0, None
    )
    annual_rain = np.clip(
        np.asarray(data["annual_precipitation"], dtype=np.float64), 0.0, None
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    temperature_memory = _antecedent(
        temperature, 1.0 - np.exp(-1.0 / 24.0)
    )
    open_natural = natural * 10.0 / (canopy + 10.0)
    cold_forest = (
        natural
        * canopy / (canopy + 8.0)
        * _falling(temperature_memory, 1.0 / 3.0, 8.0)
    )
    humid_closed_canopy = (
        _rising(temperature, 0.5, 20.0)
        * _rising(annual_rain, 1.0 / 250.0, 1200.0)
        * _rising(canopy, 1.0 / 3.0, 15.0)
        * _rising(leaf_area, 2.0, 2.5)
        * natural
    )
    canopy_access = np.exp(-4.0 * humid_closed_canopy)
    burnable_land = np.clip(
        (open_natural + cold_forest + 0.5 * rangeland) * canopy_access,
        0.0,
        1.0,
    )

    trailing = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        annual = np.asarray(prediction[start : time + 1], dtype=np.float64).sum(
            axis=0
        )
        annual *= 12.0 / (time - start + 1)
        trailing[time] = annual
    opportunity_gap = 1.0 / (1.0 + trailing / 0.2)
    ignition = (
        scale
        * lightning_chance
        * rain_window
        * thermal_window
        * fuel_continuity
        * burnable_land
        * opportunity_gap
    )
    precipitation_memory = _antecedent(
        rain, 1.0 - np.exp(-1.0 / 12.0)
    )
    rain_built_fuel = (
        precipitation_memory / (precipitation_memory + 25.0)
        * _rising(annual_rain, 1.0 / 60.0, 250.0)
        * _falling(annual_rain, 1.0 / 250.0, 1400.0)
    )
    low_woody_biomass = 1.0 / (1.0 + biomass / 0.7)
    open_fuel_land = np.clip(open_natural, 0.0, 1.0)
    dry_combustion = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    dry_combustion = dry_combustion / (dry_combustion + 500.0)
    fuel_drying = np.maximum(
        (precipitation_memory - rain)
        / (precipitation_memory + rain + 10.0),
        0.0,
    )
    ignition_access = lightning_chance
    pulse_half = max(
        float(p.get("rain_pulse_opportunity_half", 0.02)), 1e-4
    )
    pulse_gap = 1.0 / (
        1.0 + np.power(np.maximum(trailing, 0.0) / pulse_half, 0.75)
    )
    rain_pulse_ignition = (
        max(float(p.get("rain_pulse_ignition_scale", 0.0)), 0.0)
        * rain_built_fuel
        * low_woody_biomass
        * open_fuel_land
        * dry_combustion
        * fuel_drying
        * thermal_window
        * ignition_access
        * pulse_gap
    )
    ignition += rain_pulse_ignition
    crown_scale = float(max(p.get("crown_fire_event_scale", 0.0), 0.0))
    if crown_scale > 0.0:
        # Crown-fire area is an event-size process, not a smooth multiplier of
        # ordinary surface fire. A cold woody landscape must first accumulate
        # a deep dry anomaly and receive lightning during a long fire-return
        # gap; only then can one ignition release a large connected event.
        dryness = np.clip(
            np.asarray(data["dryness"], dtype=np.float64), 0.0, None
        )
        dryness_memory = _antecedent(
            dryness, 1.0 - np.exp(-1.0 / 12.0)
        )
        drought_anomaly = np.maximum(
            (dryness - dryness_memory)
            / (dryness + dryness_memory + 100.0),
            0.0,
        )
        lightning_memory = _antecedent(
            lightning, 1.0 - np.exp(-1.0 / 12.0)
        )
        persistent_lightning = lightning_memory / (lightning_memory + 0.01)
        crown_fuel = cold_forest * woody_fuel
        crown_event = (
            crown_scale
            * crown_fuel
            * persistent_lightning
            * drought_anomaly
            * dry_combustion
            * thermal_window
            * pulse_gap
            * canopy_access
        )
        ignition += crown_event
    return np.asarray(
        np.clip(1.0 - (1.0 - prediction) * np.exp(-ignition), 0.0, 1.0),
        dtype=np.float32,
    )


def _state_dependent_fire_season(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Concentrate recurrent fire into the local combustible dry phase."""
    strength = float(max(p.get("fire_season_w", 0.0), 0.0))
    if "phenology" not in enabled or strength <= 0.0:
        return prediction
    fire_half = max(float(p.get("fire_season_half", 0.04)), 1e-4)
    dry_half = max(float(p.get("fire_season_dry_half", 500.0)), 1e-3)
    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    dry_phase = dryness / (dryness + dry_half)
    trailing = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        annual = np.asarray(prediction[start : time + 1], dtype=np.float64).sum(
            axis=0
        )
        trailing[time] = annual * 12.0 / (time - start + 1)
    recurrent = trailing / (trailing + fire_half)
    factor = np.exp(np.clip(strength * recurrent * dry_phase, -5.0, 5.0))
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    state = np.asarray(factor[0], dtype=np.float64).copy()
    relative = np.empty_like(factor)
    for time in range(factor.shape[0]):
        state += alpha * (factor[time] - state)
        relative[time] = factor[time] / (state + 1e-12)
    return np.asarray(
        np.clip(prediction * relative, 0.0, 1.0), dtype=np.float32
    )


def _seasonal_rainfall_capacity(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Expand burnable capacity where alternating rain builds and cures fuel."""
    strength = float(max(p.get("seasonal_rain_capacity", 0.0), 0.0))
    if "regime_capacity" not in enabled or strength <= 0.0:
        return prediction
    rain = np.asarray(data["monthly_precipitation"], dtype=np.float64)
    factor = np.empty_like(prediction, dtype=np.float64)
    for time in range(prediction.shape[0]):
        start = max(0, time - 11)
        window = rain[start : time + 1]
        rain_variability = window.std(axis=0)
        seasonal_pump = _rising(rain_variability, 1.0 / 15.0, 55.0)
        annual_fire = np.asarray(
            prediction[start : time + 1], dtype=np.float64
        ).sum(axis=0)
        annual_fire *= 12.0 / (time - start + 1)
        moderate_opportunity = (
            _rising(annual_fire, 1.0 / 0.04, 0.12)
            * _falling(annual_fire, 1.0 / 0.08, 0.35)
        )
        factor[time] = np.exp(
            np.clip(strength * seasonal_pump * moderate_opportunity, 0.0, 5.0)
        )
    return np.asarray(np.clip(prediction * factor, 0.0, 1.0), dtype=np.float32)


def _live_fuel_greenup_brake(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Suppress burning while warm vegetation is rapidly producing live tissue.

    A positive departure of current GPP from its recent local reservoir marks
    active green-up. That new biomass is moist live fuel, not cured litter, so
    it should not immediately increase spread even though productivity is high.
    The response is continuous and site-local, with one coefficient everywhere.
    """
    strength = float(max(p.get("greenup_brake", 0.0), 0.0))
    if "phenology" not in enabled or strength <= 0.0:
        return prediction
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    recent = _antecedent(gpp, 1.0 - np.exp(-1.0 / 3.0))
    greenup = np.maximum((gpp - recent) / (gpp + recent + 1e-3), 0.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    warm_growth = _rising(temperature, 1.0 / 3.0, 15.0)
    brake = np.exp(-strength * greenup * warm_growth)
    # Green-up changes when the available annual fuel burns; it does not erase
    # that fuel stock. Divide by a causal local running reference so suppressed
    # live-fuel months are reallocated toward the subsequent cured phase.
    adjusted = np.asarray(prediction, dtype=np.float64) * brake
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    baseline_state = np.asarray(prediction[0], dtype=np.float64).copy()
    adjusted_state = adjusted[0].copy()
    allocated = np.empty_like(adjusted)
    for time in range(brake.shape[0]):
        baseline_state += alpha * (prediction[time] - baseline_state)
        adjusted_state += alpha * (adjusted[time] - adjusted_state)
        allocated[time] = adjusted[time] * baseline_state / (
            adjusted_state + 1e-12
        )
    return np.asarray(allocated, dtype=np.float32)



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
        warming[0] = 0.0
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
    # Set fire potential from a trailing annual window. Seasonal timing then
    # remains a causal function of current and accumulated local state.
    prediction = _coupled_annual_correction(prediction, data, fallback, enabled)
    prediction = _ecological_regime_brakes(prediction, data, fallback, enabled)
    prediction = np.clip(
        prediction * fallback.get("annual_scale", 1.0), 0.0, 1.0
    )
    prediction = _causal_memory_gam(prediction, data, fallback, enabled)
    prediction = _causal_mechanistic_glm(prediction, data, fallback, enabled)
    prediction = _absolute_causal_glm(prediction, data, fallback, enabled)
    prediction = _ecological_fire_capacity(prediction, data, fallback, enabled)
    prediction = _seasonal_rainfall_capacity(
        prediction, data, fallback, enabled
    )
    prediction = _state_dependent_fire_season(
        prediction, data, fallback, enabled
    )
    prediction = _rare_lightning_ignition(prediction, data, fallback, enabled)
    prediction = _live_fuel_greenup_brake(
        prediction, data, fallback, enabled
    )
    return np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
