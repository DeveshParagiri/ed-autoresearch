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
INPUTS = ('dryness', 'annual_precipitation', 'monthly_precipitation', 'air_temperature', 'gpp',
          'luh2_cropland_fraction', 'luh2_rangeland_fraction',
          'aboveground_biomass',
          'luh2_primary_fraction', 'lightning_flash_rate', 'soil_carbon',
          'leaf_area_index', 'natural_canopy_height', 'secondary_canopy_height',
          'natural_vegetation_fraction', 'secondary_vegetation_fraction',
          'luh2_pasture_fraction', 'luh2_secondary_fraction', 'luh2_urban_fraction')
COMPONENTS = ('dryness', 'precipitation', 'fuel', 'temperature', 'curing',
              'cropland', 'phenology', 'regime_capacity',
              'rare_ignition', 'drought_maturation', 'dead_fuel_pool',
              'pathway_hazards', 'surface_opportunity_bank',
              'annual_regime_closure')

# Calibrate only the two pathway banks that supplied the validated structural gain.
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    'managed_bank_store': {'type': 'float', 'low': 0.2, 'high': 1.0},
    'managed_bank_release': {'type': 'float', 'low': 4.0, 'high': 24.0},
    'crop_bank_store': {'type': 'float', 'low': 0.4, 'high': 1.0},
    'crop_bank_release': {'type': 'float', 'low': 12.0, 'high': 48.0},
}

PARAMS = {'annual_scale': 1.73,
 'event_scale_half': 0.003,
 'pathway_mix_w': 0.35,
 'fire_footprint_background': 0.45792078774131323,
 'fire_footprint_w': 2.3081972320719712,
 'fire_footprint_natural_w': 0.7774112521824282,
 'fire_footprint_lightning_half': 0.06098759667228644,
 'fire_footprint_managed_half': 0.18933595753561624,
 'persistent_warm_open_brake': 5.0,
 'cold_thaw_source': 0.1,
 'surface_bank_w': 1.0,
 'surface_bank_release': 8.0,
 'managed_bank_store': 0.9429500974702053,
 'managed_bank_release': 13.738787487060241,
 'crop_bank_store': 0.9923980795361608,
 'crop_bank_release': 40.07825420342421,
 'woody_bank_store': 0.4,
 'woody_bank_release': 12.0,
 'background_bank_store': 0.1,
 'background_bank_release': 8.0,
 'conditional_allocation_w': 1.0,
 'cool_crop_brake': 4.5,
 'wet_forest_brake': 3.0,
 'cold_forest_capacity': 3.0,
 'arid_fine_fuel_capacity': 2.0,
 'productive_range_brake': 6.5,
 'seasonal_rain_capacity': 0.4,
 'fire_season_w': 0.3,
 'fire_season_half': 0.04,
 'fire_season_dry_half': 500.0,
 'drought_maturation_w': 2.0,
 'dead_fuel_pool_w': 3.0,
 'dead_fuel_decay': 0.08,
 'dead_fuel_consumption': 2.0,
 'greenup_brake': 2.0,
 'rare_ignition_scale': 0.02,
 'rain_pulse_ignition_scale': 0.24,
 'rain_pulse_opportunity_half': 0.02,
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
 'month_scale': 0.04298969468924071,
 'lag_w': 0.18862814833689176,
 'soft_w': 1.0,
 'soft_s': 2.0,
 'crop_k': 2.5,
 'crop_n': 1.514,
 'crop_rain_management_w': 0.0,
 'crop_residue_event_scale': 0.25,
}

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


def _pathway_event_scaling(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Mix surface, woody, and residue fire in local hazard space.

    The pathways share one global equation but respond to different physical
    controls. Surface fire needs connected open fine fuel and drying. Woody
    events require mature drought, anomalous warmth, and lightning. Crop fire
    is limited by a cured residue stock. A background share retains unresolved
    ignition without assigning a region or coordinate to any pathway.
    """
    annual_scale = float(max(p.get("annual_scale", 1.0), 0.0))
    event_half = float(max(p.get("event_scale_half", 0.003), 1e-8))
    connected = prediction / (prediction + event_half)
    old_scale = 1.0 + (annual_scale - 1.0) * connected
    old_burn = np.clip(prediction * old_scale, 0.0, 1.0 - 1e-7)
    if "pathway_hazards" not in enabled:
        return old_burn
    mix = float(np.clip(p.get("pathway_mix_w", 0.0), 0.0, 1.0))
    if mix <= 0.0:
        return old_burn

    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_3 = _antecedent(gpp, alpha_3)
    gpp_12 = _antecedent(gpp, alpha_12)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_6 = _antecedent(rain, alpha_6)
    rain_12 = _antecedent(rain, alpha_12)
    dry_6 = np.maximum((rain_6 - rain) / (rain_6 + rain + 10.0), 0.0)
    dry_12 = np.maximum((rain_12 - rain) / (rain_12 + rain + 10.0), 0.0)
    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    combustion = dryness / (dryness + 500.0)

    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel

    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_12 = _antecedent(temperature, alpha_12)
    warm_anomaly = _rising(temperature - temperature_12, 0.5, 3.0)
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_12 = _antecedent(lightning, alpha_12)
    ignition = lightning_12 / (lightning_12 + 0.01)
    annual_rain = np.clip(
        np.asarray(data["annual_precipitation"], dtype=np.float64), 0.0, None
    )
    humid_closed = (
        _rising(annual_rain, 1.0 / 250.0, 1200.0)
        * _rising(canopy, 1.0 / 3.0, 15.0)
        * natural
    )

    surface_available = dry_6 * combustion
    woody_available = (
        dry_12 * warm_anomaly * ignition * np.exp(-3.0 * humid_closed)
    )
    residue_curing = np.maximum(
        (gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0
    )
    crop_available = residue_curing * combustion

    background = np.full_like(surface_capacity, 0.05)
    total_capacity = (
        background + surface_capacity + woody_capacity + crop_capacity + 1e-12
    )
    q0 = background / total_capacity
    qs = surface_capacity / total_capacity
    qw = woody_capacity / total_capacity
    qc = crop_capacity / total_capacity

    surface_scale = 1.0 + 1.1 * connected * (
        0.35 + 0.65 * surface_available
    )
    woody_scale = 0.65 + 1.85 * woody_available / (woody_available + 0.015)
    crop_scale = 0.60 + 1.20 * crop_available / (crop_available + 0.06)
    new_scale = q0 * old_scale + qs * surface_scale + qw * woody_scale + qc * crop_scale

    base_hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    old_hazard = -np.log1p(-old_burn)
    new_hazard = base_hazard * new_scale
    hazard = (1.0 - mix) * old_hazard + mix * new_hazard
    return np.asarray(1.0 - np.exp(-np.clip(hazard, 0.0, 50.0)), dtype=np.float32)


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


def _rain_conditioned_crop_management(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Separate fuel removal from residue fire in cultivated landscapes.

    In weakly seasonal productive cropland, harvest, grazing, roads, and field
    boundaries remove or fragment fuel before it can support a spreading fire.
    In strongly seasonal cropland, antecedent rain can instead build residue
    that becomes burnable as the wet season ends. Rainfall variability thus
    changes the role of the same observable land use without any geographic
    dispatch or region-specific coefficient.
    """
    if "cropland" not in enabled:
        return prediction
    brake_strength = float(max(p.get("crop_rain_management_w", 0.0), 0.0))
    event_scale = float(max(p.get("crop_residue_event_scale", 0.0), 0.0))
    if brake_strength <= 0.0 and event_scale <= 0.0:
        return prediction

    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    mean = np.asarray(rain[0], dtype=np.float64).copy()
    variance = np.zeros_like(mean)
    rain_mean = np.empty_like(rain, dtype=np.float64)
    rain_spread = np.empty_like(rain, dtype=np.float64)
    for time in range(rain.shape[0]):
        departure = rain[time] - mean
        mean += alpha * departure
        variance = (1.0 - alpha) * (
            variance + alpha * np.square(departure)
        )
        rain_mean[time] = mean
        rain_spread[time] = np.sqrt(np.maximum(variance, 0.0))
    annual_rain = np.clip(
        np.asarray(data["annual_precipitation"], dtype=np.float64), 0.0, None
    )
    productive = annual_rain / (annual_rain + 400.0)
    highly_seasonal = _rising(rain_spread, 1.0 / 15.0, 60.0)
    fragmentation = crop * productive * highly_seasonal
    baseline = np.asarray(prediction, dtype=np.float64)
    adjusted = baseline * np.exp(
        -brake_strength * fragmentation
    )

    if event_scale > 0.0:
        fuel = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
        fuel_memory = _antecedent(fuel, alpha)
        residue = fuel_memory / (fuel_memory + 0.35)
        drying = np.maximum(
            (rain_mean - rain) / (rain_mean + rain + 10.0), 0.0
        )
        dryness = np.clip(
            np.asarray(data["dryness"], dtype=np.float64), 0.0, None
        )
        combustion = dryness / (dryness + 500.0)
        temperature = np.asarray(data["air_temperature"], dtype=np.float64)
        warming = np.empty_like(temperature)
        warming[0] = 0.0
        warming[1:] = np.maximum(temperature[1:] - temperature[:-1], 0.0)
        thermal_shoulder = _rising(temperature, 1.0 / 3.0, 5.0) * np.exp(
            -np.square((temperature - 18.0) / 12.0)
        )
        residue_event = (
            event_scale
            * crop
            * _falling(rain_spread, 1.0 / 15.0, 60.0)
            * residue
            * drying
            * combustion
            * thermal_shoulder
            * np.clip(warming / 5.0, 0.0, 1.0)
        )
        adjusted = 1.0 - (1.0 - adjusted) * np.exp(-residue_event)
        # Residue burning changes the timing of a cultivated fuel stock; it
        # does not manufacture additional annual fuel. Conserve that stock
        # against a causal local 12-month reference.
        baseline_state = baseline[0].copy()
        adjusted_state = adjusted[0].copy()
        allocated = np.empty_like(adjusted)
        for time in range(adjusted.shape[0]):
            baseline_state += alpha * (baseline[time] - baseline_state)
            adjusted_state += alpha * (adjusted[time] - adjusted_state)
            allocated[time] = adjusted[time] * baseline_state / (
                adjusted_state + 1e-12
            )
        adjusted = allocated
    return np.asarray(np.clip(adjusted, 0.0, 1.0), dtype=np.float32)


def _drought_maturation_response(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Distinguish mature drought from a short-lived dry anomaly.

    Woody fuel in cold or humid landscapes does not become available after one
    dry month. Fire opportunity rises only when the 12-month rainfall reservoir
    remains above the shorter reservoir, indicating that drying has persisted
    long enough to cure deep fuels. Conversely, a six-month flash anomaly is
    damped before deep curing. Open rangeland carries a weaker two-year fuel
    memory. These are continuous local-state responses with one global scale.
    """
    if "drought_maturation" not in enabled:
        return prediction
    strength = float(max(p.get("drought_maturation_w", 0.0), 0.0))
    if strength <= 0.0:
        return prediction

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_memory = {
        months: _antecedent(rain, 1.0 - np.exp(-1.0 / months))
        for months in (6.0, 12.0, 24.0)
    }
    deficit = {
        months: np.maximum(
            (state - rain) / (state + rain + 10.0), 0.0
        )
        for months, state in rain_memory.items()
    }
    mature = deficit[12.0] * np.maximum(
        deficit[12.0] - deficit[6.0], 0.0
    )
    flash = deficit[6.0] * np.maximum(
        deficit[6.0] - deficit[12.0], 0.0
    )
    legacy = deficit[24.0] * np.maximum(
        deficit[24.0] - deficit[12.0], 0.0
    )

    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_memory = _antecedent(
        temperature, 1.0 - np.exp(-1.0 / 24.0)
    )
    cold = _falling(temperature_memory, 1.0 / 3.0, 8.0)
    annual_rain = np.clip(
        np.asarray(data["annual_precipitation"], dtype=np.float64), 0.0, None
    )
    humid = _rising(annual_rain, 1.0 / 250.0, 1300.0)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    woody_fuel = (
        natural
        * canopy / (canopy + 8.0)
        * biomass / (biomass + 1.0)
    )
    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_memory = _antecedent(
        lightning, 1.0 - np.exp(-1.0 / 12.0)
    )
    ignition = lightning_memory / (lightning_memory + 0.01)
    woody_regime = woody_fuel * np.clip(cold + humid, 0.0, 1.0) * ignition

    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    fine_fuel = _antecedent(gpp, 1.0 - np.exp(-1.0 / 24.0))
    fine_fuel = fine_fuel / (fine_fuel + 0.35)
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    response = woody_regime * (mature - 0.5 * flash)
    response += 0.5 * rangeland * fine_fuel * legacy
    factor = np.exp(np.clip(strength * response, -4.0, 4.0))
    return np.asarray(
        np.clip(prediction * factor, 0.0, 1.0), dtype=np.float32
    )


def _dead_fuel_pool_response(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Track distinct causal herbaceous and woody litter pools.

    Herbaceous litter follows rapid GPP and leaf-area curing on open land.
    Woody litter accumulates slowly under biomass and canopy, but becomes
    burnable only during mature drought with anomalous warmth and lightning.
    Warm wet conditions accelerate decomposition in both stores; cold or dry
    conditions preserve litter between years. Fire consumes each available
    pool. The response is causal, site-local, and uses one global equation.
    """
    if "dead_fuel_pool" not in enabled:
        return prediction
    strength = float(max(p.get("dead_fuel_pool_w", 0.0), 0.0))
    if strength <= 0.0:
        return prediction
    decay = float(np.clip(p.get("dead_fuel_decay", 0.08), 0.001, 0.5))
    consumption = float(
        np.clip(p.get("dead_fuel_consumption", 2.0), 0.0, 20.0)
    )

    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    lai = np.clip(
        np.asarray(data["leaf_area_index"], dtype=np.float64), 0.0, None
    )
    gpp_recent = _antecedent(gpp, 1.0 - np.exp(-1.0 / 3.0))
    gpp_bank = _antecedent(gpp, 1.0 - np.exp(-1.0 / 12.0))
    lai_recent = _antecedent(lai, 1.0 - np.exp(-1.0 / 3.0))
    gpp_curing = np.maximum(
        (gpp_recent - gpp) / (gpp_recent + gpp + 0.2), 0.0
    )
    lai_curing = np.maximum(
        (lai_recent - lai) / (lai_recent + lai + 0.5), 0.0
    )
    production = gpp_bank / (gpp_bank + 0.35) * (
        0.7 * gpp_curing + 0.3 * lai_curing
    )

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_memory = _antecedent(rain, 1.0 - np.exp(-1.0 / 6.0))
    rain_long = _antecedent(rain, 1.0 - np.exp(-1.0 / 12.0))
    drying = np.maximum(
        (rain_memory - rain) / (rain_memory + rain + 10.0), 0.0
    )
    mature_drought = np.maximum(
        (rain_long - rain) / (rain_long + rain + 10.0), 0.0
    )
    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    combustion = drying * dryness / (dryness + 500.0)

    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    woody_cover = natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)

    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_memory = _antecedent(
        temperature, 1.0 - np.exp(-1.0 / 12.0)
    )
    warm = _rising(temperature, 1.0 / 4.0, 15.0)
    warm_anomaly = _rising(temperature - temperature_memory, 1.0 / 2.0, 3.0)
    wet = rain / (rain + rain_memory + 10.0)
    decomposition = decay * (0.25 + 1.5 * warm * wet)

    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_memory = _antecedent(
        lightning, 1.0 - np.exp(-1.0 / 12.0)
    )
    woody_ignition = lightning_memory / (lightning_memory + 0.01)
    woody_production = woody_cover * gpp_bank / (gpp_bank + 0.35)

    fine_stock = np.asarray(production[0], dtype=np.float64).copy()
    woody_stock = np.asarray(woody_production[0], dtype=np.float64).copy()
    available = np.empty_like(production, dtype=np.float64)
    for time in range(prediction.shape[0]):
        fine_stock = (
            fine_stock * np.exp(-decomposition[time]) + production[time]
        )
        woody_stock = (
            woody_stock * np.exp(-0.25 * decomposition[time])
            + 0.04 * woody_production[time]
        )
        burn_pressure = prediction[time] / (prediction[time] + 0.04)
        fine_available = (
            fine_stock / (fine_stock + 0.5)
            * combustion[time]
            * (0.35 + 0.65 * open_cover[time])
        )
        woody_available = (
            woody_stock / (woody_stock + 1.0)
            * mature_drought[time]
            * warm_anomaly[time]
            * woody_ignition[time]
        )
        available[time] = (
            fine_available * (0.5 + open_cover[time])
            + 0.5 * woody_available
        )
        fine_stock *= np.exp(
            -consumption * burn_pressure * fine_available
        )
        woody_stock *= np.exp(
            -0.5 * consumption * burn_pressure * woody_available
        )

    factor = np.exp(np.clip(strength * available, 0.0, 4.0))
    baseline = np.asarray(prediction, dtype=np.float64)
    adjusted = baseline * factor
    # The litter pool redistributes locally produced fuel across time. A causal
    # running reference holds the site's available fire potential fixed.
    alpha = 1.0 - np.exp(-1.0 / 12.0)
    baseline_state = baseline[0].copy()
    adjusted_state = adjusted[0].copy()
    allocated = np.empty_like(adjusted)
    for time in range(adjusted.shape[0]):
        baseline_state += alpha * (baseline[time] - baseline_state)
        adjusted_state += alpha * (adjusted[time] - adjusted_state)
        allocated[time] = adjusted[time] * baseline_state / (
            adjusted_state + 1e-12
        )
    return np.asarray(np.clip(allocated, 0.0, 1.0), dtype=np.float32)


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


def _conditional_fire_allocation(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Move local fire potential into mature fuel-supported dry windows.

    Low and moderate fire opportunity expands only when a twelve-month rain
    reservoir, antecedent productivity, vegetation curing, and anomalous warmth
    jointly support combustion. Already concentrated peak months receive a
    short-drought brake instead of further amplification. Causal running means
    conserve each independent site's available fire potential through time.
    """
    if "conditional_allocation" not in enabled:
        return prediction
    strength = float(max(p.get("conditional_allocation_w", 0.0), 0.0))
    if strength <= 0.0:
        return prediction

    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_6 = _antecedent(rain, alpha_6)
    rain_12 = _antecedent(rain, alpha_12)
    dry_6 = np.maximum((rain_6 - rain) / (rain_6 + rain + 10.0), 0.0)
    dry_12 = np.maximum((rain_12 - rain) / (rain_12 + rain + 10.0), 0.0)

    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_3 = _antecedent(gpp, alpha_3)
    gpp_12 = _antecedent(gpp, alpha_12)
    fuel_bank = gpp_12 / (gpp_12 + 0.35)
    curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_12 = _antecedent(temperature, alpha_12)
    warm_departure = np.clip(
        (temperature - temperature_12 - 3.0) / 5.0, 0.0, 1.0
    )

    baseline = np.asarray(prediction, dtype=np.float64)
    shoulder = dry_12 * (
        0.5 * fuel_bank + 0.3 * curing + 0.2 * warm_departure
    )
    wet_uncured = (1.0 - dry_6) * (1.0 - curing)
    signal = shoulder - 0.25 * wet_uncured
    factor = np.exp(np.clip(strength * signal, -4.0, 4.0))
    adjusted = baseline * factor

    baseline_state = baseline[0].copy()
    adjusted_state = adjusted[0].copy()
    allocated = np.empty_like(adjusted)
    for time in range(adjusted.shape[0]):
        baseline_state += alpha_12 * (baseline[time] - baseline_state)
        adjusted_state += alpha_12 * (adjusted[time] - adjusted_state)
        allocated[time] = adjusted[time] * baseline_state / (
            adjusted_state + 1e-12
        )
    return np.asarray(np.clip(allocated, 0.0, 1.0), dtype=np.float32)


def _surface_fire_opportunity_bank(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Store surface-fire hazard until fuel is physically combustible.

    A share of open-land fire opportunity enters a local bank each month.
    Release accelerates only when antecedent fine fuel, curing, rainfall
    deficit, atmospheric dryness and an above-background fire opportunity
    coincide. Stored and released quantities are expressed in hazard space, so
    this moves finite fire potential through time rather than multiplying it.
    Woody and crop pathways are protected by a continuous surface-fuel share.
    """
    if "surface_opportunity_bank" not in enabled:
        return prediction
    strength = float(np.clip(p.get("surface_bank_w", 0.0), 0.0, 1.0))
    if strength <= 0.0:
        return prediction

    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_3 = _antecedent(gpp, alpha_3)
    gpp_12 = _antecedent(gpp, alpha_12)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_6 = _antecedent(rain, alpha_6)
    rain_deficit = np.maximum(
        (rain_6 - rain) / (rain_6 + rain + 10.0), 0.0
    )
    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    combustion = dryness / (dryness + 500.0)

    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel
    surface_share = surface_capacity / (
        0.05 + surface_capacity + woody_capacity + crop_capacity
    )

    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    bank = np.zeros_like(hazard[0])
    hazard_state = hazard[0].copy()
    allocated = np.empty_like(hazard)
    release_rate = float(max(p.get("surface_bank_release", 8.0), 0.0))
    for time in range(hazard.shape[0]):
        relative_opportunity = hazard[time] / (
            hazard[time] + hazard_state + 1e-8
        )
        physical_window = np.sqrt(
            np.clip(
                fine_fuel[time]
                * combustion[time]
                * rain_deficit[time],
                0.0,
                1.0,
            )
        )
        physical_window *= 0.25 + 0.75 * curing[time] / (
            curing[time] + 0.05
        )
        release_opportunity = relative_opportunity * physical_window
        release_fraction = 1.0 - np.exp(
            -(1.0 / 24.0 + release_rate * release_opportunity)
        )
        stored = strength * surface_share[time] * hazard[time]
        bank += stored
        released = release_fraction * bank
        bank -= released
        allocated[time] = hazard[time] - stored + released
        hazard_state += alpha_12 * (hazard[time] - hazard_state)
    return np.asarray(
        1.0 - np.exp(-np.clip(allocated, 0.0, 50.0)), dtype=np.float32
    )


def _local_fire_footprint(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Scale event footprint by continuous fuel and ignition access.

    Open connected vegetation permits an ignition to spread beyond an isolated
    patch. Ignition access is supplied either by the trailing local lightning
    regime or smoothly by managed open land. Dense canopy removes surface-fuel
    continuity, so this equation cannot create a geographic or closed-forest
    correction. It operates on final hazard with one coefficient set globally.
    """
    if "pathway_hazards" not in enabled:
        return prediction
    background = float(max(p.get("fire_footprint_background", 0.5), 0.0))
    strength = float(max(p.get("fire_footprint_w", 0.0), 0.0))
    if strength <= 0.0:
        return prediction

    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_12 = _antecedent(
        lightning, 1.0 - np.exp(-1.0 / 12.0)
    )
    lightning_half = float(
        max(p.get("fire_footprint_lightning_half", 0.05), 1e-8)
    )
    natural_ignition = lightning_12 / (lightning_12 + lightning_half)
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    managed = np.clip(rangeland + pasture + crop, 0.0, 1.0)
    managed_half = float(max(p.get("fire_footprint_managed_half", 0.1), 1e-8))
    managed_access = managed / (managed + managed_half)
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )
    natural_weight = float(
        np.clip(p.get("fire_footprint_natural_w", 0.7), 0.0, 1.0)
    )
    activity = open_cover * (
        natural_weight * natural_ignition
        + (1.0 - natural_weight) * managed_access
    )
    surface_footprint = np.clip(background + strength * activity, 0.1, 3.0)
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = _antecedent(gpp, 1.0 - np.exp(-1.0 / 12.0))
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    residue_capacity = crop * fine_fuel
    surface_share = surface_capacity / (
        0.05 + surface_capacity + woody_capacity + residue_capacity
    )
    footprint = 1.0 + surface_share * (surface_footprint - 1.0)
    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    return np.asarray(
        1.0 - np.exp(-np.clip(hazard * footprint, 0.0, 50.0)),
        dtype=np.float32,
    )


def _annual_regime_closure(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Balance persistent warm fire and rare continental thaw events.

    Low lightning variability has opposite meanings conditional on realized
    fire opportunity. In warm, humid, naturally fuelled open systems, large
    persistent fire with little ignition variability indicates an unrealistically
    broad footprint and is damped. In cold low-fire systems, the same quiet
    ignition background can still support a rare spring event after thaw if a
    continental fine-fuel carrier, stored soil carbon, combustion dryness and
    slow lightning supply coincide. Both responses use one global state equation
    and only current or trailing local history.
    """
    if "annual_regime_closure" not in enabled:
        return prediction
    warm_strength = float(max(p.get("persistent_warm_open_brake", 0.0), 0.0))
    cold_strength = float(max(p.get("cold_thaw_source", 0.0), 0.0))
    if warm_strength <= 0.0 and cold_strength <= 0.0:
        return prediction

    baseline = np.asarray(prediction, dtype=np.float64)
    annual_fire = np.empty_like(baseline)
    annual_fire[0] = baseline[0]
    for time in range(1, baseline.shape[0]):
        annual_fire[time] = baseline[max(0, time - 12) : time].mean(axis=0)

    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_variability = np.empty_like(lightning)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_variability = np.empty_like(temperature)
    for time in range(baseline.shape[0]):
        start = max(0, time - 11)
        lightning_variability[time] = lightning[start : time + 1].std(axis=0)
        temperature_variability[time] = temperature[start : time + 1].std(axis=0)
    low_lightning_variability = _falling(
        lightning_variability, 200.0, 0.025
    )

    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)
    alpha_24 = 1.0 - np.exp(-1.0 / 24.0)
    temperature_3 = _antecedent(temperature, alpha_3)
    temperature_24 = _antecedent(temperature, alpha_24)
    lightning_12 = _antecedent(lightning, alpha_12)

    annual_rain = np.clip(
        np.asarray(data["annual_precipitation"], dtype=np.float64), 0.0, None
    )
    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_6 = _antecedent(rain, alpha_6)
    rain_deficit = np.maximum(
        (rain_6 - rain) / (rain_6 + rain + 10.0), 0.0
    )
    rain_window = (
        _rising(annual_rain, 0.01, 180.0)
        * _falling(annual_rain, 1.0 / 180.0, 900.0)
    )

    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )
    open_cover = np.clip(
        rangeland + pasture + natural * 8.0 / (canopy + 8.0), 0.0, 1.0
    )

    persistent_warm_open = (
        low_lightning_variability
        * _rising(annual_fire, 1.0 / 0.003, 0.007)
        * _rising(temperature_24, 0.25, 18.0)
        * _rising(annual_rain, 1.0 / 220.0, 900.0)
        * open_cover
        * _rising(natural, 10.0, 0.2)
        * _rising(biomass, 20.0, 0.075)
    )

    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    combustion = dryness / (dryness + 250.0) * (
        0.35 + 0.65 * rain_deficit
    )
    thaw = (
        _rising(temperature, 1.0 / 3.0, 1.0)
        * _rising(temperature - temperature_3, 0.5, 2.0)
    )
    soil_carbon = np.clip(
        np.asarray(data["soil_carbon"], dtype=np.float64), 0.0, None
    )
    continental_natural_carrier = (
        natural
        * 3.0 / (canopy + 3.0)
        * _rising(temperature_variability, 2.0, 11.5)
        * soil_carbon / (soil_carbon + 4.0)
        * lightning_12 / (lightning_12 + 0.004)
    )
    carrier = np.clip(
        rangeland
        + 0.5 * crop
        + 0.25 * pasture
        + continental_natural_carrier,
        0.0,
        1.0,
    )
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_12 = _antecedent(gpp, alpha_12)
    fuel = gpp_12 / (gpp_12 + 0.08)
    cold_thaw = (
        low_lightning_variability
        * _falling(annual_fire, 1000.0, 0.003)
        * _falling(temperature_24, 0.5, 7.0)
        * rain_window
        * open_cover
        * thaw
        * combustion
        * carrier
        * (0.15 + 0.85 * fuel)
    )

    hazard = -np.log1p(-np.clip(baseline, 0.0, 1.0 - 1e-7))
    adjusted_hazard = (
        hazard * np.exp(-warm_strength * persistent_warm_open)
        + cold_strength * cold_thaw
    )
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted_hazard, 0.0, 50.0)),
        dtype=np.float32,
    )


def _trailing_annual_hazard(hazard: np.ndarray) -> np.ndarray:
    """Return a causal current-inclusive trailing-12 hazard sum."""
    output = np.empty_like(hazard, dtype=np.float64)
    accumulator = np.zeros_like(hazard[0], dtype=np.float64)
    for time in range(hazard.shape[0]):
        accumulator += hazard[time]
        if time >= 12:
            accumulator -= hazard[time - 12]
        output[time] = accumulator * 12.0 / min(time + 1, 12)
    return output


def _pathway_bank_delta(
    hazard: np.ndarray,
    pathway_share: np.ndarray,
    readiness: np.ndarray,
    storage_fraction: float,
    release_gain: float,
    storage_gate: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Redistribute one pathway's finite hazard through a causal local bank."""
    bank = np.zeros_like(hazard[0], dtype=np.float64)
    hazard_state = np.asarray(hazard[0], dtype=np.float64).copy()
    allocated = np.empty_like(hazard, dtype=np.float64)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)

    for time in range(hazard.shape[0]):
        relative_opportunity = hazard[time] / (
            hazard[time] + hazard_state + 1e-8
        )
        gate = storage_gate[time] if not np.isscalar(storage_gate) else storage_gate
        stored = (
            storage_fraction
            * pathway_share[time]
            * gate
            * hazard[time]
        )
        release_fraction = 1.0 - np.exp(
            -(1.0 / 24.0 + release_gain * relative_opportunity * readiness[time])
        )
        bank += stored
        released = release_fraction * bank
        bank -= released
        allocated[time] = hazard[time] - stored + released
        hazard_state += alpha_12 * (hazard[time] - hazard_state)

    return allocated - hazard


def _multi_pathway_opportunity_bank(
    prediction: np.ndarray,
    data: Mapping[str, np.ndarray],
    p: Mapping[str, float],
    enabled: set[str],
) -> np.ndarray:
    """Release finite pathway hazards in their own combustible windows.

    Natural surface-fire timing is handled by the earlier surface bank. This
    extension separates managed fine fuel, crop residue, woody fuel, and an
    unresolved background share. Each pathway stores only its own existing
    hazard and releases that stock through globally shared causal local-state
    equations; no burned area is fitted or geographically dispatched.
    """
    if "surface_opportunity_bank" not in enabled:
        return prediction

    hazard = -np.log1p(-np.clip(prediction, 0.0, 1.0 - 1e-7))
    alpha_3 = 1.0 - np.exp(-1.0 / 3.0)
    alpha_6 = 1.0 - np.exp(-1.0 / 6.0)
    alpha_12 = 1.0 - np.exp(-1.0 / 12.0)

    rain = np.clip(
        np.asarray(data["monthly_precipitation"], dtype=np.float64), 0.0, None
    )
    rain_6 = _antecedent(rain, alpha_6)
    rain_12 = _antecedent(rain, alpha_12)
    deficit_6 = np.maximum(
        (rain_6 - rain) / (rain_6 + rain + 10.0), 0.0
    )
    deficit_12 = np.maximum(
        (rain_12 - rain) / (rain_12 + rain + 10.0), 0.0
    )
    wet_anomaly = np.maximum(
        (rain - rain_12) / (rain + rain_12 + 10.0), 0.0
    )

    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp_3 = _antecedent(gpp, alpha_3)
    gpp_12 = _antecedent(gpp, alpha_12)
    fine_fuel = gpp_12 / (gpp_12 + 0.35)
    curing = np.maximum((gpp_3 - gpp) / (gpp_3 + gpp + 0.2), 0.0)
    curing_gate = curing / (curing + 0.05)

    dryness = np.clip(
        np.asarray(data["dryness"], dtype=np.float64), 0.0, None
    )
    combustion = dryness / (dryness + 500.0)
    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature_3 = _antecedent(temperature, alpha_3)
    temperature_12 = _antecedent(temperature, alpha_12)
    thermal_window = _rising(temperature, 0.25, 5.0)
    warm_departure_3 = _rising(temperature - temperature_3, 0.5, 1.0)
    warm_departure_12 = _rising(temperature - temperature_12, 0.5, 2.0)

    lightning = np.clip(
        np.asarray(data["lightning_flash_rate"], dtype=np.float64), 0.0, None
    )
    lightning_12 = _antecedent(lightning, alpha_12)
    ignition_12 = lightning_12 / (lightning_12 + 0.01)

    crop = np.clip(
        np.asarray(data["luh2_cropland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    natural = np.clip(
        np.asarray(data["natural_vegetation_fraction"], dtype=np.float64),
        0.0,
        1.0,
    )
    rangeland = np.clip(
        np.asarray(data["luh2_rangeland_fraction"], dtype=np.float64), 0.0, 1.0
    )
    pasture = np.clip(
        np.asarray(data["luh2_pasture_fraction"], dtype=np.float64), 0.0, 1.0
    )
    canopy = np.clip(
        np.asarray(data["natural_canopy_height"], dtype=np.float64), 0.0, None
    )
    biomass = np.clip(
        np.asarray(data["aboveground_biomass"], dtype=np.float64), 0.0, None
    )

    open_natural = natural * 8.0 / (canopy + 8.0)
    open_cover = np.clip(rangeland + pasture + open_natural, 0.0, 1.0)
    surface_capacity = (1.0 - crop) * fine_fuel * open_cover
    woody_capacity = (
        natural * canopy / (canopy + 8.0) * biomass / (biomass + 1.0)
    )
    crop_capacity = crop * fine_fuel
    total_capacity = 0.05 + surface_capacity + woody_capacity + crop_capacity
    managed_share = (
        np.clip(rangeland + pasture, 0.0, 1.0)
        * fine_fuel
        / total_capacity
    )
    crop_share = crop_capacity / total_capacity
    woody_share = woody_capacity / total_capacity
    background_share = 0.05 / total_capacity

    managed_readiness = deficit_6 * combustion * curing_gate * thermal_window
    crop_readiness = (
        combustion * curing_gate * thermal_window * warm_departure_3
    )
    woody_readiness = (
        deficit_12
        * warm_departure_12
        * thermal_window
        * ignition_12
        * (1.0 - wet_anomaly)
    )
    background_readiness = deficit_12 * combustion * thermal_window
    annual_hazard = _trailing_annual_hazard(hazard)
    managed_storage_gate = 0.2 / (annual_hazard + 0.2)

    adjusted_hazard = hazard.copy()
    adjusted_hazard += _pathway_bank_delta(
        hazard,
        managed_share,
        managed_readiness,
        p.get("managed_bank_store", 0.0),
        p.get("managed_bank_release", 0.0),
        managed_storage_gate,
    )
    adjusted_hazard += _pathway_bank_delta(
        hazard,
        crop_share,
        crop_readiness,
        p.get("crop_bank_store", 0.0),
        p.get("crop_bank_release", 0.0),
    )
    adjusted_hazard += _pathway_bank_delta(
        hazard,
        woody_share,
        woody_readiness,
        p.get("woody_bank_store", 0.0),
        p.get("woody_bank_release", 0.0),
    )
    adjusted_hazard += _pathway_bank_delta(
        hazard,
        background_share,
        background_readiness,
        p.get("background_bank_store", 0.0),
        p.get("background_bank_release", 0.0),
    )
    return np.asarray(
        1.0 - np.exp(-np.clip(adjusted_hazard, 0.0, 50.0)),
        dtype=np.float32,
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
    if "curing" in enabled:
        rate = rate * _curing(data, fallback)
    if "lag" in enabled:
        rate = _lag(rate, fallback)
    prediction = _transform(rate, fallback)
    prediction = _ecological_regime_brakes(prediction, data, fallback, enabled)
    prediction = _pathway_event_scaling(prediction, data, fallback, enabled)
    prediction = _ecological_fire_capacity(prediction, data, fallback, enabled)
    prediction = _seasonal_rainfall_capacity(
        prediction, data, fallback, enabled
    )
    prediction = _state_dependent_fire_season(
        prediction, data, fallback, enabled
    )
    prediction = _rare_lightning_ignition(prediction, data, fallback, enabled)
    prediction = _rain_conditioned_crop_management(
        prediction, data, fallback, enabled
    )
    prediction = _drought_maturation_response(
        prediction, data, fallback, enabled
    )
    prediction = _dead_fuel_pool_response(
        prediction, data, fallback, enabled
    )
    prediction = _conditional_fire_allocation(
        prediction, data, fallback, enabled
    )
    prediction = _live_fuel_greenup_brake(
        prediction, data, fallback, enabled
    )
    prediction = _surface_fire_opportunity_bank(
        prediction, data, fallback, enabled
    )
    prediction = _local_fire_footprint(
        prediction, data, fallback, enabled
    )
    prediction = _annual_regime_closure(
        prediction, data, fallback, enabled
    )
    prediction = _multi_pathway_opportunity_bank(
        prediction, data, fallback, enabled
    )
    return np.asarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
