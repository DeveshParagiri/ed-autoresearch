"""The mechanism terms that turn Model C into Models D through J, in ONE place.

WHY THIS MODULE EXISTS. Model H is Model C plus the GDP term. That term lived only inside
scripts/optimize_modelC_coupled.py, behind the env var GDP_TERM=1, and scripts/reproduce_modelC.py
had no idea it existed. So rebuilding H from models/C/params.H.json silently produced Model C driven
by H's parameters, scoring 0.5582 against H's true 0.6819, with no error and no warning. Dev hit
this from outside the group on 2026-08-22 and reported it as a discrepancy in our recorded scores.

The recorded scores were right. The model was not rebuildable. Ten env flags change the model
equation and NONE of them were written into the params file, so every fitted version was a params
file plus an invisible environment that survived only in a shell script or an optimizer log, and we
had run scripts for two versions out of ten.

The fix has two halves and this module is the first. Every term is here as a pure function taking
explicit arguments, imported by BOTH the optimizer and scripts/reproduce_model.py, so there is one
copy that cannot drift. The second half is that the optimizer now stamps `environment` into the
params JSON it writes, so a params file says which of these terms it needs.

fire_C itself already lived in reproduce_modelC.py and was already imported by the optimizer. That
part was never the problem, which is why the GDP-less rebuild produced a plausible-looking field
rather than a crash.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# The env vars that change the model EQUATION, as opposed to the search. These are what a params
# file has to carry for the model to be rebuildable. Anything that only steers the optimizer
# (N_TRIALS, SAMPLER, SEED, WARM, FP_MIN, the objective weights) is deliberately NOT here, since it
# cannot change the prediction once the parameters are fixed.
#
# name -> (kind, default). kind is how the string is parsed.
MECH_FLAGS = {
    "GDP_TERM":           ("bool",  False),
    "POP_TERM":           ("bool",  False),
    "LANDUSE_TERM":       ("bool",  False),
    "CURING":             ("bool",  False),
    "RATE_AMP":           ("bool",  False),
    "FUEL_AMP":           ("bool",  False),
    "SEASONAL_TRANSFORM": ("bool",  False),
    "DUMP_CLIMATE":       ("bool",  False),
    "TROP_MASK":          ("bool",  True),      # note the default is ON, unlike the rest
    "FUEL_WINDOW":        ("int",   0),
    "FIRE_MAX_RATE":      ("float", 5.0),
    "GDP_MLO":            ("float", 0.15),
    "GDP_MHI":            ("float", 6.0),
    "REGION":             ("str",   ""),
    "FIT_Y0":             ("int",   2001),
    "FIT_YF":             ("int",   2016),
}

_PARSE = {
    "bool":  lambda s: s == "1",
    "int":   int,
    "float": float,
    "str":   str,
}


def read_env():
    """The mechanism environment this process is running under, as a plain dict."""
    out = {}
    for name, (kind, default) in MECH_FLAGS.items():
        raw = os.environ.get(name)
        out[name] = default if raw is None else _PARSE[kind](raw)
    return out


def env_of(params_json):
    """The mechanism environment a params file was fitted under.

    Files written before the stamp existed have no `environment` block. Rather than guess, this
    returns None so the caller can refuse to rebuild instead of quietly rebuilding the wrong model,
    which is the failure this whole module exists to prevent.
    """
    env = params_json.get("environment")
    if env is None:
        return None
    return {name: env.get(name, default) for name, (_, default) in MECH_FLAGS.items()}


def describe(env):
    """The non-default flags, for a one-line log of what model is actually being built."""
    on = []
    for k, (_, d) in MECH_FLAGS.items():
        v = env.get(k, d)
        if v != d:
            on.append(f"{k}={v}")
    return ", ".join(on) if on else "Model C baseline (no mechanism flags set)"


# -- drivers ------------------------------------------------------------------
# The loader lives here rather than in either caller because DUMP_CLIMATE, FUEL_WINDOW and
# LANDUSE_TERM all change WHICH FIELDS the model is driven by, not just how they are combined.
# Rebuilding a model from the wrong drivers is the same class of silent error as rebuilding it
# without the GDP term, and it has already happened once: the POP_TERM=0 control was run without
# DUMP_CLIMATE=1 and compared against coupling-mode models, giving a warm start of 0.4320 against
# the 0.6094 it should have shown. That result had to be voided. See HANDOFF_NOTE 2026-08-19.

DUMP_NC = REPO / "global_baseline_modelC_inputs_1997-2016.nc"
DUMP_START_YEAR = 1997
FIT_START_YEAR = 2001
N_MONTHS = 192                                        # 2001-01 .. 2016-12
DUMP_SLICE_START = (FIT_START_YEAR - DUMP_START_YEAR) * 12    # 48
DUMP_SLICE_STOP = DUMP_SLICE_START + N_MONTHS                 # 240


def load_drivers(env, coarsen, trailing_mean, verbose=True):
    """Hybrid drivers at 1 degree: CRUJRA climate plus coupled GPP, or all-ED under DUMP_CLIMATE.

    coarsen and trailing_mean are passed in rather than imported so this module stays free of a
    circular import with reproduce_modelC, which owns fire_C and the grid helpers.
    """
    import xarray as xr

    ds = xr.open_dataset(DUMP_NC)
    sl = slice(DUMP_SLICE_START, DUMP_SLICE_STOP)

    def grab(name):
        return np.nan_to_num(ds[name].isel(time=sl).values.astype(np.float32), nan=0.0)

    if env["DUMP_CLIMATE"]:
        if verbose:
            print(f"[setup] COUPLING-READY: climate (D_bar/T_air/P_*) + GPP all from {DUMP_NC.name} ...")
        d = {
            "dbar":    coarsen(grab("D_bar")),
            "p_ann":   coarsen(grab("P_ann")),
            "p_month": coarsen(grab("P_month")),
            "t_air":   coarsen(grab("T_air")),
        }
    else:
        if verbose:
            print(f"[setup] CRUJRA climate from data/crujra/, GPP from {DUMP_NC.name} ...")
        cru = REPO / "data" / "crujra"
        d = {
            "dbar":    np.load(cru / "dbar_monthly.npy").astype(np.float32),
            "p_ann":   np.load(cru / "p_ann_monthly.npy").astype(np.float32),
            "p_month": np.load(cru / "p_month_monthly.npy").astype(np.float32),
            "t_air":   np.load(cru / "t_air_monthly.npy").astype(np.float32),
        }

    gpp_n = np.clip(grab("GPP_month_ntrl"), 0, None)
    gpp_s = np.clip(grab("GPP_month_scnd"), 0, None)
    gpp_p = np.clip(grab("GPP_month_past"), 0, None)
    af_n = grab("area_frac_ntrl")
    af_s = grab("area_frac_scnd")
    af_p = grab("area_frac_past")
    gpp_total = (gpp_n * af_n + gpp_s * af_s + gpp_p * af_p).astype(np.float32)
    d["gpp_monthly"] = coarsen(gpp_total)
    if env["LANDUSE_TERM"]:
        d["f_past"] = coarsen(af_p)
        d["f_scnd"] = coarsen(af_s)

    # FUEL_WINDOW (months) replaces the fuel term's static whole-record GPP mean with a causal
    # trailing mean, so the term can run forward and back to 1850. Built from the FULL dump and
    # sliced afterwards, so the fit window opens with a real spun-up window rather than a one-month
    # mean. FUEL_WINDOW=0 keeps the old static field, so every already-fitted set reproduces.
    fw = env["FUEL_WINDOW"]
    if fw > 0:
        full = np.zeros_like(ds["GPP_month_ntrl"].values, dtype=np.float32)
        for tag in ("ntrl", "scnd", "past"):
            g = np.clip(np.nan_to_num(ds[f"GPP_month_{tag}"].values.astype(np.float32), nan=0.0),
                        0, None)
            a = np.nan_to_num(ds[f"area_frac_{tag}"].values.astype(np.float32), nan=0.0)
            full += g * a
        d["gpp_fuel"] = coarsen(trailing_mean(full, fw))[sl]
        if verbose:
            spin = min(fw, DUMP_SLICE_START)
            print(f"[fuel] FUEL_WINDOW={fw} months, causal trailing mean, "
                  f"{spin} months of spin-up ahead of the fit window")
    ds.close()

    # AGB for vegetation-aware suppression
    agb_p = REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc"
    if agb_p.exists():
        ds_agb = xr.open_dataset(agb_p)
        agb = np.nan_to_num(ds_agb["AGB"].isel(time=sl).values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        ds_agb.close()
        d["agb"] = coarsen(agb)
    return d


def load_gfed5_1deg(coarsen, verbose=True):
    """GFED5 reference, 0.5 deg percent -> 1 deg fraction, 2001-2016."""
    import xarray as xr

    ref = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    if verbose:
        print(f"[setup] loading GFED5 reference from {ref.name} ...")
    ds = xr.open_dataset(ref)
    a = ds["burntArea"].isel(time=slice(0, N_MONTHS)).values.astype(np.float32)
    a = np.nan_to_num(a, nan=0.0) / 100.0
    ds.close()
    return coarsen(a)


# -- the mechanism terms ------------------------------------------------------
# Each is a factory returning a closure over the precomputed fields, matching how the optimizer
# builds them once at setup and calls them per trial. Transcribed from optimize_modelC_coupled.py
# so the numerics are identical. Do not tidy the arithmetic.

def make_gdp_mult(land_mask, mlo=0.15, mhi=6.0, verbose=True):
    """Human suppression keyed to GDP per capita (George, 2026-07-23).

    M = clip(10^(gamma*(w0 - log10 GDPpc)), mlo, mhi): amplifies fire where poor, about 1 at the
    median wealth of burnable land, suppresses where wealthy. Applied to the fire RATE.
    """
    gdp = np.load(REPO / "data_human" / "gdp_pcap_grid_1deg.npy").astype(np.float64)
    gw = np.log10(np.clip(gdp, 50.0, None))
    have = np.isfinite(gdp) & (gdp > 0)
    mask = have & land_mask
    w0 = float(np.median(gw[mask])) if mask.any() else float(np.median(gw[have]))
    if verbose:
        print(f"[gdp] GDP_TERM ON: pivot ${10**w0:,.0f}/cap, "
              f"coverage {100*mask.mean():.0f}% of grid, clip [{mlo},{mhi}]x")

    def gdp_mult(gamma):
        M = np.power(10.0, gamma * (w0 - gw))
        M[~have] = 1.0
        return np.clip(M, mlo, mhi).astype(np.float64)

    return gdp_mult


def make_pop_mult(verbose=True):
    """Human activity keyed to HYDE gridded population density.

    A HUMP in log10 density, not a monotonic suppression, because that is what GFED5 shows. Mean
    burned fraction rises from 0.021 in the sparsest cells to 0.097 around 10 to 35 people per km2,
    then falls to 0.036 in the densest. People both start fires and put them out and the two effects
    peak at different densities. Reaches 1700, so unlike GDP it is coupling-legal.
    """
    pop = np.load(REPO / "data_human" / "pop_density_1deg_2001_2016.npy")
    lpop = np.log10(pop.astype(np.float32) + 0.1)
    npop = int((pop[0] > 0).sum())
    del pop
    if verbose:
        print(f"[pop] POP_TERM ON: HYDE density, {npop} populated cells, "
              f"log10 range {lpop.min():.2f} to {lpop.max():.2f}")

    def pop_mult(p):
        mu = np.log10(p["pop_peak"] + 0.1)
        z = (lpop - mu) / (p["pop_sig"] + 1e-9)
        return 1.0 + p["pop_amp"] * np.exp(-0.5 * z * z)

    return pop_mult


def make_landuse_mult(drivers):
    """Human activity from ED's own LUH2 tiles, time-varying by construction.

    Caveat kept in view: the GPP driver is already an area-weighted sum over these same tiles, so
    some of this signal is double counted and the fit may return near zero.
    """
    f_past, f_scnd = drivers["f_past"], drivers["f_scnd"]

    def landuse_mult(p):
        return ((1.0 + p["lu_past"] * f_past) * (1.0 + p["lu_scnd"] * f_scnd)).astype(np.float64)

    return landuse_mult


def make_curing_term(drivers, sig, verbose=True):
    """Grass-curing pathway, additive to the fire rate.

    Lets temperate dry grassland (Kazakh steppe, Australia) burn without the high GPP the
    tropical-savanna-tuned hump demands. Gated to grassland by PRECIP rather than GPP, precisely
    because the steppe's low GPP is what must not be penalised, gated out of forest by AGB and out
    of desert by precip, dryness modulated, and boosted where fuel is fine and sparse.
    """
    p_ann_cell = drivers["p_ann"].mean(0, keepdims=True).astype(np.float64)
    gpp_cell = drivers["gpp_monthly"].mean(0, keepdims=True).astype(np.float64)
    agb_cell = drivers["agb"].mean(0, keepdims=True).astype(np.float64)
    cured_sat = sig(drivers["dbar"], 0.001, 3000.0).astype(np.float64)
    if verbose:
        print("[curing] CURING ON: grass-curing pathway (precip-gated, forest/desert-gated, "
              "inverse-GPP)")

    def curing_term(p):
        grass_zone = 1.0 / (1.0 + np.exp(-0.02 * (p_ann_cell - p["cure_p_min"])))
        grass_gate = 1.0 / (1.0 + np.power(np.clip(agb_cell / p["cure_agb_crit"], 0, None), 3.0))
        inv_gpp = 1.0 / (1.0 + gpp_cell / (p["cure_gpp_ref"] + 1e-9))
        return (p["cure_k"] * grass_zone * grass_gate * inv_gpp * cured_sat).astype(np.float64)

    return curing_term


# Parameter names owned by each mechanism. predict() strips these before calling fire_C, which
# would not recognise them.
POP_KEYS = ("pop_amp", "pop_peak", "pop_sig")
LU_KEYS = ("lu_past", "lu_scnd")
CURE_KEYS = ("cure_k", "cure_p_min", "cure_gpp_ref", "cure_agb_crit")
GDP_KEYS = ("gdp_gamma",)
MECH_KEYS = POP_KEYS + LU_KEYS + CURE_KEYS + GDP_KEYS


def ed_transform(rate_yr, fire_max_rate=5.0, seasonal=False, dt_years=1.0):
    """ED's saturating annual-to-monthly disturbance transform.

    seasonal=False is the legacy even spread (1-exp(-r))/12, which hard-caps any month at 1/12 and
    flattens seasonality. seasonal=True is the per-month form 1-exp(-r/12), which lifts dry-season
    cells and raises the spatial std ratio toward 1. The canonical versions were fitted under the
    legacy form, so it stays the default.
    """
    rate_capped = np.minimum(rate_yr, fire_max_rate)
    if seasonal:
        return (1.0 - np.exp(-rate_capped * dt_years / 12.0)).astype(np.float32)
    return ((1.0 - np.exp(-rate_capped * dt_years)) / 12.0).astype(np.float32)


def write_ba_nc(pred, path, land_mask, env, uncoarsen, add_cf_bounds, years=None):
    """Write a 1 degree monthly-fraction prediction to a 0.5 degree CF NetCDF.

    Shared by the optimizer and the rebuilder so a rebuilt file is encoded byte for byte the way a
    fitted one is. ILAMB is sensitive to encoding and fill values, so a rebuild that differed only
    in how it was written could score differently for no scientific reason at all.
    """
    import cftime
    import xarray as xr

    if years is None:
        years = list(range(env["FIT_Y0"], env["FIT_YF"] + 1))
    fire_max = env["FIRE_MAX_RATE"]
    seasonal = env["SEASONAL_TRANSFORM"]

    pred_hd = uncoarsen(np.where(land_mask[None, :, :], pred, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in years for m in range(1, 13)]
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(-179.75, 180.0, 0.5)
    ds = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), pred_hd,
                       {"units": "1", "standard_name": "burnt_area_fraction",
                        "long_name": "Burnt Area Fraction"})},
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={"title": "ED-ModelC-final (coupled-consistent retune, GFED5)",
               "Conventions": "CF-1.7",
               "transform": (f"monthly_frac = 1 - exp(-min(rate_yr, {fire_max}) / 12)"
                             if seasonal else
                             f"monthly_frac = (1 - exp(-min(rate_yr, {fire_max}) * 1yr)) / 12")})
    ds = add_cf_bounds(ds)
    enc = {"burntArea":   {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":        {"units": "days since 2001-01-01 00:00:00", "calendar": "noleap",
                           "dtype": "float64"},
           "time_bounds": {"units": "days since 2001-01-01 00:00:00", "calendar": "noleap",
                           "dtype": "float64"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, path)
    return path


def apply_mechanisms(rate, params, env, land_mask, *, gdp_mult=None, pop_mult=None,
                     landuse_mult=None, curing_term=None):
    """Everything predict() does between fire_C and ed_transform, in the optimizer's order.

    The order matters and does not commute with the land mask. Curing is added BEFORE masking, the
    three multipliers apply after it.
    """
    if env["CURING"] and curing_term is not None and "cure_k" in params:
        rate = rate + curing_term(params)
    rate = rate * land_mask[None, :, :]
    if env["GDP_TERM"] and gdp_mult is not None and "gdp_gamma" in params:
        rate = rate * gdp_mult(params["gdp_gamma"])[None, :, :]
    if env["POP_TERM"] and pop_mult is not None and "pop_amp" in params:
        rate = rate * pop_mult(params)
    if env["LANDUSE_TERM"] and landuse_mult is not None and "lu_past" in params:
        rate = rate * landuse_mult(params)
    return rate
