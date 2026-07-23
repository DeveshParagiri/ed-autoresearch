"""Coupling-ready SMOOTH regional model: dump-climate per-continent fits, blended
smoothly (no hard seams). This is the production Option 3 build.

Drivers: ALL from ED's dump (D_bar/T_air/P_*/GPP/AGB) -> one dryness definition.
Params:  5 continental dump-climate fits + a global dump fallback (N.America/Australia).
Smooth:  each parameter field Gaussian-blended across borders (SIGMA deg) -> no seams.

Writes ilamb/MODELS_SMOOTH_COUPLED/ED-ModelC-{hard,smooth}/burntArea.nc + regional Mha.
"""
import json, os, sys
from pathlib import Path
import numpy as np, xarray as xr, cftime
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO = Path("."); YEARS = list(range(2001, 2017)); FIRE_MAX = 5.0
SIGMA = float(os.environ.get("SIGMA", 4.0))
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"
FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
sl = slice(48, 240)

REGION_BOX = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
              "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
              "Europe": (-12, 40, 36, 72), "N.America": (-168, -52, 14, 74),
              "Australia": (112, 154, -44, -10)}
REGION_PARAMS = {"Africa": "params.coupledE_af.json", "Boreal": "params.coupledE_bor.json",
                 "S.America": "params.coupledE_sam.json", "SEAsia": "params.coupledE_sea.json",
                 "Europe": "params.coupledE_eur.json", "N.America": "params.coupledE_nam.json",
                 "Australia": "params.coupledE_aus.json"}
FALLBACK = "params.coupledE_fx.json"          # global dump fit -> unfit regions, elsewhere
# DROP_REGIONS (env, comma-sep) excludes a region even if its fit exists, so a fit that
# REGRESSES its region can be reverted to the global fallback without deleting anything.
_drop = {s.strip() for s in os.environ.get("DROP_REGIONS", "").split(",") if s.strip()}
# keep only regions whose param file exists and that are not dropped
REGION_PARAMS = {r: f for r, f in REGION_PARAMS.items()
                 if (REPO / "models" / "C" / f).exists() and r not in _drop}
REGION_BOX = {r: b for r, b in REGION_BOX.items() if r in REGION_PARAMS}
print(f"[regions] using per-continent fits: {sorted(REGION_PARAMS)}"
      + (f"   dropped: {sorted(_drop)}" if _drop else ""))
BASE_KEYS = ["k1", "D_low", "k2", "D_high", "fire_exp", "P_half", "pre_dampen_half",
             "gpp_af", "gpp_b", "gpp_d", "ign_k", "ign_c", "trop_agb_crit", "trop_k_veg",
             "fuel_k", "fuel_half"]

lp = lambda n: json.load(open(REPO / "models" / "C" / n))["params"]
FB = lp(FALLBACK); RP = {r: lp(f) for r, f in REGION_PARAMS.items()}

# ---- dump-climate drivers (1-deg) ----
ds = xr.open_dataset(DUMP)
grab = lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32), nan=0.0)
d = {"dbar": coarsen(grab("D_bar")), "t_air": coarsen(grab("T_air")),
     "p_ann": coarsen(grab("P_ann")), "p_month": coarsen(grab("P_month"))}
d["gpp_monthly"] = coarsen((np.clip(grab("GPP_month_ntrl"), 0, None) * grab("area_frac_ntrl")
                          + np.clip(grab("GPP_month_scnd"), 0, None) * grab("area_frac_scnd")
                          + np.clip(grab("GPP_month_past"), 0, None) * grab("area_frac_past")).astype(np.float32))
ds.close()
da = xr.open_dataset(FUEL); d["agb"] = coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()
gpp_cell = d["gpp_monthly"].mean(0, keepdims=True)
lat1 = -89.5 + np.arange(180); lon1 = -179.5 + np.arange(360); LON, LAT = np.meshgrid(lon1, lat1)

# ---- hard region-id field (first-come) ----
assigned = np.zeros((180, 360), bool); region_of = np.full((180, 360), "fb", dtype=object)
for r, b in REGION_BOX.items():
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]) & ~assigned
    region_of[box] = r; assigned |= box

def hard_field(key):
    fld = np.full((180, 360), FB[key], float)
    for r in REGION_BOX:
        fld[region_of == r] = RP[r][key]
    return fld

def smooth_log(f):
    return np.exp(gaussian_filter(np.log(np.clip(f, 1e-30, None)), SIGMA, mode="nearest"))

def hump_f(x, b, dec):
    b = np.maximum(b, 1e-9); dec = np.maximum(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))

def fire(F):
    onset = sig(d["dbar"], F["k1"], F["D_low"]); supr = supp(d["dbar"], F["k2"], F["D_high"])
    p_flr = d["p_ann"] / (d["p_ann"] + F["P_half"] + 1e-12)
    p_dmp = 1.0 / (1.0 + d["p_month"] / (F["pre_dampen_half"] + 1e-12))
    gpp_m = hump_f(F["gpp_af"] * d["gpp_monthly"], F["gpp_b"], F["gpp_d"])
    ign = sig(d["t_air"], F["ign_k"], F["ign_c"])
    base = onset * supr * p_flr * p_dmp * gpp_m * ign
    trop = (np.abs(lat1) < 23.5).astype(float)[None, :, None]
    ratio = np.clip(d["agb"] / (F["trop_agb_crit"] + 1e-12), 0, None)
    canopy = 1.0 / (1.0 + np.power(ratio, F["trop_k_veg"]))
    prod = base * (trop * canopy + (1.0 - trop))
    fuel = gpp_cell / (gpp_cell + F["fuel_half"] + 1e-9)
    rate = np.power(np.clip(prod, 0, None), F["fire_exp"]) * (1.0 + F["fuel_k"] * fuel)
    return (1.0 - np.exp(-np.minimum(rate, FIRE_MAX) / 12.0)).astype(np.float32)

Fhard = {k: hard_field(k) for k in BASE_KEYS}
Fsmooth = {k: smooth_log(Fhard[k]) for k in BASE_KEYS}
ba_hard, ba_smooth = fire(Fhard), fire(Fsmooth)

can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_1d = np.isfinite(can.values[0]).reshape(180, 2, 360, 2).any((1, 3))

def write_nc(ba, name):
    hd = uncoarsen(np.where(land_1d[None], ba, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    dso = xr.Dataset({"burntArea": (("time", "lat", "lon"), hd, {"units": "1", "standard_name": "burnt_area_fraction"})},
                     coords={"time": times, "lat": can.lat.values, "lon": can.lon.values},
                     attrs={"title": name, "Conventions": "CF-1.7"})
    dso = add_cf_bounds(dso); tu = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"}}
    out = REPO / "ilamb" / "MODELS_SMOOTH_COUPLED" / name / "burntArea.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    dso.to_netcdf(out, encoding=enc, format="NETCDF4_CLASSIC"); print(f"[write] {out}")

write_nc(ba_hard, "ED-ModelC-hard"); write_nc(ba_smooth, "ED-ModelC-smooth")

dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gfed = coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0); dg.close()
R = 6371e3; area = (R**2 * np.deg2rad(1.0)**2 * np.cos(np.deg2rad(lat1)))[:, None] * np.ones((1, 360))
def tot(ba, b=None):
    ann = ba.reshape(16, 12, 180, 360).sum(1).mean(0)
    m = np.ones((180, 360), bool) if b is None else ((LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]))
    return (ann * area * m).sum() / 1e10
regs = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
        "India+SEA": (60, 150, -11, 35), "S.Amer": (-82, -34, -56, 14), "N.Amer": (-168, -52, 14, 74)}
print(f"\nSIGMA={SIGMA}   {'model':10s} {'GLOBAL':>7s} " + " ".join(f"{r:>9s}" for r in regs))
for nm, ba in [("GFED5", gfed), ("hard", ba_hard), ("smooth", ba_smooth)]:
    print(f"{'':17s}{nm:10s} {tot(ba):7.0f} " + " ".join(f"{tot(ba, b):9.0f}" for b in regs.values()))
