"""Option 3 prototype: smoothly-varying per-continent parameters (no hard seams).

Build a global field for EACH parameter (per-continent value inside each box, fallback
elsewhere), then Gaussian-smooth the fields so borders blend gradually. Compute fire per
cell from its local blended params. Compare HARD (== paper E assembly, has seams) vs
SMOOTH. CRUJRA drivers (viability test; port to dump climate if it works).

Writes ilamb/MODELS_SMOOTH/ED-ModelC-{hard,smooth}/burntArea.nc and prints regional Mha.
"""
import json, sys
from pathlib import Path
import numpy as np, xarray as xr, cftime
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import load_drivers, coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO = Path(".")
YEARS = list(range(2001, 2017))
FIRE_MAX = 5.0
SIGMA = 4.0  # smoothing length (deg) for the parameter fields

REGION_BOX = {  # first-come priority, matches assemble_continental clean preset
    "Africa":   (-20, 52, -36, 18),  "Boreal": (40, 180, 48, 78),
    "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
    "Europe":   (-12, 40, 36, 72),
}
REGION_PARAMS = {"Africa": "params.africafuel.json", "Boreal": "params.boreal.json",
                 "S.America": "params.samerica.json", "SEAsia": "params.seasia.json",
                 "Europe": "params.europe.json"}
FALLBACK = "params.spatial.k1.json"
BASE_KEYS = ["k1", "D_low", "k2", "D_high", "fire_exp", "P_half", "pre_dampen_half",
             "gpp_af", "gpp_b", "gpp_d", "ign_k", "ign_c", "trop_agb_crit", "trop_k_veg"]

lp = lambda n: json.load(open(REPO / "models" / "C" / n))["params"]
FB = lp(FALLBACK)
RP = {r: lp(f) for r, f in REGION_PARAMS.items()}

d = load_drivers()
gpp_cell = d["gpp_monthly"].mean(0, keepdims=True)
lat1 = -89.5 + np.arange(180); lon1 = -179.5 + np.arange(360)
LON, LAT = np.meshgrid(lon1, lat1)

# hard region-id field (first-come); -1 = fallback
assigned = np.zeros((180, 360), bool)
region_of = np.full((180, 360), "fallback", dtype=object)
for r, b in REGION_BOX.items():
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]) & ~assigned
    region_of[box] = r; assigned |= box


def hard_field(key):
    fld = np.full((180, 360), FB.get(key, 0.0), float)
    for r in REGION_BOX:
        fld[region_of == r] = RP[r].get(key, FB.get(key, 0.0))
    return fld


# amplitude field A (per cell): Africa uses fuel form, others scalar fire_amp
def amp_hard():
    A = np.full((180, 360), FB.get("fire_amp", 1.0), float)
    for r in REGION_BOX:
        p = RP[r]
        if "fuel_k" in p:                       # Africa fuel form (cell-varying)
            fuel = gpp_cell[0] / (gpp_cell[0] + p.get("fuel_half", 1.0) + 1e-9)
            A = np.where(region_of == r, 1.0 + p["fuel_k"] * fuel, A)
        else:
            A = np.where(region_of == r, p.get("fire_amp", 1.0), A)
    return A


def smooth_log(fld):
    return np.exp(gaussian_filter(np.log(np.clip(fld, 1e-30, None)), SIGMA, mode="nearest"))


def hump_f(x, b, dec):
    b = np.maximum(b, 1e-9); dec = np.maximum(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))


def fire_from_fields(F, A):
    onset = sig(d["dbar"], F["k1"], F["D_low"])
    supr  = supp(d["dbar"], F["k2"], F["D_high"])
    p_flr = d["p_ann"] / (d["p_ann"] + F["P_half"] + 1e-12)
    p_dmp = 1.0 / (1.0 + d["p_month"] / (F["pre_dampen_half"] + 1e-12))
    gpp_m = hump_f(F["gpp_af"] * d["gpp_monthly"], F["gpp_b"], F["gpp_d"])
    ign   = sig(d["t_air"], F["ign_k"], F["ign_c"])
    base  = onset * supr * p_flr * p_dmp * gpp_m * ign
    trop  = (np.abs(lat1) < 23.5).astype(float)[None, :, None]
    ratio = np.clip(d["agb"] / (F["trop_agb_crit"] + 1e-12), 0, None)
    canopy = 1.0 / (1.0 + np.power(ratio, F["trop_k_veg"]))
    prod = base * (trop * canopy + (1.0 - trop))
    rate = np.power(np.clip(prod, 0, None), F["fire_exp"]) * A
    rc = np.minimum(rate, FIRE_MAX)
    return (1.0 - np.exp(-rc / 12.0)).astype(np.float32)   # SEASONAL_TRANSFORM


# build hard + smoothed fields
Fhard = {k: hard_field(k) for k in BASE_KEYS}
Ahard = amp_hard()
Fsmooth = {k: smooth_log(Fhard[k]) for k in BASE_KEYS}
Asmooth = smooth_log(Ahard)

ba_hard = fire_from_fields(Fhard, Ahard)
ba_smooth = fire_from_fields(Fsmooth, Asmooth)

# land mask from canonical, write 0.5deg
can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_hd = np.isfinite(can.values[0]); land_1d = land_hd.reshape(180, 2, 360, 2).any((1, 3))


def write_nc(ba, name):
    hd = uncoarsen(np.where(land_1d[None], ba, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    ds = xr.Dataset({"burntArea": (("time", "lat", "lon"), hd,
                     {"units": "1", "standard_name": "burnt_area_fraction"})},
                    coords={"time": times, "lat": can.lat.values, "lon": can.lon.values},
                    attrs={"title": name, "Conventions": "CF-1.7"})
    ds = add_cf_bounds(ds)
    tu = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"}}
    out = REPO / "ilamb" / "MODELS_SMOOTH" / name / "burntArea.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding=enc, format="NETCDF4_CLASSIC")
    print(f"[write] {out}")


write_nc(ba_hard, "ED-ModelC-hard")
write_nc(ba_smooth, "ED-ModelC-smooth")

# regional Mha check
dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gfed = coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0)
dg.close()
R = 6371e3; area = (R**2 * np.deg2rad(1.0)**2 * np.cos(np.deg2rad(lat1)))[:, None] * np.ones((1, 360))
def tot(ba, b=None):
    ann = ba.reshape(16, 12, 180, 360).sum(1).mean(0)
    m = np.ones((180, 360), bool) if b is None else (
        (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]))
    return (ann * area * m).sum() / 1e10
regions = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
           "India+SEA": (60, 150, -11, 35), "S.Amer": (-82, -34, -56, 14)}
print(f"\n{'model':10s} {'GLOBAL':>7s} " + " ".join(f"{r:>9s}" for r in regions))
for nm, ba in [("GFED5", gfed), ("hard", ba_hard), ("smooth", ba_smooth)]:
    print(f"{nm:10s} {tot(ba):7.0f} " + " ".join(f"{tot(ba, b):9.0f}" for b in regions.values()))
