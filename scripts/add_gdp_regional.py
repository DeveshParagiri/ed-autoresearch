"""Regional (biome-ish) GDP human term: gdp_gamma varies by region, smooth-blended
(no hard seams), so African savanna amplifies without blowing up poor monsoon Asia.

George's biome-specific direction, landing on the human predictor. Fixes the joint
single-global refit's India+SEA over-burn (3.3x): one global wealth coefficient can
not serve both African savanna (needs amplification) and poor monsoon Asia (must not
be amplified) -- the cerrado-vs-Africa indistinguishability, now in the human term.

Base params: the joint refit (params.coupledE_gdp.json), scalar gdp_gamma stripped.
Per-region gamma is fit by coordinate descent to minimize global area-weighted RMSE
vs GFED5, with the global magnitude PINNED to GFED5 (also cures the 1.18x creep).
Smooth-blend the gamma field (Gaussian, SIGMA deg) -> no seams. Writes base + regional
BA for ILAMB and prints regional Mha.
"""
import json, os, sys
from pathlib import Path
import numpy as np, xarray as xr, cftime
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO = Path("."); YEARS = list(range(2001, 2017)); FIRE_MAX = 5.0
SIGMA = float(os.environ.get("SIGMA", 4.0))
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"; FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
sl = slice(48, 240)
BASE = os.environ.get("BASE", "params.coupledE_gdp.json")
FB = json.load(open(REPO / "models" / "C" / BASE))["params"]
BASE_KEYS = ["k1", "D_low", "k2", "D_high", "fire_exp", "P_half", "pre_dampen_half", "gpp_af",
             "gpp_b", "gpp_d", "ign_k", "ign_c", "trop_agb_crit", "trop_k_veg", "fuel_k", "fuel_half"]
REGION_BOX = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
              "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
              "Europe": (-12, 40, 36, 72), "N.America": (-168, -52, 14, 74),
              "Australia": (112, 154, -44, -10)}

# ---------------- dump drivers (1deg) ----------------
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

# ---------------- GDP field + region ids ----------------
gdp1 = np.load(REPO / "data_human" / "gdp_pcap_grid_1deg.npy")
w = np.log10(np.clip(gdp1, 50, None)); have = np.isfinite(gdp1) & (gdp1 > 0)
land = np.isfinite(xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0]
                   ).reshape(180, 2, 360, 2).any((1, 3))
w0 = float(np.median(w[have & land]))
region_of = np.full((180, 360), "fb", dtype=object); assigned = np.zeros((180, 360), bool)
for r, b in REGION_BOX.items():
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]) & ~assigned
    region_of[box] = r; assigned |= box
REGIONS = list(REGION_BOX) + ["fb"]

def gamma_field(gvec):                                   # gvec: dict region->gamma, smooth-blended
    f = np.zeros((180, 360))
    for r in REGIONS:
        f[region_of == r] = gvec[r]
    return gaussian_filter(f, SIGMA, mode="nearest")

def human_mult(gfield, Mlo=0.15, Mhi=6.0):
    M = np.power(10.0, gfield * (w0 - w)); M[~have] = 1.0
    return np.clip(M, Mlo, Mhi)[None]

# ---------------- base fire rate (pre-gamma) ----------------
def hump_f(x, b, dec):
    b = np.maximum(b, 1e-9); dec = np.maximum(dec, 1e-9)
    return (1.0 - np.exp(-np.clip(x / b, 0, 500))) * np.exp(-np.clip(x / dec, 0, 500))
def base_rate(F):
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
    return np.power(np.clip(prod, 0, None), F["fire_exp"]) * (1.0 + F["fuel_k"] * fuel)
def ba_from_rate(rate, s):
    return (1.0 - np.exp(-np.minimum(s * rate, FIRE_MAX) / 12.0)).astype(np.float32)
R0 = base_rate({k: float(FB[k]) for k in BASE_KEYS})

# ---------------- GFED5 target + magnitude helpers ----------------
dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf = coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0); dg.close()
gf_ann = gf.reshape(16, 12, 180, 360).sum(1).mean(0)
R = 6371e3; area = (R**2 * np.deg2rad(1.0)**2 * np.cos(np.deg2rad(lat1)))[:, None] * np.ones((1, 360))
def ann(ba): return ba.reshape(16, 12, 180, 360).sum(1).mean(0)
gfed_tot = float((gf_ann * area).sum())
def s_for_total(rate, target, lo=0.02, hi=8.0):
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if float((ann(ba_from_rate(rate, mid)) * area).sum()) < target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
def wrmse(a):
    return float(np.sqrt((area[land] * (a[land] - gf_ann[land])**2).sum() / area[land].sum()))
def rmse_at(gvec, s):                                    # cheap: one transform, fixed scale
    return wrmse(ann(ba_from_rate(R0 * human_mult(gamma_field(gvec)), s)))

# ---------------- fit per-region gamma by coordinate descent ----------------
# Hold the global scale s fixed within a pass (so each eval is one transform), re-pin
# it to the GFED5 total between passes. Keeps magnitude controlled without a bisection
# inside every line-search step.
gvec = {r: float(FB.get("gdp_gamma", 0.6)) for r in REGIONS}   # start from the joint-refit scalar
grid = np.linspace(0.0, 1.6, 17)
s = s_for_total(R0 * human_mult(gamma_field(gvec)), gfed_tot)
print(f"[fit] start rmse={rmse_at(gvec, s):.4f}  (gamma0={gvec['Africa']:.2f} everywhere, s={s:.2f})")
for it in range(4):
    for r in REGIONS:
        gvec[r] = float(min(((rmse_at({**gvec, r: float(gv)}, s), gv) for gv in grid))[1])
    s = s_for_total(R0 * human_mult(gamma_field(gvec)), gfed_tot)   # re-pin between passes
    print(f"[fit] pass {it+1}: rmse={rmse_at(gvec, s):.4f} s={s:.2f}  "
          + " ".join(f"{r[:3]}={gvec[r]:.2f}" for r in REGIONS))

s_f = s_for_total(R0 * human_mult(gamma_field(gvec)), gfed_tot)
rmse_f = rmse_at(gvec, s_f); ba_reg = ba_from_rate(R0 * human_mult(gamma_field(gvec)), s_f)
ba_base = ba_from_rate(R0, s_for_total(R0, gfed_tot))     # same base, gamma=0, magnitude pinned
json.dump({"per_region_gamma": gvec, "sigma": SIGMA, "s": s_f, "base": BASE, "w0_gdp": 10**w0},
          open(REPO / "data_human" / "gdp_regional_gamma.json", "w"), indent=2)

# ---------------- write ILAMB inputs ----------------
def write_nc(ba, name):
    hd = uncoarsen(np.where(land[None], ba, np.nan).astype(np.float32))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    dso = xr.Dataset({"burntArea": (("time", "lat", "lon"), hd, {"units": "1", "standard_name": "burnt_area_fraction"})},
                     coords={"time": times, "lat": np.arange(-89.75, 90, 0.5), "lon": np.arange(-179.75, 180, 0.5)},
                     attrs={"title": name, "Conventions": "CF-1.7"})
    dso = add_cf_bounds(dso); tu = "days since 2001-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"}}
    out = REPO / "ilamb" / "MODELS_GDP_REGIONAL" / name / "burntArea.nc"; out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob("._*"): stale.unlink()
    dso.to_netcdf(out, encoding=enc, format="NETCDF4_CLASSIC"); print(f"[write] {out}")
write_nc(ba_base, "ED-ModelC-base"); write_nc(ba_reg, "ED-ModelC-gdpreg")

# ---------------- regional Mha ----------------
def tot(ba, b=None):
    a = ann(ba)
    m = np.ones((180, 360), bool) if b is None else ((LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]))
    return (a * area * m).sum() / 1e10
regs = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78), "India+SEA": (60, 150, -11, 35),
        "S.Amer": (-82, -34, -56, 14), "N.Amer": (-168, -52, 14, 74), "Europe": (-12, 40, 36, 72)}
print(f"\nfitted gamma: " + " ".join(f"{r}={gvec[r]:.2f}" for r in REGIONS))
print(f"{'model':10s} {'GLOBAL':>7s} " + " ".join(f"{r:>9s}" for r in regs))
for nm, ba in [("GFED5", gf), ("base(g=0)", ba_base), ("gdp-regional", ba_reg)]:
    print(f"{nm:10s} {tot(ba):7.0f} " + " ".join(f"{tot(ba, b):9.0f}" for b in regs.values()))
