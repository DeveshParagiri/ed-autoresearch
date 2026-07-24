"""Add a physical human-suppression term (GDP per capita) to the single-global
dump model and measure what it buys. George's "bolt it onto the fire model".

Base model: params.coupledE.k2.json (best single-global dump fit, one global form,
no regional hard-coding -> clean attribution for the human term).

Human term (one physical predictor, minimal DOF):
    M(cell) = clip(10^( gamma * (w0 - log10 GDPpc) ),  Mlo, Mhi)
  amplifies fire where poor, ~1 at pivot wealth w0, suppresses where wealthy.
Applied multiplicatively to the fire RATE (same place fuel_k acts), with one
global rescale s so the magnitude stays honest. (gamma, s) fit to GFED5.

Writes ilamb/MODELS_GDP/ED-ModelC-{base,gdp}/burntArea.nc for ILAMB, prints
regional Mha for base vs +GDP vs GFED5.
"""
import json, os, sys
from pathlib import Path
import numpy as np, xarray as xr, cftime, pandas as pd, glob
import shapefile, shapely
from shapely.geometry import shape
from shapely import STRtree
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO = Path("."); YEARS = list(range(2001, 2017)); FIRE_MAX = 5.0
SP = "/private/tmp/claude-501/-Volumes-RICHIE---T7-FIRE-OFFLINE/c532496d-370d-40db-969d-b2a6dd880db0/scratchpad/humandata"
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"; FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"; sl = slice(48, 240)
BASE = os.environ.get("BASE", "params.coupledE.k2.json")
FB = json.load(open(REPO / "models" / "C" / BASE))["params"]
BASE_KEYS = ["k1", "D_low", "k2", "D_high", "fire_exp", "P_half", "pre_dampen_half", "gpp_af",
             "gpp_b", "gpp_d", "ign_k", "ign_c", "trop_agb_crit", "trop_k_veg", "fuel_k", "fuel_half"]

# ---------------- dump drivers (1deg, same as assemble_smooth_coupledE) ----------
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
lat1 = -89.5 + np.arange(180); lon1 = -179.5 + np.arange(360)

# ---------------- gridded GDP per capita (cache) ---------------------------------
GCACHE = REPO / "data_human" / "gdp_pcap_grid_1deg.npy"
if GCACHE.exists():
    gdp1 = np.load(GCACHE)
else:
    lat05 = -89.75 + 0.5 * np.arange(360); lon05 = -179.75 + 0.5 * np.arange(720)
    LON, LAT = np.meshgrid(lon05, lat05)
    tree = STRtree(shapely.points(LON.ravel(), LAT.ravel()))
    cidx = np.full(LAT.size, -1, np.int64)
    sf = shapefile.Reader(NE); col = {f[0]: i for i, f in enumerate(sf.fields[1:])}
    isos = []
    for k, (rec, shp) in enumerate(zip(sf.records(), sf.shapes())):
        cidx[tree.query(shapely.make_valid(shape(shp.__geo_interface__)), predicate="contains")] = k
        isos.append(rec[col["ISO_A3_EH"]])
    f = glob.glob(f"{SP}/gdp/API_*.csv")[0]; wbd = pd.read_csv(f, skiprows=4)
    yrs = [str(y) for y in range(2001, 2021) if str(y) in wbd.columns]
    wbd["g"] = wbd[yrs].mean(axis=1); g_of = dict(zip(wbd["Country Code"], wbd["g"]))
    gdp05 = np.full(LAT.size, np.nan); cflat = cidx.ravel()
    for k, iso in enumerate(isos):
        v = g_of.get(iso)
        if v is not None and np.isfinite(v) and v > 0:
            gdp05[cflat == k] = v
    with np.errstate(invalid="ignore"):                      # NaN-aware 2x2 -> 1deg block mean
        gdp1 = np.nanmean(gdp05.reshape(180, 2, 360, 2), axis=(1, 3))
    from scipy.ndimage import distance_transform_edt          # fill every cell from nearest country
    miss = ~np.isfinite(gdp1)
    ij = distance_transform_edt(miss, return_distances=False, return_indices=True)
    gdp1 = gdp1[tuple(ij)]
    os.makedirs(REPO / "data_human", exist_ok=True)
    np.save(GCACHE, gdp1); print(f"[gdp] cached {GCACHE}")

w = np.log10(np.clip(gdp1, 50, None))                       # log10 GDP/cap, 1deg
have = np.isfinite(gdp1) & (gdp1 > 0)
w0 = float(np.nanmedian(w[have]))                            # pivot: M=1 at median wealth
print(f"[gdp] pivot w0=10^{w0:.2f}=${10**w0:,.0f}/cap   coverage {100*have.mean():.0f}% of cells")

# ---------------- fire physics (returns RATE, so we can inject the human term) ----
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

def human_mult(gamma, Mlo=0.15, Mhi=6.0):
    M = np.power(10.0, gamma * (w0 - w)); M[~have] = 1.0
    return np.clip(M, Mlo, Mhi)[None]

def ba_from_rate(rate, s):
    return (1.0 - np.exp(-np.minimum(s * rate, FIRE_MAX) / 12.0)).astype(np.float32)

R0 = base_rate({k: float(FB[k]) for k in BASE_KEYS})

# ---------------- GFED5 target (1deg annual burned fraction) ---------------------
dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf = coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0); dg.close()
gf_ann = gf.reshape(16, 12, 180, 360).sum(1).mean(0)
R = 6371e3; area = (R**2 * np.deg2rad(1.0)**2 * np.cos(np.deg2rad(lat1)))[:, None] * np.ones((1, 360))
can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_1d = np.isfinite(can.values[0]).reshape(180, 2, 360, 2).any((1, 3))
def ann_frac(ba): return ba.reshape(16, 12, 180, 360).sum(1).mean(0)
def wrmse(mod_ann):
    m = land_1d
    return float(np.sqrt((area[m] * (mod_ann[m] - gf_ann[m])**2).sum() / area[m].sum()))

# ---------------- fit gamma with magnitude PINNED to the native base -------------
# base = the published single-global model untouched (s=1). The GDP term only
# REDISTRIBUTES fire by wealth; we hold the global total fixed (fit s per gamma so
# total(+GDP)=total(base)) so ILAMB gain is pattern, not a magnitude re-tune.
ba_base = ba_from_rate(R0, 1.0)
base_tot = float(ann_frac(ba_base).sum())
def s_for_total(rate, target, lo=0.05, hi=5.0):             # bisect s so global total matches
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float(ann_frac(ba_from_rate(rate, mid)).sum()) < target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
best = (wrmse(ann_frac(ba_base)), 0.0, 1.0)
for gamma in np.linspace(0.0, 1.4, 29):
    rate = R0 * human_mult(gamma); s = s_for_total(rate, base_tot)
    e = wrmse(ann_frac(ba_from_rate(rate, s)))
    if e < best[0]: best = (e, gamma, s)
rmse0, gam, s_gdp = best
print(f"[fit] base rmse={wrmse(ann_frac(ba_base)):.4f} (native, total={base_tot/1e10:.0f} Mha)   "
      f"+GDP: gamma={gam:.2f} s={s_gdp:.2f} rmse={rmse0:.4f}   "
      f"({100*(1-rmse0/wrmse(ann_frac(ba_base))):.1f}% RMSE cut, magnitude pinned)")
ba_gdp = ba_from_rate(R0 * human_mult(gam), s_gdp)
np.savez("data_human/gdp_term_fields.npz", lat1=lat1, lon1=lon1, gdp1=gdp1,
         M=human_mult(gam)[0], gam=gam, land=land_1d,
         base_ann=ann_frac(ba_base), gdp_ann=ann_frac(ba_gdp), gfed_ann=gf_ann)   # fig inputs

# ---------------- write ILAMB inputs --------------------------------------------
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
    out = REPO / "ilamb" / "MODELS_GDP" / name / "burntArea.nc"; out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob("._*"): stale.unlink()
    dso.to_netcdf(out, encoding=enc, format="NETCDF4_CLASSIC"); print(f"[write] {out}")
write_nc(ba_base, "ED-ModelC-base"); write_nc(ba_gdp, "ED-ModelC-gdp")
# gamma sweep (magnitude pinned) so ILAMB picks the human-term strength directly
for gv in [0.30, 0.50, 0.70]:
    rate = R0 * human_mult(gv); sv = s_for_total(rate, base_tot)
    write_nc(ba_from_rate(rate, sv), f"ED-ModelC-gdp{int(gv*100):02d}")

# ---------------- regional Mha table --------------------------------------------
LON1, LAT1 = np.meshgrid(lon1, lat1)
def tot(ba, b=None):
    ann = ann_frac(ba)
    m = np.ones((180, 360), bool) if b is None else ((LON1 >= b[0]) & (LON1 <= b[1]) & (LAT1 >= b[2]) & (LAT1 <= b[3]))
    return (ann * area * m).sum() / 1e10
regs = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78), "India+SEA": (60, 150, -11, 35),
        "S.Amer": (-82, -34, -56, 14), "N.Amer": (-168, -52, 14, 74), "Europe": (-12, 40, 36, 72)}
print(f"\n{'model':8s} {'GLOBAL':>7s} " + " ".join(f"{r:>9s}" for r in regs))
for nm, ba in [("GFED5", gf), ("base", ba_base), ("+GDP", ba_gdp)]:
    print(f"{nm:8s} {tot(ba):7.0f} " + " ".join(f"{tot(ba, b):9.0f}" for b in regs.values()))
