"""
Workstream C, Africa form-change scouting: WHAT signal is Model C missing in
Africa? The headroom diagnostic says the drivers support r~0.68 in Africa but the
current form reaches ~0.50, and regional re-tuning did NOT close it -> the form
lacks a term. This finds which term.

On Africa active-fire cells (GFED5 annual > 1%), for the current model (spatial-k1):
  corr(GFED, feature)   - what GFED's pattern keys on
  corr(modelC, feature) - what the model keys on
  corr(residual, feature) where residual = GFED - best-scaled modelC
                          -> the feature most correlated with the RESIDUAL is the
                          missing term to add to fire_C.
Features: each driver, squares, and physically-motivated interactions (fuel x
dryness curing, precip x temperature, etc.).

Run: SEASONAL_TRANSFORM=1 python scripts/diag_africa_residual.py
"""
import importlib.util, sys, types, json, os
from pathlib import Path
import numpy as np, xarray as xr

os.environ.setdefault("SEASONAL_TRANSFORM", "1")
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)


def gfed5_annual_1deg():
    g = xr.open_dataset(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")["burntArea"]
    yrs = np.array([t.year for t in g.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(g.values[sel].astype(np.float64), nan=0.0)
    if g.attrs.get("units") in ("%", "percent"): arr = arr / 100.0
    return rmc.coarsen(arr.mean(axis=0)[None])[0] * 12.0


d = rmc.load_drivers()
gfed = gfed5_annual_1deg()
params = json.load(open(REPO / "models/C/params.spatial.k1.json"))["params"]
with np.errstate(over="ignore", invalid="ignore"):
    rate = np.minimum(rmc.fire_C(d, params), rmc.FIRE_MAX_RATE)
modelC = (1.0 - np.exp(-rate / 12.0)).mean(axis=0) * 12.0     # annual frac

# period-mean drivers
P = {k: d[k].mean(axis=0) for k in ("dbar", "p_ann", "p_month", "t_air", "gpp_monthly")}
if "agb" in d: P["agb"] = d["agb"].mean(axis=0)

# Africa active-fire mask
lat = -89.5 + np.arange(180) * 1.0; lon = -179.5 + np.arange(360) * 1.0
LON, LAT = np.meshgrid(lon, lat)
mask = ((LON >= -20) & (LON <= 52) & (LAT >= -36) & (LAT <= 18) & (gfed > 0.01))
n = int(mask.sum())

y = gfed[mask]; mc = modelC[mask]
b = np.dot(mc, y) / (np.dot(mc, mc) + 1e-30)        # best scale of modelC onto GFED
resid = y - b * mc


def corr(a, c):
    a = a - a.mean(); c = c - c.mean()
    return float((a * c).mean() / (a.std() * c.std() + 1e-30))


# Feature set: drivers, squares, and interactions (named)
feats = {}
for k, v in P.items():
    feats[k] = v[mask]
    feats[k + "^2"] = v[mask] ** 2
g_ = P["gpp_monthly"][mask]; db = P["dbar"][mask]; pa = P["p_ann"][mask]
ta = P["t_air"][mask]; pm = P["p_month"][mask]
feats["gpp*dbar (curing)"] = g_ * db
feats["gpp*Pann"] = g_ * pa
feats["dbar*Tair"] = db * ta
feats["gpp/(Pann+1)"] = g_ / (pa + 1.0)
feats["Pann*dbar"] = pa * db
feats["log_gpp"] = np.log1p(np.clip(g_, 0, None))

print(f"Africa active-fire cells n={n}\n")
print(f"{'feature':22s} {'corr_GFED':>10s} {'corr_modelC':>12s} {'corr_RESID':>11s}")
rows = []
for name, f in feats.items():
    rows.append((name, corr(y, f), corr(mc, f), corr(resid, f)))
for name, cg, cm, cr in sorted(rows, key=lambda r: -abs(r[3])):
    print(f"{name:22s} {cg:10.3f} {cm:12.3f} {cr:11.3f}")

print("\ncorr_RESID = correlation of (GFED - best-scaled modelC) with the feature. The "
      "top |corr_RESID| features are the signal Model C is MISSING in Africa -> candidate "
      "new term(s) for fire_C. corr_GFED vs corr_modelC shows where the model over/under-uses a driver.")
