"""
Workstream C, scouting step: per-continent DRIVER HEADROOM. For each continent,
how well can the EXISTING drivers (D_bar, precip annual/monthly, T_air, GPP, AGB)
explain GFED5's spatial fire pattern, vs what Model C currently achieves there?

For each continent on active-fire cells (GFED5 annual > 1%):
  modelC_r : spatial correlation of the current Model C (spatial-k1) with GFED5
  driver_R : held-out (50/50 split) multiple-regression R of GFED5 on the drivers
             + their squares -> a proxy for the BEST r the drivers can support
  gap      : driver_R - modelC_r

Reading:
  gap large  -> Model C underperforms what the drivers allow HERE: a continent-
                specific FORM/parameter set can raise r (build structure, workstream C).
  driver_R low (and ~ modelC_r) -> the DRIVERS themselves lack the signal here:
                no parameter change helps; this region needs NEW drivers (human
                ignition, lightning, land use). This is the honest ceiling.

This decides HOW to build C (restructure vs new drivers) before committing.
Run: python scripts/diag_continent_headroom.py
"""
import importlib.util, sys, types, json
from pathlib import Path
import numpy as np
import xarray as xr

sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)


def gfed5_annual_1deg():
    """GFED5 reference annual burned fraction on the 1deg compute grid (180,360)."""
    g = xr.open_dataset(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")["burntArea"]
    yrs = np.array([t.year for t in g.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(g.values[sel].astype(np.float64), nan=0.0)
    if g.attrs.get("units") in ("%", "percent"):
        arr = arr / 100.0
    pm_hd = arr.mean(axis=0)                       # (360,720) period-mean monthly frac
    pm_1d = rmc.coarsen(pm_hd[None])[0]            # -> (180,360)
    return pm_1d * 12.0                            # annual fraction

REGIONS = {  # (lon_min, lon_max, lat_min, lat_max) on the 1deg compute grid
    "Africa":         (-20,  52, -36,  18),
    "S.America":      (-82, -34, -56,  14),
    "N.America":      (-168, -52, 14,  74),
    "Boreal Eurasia": (40, 180, 48,  78),
    "Trop/SE Asia":   (60, 150, -11,  30),
    "Australia":      (112, 154, -44, -10),
    "Europe":         (-12,  40,  36,  72),
}

d = rmc.load_drivers()                                  # each (192,180,360) on 1deg
gfed_annual = gfed5_annual_1deg()                       # (180,360) GFED5 annual fraction

# Model C current = spatial-k1 (the A+B refit winner), via fire_C + seasonal transform
params = json.load(open(REPO / "models/C/params.spatial.k1.json"))["params"]
with np.errstate(over="ignore", invalid="ignore"):
    rate = np.minimum(rmc.fire_C(d, params), rmc.FIRE_MAX_RATE)
modelC_pm = (1.0 - np.exp(-rate / 12.0)).mean(axis=0)   # period-mean monthly frac (180,360)

# Driver feature stack: period-mean of each driver per cell, + squares (standardized later)
feat = {k: d[k].mean(axis=0) for k in ("dbar", "p_ann", "p_month", "t_air", "gpp_monthly")}
if "agb" in d:
    feat["agb"] = d["agb"].mean(axis=0)
names = list(feat.keys())

lat = -89.5 + np.arange(180) * 1.0
lon = -179.5 + np.arange(360) * 1.0
LON, LAT = np.meshgrid(lon, lat)
land = gfed_annual > 0  # GFED defined / burnable anywhere on the 1deg grid


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-30))


def regression_R(X, y):
    """Held-out (deterministic 50/50 interleaved split) multiple-regression R."""
    n = len(y)
    idx = np.arange(n)
    tr, te = idx[idx % 2 == 0], idx[idx % 2 == 1]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xtr = np.c_[np.ones(len(tr)), Xs[tr]]; Xte = np.c_[np.ones(len(te)), Xs[te]]
    beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
    return corr(Xte @ beta, y[te])


print(f"{'region':16s} {'n':>6s} {'modelC_r':>9s} {'driver_R':>9s} {'gap':>6s}  verdict")
for name, (lo0, lo1, la0, la1) in REGIONS.items():
    box = (LON >= lo0) & (LON <= lo1) & (LAT >= la0) & (LAT <= la1)
    mask = land & box & (gfed_annual > 0.01)
    n = int(mask.sum())
    if n < 50:
        continue
    y = gfed_annual[mask].astype(np.float64)
    mc_r = corr(modelC_pm[mask].astype(np.float64), y)
    base = np.column_stack([feat[k][mask] for k in names]).astype(np.float64)
    X = np.column_stack([base, base ** 2])              # drivers + squares
    dR = regression_R(X, y)
    gap = dR - mc_r
    if dR < 0.45:
        verdict = "DRIVER-LIMITED (needs new drivers)"
    elif gap > 0.12:
        verdict = "FORM headroom (build C structure)"
    else:
        verdict = "near driver ceiling"
    print(f"{name:16s} {n:6d} {mc_r:9.3f} {dR:9.3f} {gap:+6.3f}  {verdict}")

print("\nmodelC_r = current spatial correlation (spatial-k1); driver_R = best a free "
      "regression of GFED5 on the drivers+squares achieves (held-out). gap = the r a "
      "continent-specific Model C could still gain. Low driver_R = the drivers, not the "
      "model, are the ceiling there.")
