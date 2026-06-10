"""
DIAGNOSTIC (no canonical change): does letting the fire RATE exceed 1.0 lift the
per-cell 1:1 band toward GFED5's 0.104? Sweeps the new `fire_amp` multiplier with
the SEASONAL_TRANSFORM on, starting from the seasonal-k1 params, and reports per
cell ceiling, spatial std ratio (sigma) vs GFED5, and the magnitude ratio.

This is the "Cause 1" lever for George's 1:1 bar. fire_amp scales the bounded
[0,1] fire rate above 1 (savanna multi-burn frequency). Raising it lifts the
ceiling AND the global total, so the optimizer will need to rebalance — this
diagnostic only checks the dynamic-range potential, it does not re-fit.

Run: SEASONAL_TRANSFORM=1 python scripts/diag_rate_amp.py
"""
import importlib.util, sys, types, json, os
from pathlib import Path
import numpy as np, xarray as xr

os.environ.setdefault("SEASONAL_TRANSFORM", "1")
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)

# Baseline params = the seasonal-transform fit (seasonal-k1). Falls back to canonical.
pj = REPO / "models" / "C" / "params.seasonal.k1.json"
if not pj.exists():
    pj = REPO / "models" / "C" / "params.json"
params = json.load(open(pj))["params"]
print(f"baseline params: {pj.name}  (SEASONAL_TRANSFORM={rmc.SEASONAL_TRANSFORM})")

d = rmc.load_drivers()

# land mask + cos-lat weights on the 1deg compute grid, from the shipped 0.5deg file
can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_hd = np.isfinite(can.values[0])
land_1d = land_hd.reshape(180, 2, 360, 2).any(axis=(1, 3))
lat1 = np.arange(-89.5, 90.0, 1.0)
w = np.cos(np.deg2rad(lat1))[None, :, None] * land_1d[None]

# GFED5 reference period-mean on its native 0.5deg grid (for ceiling + sigma + magnitude)
g = xr.open_dataset(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")["burntArea"]
gyr = np.array([t.year for t in g.time.values]); gm = (gyr >= 2001) & (gyr <= 2016)
gfed_hd = np.nan_to_num(g.values[gm], nan=0.0)
if g.attrs.get("units") in ("%", "percent"): gfed_hd = gfed_hd / 100.0
gfed_pm = gfed_hd.mean(axis=0)
gfed_pm[~land_hd] = np.nan
gland = gfed_pm[land_hd]
gfed_lm = float(np.nanmean(np.where(np.isfinite(gfed_pm), gfed_pm, np.nan)))

print(f"\nGFED5 ref: per-cell max {np.nanmax(gfed_pm):.4f}  "
      f"std {np.nanstd(gland):.5f}  land-mean {gfed_lm:.6f}\n")
print(f"{'fire_amp':>8s} {'rate_max':>9s} {'cell_max':>9s} {'sigma':>7s} {'magx':>6s}")

for amp in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    p = dict(params)
    if amp != 1.0:
        p["fire_amp"] = amp
    else:
        p.pop("fire_amp", None)
    with np.errstate(over="ignore", invalid="ignore"):
        rate = rmc.fire_C(d, p) * land_1d[None]
    rc = np.minimum(rate, rmc.FIRE_MAX_RATE)
    pred = (1.0 - np.exp(-rc / 12.0)) * land_1d[None]          # seasonal transform
    pred_hd = rmc.uncoarsen(np.where(land_1d[None], pred, np.nan).astype(np.float32))
    pred_pm = np.nanmean(pred_hd, axis=0)
    aland = pred_pm[land_hd]
    sigma = np.nanstd(aland) / np.nanstd(gland)
    magx = float(np.nanmean(aland)) / gfed_lm
    print(f"{amp:8.1f} {float(rate.max()):9.3f} {np.nanmax(pred_pm):9.4f} "
          f"{sigma:7.3f} {magx:6.2f}")

print("\nNote: cell_max should climb toward GFED5 0.1044 and sigma toward 1.0 as "
      "fire_amp rises; magx also rises (more total burn) so the optimizer must "
      "rebalance suppression. This only shows the dynamic-range potential.")
