"""
PROTOTYPE (tagged, does NOT touch canonical): replace the annual->monthly fire
transform  (1 - exp(-rate))/12  with the physically-correct per-month
disturbance fraction  1 - exp(-rate/12).

The current /12 even-spread caps any month at 1/12 and flattens seasonality;
the corrected form concentrates fire in high-rate (dry-season) months and lifts
the per-cell period-mean, which is what the 1:1 plot needs.

Writes:
  ilamb/MODELS_SEASONAL_PROTO/ED-ModelC-seasonal/burntArea.nc   (scoreable later)
  ED-ModelC-tropfix2-k4/figures/per_cell_scatter_seasonal_proto.png
Reuses the canonical land mask + grid from the shipped k4 burntArea.nc.
"""
import importlib.util, sys, types, json
from pathlib import Path
import numpy as np, xarray as xr, cftime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)

YEARS = list(range(2001, 2017))
params = json.load(open(REPO / "models" / "C" / "params.json"))["params"]
d = rmc.load_drivers()
with np.errstate(over="ignore", invalid="ignore"):
    rate = rmc.fire_C(d, params)                 # (192,180,360) yr^-1

# land mask + area weights, taken from the canonical shipped file (0.5deg) then
# coarsened to the 1deg compute grid so the prototype is apples-to-apples.
can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_hd = np.isfinite(can.values[0])             # (360,720)
land_1d = land_hd.reshape(180, 2, 360, 2).any(axis=(1, 3))   # (180,360)
lat1 = np.arange(-89.5, 90.0, 1.0)
w = np.cos(np.deg2rad(lat1))[None, :, None] * land_1d[None]

def transform(r, kind):
    rc = np.minimum(r, rmc.FIRE_MAX_RATE)
    if kind == "current":   return (1.0 - np.exp(-rc)) / 12.0
    if kind == "corrected": return 1.0 - np.exp(-rc / 12.0)

cur = transform(rate, "current") * land_1d[None]
new = transform(rate, "corrected") * land_1d[None]

# Reference GFED5 land-mean (fraction) on its own grid, for magnitude ratio
g = xr.open_dataset(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")["burntArea"]
gyr = np.array([t.year for t in g.time.values]); gm = (gyr >= 2001) & (gyr <= 2016)
gfed_hd = np.nan_to_num(g.values[gm], nan=0.0)
if g.attrs.get("units") in ("%", "percent"): gfed_hd = gfed_hd / 100.0
gfed_pm_hd = gfed_hd.mean(axis=0)                # (360,720) period-mean fraction

def lm(x): return float((x * w).sum() / w.sum())
print(f"land-mean monthly fraction:  current {lm(cur):.6f}   corrected {lm(new):.6f}")
print(f"corrected magnitude vs current: x{lm(new)/lm(cur):.3f}  "
      f"(current is calibrated to 1.11x GFED5 -> corrected ~{1.11*lm(new)/lm(cur):.2f}x; NEEDS RE-TUNE)")

# ---- write the prototype nc (uncoarsen to 0.5, apply canonical NaN mask) ----
pred_hd = rmc.uncoarsen(np.where(land_1d[None], new, np.nan).astype(np.float32))
times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
ds = xr.Dataset(
    {"burntArea": (("time", "lat", "lon"), pred_hd,
                   {"units": "1", "standard_name": "burnt_area_fraction",
                    "long_name": "Burnt Area Fraction"})},
    coords={"time": times, "lat": can.lat.values, "lon": can.lon.values},
    attrs={"title": "ED-ModelC seasonal-transform PROTOTYPE (1-exp(-rate/12))",
           "Conventions": "CF-1.7", "note": "prototype, not canonical, not re-tuned"})
ds = rmc.add_cf_bounds(ds)
outdir = REPO / "ilamb/MODELS_SEASONAL_PROTO/ED-ModelC-seasonal"; outdir.mkdir(parents=True, exist_ok=True)
tu = "days since 2001-01-01 00:00:00"
enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
       "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
       "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"}}
ds.to_netcdf(outdir / "burntArea.nc", encoding=enc, format="NETCDF4_CLASSIC")
print("wrote", outdir / "burntArea.nc")

# ---- 1:1 scatter: canonical k4 (current /12) vs prototype (corrected) vs GFED5 ----
cur_pm = np.where(land_hd, rmc.uncoarsen(cur)[0]*0, np.nan)  # placeholder
cur_pm = rmc.uncoarsen(cur).mean(axis=0); cur_pm[~land_hd] = np.nan
new_pm = pred_hd.mean(axis=0)
m = np.isfinite(new_pm) & np.isfinite(gfed_pm_hd) & ((gfed_pm_hd > 0) | (new_pm > 0))
G, C, N = gfed_pm_hd[m], cur_pm[m], new_pm[m]

def stat(model, ref):
    dd = model - ref; return dd.mean(), np.sqrt((dd**2).mean())

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle("Per-grid-cell burned area: seasonal-transform prototype vs GFED5", fontsize=14, y=1.02)
def panel(a, model, title, lim=0.12):
    hb = a.hexbin(G, model, gridsize=80, extent=(0, lim, 0, lim),
                  norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
    a.plot([0, lim], [0, lim], "r--", lw=1, label="1:1")
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("GFED5 burned fraction"); a.set_ylabel("Model burned fraction")
    md, rm = stat(model, G)
    a.set_title(f"{title}\nmean diff {md:+.4f}  RMSE {rm:.4f}  max {model.max():.3f}", fontsize=11)
    a.legend(loc="upper right", fontsize=9); return hb
panel(ax[0], C, "canonical k4 (current /12)")
hb = panel(ax[1], N, "PROTOTYPE 1-exp(-rate/12)")
fig.colorbar(hb, ax=ax[1], fraction=0.046, pad=0.02).set_label("cell count (log)")
bins = np.linspace(-0.075, 0.075, 80)
ax[2].hist(C - G, bins=bins, color="indianred", alpha=0.55, label=f"k4 /12  |d| {np.abs(C-G).mean():.4f}")
ax[2].hist(N - G, bins=bins, color="steelblue", alpha=0.55, label=f"proto   |d| {np.abs(N-G).mean():.4f}")
ax[2].axvline(0, color="k", lw=1); ax[2].set_yscale("log")
ax[2].set_xlabel("Model - GFED5 (burned fraction)"); ax[2].set_ylabel("cell count (log)")
ax[2].set_title("Distribution of per-cell difference", fontsize=11); ax[2].legend(fontsize=9)
fig.tight_layout()
fdir = REPO / "NEW MAPS" / "proto_seasonal"; fdir.mkdir(parents=True, exist_ok=True)
f = fdir / "per_cell_scatter_seasonal_proto.png"
fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)

print("\nper-cell period-mean max:  k4 /12 = %.4f   prototype = %.4f   GFED5 = %.4f"
      % (np.nanmax(cur_pm), np.nanmax(new_pm), np.nanmax(gfed_pm_hd)))
