"""
Quantify and map Model C false positives vs GFED5 burned area.
A "false positive" cell = model burns meaningfully where GFED5 is near-zero.
Also decompose the global over-prediction into (a) false-positive area and
(b) hotspot intensity error.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "Seasonal"
OUT.mkdir(parents=True, exist_ok=True)
NY = 16


def load_ba(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    units = da.attrs.get("units", "1")
    if units in ("1", "fraction", ""):
        arr = arr * 100.0
    annual = arr.reshape(NY, 12, *arr.shape[1:]).sum(axis=1).mean(axis=0)  # % per yr
    return annual, da.lat.values, da.lon.values


ours, lat, lon = load_ba(REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc")
truth, _, _    = load_ba(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")

# Cell area (m^2) and Mha
R = 6.371e6; dlon = np.deg2rad(0.5)
area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
area2d = np.abs(np.broadcast_to(area_lat[:, None], (len(lat), len(lon))))  # m^2
cell_Mha = area2d / 1e10

# Burnable land mask: cells where either burns at all
land = (truth > 0) | (ours > 0)

# Classify cells
QUIET = 0.1   # GFED < 0.1 %/yr = "GFED quiet"
FP_THRESH = 0.5  # model > 0.5 %/yr where GFED quiet = false positive
fp_mask  = (truth < QUIET) & (ours > FP_THRESH)
hot_mask = (truth >= QUIET)  # cells GFED considers real fire

# Burned area totals (Mha/yr) = sum( frac * cell_area )
ba_ours_total  = float(((ours  / 100.0) * cell_Mha).sum())
ba_truth_total = float(((truth / 100.0) * cell_Mha).sum())
ba_fp_total    = float(((ours  / 100.0) * cell_Mha * fp_mask).sum())
ba_hot_ours    = float(((ours  / 100.0) * cell_Mha * hot_mask).sum())
ba_hot_truth   = float(((truth / 100.0) * cell_Mha * hot_mask).sum())

print("="*64)
print("GLOBAL BURNED AREA DECOMPOSITION (Mha/yr)")
print("="*64)
print(f"  GFED5 total:                 {ba_truth_total:8.1f} Mha/yr")
print(f"  Model C total:               {ba_ours_total:8.1f} Mha/yr  ({ba_ours_total/ba_truth_total:.2f}x GFED5)")
print()
print(f"  -- in GFED-active cells (GFED >= {QUIET}%/yr):")
print(f"       GFED5:                  {ba_hot_truth:8.1f} Mha/yr")
print(f"       Model C:                {ba_hot_ours:8.1f} Mha/yr  ({ba_hot_ours/ba_hot_truth:.2f}x)")
print()
print(f"  -- false-positive area (GFED < {QUIET}%, model > {FP_THRESH}%):")
print(f"       Model C extra:          {ba_fp_total:8.1f} Mha/yr  "
      f"({100*ba_fp_total/ba_ours_total:.0f}% of our total)")
print()
n_fp  = int(fp_mask.sum())
n_hot = int(hot_mask.sum())
n_land = int(land.sum())
print(f"  false-positive cells:        {n_fp:6d}  ({100*n_fp/n_land:.1f}% of burnable land cells)")
print(f"  GFED-active cells:           {n_hot:6d}")
print("="*64)

# Regional breakdown of false-positive area
REGIONS = {
    "Amazonia/S.America interior": (-20, 8, -80, -45),
    "Eastern N. America":          (25, 50, -100, -60),
    "Europe":                      (38, 60, -10, 40),
    "Boreal Asia":                 (50, 70, 60, 180),
    "SE Asia / Indonesia":         (-10, 25, 90, 150),
    "Central Africa forest":       (-5, 5, 10, 30),
}
print("\nFALSE-POSITIVE AREA BY REGION (Mha/yr):")
for name, (la0, la1, lo0, lo1) in REGIONS.items():
    li = (lat >= la0) & (lat <= la1); lj = (lon >= lo0) & (lon <= lo1)
    sub = np.zeros_like(fp_mask); sub[np.ix_(li, lj)] = True
    fp_area = float(((ours / 100.0) * cell_Mha * fp_mask * sub).sum())
    print(f"  {name:32s} {fp_area:7.1f}")

# ---- MAP: where false positives concentrate ----
fp_area_map = (ours / 100.0) * cell_Mha * fp_mask  # Mha/yr per cell
fp_area_map = np.where(fp_mask, fp_area_map, np.nan)

fig, axes = plt.subplots(2, 1, figsize=(12, 11), subplot_kw={"projection": ccrs.Robinson()})

# Panel 1: false-positive burned area
ax = axes[0]
ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
im = ax.pcolormesh(lon, lat, fp_area_map * 1000, transform=ccrs.PlateCarree(),
                   cmap="Reds", vmin=0, vmax=np.nanpercentile(fp_area_map*1000, 98),
                   shading="auto")
ax.set_title(f"False-positive burned area: model burns > {FP_THRESH}%/yr where GFED5 < {QUIET}%/yr\n"
             f"({ba_fp_total:.0f} Mha/yr, {100*ba_fp_total/ba_ours_total:.0f}% of our total)", fontsize=11)
cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
cb.set_label("burned area in false-positive cells (1000 ha / yr per cell)", fontsize=8)

# Panel 2: signed bias (model - GFED) for context
ax = axes[1]
ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
bias = np.where(land, ours - truth, np.nan)
im = ax.pcolormesh(lon, lat, bias, transform=ccrs.PlateCarree(),
                   cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
ax.set_title("Signed bias: Model C minus GFED5 (% per year)", fontsize=11)
cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
cb.set_label("model minus obs (% per year)", fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "9_false_positive_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nwrote 9_false_positive_map.png")
