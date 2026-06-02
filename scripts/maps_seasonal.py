"""
Seasonal visualization of the shipped Model C (NSGA-II refit) vs GFED5.
Four figures:
  1. Global monthly time series (2001-2016)
  2. Regional seasonal climatology (12-month cycle) for major fire regions
  3. Month-of-peak-burning map (model vs GFED5)
  4. Hovmoller diagram: latitude x month
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "Seasonal"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2001, 2017))
NY = len(YEARS)
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def load_ba(path, to_pct=True):
    """Return monthly burned-fraction array (192, lat, lon) on the 0.5deg grid, plus lat/lon."""
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[m].astype(np.float64)
    units = da.attrs.get("units", "1")
    arr = np.nan_to_num(arr, nan=0.0)
    if to_pct and units in ("1", "fraction", ""):
        arr = arr * 100.0   # fraction -> percent
    # GFED5 ref is already in %
    return arr, da.lat.values, da.lon.values


ours, lat, lon = load_ba(REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc")
truth, _, _    = load_ba(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")

# Area weighting: cos(lat) * cell area for global aggregation
R = 6.371e6
dlat = np.deg2rad(0.5); dlon = np.deg2rad(0.5)
area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
area2d = np.abs(np.broadcast_to(area_lat[:, None], (len(lat), len(lon))))  # m^2 per cell

# Convert % burned to burned AREA (Mha) per cell-month: frac * cell_area
def burned_area_Mha(monthly_pct):
    # monthly_pct in %, /100 -> fraction; * area m^2 -> m^2 burned; /1e10 -> Mha
    return (monthly_pct / 100.0) * area2d[None, :, :] / 1e10


# ============ FIGURE 1: Global monthly time series ============
ours_ts  = burned_area_Mha(ours).sum(axis=(1, 2))   # Mha per month, 192 pts
truth_ts = burned_area_Mha(truth).sum(axis=(1, 2))
t_axis = np.arange(192) / 12.0 + 2001

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(t_axis, truth_ts, color="k", lw=1.3, label="GFED5")
ax.plot(t_axis, ours_ts,  color="firebrick", lw=1.3, label="ED-ModelC (ours)")
ax.set_xlabel("Year"); ax.set_ylabel("Global burned area (Mha / month)")
ax.set_title("Global monthly burned area, 2001-2016")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "1_global_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 1_global_timeseries.png")


# ============ FIGURE 2: Regional seasonal climatology ============
# Regions defined by lat/lon boxes (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    "N. Hemisphere Africa":  (0, 20, -20, 50),
    "S. Hemisphere Africa":  (-35, 0, 10, 50),
    "Australia":             (-45, -10, 110, 155),
    "South America":         (-35, 10, -82, -34),
    "Boreal (N. of 50N)":    (50, 75, -170, 180),
    "Central Asia":          (35, 55, 40, 110),
}


def regional_clim(monthly_pct, box):
    la0, la1, lo0, lo1 = box
    li = (lat >= la0) & (lat <= la1)
    lj = (lon >= lo0) & (lon <= lo1)
    ba = burned_area_Mha(monthly_pct)[:, li, :][:, :, lj].sum(axis=(1, 2))  # Mha/month, 192
    return ba.reshape(NY, 12).mean(axis=0)  # 12-month climatology


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (name, box) in zip(axes.flat, REGIONS.items()):
    o = regional_clim(ours, box)
    t = regional_clim(truth, box)
    x = np.arange(12)
    ax.plot(x, t, "k-o", ms=3, lw=1.2, label="GFED5")
    ax.plot(x, o, color="firebrick", marker="o", ms=3, lw=1.2, label="ours")
    ax.set_title(name, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, fontsize=7, rotation=45)
    ax.set_ylabel("Mha / month", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Regional seasonal cycle of burned area (2001-2016 climatology)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "2_regional_seasonal_cycles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 2_regional_seasonal_cycles.png")


# ============ FIGURE 3: Month-of-peak-burning map ============
def peak_month(monthly_pct):
    clim = monthly_pct.reshape(NY, 12, len(lat), len(lon)).mean(axis=0)  # (12, lat, lon)
    total = clim.sum(axis=0)
    pm = np.argmax(clim, axis=0).astype(float)  # 0-11
    pm[total < 0.05] = np.nan  # mask near-zero-fire cells (annual < 0.05%)
    return pm


pm_ours  = peak_month(ours)
pm_truth = peak_month(truth)
cyclic = plt.cm.hsv

fig, axes = plt.subplots(2, 1, figsize=(11, 10), subplot_kw={"projection": ccrs.Robinson()})
for ax, data, title in [(axes[0], pm_truth, "GFED5 — peak burning month"),
                         (axes[1], pm_ours, "ED-ModelC (ours) — peak burning month")]:
    ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap=cyclic, vmin=0, vmax=11, shading="auto")
    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7, ticks=range(12))
    cb.ax.set_xticklabels(MONTHS, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "3_peak_month_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 3_peak_month_map.png")


# ============ FIGURE 4: Hovmoller (latitude x month) ============
def hovmoller(monthly_pct):
    ba = burned_area_Mha(monthly_pct)  # (192, lat, lon)
    clim = ba.reshape(NY, 12, len(lat), len(lon)).mean(axis=0)  # (12, lat, lon)
    return clim.sum(axis=2).T  # (lat, 12) — sum over lon

hov_ours  = hovmoller(ours)
hov_truth = hovmoller(truth)
vmax = max(np.nanpercentile(hov_truth, 99), np.nanpercentile(hov_ours, 99))

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
for ax, data, title in [(axes[0], hov_truth, "GFED5"), (axes[1], hov_ours, "ED-ModelC (ours)")]:
    im = ax.pcolormesh(np.arange(12), lat, data, cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(12)); ax.set_xticklabels(MONTHS, fontsize=8, rotation=45)
    ax.set_xlabel("Month")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cb.set_label("Burned area (Mha / month / lat band)", fontsize=8)
axes[0].set_ylabel("Latitude")
fig.suptitle("Hovmoller: latitudinal migration of fire season (2001-2016 climatology)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "4_hovmoller_lat_month.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 4_hovmoller_lat_month.png")
print("done")
