"""
Seasonal visualization of fire CARBON EMISSIONS (fFire) for the shipped Model C
(NSGA-II refit + retuned betas) vs GFED5. Four figures, mirroring maps_seasonal.py:
  1. Global monthly time series (PgC / month)
  2. Regional seasonal climatology (12-month cycle)
  3. Month-of-peak-emission map (model vs GFED5)
  4. Hovmoller diagram: latitude x month
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
YEARS = list(range(2001, 2017))
NY = len(YEARS)
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SEC_PER_MONTH = (365.25 / 12) * 86400.0


def load_ffire(path):
    """Return monthly fFire (192, lat, lon) in kg m-2 s-1, sliced to 2001-2016, + lat/lon."""
    da = xr.open_dataset(path)["fFire"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    return arr, da.lat.values, da.lon.values


ours, lat, lon = load_ffire(REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-Hybrid" / "fFire.nc")
truth, _, _    = load_ffire(REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc")

# Cell area (m^2)
R = 6.371e6
dlon = np.deg2rad(0.5)
area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
area2d = np.abs(np.broadcast_to(area_lat[:, None], (len(lat), len(lon))))  # m^2


def emis_PgC(monthly_flux):
    """kg m-2 s-1 -> PgC per month per cell. flux * sec_per_month * area / 1e12."""
    return monthly_flux * SEC_PER_MONTH * area2d[None, :, :] / 1e12


# ============ FIGURE 1: Global monthly time series ============
ours_ts  = emis_PgC(ours).sum(axis=(1, 2))   # PgC/month
truth_ts = emis_PgC(truth).sum(axis=(1, 2))
t_axis = np.arange(192) / 12.0 + 2001

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(t_axis, truth_ts, color="k", lw=1.3, label="GFED5")
ax.plot(t_axis, ours_ts,  color="darkorange", lw=1.3, label="ED-ModelC (ours)")
ax.set_xlabel("Year"); ax.set_ylabel("Global fire emissions (PgC / month)")
ax.set_title("Global monthly fire carbon emissions, 2001-2016")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "5_ffire_global_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 5_ffire_global_timeseries.png")


# ============ FIGURE 2: Regional seasonal climatology ============
REGIONS = {
    "N. Hemisphere Africa":  (0, 20, -20, 50),
    "S. Hemisphere Africa":  (-35, 0, 10, 50),
    "Australia":             (-45, -10, 110, 155),
    "South America":         (-35, 10, -82, -34),
    "Boreal (N. of 50N)":    (50, 75, -170, 180),
    "Central Asia":          (35, 55, 40, 110),
}


def regional_clim(monthly_flux, box):
    la0, la1, lo0, lo1 = box
    li = (lat >= la0) & (lat <= la1)
    lj = (lon >= lo0) & (lon <= lo1)
    e = emis_PgC(monthly_flux)[:, li, :][:, :, lj].sum(axis=(1, 2))  # PgC/month, 192
    return e.reshape(NY, 12).mean(axis=0) * 1000.0  # -> TgC/month for readability


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (name, box) in zip(axes.flat, REGIONS.items()):
    o = regional_clim(ours, box)
    t = regional_clim(truth, box)
    x = np.arange(12)
    ax.plot(x, t, "k-o", ms=3, lw=1.2, label="GFED5")
    ax.plot(x, o, color="darkorange", marker="o", ms=3, lw=1.2, label="ours")
    ax.set_title(name, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, fontsize=7, rotation=45)
    ax.set_ylabel("TgC / month", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Regional seasonal cycle of fire emissions (2001-2016 climatology)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "6_ffire_regional_seasonal_cycles.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 6_ffire_regional_seasonal_cycles.png")


# ============ FIGURE 3: Month-of-peak-emission map ============
def peak_month(monthly_flux):
    clim = monthly_flux.reshape(NY, 12, len(lat), len(lon)).mean(axis=0)
    total = clim.sum(axis=0)
    pm = np.argmax(clim, axis=0).astype(float)
    thresh = np.nanpercentile(total[total > 0], 20) if (total > 0).any() else 0
    pm[total < thresh] = np.nan
    return pm


pm_ours  = peak_month(ours)
pm_truth = peak_month(truth)
cyclic = plt.cm.hsv

fig, axes = plt.subplots(2, 1, figsize=(11, 10), subplot_kw={"projection": ccrs.Robinson()})
for ax, data, title in [(axes[0], pm_truth, "GFED5 — peak emission month"),
                         (axes[1], pm_ours, "ED-ModelC (ours) — peak emission month")]:
    ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap=cyclic, vmin=0, vmax=11, shading="auto")
    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7, ticks=range(12))
    cb.ax.set_xticklabels(MONTHS, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "7_ffire_peak_month_map.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 7_ffire_peak_month_map.png")


# ============ FIGURE 4: Hovmoller (latitude x month) ============
def hovmoller(monthly_flux):
    e = emis_PgC(monthly_flux)
    clim = e.reshape(NY, 12, len(lat), len(lon)).mean(axis=0)  # PgC/month/cell
    return clim.sum(axis=2).T * 1000.0  # TgC/month/lat band, (lat, 12)

hov_ours  = hovmoller(ours)
hov_truth = hovmoller(truth)
vmax = max(np.nanpercentile(hov_truth, 99), np.nanpercentile(hov_ours, 99))

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
for ax, data, title in [(axes[0], hov_truth, "GFED5"), (axes[1], hov_ours, "ED-ModelC (ours)")]:
    im = ax.pcolormesh(np.arange(12), lat, data, cmap="OrRd", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(12)); ax.set_xticklabels(MONTHS, fontsize=8, rotation=45)
    ax.set_xlabel("Month")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cb.set_label("Emissions (TgC / month / lat band)", fontsize=8)
axes[0].set_ylabel("Latitude")
fig.suptitle("Hovmoller: latitudinal migration of fire emissions (2001-2016 climatology)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "8_ffire_hovmoller_lat_month.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 8_ffire_hovmoller_lat_month.png")
print("done")
