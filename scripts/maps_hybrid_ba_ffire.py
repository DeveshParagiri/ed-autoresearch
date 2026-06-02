"""
Map comparison for the hybrid retune: BA + fFire, each vs GFED5 + 3 TRENDY models.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "Hybrid_GFED5"
OUT.mkdir(parents=True, exist_ok=True)

PLOT_LAT = xr.open_dataset(REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelC-Hybrid" / "burntArea.nc")["lat"].values
PLOT_LON = xr.open_dataset(REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelC-Hybrid" / "burntArea.nc")["lon"].values


def load_ba_pct(name):
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask]
    units = da.attrs.get("units", "1")
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(axis=1)
    if units in ("1", "fraction", ""):
        annual = annual * 100.0
    mean = annual.mean(axis=0)
    if mean.shape == (len(PLOT_LAT), len(PLOT_LON)):
        return mean
    src = xr.DataArray(mean, coords={"lat": da.lat.values, "lon": da.lon.values}, dims=("lat", "lon"))
    return src.interp(lat=PLOT_LAT, lon=PLOT_LON, method="linear").values


def load_ba_truth():
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask]
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(axis=1)
    return annual.mean(axis=0)


def load_ffire_gC_m2_yr(name):
    p = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / name / "fFire.nc"
    da = xr.open_dataset(p)["fFire"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask].astype(np.float64)
    units = da.attrs.get("units", "kg m-2 s-1")
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).mean(axis=0)
    mean = annual.mean(axis=0)
    if "s-1" in units:
        mean = mean * 86400 * 365 * 1000.0
    elif "month" in units:
        mean = mean * 12 * 1000.0
    if mean.shape == (len(PLOT_LAT), len(PLOT_LON)):
        return mean
    src = xr.DataArray(mean, coords={"lat": da.lat.values, "lon": da.lon.values}, dims=("lat", "lon"))
    return src.interp(lat=PLOT_LAT, lon=PLOT_LON, method="linear").values


def load_ffire_truth():
    p = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"
    da = xr.open_dataset(p)["fFire"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask].astype(np.float64)
    units = da.attrs.get("units", "kg m-2 s-1")
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).mean(axis=0)
    mean = annual.mean(axis=0)
    if "s-1" in units:
        mean = mean * 86400 * 365 * 1000.0
    return mean


def map_panel(ax, data, title, vmin, vmax, cmap, clabel):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(PLOT_LON, PLOT_LAT, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(clabel, fontsize=8)
    cb.ax.tick_params(labelsize=7)


# ============ BA 4-panel ============
print("Loading BA...")
ba_truth = load_ba_truth()
ba_ours  = load_ba_pct("ED-ModelC-Hybrid")
ba_clm6  = load_ba_pct("CLM6")
ba_elmfa = load_ba_pct("ELM-FATES")
print(f"  truth    mean={np.nanmean(ba_truth):.2f}% max={np.nanmax(ba_truth):.0f}")
print(f"  ours     mean={np.nanmean(ba_ours):.2f}% max={np.nanmax(ba_ours):.0f}")
print(f"  CLM6     mean={np.nanmean(ba_clm6):.2f}% max={np.nanmax(ba_clm6):.0f}")
print(f"  ELM-FATES mean={np.nanmean(ba_elmfa):.2f}% max={np.nanmax(ba_elmfa):.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Burned Area 2001-2016 mean annual (% per year)", fontsize=13)
map_panel(axes[0, 0], ba_truth, "GFED5 truth (mean 1.09% per yr)", 0, 15, "YlOrRd", "% per year")
map_panel(axes[0, 1], ba_ours,  "ED-ModelC-Hybrid (ours, ILAMB=0.6482, mean 6.27% per yr)", 0, 15, "YlOrRd", "% per year")
map_panel(axes[1, 0], ba_clm6,  "CLM6 (rank #1, ILAMB=0.6562, mean 4.86% per yr)", 0, 15, "YlOrRd", "% per year")
map_panel(axes[1, 1], ba_elmfa, "ELM-FATES (rank #2, ILAMB=0.6502, mean 6.87% per yr)", 0, 15, "YlOrRd", "% per year")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "BA_four_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote BA_four_panel.png")

# ============ fFire 4-panel ============
print("Loading fFire...")
ff_truth = load_ffire_truth()
ff_ours  = load_ffire_gC_m2_yr("ED-ModelC-Hybrid")
ff_clm6  = load_ffire_gC_m2_yr("CLM6")
ff_elmfa = load_ffire_gC_m2_yr("ELM-FATES")
print(f"  truth    mean={np.nanmean(ff_truth):.2f} g/m2/yr max={np.nanmax(ff_truth):.0f}")
print(f"  ours     mean={np.nanmean(ff_ours):.2f} g/m2/yr max={np.nanmax(ff_ours):.0f}")
print(f"  CLM6     mean={np.nanmean(ff_clm6):.2f} g/m2/yr max={np.nanmax(ff_clm6):.0f}")
print(f"  ELM-FATES mean={np.nanmean(ff_elmfa):.2f} g/m2/yr max={np.nanmax(ff_elmfa):.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Fire Carbon Emissions 2001-2016 mean annual (g C / m^2 / yr)", fontsize=13)
vmax = 200
map_panel(axes[0, 0], ff_truth, "GFED5 truth (mean 4.9 g C/m2/yr)", 0, vmax, "OrRd", "g C / m^2 / yr")
map_panel(axes[0, 1], ff_ours,  "ED-ModelC-Hybrid (ours, ILAMB=0.6465, rank #12)", 0, vmax, "OrRd", "g C / m^2 / yr")
map_panel(axes[1, 0], ff_clm6,  "CLM6 (rank #1, ILAMB=0.6913)", 0, vmax, "OrRd", "g C / m^2 / yr")
map_panel(axes[1, 1], ff_elmfa, "ELM-FATES (rank #5, ILAMB=0.6677)", 0, vmax, "OrRd", "g C / m^2 / yr")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fFire_four_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fFire_four_panel.png")

# ============ Bias stacks ============
print("Bias stacks...")
gfed_active = ba_truth > 0
fig, axes = plt.subplots(3, 1, figsize=(10, 12), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Burned Area bias vs GFED5 (% per year)", fontsize=12)
for ax, name, arr in [
    (axes[0], "ED-ModelC-Hybrid minus GFED5", ba_ours - ba_truth),
    (axes[1], "CLM6 minus GFED5", ba_clm6 - ba_truth),
    (axes[2], "ELM-FATES minus GFED5", ba_elmfa - ba_truth),
]:
    map_panel(ax, np.where(gfed_active, arr, np.nan), name, -10, 10, "RdBu_r", "pred minus obs (% per year)")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "BA_bias_stack.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote BA_bias_stack.png")

active_ff = ff_truth > 0
fig, axes = plt.subplots(3, 1, figsize=(10, 12), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("fFire bias vs GFED5 (g C / m^2 / yr)", fontsize=12)
for ax, name, arr in [
    (axes[0], "ED-ModelC-Hybrid minus GFED5", ff_ours - ff_truth),
    (axes[1], "CLM6 minus GFED5", ff_clm6 - ff_truth),
    (axes[2], "ELM-FATES minus GFED5", ff_elmfa - ff_truth),
]:
    map_panel(ax, np.where(active_ff, arr, np.nan), name, -150, 150, "RdBu_r", "pred minus obs (g C / m^2 / yr)")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "fFire_bias_stack.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fFire_bias_stack.png")
print("done")
