"""
Four-panel map: GFED5 truth, ED-ModelC-GFED5 (ours), CLASSIC, CLM6.
2001-2016 mean annual burned area in % per year.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "GFED5"
OUT.mkdir(parents=True, exist_ok=True)


def load_model_pct(name, plot_lat, plot_lon):
    """Load monthly fraction burnt-area NC and return 16-yr mean annual %."""
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    # slice to 2001-2016 (192 months)
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask]
    units = da.attrs.get("units", "1")
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(axis=1)
    if units in ("1", "fraction", ""):
        annual = annual * 100.0
    mean = annual.mean(axis=0)
    if mean.shape == (360, 720):
        return mean
    src = xr.DataArray(mean, coords={"lat": da.lat.values, "lon": da.lon.values}, dims=("lat", "lon"))
    return src.interp(lat=plot_lat, lon=plot_lon, method="linear").values


def load_gfed5_pct(plot_lat, plot_lon):
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[mask]
    # units already in %, just sum 12 months and average
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(axis=1)
    return annual.mean(axis=0)


# Use 0.5° as plotting grid
ours_ds = xr.open_dataset(REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelC-GFED5" / "burntArea.nc")
plot_lat = ours_ds["lat"].values
plot_lon = ours_ds["lon"].values
ours_ds.close()

print("Loading...")
truth = load_gfed5_pct(plot_lat, plot_lon)
ours  = load_model_pct("ED-ModelC-GFED5", plot_lat, plot_lon)
classic = load_model_pct("CLASSIC", plot_lat, plot_lon)
clm6  = load_model_pct("CLM6", plot_lat, plot_lon)
print(f"truth mean={np.nanmean(truth):.3f}% max={np.nanmax(truth):.1f}%")
print(f"ours  mean={np.nanmean(ours):.3f}% max={np.nanmax(ours):.1f}%")
print(f"classic mean={np.nanmean(classic):.3f}% max={np.nanmax(classic):.1f}%")
print(f"clm6  mean={np.nanmean(clm6):.3f}% max={np.nanmax(clm6):.1f}%")


def map_panel(ax, data, title, vmax=15):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(plot_lon, plot_lat, data, transform=ccrs.PlateCarree(),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("% per year", fontsize=9)
    cb.ax.tick_params(labelsize=8)


# 4-panel
fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0, 0], truth, "GFED5 truth, 2001-2016 mean annual burned area")
map_panel(axes[0, 1], ours, "ED-ModelC-GFED5 (ours, rank 5)")
map_panel(axes[1, 0], classic, "CLASSIC (rank 11)")
map_panel(axes[1, 1], clm6, "CLM6 (rank 1)")
fig.tight_layout()
fig.savefig(OUT / "01_four_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 01_four_panel.png")

# Stacked vertical (easier to scan)
fig, axes = plt.subplots(4, 1, figsize=(10, 16),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], truth, "GFED5 truth")
map_panel(axes[1], ours, "ED-ModelC-GFED5 (ours)")
map_panel(axes[2], classic, "CLASSIC")
map_panel(axes[3], clm6, "CLM6")
fig.tight_layout()
fig.savefig(OUT / "02_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 02_stacked.png")

# Bias maps for each model vs truth
gfed_active = truth > 0
fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                          subplot_kw={"projection": ccrs.Robinson()})
for ax, name, arr in [
    (axes[0], "ED-ModelC-GFED5 minus GFED5", ours - truth),
    (axes[1], "CLASSIC minus GFED5", classic - truth),
    (axes[2], "CLM6 minus GFED5", clm6 - truth),
]:
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    bias = np.where(gfed_active, arr, np.nan)
    im = ax.pcolormesh(plot_lon, plot_lat, bias, transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
    ax.set_title(name, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("pred minus obs (% per year)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_bias_stack.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 03_bias_stack.png")
print("done")
