"""
Side-by-side maps of the SHIP variant (ED-ModelC-ILAMB) vs GFED4.1s truth,
plus a bias map.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "SHIP"
OUT.mkdir(parents=True, exist_ok=True)


def annual_pct_from_fraction(monthly_frac):
    yrs = 16
    arr = monthly_frac.reshape(yrs, 12, *monthly_frac.shape[1:]).sum(axis=1) * 100.0
    return arr.mean(axis=0)


def annual_pct_from_percent(monthly_pct):
    yrs = 16
    arr = monthly_pct.reshape(yrs, 12, *monthly_pct.shape[1:]).sum(axis=1)
    return arr.mean(axis=0)


# Load ship variant
ours = xr.open_dataset(REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelC-ILAMB" / "burntArea.nc")
pred = ours["burntArea"].values
plot_lat = ours["lat"].values; plot_lon = ours["lon"].values
ours_units = ours["burntArea"].attrs.get("units", "1")
print(f"ship units = {ours_units}, mean = {np.nanmean(pred):.4g}")

# Load GFED reference
gfed = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc")
yrs = np.array([t.year for t in gfed["time"].values])
mask = (yrs >= 2001) & (yrs <= 2016)
obs = gfed["burntArea"].values[mask]
gfed_units = gfed["burntArea"].attrs.get("units", "1")
print(f"gfed units = {gfed_units}, mean = {np.nanmean(obs):.4g}")

# Compute annual % maps
pred_pct = annual_pct_from_fraction(pred)            # 16-yr mean annual %
obs_pct  = annual_pct_from_percent(obs)              # 16-yr mean annual %

print(f"pred_pct mean = {np.nanmean(pred_pct):.3f}%, max = {np.nanmax(pred_pct):.2f}%")
print(f"obs_pct  mean = {np.nanmean(obs_pct):.3f}%,  max = {np.nanmax(obs_pct):.2f}%")


def map_panel(ax, data, title, cmap="YlOrRd", vmin=0, vmax=15,
              label="% per year"):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(plot_lon, plot_lat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


# Side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], obs_pct, "GFED4.1s truth, 2001-2016 mean annual burned area")
map_panel(axes[1], pred_pct, "ED-ModelC retuned, 2001-2016 mean")
fig.tight_layout()
fig.savefig(OUT / "01_truth_vs_ship.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 01_truth_vs_ship.png")

# Bias map (pred - obs)
gfed_active = (obs > 0).any(0)
bias = np.where(gfed_active, pred_pct - obs_pct, np.nan)

fig, ax = plt.subplots(figsize=(10, 5),
                        subplot_kw={"projection": ccrs.Robinson()})
ax.set_global()
ax.coastlines(linewidth=0.4, color="0.3")
ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
im = ax.pcolormesh(plot_lon, plot_lat, bias, transform=ccrs.PlateCarree(),
                   cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
ax.set_title("ED-ModelC retuned minus GFED4.1s (16-yr mean annual %)", fontsize=11)
cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
cb.set_label("pred minus obs (% per year)", fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "02_bias.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 02_bias.png")

# Three-panel: truth, ship, bias
fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], obs_pct, "GFED4.1s truth, 2001-2016 mean annual burned area")
map_panel(axes[1], pred_pct, "ED-ModelC retuned, 2001-2016 mean annual burned area")
axes[2].set_global()
axes[2].coastlines(linewidth=0.4, color="0.3")
axes[2].add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
im2 = axes[2].pcolormesh(plot_lon, plot_lat, bias, transform=ccrs.PlateCarree(),
                         cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
axes[2].set_title("Difference: ED-ModelC minus GFED4.1s", fontsize=11)
cb2 = plt.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.02, shrink=0.7)
cb2.set_label("pred minus obs (% per year)", fontsize=9)
cb2.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_modelC_vs_GFED.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 03_three_panel.png")
print("done")
