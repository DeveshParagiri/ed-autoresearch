"""
Four-panel map of fire carbon emissions (annual mean g C m-2 yr-1):
GFED5 truth, ED-ModelC-ILAMB (ours, ranks above CLASSIC on both BA and fFire),
CLASSIC, CLM6.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "GFED5_FFIRE"
OUT.mkdir(parents=True, exist_ok=True)

SEC_PER_YEAR = 365.25 * 86400.0  # ~3.156e7
KG_TO_G = 1000.0


def load_ffire_annual(name):
    """Load monthly kg/m^2/s fFire NC, take 16-yr mean of monthly flux,
    convert to annual g C m^-2 yr^-1. Returns (lat, lon) array."""
    p = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / name / "fFire.nc"
    ds = xr.open_dataset(p)
    f = ds["fFire"].values  # monthly kg/m^2/s
    yrs = np.array([t.year for t in ds.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    f = f[mask]
    # Annual mean of monthly fluxes × s/yr × 1000 = g/m^2/yr
    annual_kg_per_m2_per_yr = f.mean(axis=0) * SEC_PER_YEAR
    return annual_kg_per_m2_per_yr * KG_TO_G  # g/m^2/yr


def load_gfed5_truth_annual():
    p = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"
    ds = xr.open_dataset(p)
    f = ds["fFire"].values
    yrs = np.array([t.year for t in ds.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    f = f[mask]
    annual_kg_per_m2_per_yr = f.mean(axis=0) * SEC_PER_YEAR
    return annual_kg_per_m2_per_yr * KG_TO_G, ds.lat.values, ds.lon.values


truth, lat, lon = load_gfed5_truth_annual()
ours = load_ffire_annual("ED-ModelC-ILAMB")
classic = load_ffire_annual("CLASSIC")
clm6 = load_ffire_annual("CLM6")

print(f"truth   mean={np.nanmean(truth):.3f} g/m^2/yr  max={np.nanmax(truth):.1f}")
print(f"ours    mean={np.nanmean(ours):.3f} g/m^2/yr  max={np.nanmax(ours):.1f}")
print(f"classic mean={np.nanmean(classic):.3f}  max={np.nanmax(classic):.1f}")
print(f"clm6    mean={np.nanmean(clm6):.3f}  max={np.nanmax(clm6):.1f}")


def map_panel(ax, data, title, vmax=300):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap="hot_r", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("g C m$^{-2}$ yr$^{-1}$", fontsize=9)
    cb.ax.tick_params(labelsize=8)


# 2x2
fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0, 0], truth, "GFED5 truth (2001-2016 mean fire emissions)")
map_panel(axes[0, 1], ours, "ED-ModelC retuned (ours, beats CLASSIC on BA & fFire)")
map_panel(axes[1, 0], classic, "CLASSIC")
map_panel(axes[1, 1], clm6, "CLM6")
fig.tight_layout()
fig.savefig(OUT / "01_four_panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 01_four_panel.png")

# Vertical stack
fig, axes = plt.subplots(4, 1, figsize=(10, 16),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], truth, "GFED5 truth")
map_panel(axes[1], ours, "ED-ModelC retuned (ours)")
map_panel(axes[2], classic, "CLASSIC")
map_panel(axes[3], clm6, "CLM6")
fig.tight_layout()
fig.savefig(OUT / "02_stacked.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 02_stacked.png")

# Bias maps
active = truth > 0
fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                          subplot_kw={"projection": ccrs.Robinson()})
for ax, name, arr in [
    (axes[0], "ED-ModelC retuned minus GFED5", ours - truth),
    (axes[1], "CLASSIC minus GFED5", classic - truth),
    (axes[2], "CLM6 minus GFED5", clm6 - truth),
]:
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    bias = np.where(active, arr, np.nan)
    im = ax.pcolormesh(lon, lat, bias, transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-200, vmax=200, shading="auto")
    ax.set_title(name, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("pred minus obs (g C m$^{-2}$ yr$^{-1}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_bias_stack.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 03_bias_stack.png")
print("done")
