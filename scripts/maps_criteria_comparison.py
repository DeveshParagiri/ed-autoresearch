"""
Maps comparing the three Hurtt criteria refits against GFED5 truth.
  (a) every fire equal       — ED-ModelC-GFED5
  (b) every continent equal  — ED-ModelC-GFED5cont
  (c) every fire type equal  — not yet run, placeholder
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "CRITERIA"
OUT.mkdir(parents=True, exist_ok=True)


def annual_pct_from_fraction(monthly_frac, n_years=16):
    arr = monthly_frac.reshape(n_years, 12, *monthly_frac.shape[1:]).sum(axis=1) * 100.0
    return arr.mean(axis=0)


def annual_pct_from_percent(monthly_pct, n_years=16):
    arr = monthly_pct.reshape(n_years, 12, *monthly_pct.shape[1:]).sum(axis=1)
    return arr.mean(axis=0)


def load_model_pct(name):
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    return annual_pct_from_fraction(da.values[mask])


# Truth
gfed5 = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")
yrs = np.array([t.year for t in gfed5.time.values])
mask = (yrs >= 2001) & (yrs <= 2016)
truth = annual_pct_from_percent(gfed5["burntArea"].values[mask])
plot_lat = gfed5["lat"].values
plot_lon = gfed5["lon"].values

crit_a = load_model_pct("ED-ModelC-GFED5")
crit_b = load_model_pct("ED-ModelC-GFED5cont")

print(f"truth   mean={np.nanmean(truth):.3f}%  max={np.nanmax(truth):.1f}")
print(f"crit_a  mean={np.nanmean(crit_a):.3f}%  max={np.nanmax(crit_a):.1f}")
print(f"crit_b  mean={np.nanmean(crit_b):.3f}%  max={np.nanmax(crit_b):.1f}")


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


# Three panels stacked
fig, axes = plt.subplots(3, 1, figsize=(10, 12), subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], truth, "GFED5 truth, 2001-2016 mean annual burned area")
map_panel(axes[1], crit_a, "Criterion (a), every fire equal — ED-ModelC-GFED5")
map_panel(axes[2], crit_b, "Criterion (b), every continent equal — ED-ModelC-GFED5cont")
fig.tight_layout()
fig.savefig(OUT / "01_truth_a_b.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 01_truth_a_b.png")


# Side-by-side a vs b
fig, axes = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0], crit_a, "(a) every fire equal", vmax=15)
map_panel(axes[1], crit_b, "(b) every continent equal", vmax=15)
fig.tight_layout()
fig.savefig(OUT / "02_a_vs_b.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 02_a_vs_b.png")


# Difference: b - a (where does continent-equal pull the model)
diff = crit_b - crit_a
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.Robinson()})
ax.set_global()
ax.coastlines(linewidth=0.4, color="0.3")
ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
im = ax.pcolormesh(plot_lon, plot_lat, diff, transform=ccrs.PlateCarree(),
                   cmap="RdBu_r", vmin=-5, vmax=5, shading="auto")
ax.set_title("(b) minus (a): where continent-equal weighting changes the prediction", fontsize=11)
cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
cb.set_label("(b) minus (a), % per year", fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_b_minus_a.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 03_b_minus_a.png")


# Bias maps for each criterion vs truth
active = truth > 0
fig, axes = plt.subplots(2, 1, figsize=(10, 8), subplot_kw={"projection": ccrs.Robinson()})
for ax, name, arr in [
    (axes[0], "(a) every fire equal minus GFED5", crit_a - truth),
    (axes[1], "(b) every continent equal minus GFED5", crit_b - truth),
]:
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    bias = np.where(active, arr, np.nan)
    im = ax.pcolormesh(plot_lon, plot_lat, bias, transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
    ax.set_title(name, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("pred minus obs (% per year)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "04_bias_a_b.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 04_bias_a_b.png")
print("done")
