"""
ILAMB leaderboard visualizations.

(1) Bar chart of Overall Score per model, with score components stacked.
(2) Spatial maps of the four highlight models: GFED truth, top-of-leaderboard,
    Ours, Lei.
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
OUT = REPO / "NEW MAPS" / "ILAMB"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPO / "ilamb_out_leaderboard" / "scalar_database.csv")
df = df[(df["Region"] == "global") & (df["Source"] == "GFED4.1S")]
scores = ["Bias Score", "RMSE Score", "Seasonal Cycle Score", "Spatial Distribution Score", "Overall Score"]
piv = (df[df["ScalarName"].isin(scores)]
       .pivot_table(index="Model", columns="ScalarName", values="Data", aggfunc="first"))
piv = piv[scores].sort_values("Overall Score", ascending=True)

# ---------- (1) Leaderboard bar chart ----------
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"ED-ModelC-Ours": "#1f78b4", "ED-ModelC-Lei": "#e31a1c"}
bar_colors = [colors.get(m, "#999999") for m in piv.index]
ax.barh(piv.index, piv["Overall Score"], color=bar_colors, edgecolor="black", linewidth=0.4)
for i, (m, v) in enumerate(piv["Overall Score"].items()):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0, 0.78)
ax.set_xlabel("ILAMB Overall Score (higher is better)")
ax.set_title("Burned Area, GFED4.1S benchmark, global region")
ax.axvline(piv.loc["ED-ModelC-Ours", "Overall Score"], color="#1f78b4", lw=0.5, ls="--", alpha=0.5)
ax.axvline(piv.loc["ED-ModelC-Lei", "Overall Score"], color="#e31a1c", lw=0.5, ls="--", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "30_leaderboard_overall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 30_leaderboard_overall.png")

# ---------- (2) Score components per model, grouped bars ----------
fig, ax = plt.subplots(figsize=(10, 6))
sub = piv[["Bias Score", "RMSE Score", "Seasonal Cycle Score", "Spatial Distribution Score"]]
sub = sub.sort_values("Bias Score", ascending=True)
sub.plot(kind="barh", ax=ax, width=0.85, edgecolor="black", linewidth=0.3)
ax.set_xlabel("Component score (higher is better)")
ax.set_title("ILAMB score components, burned area")
ax.set_xlim(0, 1.0)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "31_score_components.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 31_score_components.png")

# ---------- (3) Spatial map comparison ----------
def annual_pct_from_fraction(monthly_frac):
    """16-yr mean annual percent burned, from monthly fraction (units '1')."""
    yrs = 16
    arr = monthly_frac.reshape(yrs, 12, *monthly_frac.shape[1:]).sum(axis=1) * 100.0
    return arr.mean(axis=0)


def annual_pct_from_percent(monthly_percent):
    """16-yr mean annual percent burned, from monthly percent (units '%').
    GFED reference NC is already in percent per month, so just sum and average."""
    yrs = 16
    arr = monthly_percent.reshape(yrs, 12, *monthly_percent.shape[1:]).sum(axis=1)
    return arr.mean(axis=0)


def load_burnt(name):
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc"
    return xr.open_dataset(p)["burntArea"]


def to_half_deg(da, plot_lat, plot_lon):
    """Annual % mean on a regular 0.5° grid via xarray interp.
    Reads `units` attr to decide whether to multiply by 100 (fraction) or not (already %)."""
    arr = da.values
    units = da.attrs.get("units", "1")
    yrs = arr.shape[0] // 12
    annual = arr.reshape(yrs, 12, *arr.shape[1:]).sum(axis=1)
    if units in ("1", "fraction", ""):
        annual = annual * 100.0
    mean = annual.mean(axis=0)
    if mean.shape == (360, 720):
        return mean
    src = xr.DataArray(mean, coords={"lat": da.lat.values, "lon": da.lon.values}, dims=("lat", "lon"))
    return src.interp(lat=plot_lat, lon=plot_lon, method="linear").values


def load_gfed_ref():
    p = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    return da, da.values


# Top model from leaderboard
top_model = piv.sort_values("Overall Score", ascending=False).index[0]
print(f"Top model = {top_model}")

ref_da, ref_arr = load_gfed_ref()
ref_lat = ref_da.lat.values
ref_lon = ref_da.lon.values
print(f"GFED ref shape {ref_arr.shape}")

# Slice GFED reference to 2001-2016 (192 months)
yrs = np.array([t.year for t in ref_da.time.values])
mask = (yrs >= 2001) & (yrs <= 2016)
print(f"GFED months in 2001-2016 window: {mask.sum()}")
print(f"GFED units = {ref_da.attrs.get('units', '?')}")
ref_arr = ref_arr[mask]
truth = annual_pct_from_percent(ref_arr)   # GFED is in '%'
print(f"truth shape: {truth.shape}, mean: {np.nanmean(truth):.3f}%, max: {np.nanmax(truth):.3f}%")

ours_da = load_burnt("ED-ModelC-Ours")
plot_lat = ours_da["lat"].values; plot_lon = ours_da["lon"].values
ours = to_half_deg(ours_da, plot_lat, plot_lon)
lei  = to_half_deg(load_burnt("ED-ModelC-Lei"), plot_lat, plot_lon)
top  = to_half_deg(load_burnt(top_model), plot_lat, plot_lon)
print(f"shapes: ours={ours.shape} lei={lei.shape} top={top.shape}")

# Reproject GFED ref onto 0.5° if needed
if truth.shape != (360, 720):
    ref_xr = xr.DataArray(truth, coords={"lat": ref_lat, "lon": ref_lon}, dims=("lat", "lon"))
    truth = ref_xr.interp(lat=plot_lat, lon=plot_lon, method="linear").values

vmax = 15
def map_panel(ax, data, title, cmap="YlOrRd", vmin=0, vmx=vmax, label="% per year"):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(plot_lon, plot_lat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmx, shading="auto")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                          subplot_kw={"projection": ccrs.Robinson()})
map_panel(axes[0, 0], truth, "GFED4.1S truth, 16-yr mean (%)")
map_panel(axes[0, 1], top,
          f"{top_model} (ILAMB rank 1, Overall {piv.loc[top_model, 'Overall Score']:.3f})")
map_panel(axes[1, 0], ours,
          f"ED-ModelC-Ours (rank 7, Overall {piv.loc['ED-ModelC-Ours', 'Overall Score']:.3f})")
map_panel(axes[1, 1], lei,
          f"ED-ModelC-Lei (rank 11, Overall {piv.loc['ED-ModelC-Lei', 'Overall Score']:.3f})")
fig.tight_layout()
fig.savefig(OUT / "32_four_panel_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 32_four_panel_comparison.png")

# Bias maps for ours, lei, and top
def bias_panel(ax, data, title):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(plot_lon, plot_lat, data, transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-10, vmax=10, shading="auto")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label("pred minus obs (% per year)", fontsize=8)
    cb.ax.tick_params(labelsize=7)


fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                          subplot_kw={"projection": ccrs.Robinson()})
bias_panel(axes[0], top - truth, f"{top_model} bias")
bias_panel(axes[1], ours - truth, "ED-ModelC-Ours bias")
bias_panel(axes[2], lei - truth, "ED-ModelC-Lei bias")
fig.tight_layout()
fig.savefig(OUT / "33_three_panel_bias.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 33_three_panel_bias.png")
print("done")
