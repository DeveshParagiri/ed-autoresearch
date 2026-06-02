"""Map stack: GFED5 fFire truth vs ED-ModelC-Emissions vs ED-ModelC-EmpiricalEmit."""
from __future__ import annotations
from pathlib import Path
import numpy as np, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "EMISSIONS"
OUT.mkdir(parents=True, exist_ok=True)
SEC_PER_MONTH = (365.25 / 12) * 86400.0


def annual_gC_per_m2(monthly_kg_per_s):
    return np.nanmean(monthly_kg_per_s, axis=0) * SEC_PER_MONTH * 12 * 1000.0


def load_ffire(path):
    ds = xr.open_dataset(path)
    return ds["fFire"].values, ds["lat"].values, ds["lon"].values


truth_path = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"
truth, lat, lon = load_ffire(truth_path)
ds_t = xr.open_dataset(truth_path)
yrs = np.array([t.year for t in ds_t["time"].values])
mask = (yrs >= 2001) & (yrs <= 2016)
truth = truth[mask]

emissions, _, _ = load_ffire(REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-Emissions" / "fFire.nc")
empirical, _, _ = load_ffire(REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-EmpiricalEmit" / "fFire.nc")

t_a = annual_gC_per_m2(truth)
e_a = annual_gC_per_m2(emissions)
p_a = annual_gC_per_m2(empirical)
vmax = float(np.nanquantile(t_a, 0.99))


def panel(ax, data, title):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    return im


fig, axes = plt.subplots(3, 1, figsize=(11, 13), subplot_kw={"projection": ccrs.Robinson()})
im = panel(axes[0], t_a, "GFED5 fFire, 2001-2016 mean annual emissions")
panel(axes[1], e_a, "ED-ModelC-Emissions (process-based, ILAMB 0.6582, 3.45 PgC/yr)")
panel(axes[2], p_a, "ED-ModelC-EmpiricalEmit (empirical EF, ILAMB 0.6611)")
cax = fig.add_axes([0.12, 0.05, 0.76, 0.012])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("annual fire carbon emissions, gC/m² per year", fontsize=10)
cb.ax.tick_params(labelsize=9)
fig.suptitle("Fire carbon emissions: GFED5 vs ED Model C variants", fontsize=13, y=0.94)
out_p = OUT / "FIG_ffire_stack_truth_vs_ED_variants.png"
fig.savefig(out_p, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_p}")
print(f"truth mean = {np.nanmean(t_a):.2f} gC/m^2/yr")
print(f"Emissions mean = {np.nanmean(e_a):.2f}")
print(f"EmpiricalEmit mean = {np.nanmean(p_a):.2f}")
