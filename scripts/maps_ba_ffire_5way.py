"""Two figure stacks for the paper:
  1. Burned area: GFED5 truth + our BA + CLM6 + CLASSIC + JSBACH + EDv3.
  2. Fire carbon emissions: GFED5 truth + our Emissions + CLM6 + CLASSIC + JSBACH + EDv3.
"""
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


def years_of(ds):
    tv = ds.time.values
    if hasattr(tv[0], "year"):
        return np.array([t.year for t in tv])
    if isinstance(tv[0], np.datetime64):
        dt = tv.astype("datetime64[s]").astype(object)
        return np.array([d.year for d in dt])
    return np.array([int(np.floor(float(t))) for t in tv])


def annual_pct_BA(monthly_arr, units, n_years):
    """Average annual burned area, percent per year. Handles 'fraction' (1)
    and percent units."""
    arr = monthly_arr.reshape(n_years, 12, *monthly_arr.shape[1:]).sum(axis=1)
    if units in ("%", "percent"):
        return arr.mean(axis=0)
    # default: fraction (units = '1' or empty); convert to percent
    return arr.mean(axis=0) * 100.0


def annual_gC_per_m2(monthly_fFire):
    """Convert kg/m2/s monthly to gC/m2/yr annual mean."""
    return np.nanmean(monthly_fFire, axis=0) * SEC_PER_MONTH * 12 * 1000.0


# ---------------- BURNED AREA stack ----------------
def load_ba_pct(path):
    ds = xr.open_dataset(path)
    yrs = years_of(ds)
    m = (yrs >= 2001) & (yrs <= 2016)
    if m.sum() == 0:
        ds.close(); return None, None, None
    arr = ds["burntArea"].values[m]
    units = ds["burntArea"].attrs.get("units", "1")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    pct = annual_pct_BA(arr, units, n_years=int(m.sum() / 12))
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat = ds[lat_name].values; lon = ds[lon_name].values
    ds.close()
    return pct, lat, lon


truth_ba_pct, truth_lat, truth_lon = load_ba_pct(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")

ba_layers = [("GFED5 truth", truth_ba_pct, truth_lat, truth_lon)]
for name in ["ED-ModelC-GFED5", "CLM6", "CLASSIC", "JSBACH", "EDv3"]:
    pct, la, lo = load_ba_pct(REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc")
    if pct is not None:
        ba_layers.append((name, pct, la, lo))


def panel(ax, data, lat_arr, lon_arr, title, vmax, cmap="YlOrRd"):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon_arr, lat_arr, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=10)
    return im


vmax_ba = float(np.nanquantile(truth_ba_pct, 0.99))
fig, axes = plt.subplots(2, 3, figsize=(18, 8),
                         subplot_kw={"projection": ccrs.Robinson()})
im = None
for ax, (name, data, la, lo) in zip(axes.flat, ba_layers):
    im = panel(ax, data, la, lo, name, vmax_ba)
for ax in list(axes.flat)[len(ba_layers):]:
    ax.set_visible(False)
cax = fig.add_axes([0.20, 0.05, 0.60, 0.014])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("annual burned area, percent per year", fontsize=10)
cb.ax.tick_params(labelsize=9)
fig.suptitle("Burned area: GFED5 vs five fire models", fontsize=14, y=0.97)
fig.subplots_adjust(top=0.93, bottom=0.10, wspace=0.05, hspace=0.18)
out_p = OUT / "FIG_burnedArea_stack_5models.png"
fig.savefig(out_p, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_p}")


# ---------------- fFIRE stack ----------------
def load_ff(path):
    ds = xr.open_dataset(path)
    yrs = years_of(ds)
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = ds["fFire"].values[m]
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    la = ds[lat_name].values; lo = ds[lon_name].values
    ds.close()
    return annual_gC_per_m2(arr), la, lo


truth_ff_a, ff_lat, ff_lon = load_ff(REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc")
ff_layers = [("GFED5 truth", truth_ff_a, ff_lat, ff_lon)]
for name in ["ED-ModelC-Emissions", "CLM6", "CLASSIC", "JSBACH", "EDv3"]:
    p = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / name / "fFire.nc"
    d, la, lo = load_ff(p)
    ff_layers.append((name, d, la, lo))

vmax_ff = float(np.nanquantile(truth_ff_a, 0.99))
fig, axes = plt.subplots(2, 3, figsize=(18, 8),
                         subplot_kw={"projection": ccrs.Robinson()})
for ax, (name, data, la, lo) in zip(axes.flat, ff_layers):
    im = panel(ax, data, la, lo, name, vmax_ff)
for ax in list(axes.flat)[len(ff_layers):]:
    ax.set_visible(False)
cax = fig.add_axes([0.20, 0.05, 0.60, 0.014])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("annual fire carbon emissions, gC/m^2 per year", fontsize=10)
cb.ax.tick_params(labelsize=9)
fig.suptitle("Fire carbon emissions: GFED5 vs five fire models", fontsize=14, y=0.97)
fig.subplots_adjust(top=0.93, bottom=0.10, wspace=0.05, hspace=0.18)
out_p = OUT / "FIG_fFire_stack_5models.png"
fig.savefig(out_p, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_p}")
