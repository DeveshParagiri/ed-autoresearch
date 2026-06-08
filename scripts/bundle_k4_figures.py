"""Generate the full figure set for the promoted canonical (tropfix2-k4) into the
self-contained bundle ED-ModelC-tropfix2-k4/figures/. All annotations use the
NEW official scores. Reads the now-canonical k4 BA/fFire (ED-ModelC-Hybrid).
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "ED-ModelC-tropfix2-k4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# NEW official ILAMB scores (vs GFED5)
BA = {"ours": 0.6473, "CLM6": 0.6562, "ELM-FATES": 0.6502}
FF = {"ours": 0.6534, "CLM6": 0.6913, "ELM-FATES": 0.6677}

PLAT = xr.open_dataset(REPO/"ilamb"/"MODELS_LEADERBOARD"/"ED-ModelC-Hybrid"/"burntArea.nc")["lat"].values
PLON = xr.open_dataset(REPO/"ilamb"/"MODELS_LEADERBOARD"/"ED-ModelC-Hybrid"/"burntArea.nc")["lon"].values


def _regrid(mean, la, lo):
    if mean.shape == (len(PLAT), len(PLON)):
        return mean
    return xr.DataArray(mean, coords={"lat": la, "lon": lo}, dims=("lat", "lon")
                        ).interp(lat=PLAT, lon=PLON, method="linear").values


def ba_pct(name):
    da = xr.open_dataset(REPO/"ilamb"/"MODELS_LEADERBOARD"/name/"burntArea.nc")["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m]); n = arr.shape[0]//12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(1)
    if da.attrs.get("units", "1") in ("1", "fraction", ""): annual = annual*100.0
    return _regrid(annual.mean(0), da.lat.values, da.lon.values)


def ba_truth():
    da = xr.open_dataset(REPO/"ilamb_ref_official"/"DATA"/"burntArea"/"GFED5"/"burntArea.nc")["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m]); n = arr.shape[0]//12
    return arr.reshape(n, 12, *arr.shape[1:]).sum(1).mean(0)


def ff_gcm2(name):
    da = xr.open_dataset(REPO/"ilamb"/"MODELS_LEADERBOARD_FFIRE_GFED5"/name/"fFire.nc")["fFire"]
    yrs = np.array([t.year for t in da.time.values]); m = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[m].astype(np.float64); u = da.attrs.get("units", "kg m-2 s-1")
    mean = arr.reshape(arr.shape[0]//12, 12, *arr.shape[1:]).mean(0).mean(0)
    if "s-1" in u: mean = mean*86400*365*1000.0
    elif "month" in u: mean = mean*12*1000.0
    return _regrid(mean, da.lat.values, da.lon.values)


def ff_truth():
    da = xr.open_dataset(REPO/"ilamb_ref_official"/"DATA"/"fFire"/"GFED5"/"fFire.nc")["fFire"]
    yrs = np.array([t.year for t in da.time.values]); m = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[m].astype(np.float64); u = da.attrs.get("units", "kg m-2 s-1")
    mean = arr.reshape(arr.shape[0]//12, 12, *arr.shape[1:]).mean(0).mean(0)
    if "s-1" in u: mean = mean*86400*365*1000.0
    return mean


def panel(ax, data, title, vmin, vmax, cmap, clabel):
    ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(PLON, PLAT, data, transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)


# ---------- BA 4-panel ----------
bt, bo, bc, be = ba_truth(), ba_pct("ED-ModelC-Hybrid"), ba_pct("CLM6"), ba_pct("ELM-FATES")
fig, ax = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Burned Area 2001-2016 mean annual (% per year)", fontsize=13)
panel(ax[0,0], bt, f"GFED5 truth (mean {np.nanmean(bt):.2f}%/yr)", 0, 15, "YlOrRd", "% per year")
panel(ax[0,1], bo, f"ED-ModelC-k4 (ours, ILAMB={BA['ours']:.4f}, rank #3, 1.11x, mean {np.nanmean(bo):.2f}%/yr)", 0, 15, "YlOrRd", "% per year")
panel(ax[1,0], bc, f"CLM6 (rank #1, ILAMB={BA['CLM6']:.4f})", 0, 15, "YlOrRd", "% per year")
panel(ax[1,1], be, f"ELM-FATES (rank #2, ILAMB={BA['ELM-FATES']:.4f})", 0, 15, "YlOrRd", "% per year")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(OUT/"BA_four_panel.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("BA_four_panel.png")

# ---------- fFire 4-panel ----------
ft, fo, fc, fe = ff_truth(), ff_gcm2("ED-ModelC-Hybrid"), ff_gcm2("CLM6"), ff_gcm2("ELM-FATES")
fig, ax = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Fire Carbon Emissions 2001-2016 mean annual (g C / m^2 / yr)", fontsize=13)
vmax = 200
panel(ax[0,0], ft, f"GFED5 truth (mean {np.nanmean(ft):.1f} gC/m2/yr)", 0, vmax, "OrRd", "g C / m^2 / yr")
panel(ax[0,1], fo, f"ED-ModelC-k4 (ours, ILAMB={FF['ours']:.4f}, rank #4)", 0, vmax, "OrRd", "g C / m^2 / yr")
panel(ax[1,0], fc, f"CLM6 (rank #1, ILAMB={FF['CLM6']:.4f})", 0, vmax, "OrRd", "g C / m^2 / yr")
panel(ax[1,1], fe, f"ELM-FATES (rank #2, ILAMB={FF['ELM-FATES']:.4f})", 0, vmax, "OrRd", "g C / m^2 / yr")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(OUT/"fFire_four_panel.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("fFire_four_panel.png")

# ---------- bias panels ----------
fig, ax = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={"projection": ccrs.Robinson()})
panel(ax[0], np.where(bt > 0, bo - bt, np.nan), "BA: ED-ModelC-k4 minus GFED5", -10, 10, "RdBu_r", "% per year")
panel(ax[1], np.where(ft > 0, fo - ft, np.nan), "fFire: ED-ModelC-k4 minus GFED5", -150, 150, "RdBu_r", "g C / m^2 / yr")
fig.suptitle("Bias vs GFED5 (red = over-predict, blue = under-predict)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"bias_vs_gfed5.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("bias_vs_gfed5.png")

# ---------- global timeseries (BA + fFire) ----------
R = 6.371e6; dlon = np.deg2rad(0.5)
al = (R**2)*dlon*(np.sin(np.deg2rad(PLAT+0.25))-np.sin(np.deg2rad(PLAT-0.25)))
area = np.abs(al)[:, None]


def ts_ba(name, truth=False):
    p = (REPO/"ilamb_ref_official"/"DATA"/"burntArea"/"GFED5"/"burntArea.nc") if truth \
        else (REPO/"ilamb"/"MODELS_LEADERBOARD"/name/"burntArea.nc")
    da = xr.open_dataset(p)["burntArea"]; yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016); arr = np.nan_to_num(da.values[m].astype(np.float64))
    if not truth and da.attrs.get("units", "1") in ("1", "fraction", ""): arr = arr*100.0
    return ((arr/100.0)*area[None]/1e10).sum((1, 2))


def ts_ff(name, truth=False):
    p = (REPO/"ilamb_ref_official"/"DATA"/"fFire"/"GFED5"/"fFire.nc") if truth \
        else (REPO/"ilamb"/"MODELS_LEADERBOARD_FFIRE_GFED5"/name/"fFire.nc")
    da = xr.open_dataset(p)["fFire"]; yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016); arr = np.nan_to_num(da.values[m].astype(np.float64))
    # kg/m2/s -> PgC/month: * area * seconds-in-month; approx month=365/12 days
    sec_mo = 86400*365/12.0
    return (arr*area[None]*sec_mo/1e12).sum((1, 2))


t = np.arange(192)/12.0 + 2001
fig, ax = plt.subplots(2, 1, figsize=(14, 8))
ax[0].plot(t, ts_ba("", True), "k", lw=1.2, label="GFED5")
ax[0].plot(t, ts_ba("ED-ModelC-Hybrid"), color="tab:blue", lw=1.2, label="ED-ModelC-k4 (1.11x)")
ax[0].set_ylabel("BA (Mha / month)"); ax[0].set_title("Global monthly burned area"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t, ts_ff("", True), "k", lw=1.2, label="GFED5")
ax[1].plot(t, ts_ff("ED-ModelC-Hybrid"), color="darkorange", lw=1.2, label="ED-ModelC-k4")
ax[1].set_ylabel("fFire (PgC / month)"); ax[1].set_xlabel("Year"); ax[1].set_title("Global monthly fire carbon emissions"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"global_timeseries.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("global_timeseries.png")
print(f"done -> {OUT}")
