"""
Figure set for the NEW continental Model C (per-continent + fuel form) vs GFED5.
All outputs -> NEW MAPS/continental_model/.

  01_burned_area_map.png   annual burned fraction: model | GFED5 | difference
  02_ffire_map.png         annual fire carbon: model | GFED5 | difference
  03_global_timeseries.png global monthly BA (Mha) and fFire (PgC/month), model vs GFED5
  04_regional_seasonal.png per-continent seasonal cycle (mean monthly burned %), model vs GFED5
  05_per_cell_scatter.png  per-grid-cell 1:1 burned-fraction scatter, model vs GFED5

Run: python scripts/figures_continental_model.py
Note: fFire uses the current (k4-tuned) combustion betas, not yet retuned for this BA,
so its magnitude (~2.7 PgC/yr vs GFED5 3.40) is indicative; the spatial pattern is the point.
"""
from pathlib import Path
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "continental_model"; OUT.mkdir(parents=True, exist_ok=True)
BA_MODEL = REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc"
BA_GFED = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
FF_MODEL = REPO / "ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-continental-Hurtt/fFire.nc"
FF_GFED = REPO / "ilamb_ref_official/DATA/fFire/GFED5/fFire.nc"
YEARS = list(range(2001, 2017)); SEC_YR = 365.25 * 86400.0
FIRE_CMAP = LinearSegmentedColormap.from_list("fire",
            ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"])
REGIONS = {"Africa": (-20, 52, -36, 18), "S.America": (-82, -34, -56, 14),
           "N.America": (-168, -52, 14, 74), "Boreal Eurasia": (40, 180, 48, 78),
           "Trop/SE Asia": (60, 150, -11, 30), "Australia": (112, 154, -44, -10)}


def load_var(path, var):
    da = xr.open_dataset(path)[var]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[sel].astype(np.float64)
    return arr, da.lat.values, da.lon.values, da.attrs.get("units", "")


def cell_area_km2(lat, lon):
    dlat = abs(lat[1] - lat[0]); dlon = 360.0 / len(lon)
    return ((111.32 * dlat) * (111.32 * dlon * np.cos(np.deg2rad(lat))))[:, None] * np.ones((1, len(lon)))


def draw_map(ax, data, lat, lon, title, cmap, vmin, vmax, label):
    ax.set_global(); ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
    m = ax.pcolormesh(lon, lat, np.where(np.abs(data) < 1e-12, np.nan, data),
                      transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85, extend="both")
    cb.set_label(label, fontsize=9); cb.ax.tick_params(labelsize=8)


# ---------- 01 burned-area map ----------
ba_m, lat, lon, ba_u = load_var(BA_MODEL, "burntArea")
ba_g, _, _, ba_gu = load_var(BA_GFED, "burntArea")
if ba_gu in ("%", "percent"): ba_g = ba_g / 100.0
ann_m = ba_m.reshape(16, 12, *ba_m.shape[1:]).sum(1).mean(0) * 100.0   # %/yr
ann_g = np.nan_to_num(ba_g.reshape(16, 12, *ba_g.shape[1:]).sum(1).mean(0)) * 100.0
fig, ax = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Annual burned-area fraction (% / yr), 2001-2016 — continental Model C vs GFED5  "
             "(magnitude 1.03x, spatial r 0.70)", fontsize=13, y=1.04)
draw_map(ax[0], ann_m, lat, lon, "Model C continental", FIRE_CMAP, 0, 30, "% burned / yr")
draw_map(ax[1], ann_g, lat, lon, "GFED5 (observed)", FIRE_CMAP, 0, 30, "% burned / yr")
draw_map(ax[2], ann_m - ann_g, lat, lon, "Model - GFED5", "RdBu_r", -15, 15, "% / yr")
fig.tight_layout(); fig.savefig(OUT / "01_burned_area_map.png", dpi=140, bbox_inches="tight")
plt.close(fig); print("wrote 01_burned_area_map.png")

# ---------- 02 fFire map ----------
ff_m, latf, lonf, _ = load_var(FF_MODEL, "fFire")
ff_g, latg, long_, ffgu = load_var(FF_GFED, "fFire")
gC_m = np.nan_to_num(ff_m.mean(0)) * SEC_YR * 1000.0                   # gC/m2/yr
gC_g = np.nan_to_num(ff_g.mean(0)) * SEC_YR * 1000.0
fig, ax = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Annual fire carbon flux (gC / m2 / yr), 2001-2016 — continental Model C vs GFED5  "
             "(betas retuned for this BA: 3.42 vs 3.40 PgC/yr)", fontsize=13, y=1.04)
draw_map(ax[0], gC_m, latf, lonf, "Model C continental", FIRE_CMAP, 0, 500, "gC/m2/yr")
draw_map(ax[1], gC_g, latg, long_, "GFED5 (observed)", FIRE_CMAP, 0, 500, "gC/m2/yr")
if gC_m.shape == gC_g.shape:
    draw_map(ax[2], gC_m - gC_g, latf, lonf, "Model - GFED5", "RdBu_r", -300, 300, "gC/m2/yr")
fig.tight_layout(); fig.savefig(OUT / "02_ffire_map.png", dpi=140, bbox_inches="tight")
plt.close(fig); print("wrote 02_ffire_map.png")

# ---------- 03 global timeseries ----------
area = cell_area_km2(lat, lon)
ba_mha_m = (ba_m * area[None] / 1e4).sum((1, 2))                       # Mha per month
ba_mha_g = (np.nan_to_num(ba_g) * area[None] / 1e4).sum((1, 2))
areaf = cell_area_km2(latf, lonf) * 1e6                                # m2
ff_pg_m = (ff_m * areaf[None]).sum((1, 2)) * (SEC_YR / 12) / 1e12      # PgC per month
ff_pg_g = (np.nan_to_num(ff_g) * (cell_area_km2(latg, long_) * 1e6)[None]).sum((1, 2)) * (SEC_YR / 12) / 1e12
tmonths = np.arange(192) / 12.0 + 2001
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(tmonths, ba_mha_g, color="k", lw=1.2, label=f"GFED5  ({ba_mha_g.mean()*12:.0f} Mha/yr)")
ax[0].plot(tmonths, ba_mha_m, color="#e31a1c", lw=1.2, label=f"Model C  ({ba_mha_m.mean()*12:.0f} Mha/yr)")
ax[0].set_ylabel("Burned area (Mha / month)"); ax[0].legend(fontsize=9); ax[0].set_title("Global burned area")
ax[1].plot(tmonths, ff_pg_g, color="k", lw=1.2, label=f"GFED5  ({ff_pg_g.sum()/16:.2f} PgC/yr)")
ax[1].plot(tmonths, ff_pg_m, color="#e31a1c", lw=1.2, label=f"Model C  ({ff_pg_m.sum()/16:.2f} PgC/yr)")
ax[1].set_ylabel("Fire carbon (PgC / month)"); ax[1].set_xlabel("Year"); ax[1].legend(fontsize=9)
ax[1].set_title("Global fire carbon emissions")
fig.suptitle("Global monthly time series 2001-2016 — continental Model C vs GFED5", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "03_global_timeseries.png", dpi=140, bbox_inches="tight")
plt.close(fig); print("wrote 03_global_timeseries.png")

# ---------- 04 regional seasonal cycle ----------
LON, LAT = np.meshgrid(lon, lat)
cyc_m = ba_m.reshape(16, 12, *ba_m.shape[1:]).mean(0)                  # (12, lat, lon) monthly frac
cyc_g = np.nan_to_num(ba_g).reshape(16, 12, *ba_g.shape[1:]).mean(0)
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
mon = np.arange(1, 13)
for ax, (name, (lo0, lo1, la0, la1)) in zip(axs.ravel(), REGIONS.items()):
    box = (LON >= lo0) & (LON <= lo1) & (LAT >= la0) & (LAT <= la1)
    w = (np.cos(np.deg2rad(LAT)) * box)
    sm = np.array([(cyc_m[k] * w).sum() / w.sum() * 100 for k in range(12)])
    sg = np.array([(cyc_g[k] * w).sum() / w.sum() * 100 for k in range(12)])
    ax.plot(mon, sg, "k-o", ms=3, lw=1.2, label="GFED5")
    ax.plot(mon, sm, "-o", color="#e31a1c", ms=3, lw=1.2, label="Model C")
    ax.set_title(name, fontsize=11); ax.set_xticks([1, 4, 7, 10])
    ax.set_xlabel("month"); ax.set_ylabel("burned % / month"); ax.legend(fontsize=8)
fig.suptitle("Regional seasonal cycle of burned area — continental Model C vs GFED5", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "04_regional_seasonal.png", dpi=140, bbox_inches="tight")
plt.close(fig); print("wrote 04_regional_seasonal.png")

# ---------- 05 per-cell scatter ----------
pm_m = ba_m.mean(0); pm_g = np.nan_to_num(ba_g).mean(0)
mask = np.isfinite(pm_g) & ((pm_g * 12) > 0.01)
mm, gg = pm_m[mask], pm_g[mask]
r = np.corrcoef(mm, gg)[0, 1]; sigma = mm.std() / gg.std(); slope = r * sigma
fig, ax = plt.subplots(figsize=(6.5, 6))
hb = ax.hexbin(gg, mm, gridsize=80, extent=(0, 0.12, 0, 0.12),
               norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
ax.plot([0, 0.12], [0, 0.12], "r--", lw=1, label="1:1 (target)")
ax.plot([0, 0.12], [0, 0.12 * slope], "w-", lw=1.4, alpha=0.85, label=f"fit slope {slope:.2f}")
ax.set_xlim(0, 0.12); ax.set_ylim(0, 0.12)
ax.set_xlabel("GFED5 burned fraction"); ax.set_ylabel("Model C burned fraction")
ax.set_title(f"Per-grid-cell burned area (active-fire cells)\nr {r:.2f}   sigma {sigma:.2f}   slope {slope:.2f}")
ax.legend(loc="upper right", fontsize=9)
plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02).set_label("cell count (log)")
fig.tight_layout(); fig.savefig(OUT / "05_per_cell_scatter.png", dpi=140, bbox_inches="tight")
plt.close(fig); print("wrote 05_per_cell_scatter.png")
print(f"\nAll figures -> {OUT}")
