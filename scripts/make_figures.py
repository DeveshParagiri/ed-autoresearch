"""
make_figures.py
Generate all figures requested by project lead:
  Fig 1 — Shapley ablation bar chart
  Fig 2 — Regional time series (Sub-Saharan Africa + South America)
  Fig 3 — Global time series
  Fig 4 — Regional maps: period mean + difference maps vs GFED

Run from repo root:
    python scripts/make_figures.py
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

REPO      = Path(__file__).resolve().parents[1]
MODELS_DIR= REPO / "models"
GFED_REF  = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc"
TRENDY_DIR= Path(r"C:\Users\owusu\OneDrive\Documents\UMD STUFF\ED AUTORESEARCH\fire_autoresearch\data\raw\trendy")
OUT_DIR   = REPO / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Colours
COL = {"A":       "#E07B39",
       "B":       "#5B8DB8",
       "C":       "#2E8B57",
       "GFED":    "#111111",
       "CLASSIC": "#9B59B6",
       "EDv3":    "#CC3333"}
LABELS = {"A":       "ED-Model A (8 mech, 27 params)",
          "B":       "ED-Model B (5 mech, 17 params)",
          "C":       "ED-Model C (3 mech, 12 params) ★",
          "GFED":    "GFED4.1s (reference)",
          "CLASSIC": "CLASSIC (best TRENDY)",
          "EDv3":    "EDv3 stock (baseline)"}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(name):
    """Load burntArea (fraction) from Dev's final NC, shape (192,360,720)."""
    p = MODELS_DIR / f"ED-Model{name}-final" / "burntArea.nc"
    ds = xr.open_dataset(p, decode_times=False)
    arr = ds["burntArea"].values.astype(np.float32)   # (192,360,720)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    ds.close()
    return arr

def load_trendy(filename, var="burntArea", start_year=1700):
    """Load a TRENDY model, extracting 2001-2016 (192 months).
    Returns (192, nlat, nlon) float32 in fraction units.
    Handles both 1-deg (180x360) and 0.5-deg (360x720) grids.
    """
    p = TRENDY_DIR / filename
    ds = xr.open_dataset(p, decode_times=False)
    arr = ds[var].values.astype(np.float32)
    ds.close()
    start_idx = (2001 - start_year) * 12
    arr = arr[start_idx: start_idx + 192]
    arr = np.where(np.isfinite(arr) & (arr >= 0), arr, np.nan)
    return arr

def upsample_1deg_to_05deg(arr):
    """Repeat each 1-deg cell into 2x2 block → 0.5-deg. Works on (...,180,360)."""
    return np.repeat(np.repeat(arr, 2, axis=-2), 2, axis=-1)

def load_gfed():
    """Load GFED4.1S (%) for 2001-2016 → fraction, shape (192,360,720)."""
    ds = xr.open_dataset(GFED_REF, decode_times=False)
    arr = ds["burntArea"].values.astype(np.float32)   # (240,360,720) 1997-2016
    ds.close()
    # 1997 = index 0; 2001 = index 48 (4 years × 12)
    arr = arr[48:240]   # (192,360,720)
    arr = arr / 100.0   # % → fraction
    arr = np.where(arr < 0, np.nan, arr)   # mask fill=-999
    return arr

def global_monthly_mean(arr, lat):
    """Weighted spatial mean → (192,) time series."""
    cos_lat = np.cos(np.deg2rad(lat))[np.newaxis, :, np.newaxis]
    w = np.broadcast_to(cos_lat, arr.shape)
    mask = np.isfinite(arr)
    num  = np.where(mask, arr * w, 0.0).sum(axis=(1,2))
    den  = np.where(mask, w,       0.0).sum(axis=(1,2))
    return num / (den + 1e-20)

def region_monthly_mean(arr, lat, lon, lat_range, lon_range):
    """Weighted spatial mean over a lat/lon box → (192,)."""
    la0, la1 = lat_range
    lo0, lo1 = lon_range
    lat_mask = (lat >= la0) & (lat <= la1)
    lon_mask = (lon >= lo0) & (lon <= lo1)
    sub  = arr[:, lat_mask, :][:, :, lon_mask]
    slat = lat[lat_mask]
    cos_lat = np.cos(np.deg2rad(slat))[np.newaxis, :, np.newaxis]
    w = np.broadcast_to(cos_lat, sub.shape)
    mask = np.isfinite(sub)
    num = np.where(mask, sub * w, 0.0).sum(axis=(1,2))
    den = np.where(mask, w,       0.0).sum(axis=(1,2))
    return num / (den + 1e-20)

def annual_from_monthly(ts):
    """Convert monthly (192,) → annual (16,) mean."""
    return ts.reshape(16, 12).mean(axis=1)

def period_mean(arr):
    """Time-mean over 192 months, ignoring NaN. Returns (360,720)."""
    return np.nanmean(arr, axis=0)

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
gfed    = load_gfed()
mA      = load_model("A")
mB      = load_model("B")
mC      = load_model("C")
classic = load_trendy("CLASSIC_S3_burntArea.nc")          # (192,180,360) 1-deg
edv3    = load_trendy("EDv3_S3_burntArea.nc")             # (192,360,720) 0.5-deg

# Upsample CLASSIC to 0.5-deg for map comparisons
classic_05 = upsample_1deg_to_05deg(classic)              # (192,360,720)

# 0.5-deg lat/lon (for models A/B/C, GFED, EDv3)
ds_tmp = xr.open_dataset(REPO / "models" / "ED-ModelC-final" / "burntArea.nc", decode_times=False)
lat    = ds_tmp["lat"].values   # (360,)
lon    = ds_tmp["lon"].values   # (720,)
ds_tmp.close()
# 1-deg lat/lon (for CLASSIC time series)
lat1   = np.arange(-89.5, 90.0, 1.0)
lon1   = np.arange(-179.5, 180.0, 1.0)

years     = np.arange(2001, 2017)
t_monthly = np.array([2001 + i / 12.0 for i in range(192)])

print("  GFED loaded:", gfed.shape)
print("  Models A/B/C loaded:", mA.shape)
print("  CLASSIC loaded:", classic.shape)
print("  EDv3 loaded:", edv3.shape)

# ── Figure 1: Shapley Ablation ─────────────────────────────────────────────────
print("\nFig 1: Shapley ablation...")

with open(MODELS_DIR / "shapley.json") as f:
    shap = json.load(f)

sorted_mech = shap["shapley_sorted"]
names_raw   = [s["mechanism"] for s in sorted_mech]
phis        = [s["phi"]       for s in sorted_mech]
pcts        = [s["percent"]   for s in sorted_mech]

# Human-readable labels
MECH_LABELS = {
    "t_air_ign":  "Monthly air temp (T_air ignition)",
    "precip":     "Annual + monthly precipitation",
    "gpp_monthly":"Monthly GPP hump",
    "gpp_anom":   "Per-cell GPP anomaly",
    "t_surf":     "Monthly surface soil temp",
    "soil_temp":  "Deep soil temp + rate",
    "height":     "Canopy height",
    "fuel":       "Fuel load",
}

labels = [MECH_LABELS.get(n, n) for n in names_raw]

# Colour by model membership
def bar_color(mech):
    top3 = {"t_air_ign", "precip", "gpp_monthly"}
    top5 = {"gpp_anom", "t_surf"}
    if mech in top3: return COL["C"]
    if mech in top5: return COL["B"]
    return "#AAAAAA"

colors = [bar_color(n) for n in names_raw]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(labels[::-1], phis[::-1], color=colors[::-1], edgecolor="white", height=0.65)

# Annotate with %
for bar, pct in zip(bars, pcts[::-1]):
    ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", ha="left", fontsize=9)

ax.set_xlabel("Shapley φ  (contribution to Overall Score)", fontsize=11)
ax.set_title("Figure 1 — Shapley Mechanism Ablation\nModel A (8 mechanisms, 500 Optuna trials per subset)", fontsize=12)
ax.axvline(0, color="black", lw=0.8)

patch_c = mpatches.Patch(color=COL["C"],  label="In Model C (top 3) ★")
patch_b = mpatches.Patch(color=COL["B"],  label="Added in Model B (top 5)")
patch_x = mpatches.Patch(color="#AAAAAA", label="Dropped (bottom 3)")
ax.legend(handles=[patch_c, patch_b, patch_x], loc="lower right", fontsize=9)

ax.set_xlim(0, max(phis) * 1.25)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_shapley_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_shapley_ablation.png")


# ── Figure 3: Global time series ───────────────────────────────────────────────
print("\nFig 3: Global time series...")

ts_gfed    = global_monthly_mean(gfed,    lat)  * 100
ts_A       = global_monthly_mean(mA,      lat)  * 100
ts_B       = global_monthly_mean(mB,      lat)  * 100
ts_C       = global_monthly_mean(mC,      lat)  * 100
ts_classic = global_monthly_mean(classic, lat1) * 100
ts_edv3    = global_monthly_mean(edv3,    lat)  * 100

ann_gfed    = annual_from_monthly(ts_gfed)
ann_A       = annual_from_monthly(ts_A)
ann_B       = annual_from_monthly(ts_B)
ann_C       = annual_from_monthly(ts_C)
ann_classic = annual_from_monthly(ts_classic)
ann_edv3    = annual_from_monthly(ts_edv3)

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
fig.suptitle("Figure 3 — Global Burned Area: Monthly & Annual Time Series", fontsize=13, y=1.01)

ax = axes[0]
ax.plot(t_monthly, ts_gfed,    color=COL["GFED"],    lw=2.2, label=LABELS["GFED"])
ax.plot(t_monthly, ts_classic, color=COL["CLASSIC"], lw=1.4, ls="--", alpha=0.9, label=LABELS["CLASSIC"])
ax.plot(t_monthly, ts_edv3,    color=COL["EDv3"],    lw=1.4, ls=":",  alpha=0.9, label=LABELS["EDv3"])
ax.plot(t_monthly, ts_A,       color=COL["A"],       lw=1.2, alpha=0.8, label=LABELS["A"])
ax.plot(t_monthly, ts_B,       color=COL["B"],       lw=1.2, alpha=0.8, label=LABELS["B"])
ax.plot(t_monthly, ts_C,       color=COL["C"],       lw=1.8, label=LABELS["C"])
ax.set_ylabel("Burned Area Fraction (%)", fontsize=10)
ax.set_title("Monthly global mean", fontsize=10)
ax.legend(fontsize=8, ncol=3)
ax.set_xlim(2001, 2017)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(years, ann_gfed,    "o-", color=COL["GFED"],    lw=2.2, ms=6, label=LABELS["GFED"])
ax.plot(years, ann_classic, "s--",color=COL["CLASSIC"], lw=1.4, ms=5, alpha=0.9, label=LABELS["CLASSIC"])
ax.plot(years, ann_edv3,    "^:", color=COL["EDv3"],    lw=1.4, ms=5, alpha=0.9, label=LABELS["EDv3"])
ax.plot(years, ann_A,       "s-", color=COL["A"],       lw=1.2, ms=4, alpha=0.8, label=LABELS["A"])
ax.plot(years, ann_B,       "^-", color=COL["B"],       lw=1.2, ms=4, alpha=0.8, label=LABELS["B"])
ax.plot(years, ann_C,       "D-", color=COL["C"],       lw=1.8, ms=5, label=LABELS["C"])
ax.set_ylabel("Burned Area Fraction (%)", fontsize=10)
ax.set_xlabel("Year", fontsize=10)
ax.set_title("Annual mean", fontsize=10)
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_global_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_global_timeseries.png")


# ── Figure 2: Regional time series ─────────────────────────────────────────────
print("\nFig 2: Regional time series...")

REGIONS = {
    "Sub-Saharan Africa":  ((-35, 15),  (10,  45)),
    "South America":       ((-25, 10),  (-75, -45)),
}

fig, axes = plt.subplots(len(REGIONS), 2, figsize=(15, 9))
fig.suptitle("Figure 2 — Regional Burned Area Time Series\n(Monthly left, Annual right)", fontsize=13)

for row, (region_name, (lat_r, lon_r)) in enumerate(REGIONS.items()):
    rg  = region_monthly_mean(gfed,    lat,  lon,  lat_r, lon_r) * 100
    rA  = region_monthly_mean(mA,      lat,  lon,  lat_r, lon_r) * 100
    rB  = region_monthly_mean(mB,      lat,  lon,  lat_r, lon_r) * 100
    rC  = region_monthly_mean(mC,      lat,  lon,  lat_r, lon_r) * 100
    rCL = region_monthly_mean(classic, lat1, lon1, lat_r, lon_r) * 100
    rED = region_monthly_mean(edv3,    lat,  lon,  lat_r, lon_r) * 100

    # Monthly
    ax = axes[row, 0]
    ax.plot(t_monthly, rg,  color=COL["GFED"],    lw=2.2, label=LABELS["GFED"])
    ax.plot(t_monthly, rCL, color=COL["CLASSIC"], lw=1.4, ls="--", alpha=0.9, label=LABELS["CLASSIC"])
    ax.plot(t_monthly, rED, color=COL["EDv3"],    lw=1.4, ls=":",  alpha=0.9, label=LABELS["EDv3"])
    ax.plot(t_monthly, rA,  color=COL["A"],       lw=1.2, alpha=0.8, label=LABELS["A"])
    ax.plot(t_monthly, rB,  color=COL["B"],       lw=1.2, alpha=0.8, label=LABELS["B"])
    ax.plot(t_monthly, rC,  color=COL["C"],       lw=1.8, label=LABELS["C"])
    ax.set_title(f"{region_name} — Monthly", fontsize=10)
    ax.set_ylabel("Burned Area (%)", fontsize=9)
    ax.set_xlim(2001, 2017)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Annual
    ax = axes[row, 1]
    ax.plot(years, annual_from_monthly(rg),  "o-",  color=COL["GFED"],    lw=2.2, ms=6, label=LABELS["GFED"])
    ax.plot(years, annual_from_monthly(rCL), "s--", color=COL["CLASSIC"], lw=1.4, ms=5, alpha=0.9, label=LABELS["CLASSIC"])
    ax.plot(years, annual_from_monthly(rED), "^:",  color=COL["EDv3"],    lw=1.4, ms=5, alpha=0.9, label=LABELS["EDv3"])
    ax.plot(years, annual_from_monthly(rA),  "s-",  color=COL["A"],       lw=1.2, ms=4, alpha=0.8, label=LABELS["A"])
    ax.plot(years, annual_from_monthly(rB),  "^-",  color=COL["B"],       lw=1.2, ms=4, alpha=0.8, label=LABELS["B"])
    ax.plot(years, annual_from_monthly(rC),  "D-",  color=COL["C"],       lw=1.8, ms=5, label=LABELS["C"])
    ax.set_title(f"{region_name} — Annual", fontsize=10)
    ax.set_ylabel("Burned Area (%)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_regional_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_regional_timeseries.png")


# ── Figure 4: Regional Maps ─────────────────────────────────────────────────────
print("\nFig 4: Maps...")

# Period means (all in %)
pm_gfed    = period_mean(gfed)       * 100
pm_A       = period_mean(mA)         * 100
pm_B       = period_mean(mB)         * 100
pm_C       = period_mean(mC)         * 100
pm_classic = period_mean(classic_05) * 100   # upsampled to 0.5-deg
pm_edv3    = period_mean(edv3)       * 100

# Difference maps vs GFED
def diff_map(model_pm): return model_pm - pm_gfed

LON2D, LAT2D = np.meshgrid(lon, lat)
land_mask_2d = np.isfinite(pm_gfed) & (pm_gfed >= 0)

def plot_map(ax, data, title, cmap, vmin, vmax, land_only=True):
    if land_only:
        data = np.where(land_mask_2d, data, np.nan)
    im = ax.pcolormesh(LON2D, LAT2D, data, cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#D0E8F0")
    return im

def peak_month(arr):
    clim  = arr.reshape(16, 12, *arr.shape[1:]).mean(axis=0)
    clim  = np.where(np.isfinite(clim), clim, np.nan)
    valid = np.any(np.isfinite(clim), axis=0)
    pk    = np.full(clim.shape[1:], np.nan, dtype=float)
    pk[valid] = np.nanargmax(clim[:, valid], axis=0) + 1
    return pk

vmax_mean = 4.0
vmax_diff = 1.5

# ---- Row layout: 3 rows × 6 cols ----
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Figure 4 — Period Mean Burned Area (2001–2016), Difference Maps & Peak Fire Month",
             fontsize=13, y=1.01)
gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.38, wspace=0.06)

# Row 0: period means — GFED | CLASSIC | EDv3 | Model A | Model B | Model C
datasets_pm = [
    (pm_gfed,    "GFED4.1s\n(Reference)"),
    (pm_classic, "CLASSIC\n(Best TRENDY)"),
    (pm_edv3,    "EDv3 stock\n(Baseline)"),
    (pm_A,       "ED-Model A"),
    (pm_B,       "ED-Model B"),
    (pm_C,       "ED-Model C ★"),
]
axs0 = [fig.add_subplot(gs[0, c]) for c in range(6)]
im0  = None
for ax, (data, title) in zip(axs0, datasets_pm):
    im0 = plot_map(ax, data, title, "YlOrRd", 0, vmax_mean)
cb0 = fig.colorbar(im0, ax=axs0, orientation="horizontal", fraction=0.04, pad=0.1, aspect=50)
cb0.set_label("Period Mean Burned Area (%)", fontsize=9)

# Row 1: difference maps — blank | CLASSIC−GFED | EDv3−GFED | A−GFED | B−GFED | C−GFED
ax10 = fig.add_subplot(gs[1, 0]); ax10.axis("off")
ax10.text(0.5, 0.5, "Difference Maps\n(Model − GFED)\n\nRed = overburn\nBlue = underburn",
          ha="center", va="center", fontsize=9, transform=ax10.transAxes)
diff_data = [
    (diff_map(pm_classic), "CLASSIC − GFED"),
    (diff_map(pm_edv3),    "EDv3 − GFED"),
    (diff_map(pm_A),       "Model A − GFED"),
    (diff_map(pm_B),       "Model B − GFED"),
    (diff_map(pm_C),       "Model C − GFED"),
]
axs1 = [fig.add_subplot(gs[1, c+1]) for c in range(5)]
imD  = None
for ax, (data, title) in zip(axs1, diff_data):
    imD = plot_map(ax, data, title, "RdBu_r", -vmax_diff, vmax_diff)
cb1 = fig.colorbar(imD, ax=[ax10]+axs1, orientation="horizontal", fraction=0.04, pad=0.1, aspect=50)
cb1.set_label("Difference in Burned Area (%)", fontsize=9)

# Row 2: peak fire month
pk_datasets = [
    (peak_month(gfed),       "GFED peak month"),
    (peak_month(classic_05), "CLASSIC peak month"),
    (peak_month(edv3),       "EDv3 peak month"),
    (peak_month(mA),         "Model A peak month"),
    (peak_month(mB),         "Model B peak month"),
    (peak_month(mC),         "Model C peak month"),
]
axs2 = [fig.add_subplot(gs[2, c]) for c in range(6)]
im2  = None
for ax, (data, title) in zip(axs2, pk_datasets):
    im2 = plot_map(ax, data, title, "hsv", 1, 12)
cb2 = fig.colorbar(im2, ax=axs2, orientation="horizontal", fraction=0.04, pad=0.1, aspect=50,
                   ticks=range(1,13))
cb2.set_label("Peak Fire Month", fontsize=9)
cb2.set_ticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

fig.savefig(OUT_DIR / "fig4_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_maps.png")

print(f"\nAll figures saved to: {OUT_DIR}")
