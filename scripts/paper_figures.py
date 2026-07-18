"""Generate paper figures for Models C, D, E vs GFED5.

Outputs under paper/figures/ (and a copy under figures/paper/):
  fig_maps_ladder.png       annual BA maps (GFED5, ED-stock, C, D, E)
  fig_diff_ladder.png       model - GFED5 difference maps
  fig_scatter.png           per-cell scatter C / D / E
  fig_seasonal.png          regional seasonal cycles (C, D, E)
  fig_emissions.png         Model E fFire vs GFED5
  fig_timeseries.png        global BA (+ fFire if present) timeseries for E

Requires prior: BA NetCDFs under ilamb/MODELS/paper/ (see reproduce_paper.py)
               and/or official ILAMB verify staging.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, LogNorm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
# Mirror for people who look under figures/
(REPO / "figures" / "paper").mkdir(parents=True, exist_ok=True)

def _save(fig, name):
    for d in (OUT, REPO / "figures" / "paper"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=160, bbox_inches="tight")
    print("wrote", name, "-> paper/figures/ and figures/paper/")

YEARS = list(range(2001, 2017))
SEC_YR = 365.25 * 86400.0
FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire", ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
)
REGIONS = {
    "Africa": (-20, 52, -36, 18),
    "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74),
    "Boreal Eurasia": (40, 180, 48, 78),
    "Trop/SE Asia": (60, 150, -11, 30),
    "Australia": (112, 154, -44, -10),
}

BA_PATHS = {
    "ED-stock": [
        REPO / "ilamb/MODELS/paper/ED-stock/burntArea.nc",
        REPO / "ilamb/MODELS/EDv3/burntArea.nc",
    ],
    "C": [
        REPO / "ilamb/MODELS/paper/Model-C/burntArea.nc",
        REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc",
    ],
    "D": [
        REPO / "ilamb/MODELS/paper/Model-D/burntArea.nc",
        REPO / "ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc",
    ],
    "E": [
        REPO / "ilamb/MODELS/paper/Model-E/burntArea.nc",
        REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc",
    ],
}
FF_PATHS = [
    REPO / "ilamb/MODELS/paper/Model-E-fFire/fFire.nc",
    REPO / "ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-continental-percont/fFire.nc",
]
GFED_BA = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
GFED_FF = REPO / "ilamb_ref_official/DATA/fFire/GFED5/fFire.nc"


def first_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None


def load_var(path, var):
    da = xr.open_dataset(path)[var]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[sel].astype(np.float64)
    units = da.attrs.get("units", "")
    lat, lon = da.lat.values, da.lon.values
    da.close()
    return arr, lat, lon, units


def annual_pct(monthly, units):
    arr = np.nan_to_num(monthly, nan=0.0)
    if units in ("%", "percent"):
        return arr.reshape(16, 12, *arr.shape[1:]).sum(1).mean(0)
    return arr.reshape(16, 12, *arr.shape[1:]).sum(1).mean(0) * 100.0


def period_mean_frac(monthly, units):
    arr = np.nan_to_num(monthly, nan=0.0)
    if units in ("%", "percent"):
        arr = arr / 100.0
    return arr.mean(0)


def cell_area_km2(lat, lon):
    dlat = abs(lat[1] - lat[0])
    dlon = 360.0 / len(lon)
    return ((111.32 * dlat) * (111.32 * dlon * np.cos(np.deg2rad(lat))))[:, None] * np.ones(
        (1, len(lon))
    )


def draw_map(ax, data, lat, lon, title, cmap, vmin, vmax, label):
    ax.set_global()
    ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
    m = ax.pcolormesh(
        lon,
        lat,
        np.where(np.abs(data) < 1e-12, np.nan, data),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85, extend="both")
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def load_ba_stack():
    g, lat, lon, gu = load_var(GFED_BA, "burntArea")
    out = {"GFED5": annual_pct(g, gu)}
    monthly = {"GFED5": (g, gu)}
    for key, paths in BA_PATHS.items():
        p = first_existing(paths)
        if p is None:
            print(f"[warn] missing BA for Model {key}")
            continue
        arr, la, lo, u = load_var(p, "burntArea")
        out[key] = annual_pct(arr, u)
        monthly[key] = (arr, u)
        print(f"loaded Model {key}: {p.relative_to(REPO)}")
    return out, monthly, lat, lon


def fig_maps(ann, lat, lon):
    labels = ["GFED5", "ED-stock", "C", "D", "E"]
    present = [k for k in labels if k in ann]
    n = len(present)
    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.6), subplot_kw={"projection": ccrs.Robinson()}
    )
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Mean annual burned-area fraction (% yr$^{-1}$), 2001–2016",
        fontsize=13,
        y=1.04,
    )
    titles = {
        "GFED5": "GFED5 (observed)",
        "ED-stock": "ED-stock",
        "C": "Model C",
        "D": "Model D",
        "E": "Model E",
    }
    for ax, k in zip(axes, present):
        draw_map(ax, ann[k], lat, lon, titles[k], FIRE_CMAP, 0, 30, "% burned / yr")
    fig.tight_layout()
    _save(fig, "fig_maps_ladder.png")
    plt.close(fig)

    models = [k for k in ("C", "D", "E") if k in ann]
    if not models or "GFED5" not in ann:
        return
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(5 * len(models), 4.6),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Model − GFED5 annual burned fraction (% yr$^{-1}$)", fontsize=13, y=1.04)
    for ax, k in zip(axes, models):
        draw_map(
            ax,
            ann[k] - ann["GFED5"],
            lat,
            lon,
            f"Model {k} − GFED5",
            "RdBu_r",
            -15,
            15,
            "% / yr",
        )
    fig.tight_layout()
    fig.savefig(OUT / "fig_diff_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_diff_ladder.png")


def fig_scatter(monthly):
    models = [k for k in ("C", "D", "E") if k in monthly]
    if not models or "GFED5" not in monthly:
        return
    g = period_mean_frac(*monthly["GFED5"])
    mask = np.isfinite(g) & ((g * 12.0) > 0.01)
    gg = g[mask]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.2))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(
        "Per-grid-cell burned fraction on active-fire cells (GFED5 annual > 1%)",
        fontsize=12,
    )
    for ax, k in zip(axes, models):
        m = period_mean_frac(*monthly[k])[mask]
        r = np.corrcoef(m, gg)[0, 1]
        sigma = m.std() / (gg.std() + 1e-12)
        slope = r * sigma
        ax.hexbin(
            gg,
            m,
            gridsize=70,
            extent=(0, 0.12, 0, 0.12),
            norm=LogNorm(vmin=1, vmax=1e4),
            cmap="viridis",
            mincnt=1,
        )
        ax.plot([0, 0.12], [0, 0.12], "r--", lw=1, label="1:1")
        ax.plot([0, 0.12], [0, 0.12 * slope], "w-", lw=1.3, alpha=0.9, label=f"slope {slope:.2f}")
        ax.set_xlim(0, 0.12)
        ax.set_ylim(0, 0.12)
        ax.set_xlabel("GFED5")
        ax.set_ylabel(f"Model {k}")
        ax.set_title(f"Model {k}\nr={r:.2f}  σ={sigma:.2f}  slope={slope:.2f}")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_scatter.png")


def fig_seasonal(monthly, lat, lon):
    models = [k for k in ("C", "D", "E") if k in monthly]
    if not models or "GFED5" not in monthly:
        return
    LON, LAT = np.meshgrid(lon, lat)
    g_arr, gu = monthly["GFED5"]
    g_cyc = np.nan_to_num(g_arr).reshape(16, 12, *g_arr.shape[1:]).mean(0)
    if gu not in ("%", "percent"):
        g_cyc = g_cyc * 100.0
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    mon = np.arange(1, 13)
    colors = {"C": "#1f77b4", "D": "#ff7f0e", "E": "#e31a1c"}
    for ax, (name, (lo0, lo1, la0, la1)) in zip(axs.ravel(), REGIONS.items()):
        box = (LON >= lo0) & (LON <= lo1) & (LAT >= la0) & (LAT <= la1)
        w = np.cos(np.deg2rad(LAT)) * box
        sg = np.array([(g_cyc[k] * w).sum() / (w.sum() + 1e-12) for k in range(12)])
        ax.plot(mon, sg, "k-o", ms=3, lw=1.2, label="GFED5")
        for mk in models:
            arr, u = monthly[mk]
            cyc = np.nan_to_num(arr).reshape(16, 12, *arr.shape[1:]).mean(0)
            if u not in ("%", "percent"):
                cyc = cyc * 100.0
            sm = np.array([(cyc[k] * w).sum() / (w.sum() + 1e-12) for k in range(12)])
            ax.plot(mon, sm, "-o", color=colors[mk], ms=3, lw=1.1, label=f"Model {mk}")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlabel("month")
        ax.set_ylabel("burned % / month")
        ax.legend(fontsize=7)
    fig.suptitle("Regional seasonal cycle of burned area vs GFED5", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_seasonal.png")


def fig_emissions_and_ts(monthly, lat, lon):
    if "E" not in monthly or "GFED5" not in monthly:
        return
    ba_m, bu = monthly["E"]
    ba_g, gu = monthly["GFED5"]
    if gu in ("%", "percent"):
        ba_g = ba_g / 100.0
    if bu in ("%", "percent"):
        ba_m = ba_m / 100.0
    area = cell_area_km2(lat, lon)
    ba_mha_m = (np.nan_to_num(ba_m) * area[None] / 1e4).sum((1, 2))
    ba_mha_g = (np.nan_to_num(ba_g) * area[None] / 1e4).sum((1, 2))
    tmonths = np.arange(192) / 12.0 + 2001

    ff_path = first_existing(FF_PATHS)
    has_ff = ff_path is not None and GFED_FF.is_file()
    if has_ff:
        ff_m, latf, lonf, _ = load_var(ff_path, "fFire")
        ff_g, latg, long_, _ = load_var(GFED_FF, "fFire")
        areaf = cell_area_km2(latf, lonf) * 1e6
        areag = cell_area_km2(latg, long_) * 1e6
        ff_pg_m = (np.nan_to_num(ff_m) * areaf[None]).sum((1, 2)) * (SEC_YR / 12) / 1e12
        ff_pg_g = (np.nan_to_num(ff_g) * areag[None]).sum((1, 2)) * (SEC_YR / 12) / 1e12
        gC_m = np.nan_to_num(ff_m.mean(0)) * SEC_YR * 1000.0
        gC_g = np.nan_to_num(ff_g.mean(0)) * SEC_YR * 1000.0

        fig, ax = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": ccrs.Robinson()})
        fig.suptitle(
            "Mean annual fire carbon emissions (gC m$^{-2}$ yr$^{-1}$), Model E vs GFED5",
            fontsize=13,
            y=1.04,
        )
        draw_map(ax[0], gC_m, latf, lonf, "Model E", FIRE_CMAP, 0, 500, "gC/m2/yr")
        draw_map(ax[1], gC_g, latg, long_, "GFED5", FIRE_CMAP, 0, 500, "gC/m2/yr")
        if gC_m.shape == gC_g.shape:
            draw_map(ax[2], gC_m - gC_g, latf, lonf, "Model E − GFED5", "RdBu_r", -300, 300, "gC/m2/yr")
        fig.tight_layout()
        fig.savefig(OUT / "fig_emissions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote fig_emissions.png  ({ff_path.relative_to(REPO)})")

        fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax[0].plot(tmonths, ba_mha_g, color="k", lw=1.2, label=f"GFED5 ({ba_mha_g.mean()*12:.0f} Mha/yr)")
        ax[0].plot(tmonths, ba_mha_m, color="#e31a1c", lw=1.2, label=f"Model E ({ba_mha_m.mean()*12:.0f} Mha/yr)")
        ax[0].set_ylabel("Burned area (Mha / month)")
        ax[0].legend(fontsize=9)
        ax[0].set_title("Global burned area")
        ax[1].plot(tmonths, ff_pg_g, color="k", lw=1.2, label=f"GFED5 ({ff_pg_g.sum()/16:.2f} PgC/yr)")
        ax[1].plot(tmonths, ff_pg_m, color="#e31a1c", lw=1.2, label=f"Model E ({ff_pg_m.sum()/16:.2f} PgC/yr)")
        ax[1].set_ylabel("Fire carbon (PgC / month)")
        ax[1].set_xlabel("Year")
        ax[1].legend(fontsize=9)
        ax[1].set_title("Global fire carbon emissions")
        fig.suptitle("Global monthly time series 2001–2016 — Model E vs GFED5", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / "fig_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote fig_timeseries.png")
    else:
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.plot(tmonths, ba_mha_g, color="k", lw=1.2, label=f"GFED5 ({ba_mha_g.mean()*12:.0f} Mha/yr)")
        ax.plot(tmonths, ba_mha_m, color="#e31a1c", lw=1.2, label=f"Model E ({ba_mha_m.mean()*12:.0f} Mha/yr)")
        ax.set_ylabel("Burned area (Mha / month)")
        ax.set_xlabel("Year")
        ax.legend(fontsize=9)
        ax.set_title("Global burned area 2001–2016 — Model E vs GFED5")
        fig.tight_layout()
        fig.savefig(OUT / "fig_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote fig_timeseries.png (BA only; no fFire yet)")


def main():
    if not GFED_BA.is_file():
        raise SystemExit(
            f"Missing GFED5 BA reference: {GFED_BA}\n"
            "Place ILAMB GFED5 data under ilamb_ref_official/ or set ILAMB_ROOT."
        )
    ann, monthly, lat, lon = load_ba_stack()
    fig_maps(ann, lat, lon)
    fig_scatter(monthly)
    fig_seasonal(monthly, lat, lon)
    fig_emissions_and_ts(monthly, lat, lon)
    print(f"\nAll figures -> {OUT}")


if __name__ == "__main__":
    main()
