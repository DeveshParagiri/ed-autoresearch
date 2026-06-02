"""
Build a reference map of the world's major fire regions, with labels,
overlaid on the GFED4.1s observed mean burned fraction (2001-2010).

Output:  NEW MAPS/00_fire_regions_labeled.png
"""
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "00_fire_regions_labeled.png"


def load_gfed_halfdeg(years):
    out = np.zeros((len(years) * 12, 360, 720), dtype=np.float32)
    idx = 0
    for yr in years:
        with h5py.File(REPO / "data" / "gfed" / f"GFED4.1s_{yr}.hdf5", "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:][::-1, :]
                out[idx] = arr.reshape(360, 2, 720, 2).mean(axis=(1, 3))
                idx += 1
    return np.nan_to_num(out, nan=0.0)


# (label,  centroid lat, centroid lon,  arrow tail lat, arrow tail lon)
# Centroid = where the region actually IS.
# Tail = where the text label SITS (offset to keep map readable).
REGIONS = [
    # name,              region_lat, region_lon, label_lat, label_lon
    ("Sahel",                 13,    5,     30,    -25),
    ("Sahara (no fire)",      24,   10,     38,    -30),
    ("Congo Basin",           -3,   23,    -28,    -25),
    ("Southern African\nsavanna", -14,  25,   -38,    -10),
    ("Madagascar",           -19,   47,    -38,     65),
    ("Cerrado (Brazil)",     -12,  -49,    -38,    -75),
    ("Amazon\n(arc of deforestation)", -7, -60,  10, -90),
    ("California /\nUS West", 38, -120,    55,   -150),
    ("Pacific NW",            45, -120,    62,   -135),
    ("Boreal Canada",         58,  -100,   72,    -70),
    ("Alaskan boreal",        65,  -150,   72,   -165),
    ("Boreal Siberia",        62,  100,    78,    100),
    ("Indo-Gangetic\nplain",  26,   80,    42,     65),
    ("SE Asia\n(Myanmar/Thailand)", 18, 100, 38, 120),
    ("Indonesia /\nBorneo (peat)", -2, 113,  -22,    138),
    ("N. Australia\n(savanna)", -15, 132,   -32,    148),
    ("E. Australia",         -28,  148,    -45,    162),
    ("Mediterranean",         40,   12,     58,     20),
    ("Mexico /\nCentral America", 18, -95,   30,   -125),
]


def main():
    print("Loading GFED 2001-2010 ...")
    obs = load_gfed_halfdeg(range(2001, 2011))
    annual = obs.reshape(10, 12, 360, 720).sum(axis=1).mean(axis=0)

    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(-179.75, 180.0, 0.5)

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor="0.55")

    a = np.where(np.isfinite(annual) & (annual > 0), annual, np.nan)
    vmax = float(np.nanpercentile(a, 99))
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    m = ax.pcolormesh(lon, lat, a, transform=ccrs.PlateCarree(),
                       cmap="hot_r", norm=norm, shading="auto", alpha=0.85)
    cb = plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.04,
                       shrink=0.7, aspect=40)
    cb.set_label("GFED4.1s annual burned fraction  (mean 2001–2010)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax.set_title("Major global fire regions  —  reference map",
                  fontsize=15, weight="bold", pad=12)

    # Label each region with an arrow from the label position to the region.
    for name, rlat, rlon, llat, llon in REGIONS:
        # Marker at region
        ax.plot(rlon, rlat, marker="o", color="black",
                 markersize=4, markeredgecolor="white",
                 markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10)
        # Label box at offset position
        ax.annotate(
            name,
            xy=(rlon, rlat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(llon, llat), textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            fontsize=8.5, weight="bold", color="black",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25",
                       facecolor="white", edgecolor="0.4", linewidth=0.5,
                       alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7),
            zorder=11,
        )

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
