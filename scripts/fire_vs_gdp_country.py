"""Fire vs GDP-per-capita, one point per country (George's 07/23 assignment).

Builds the plot from his hand sketch: fire (burned fraction) on y, per-capita
wealth (GDP/person) on x. Fire high in poor countries, falling as wealth rises.

Data (all offline-cached under scratchpad/humandata):
- GFED5 burned area, 0.5deg monthly 2001-2020 (ilamb_ref_official).
- Natural Earth 50m admin_0 country polygons (ISO_A3_EH, CONTINENT).
- World Bank GDP per capita current US$ (NY.GDP.PCAP.CD) + population (SP.POP.TOTL).

Aggregates GFED5 to each country by rasterizing polygons onto the GFED5 grid,
merges World Bank wealth on ISO3, writes a country table + the scatter figure.
"""
import os
import numpy as np, pandas as pd, xarray as xr
import shapefile
import shapely
from shapely.geometry import shape, Point
from shapely import STRtree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/private/tmp/claude-501/-Volumes-RICHIE---T7-FIRE-OFFLINE/c532496d-370d-40db-969d-b2a6dd880db0/scratchpad/humandata"
GFED = "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"
Y0, Y1 = 2001, 2020                       # GFED coverage; average wealth over same span

# ---------------- GFED5 mean annual burned area per cell (km2/yr) --------------
ds = xr.open_dataset(GFED)
lat = ds.lat.values; lon = ds.lon.values
frac = np.nan_to_num(ds["burntArea"].values.astype(np.float64)) / 100.0   # % -> fraction
ds.close()
nyr = frac.shape[0] // 12
frac = frac[: nyr * 12].reshape(nyr, 12, lat.size, lon.size)
R = 6371.0                                                                # km
dlat = np.deg2rad(0.5); dlon = np.deg2rad(0.5)
cell_area = (R**2 * dlat * dlon * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))  # km2
ba_annual = (frac.sum(1) * cell_area[None]).mean(0)     # mean annual burned area, km2/yr per cell
land_area = cell_area                                    # km2 per cell (for country area)

# ---------------- rasterize country polygons onto the GFED grid ----------------
LON, LAT = np.meshgrid(lon, lat)
pts = shapely.points(LON.ravel(), LAT.ravel())
tree = STRtree(pts)
country_idx = np.full(LAT.size, -1, np.int64)

sf = shapefile.Reader(NE)
flds = [f[0] for f in sf.fields[1:]]; col = {f: i for i, f in enumerate(flds)}
recs = sf.records(); shps = sf.shapes()
meta = []                                                # (iso3, admin, continent)
for k, (rec, shp) in enumerate(zip(recs, shps)):
    geom = shapely.make_valid(shape(shp.__geo_interface__))
    cand = tree.query(geom, predicate="contains")        # cell-centers inside this country
    country_idx[cand] = k
    meta.append((rec[col["ISO_A3_EH"]], rec[col["ADMIN"]], rec[col["CONTINENT"]]))
country_idx = country_idx.reshape(LAT.shape)

# ---------------- aggregate fire + area per country ----------------------------
rows = []
ba_flat = ba_annual.ravel(); la_flat = land_area.ravel(); ci_flat = country_idx.ravel()
for k, (iso3, admin, cont) in enumerate(meta):
    m = ci_flat == k
    if not m.any():
        continue
    ba_km2 = ba_flat[m].sum()               # country burned area, km2/yr
    area_km2 = la_flat[m].sum()             # country land area on grid, km2
    rows.append(dict(iso3=iso3, admin=admin, continent=cont,
                     ba_km2yr=ba_km2, area_km2=area_km2,
                     burned_frac_pct=100.0 * ba_km2 / area_km2, ncell=int(m.sum())))
fire = pd.DataFrame(rows)

# ---------------- World Bank wealth + population (mean over GFED span) ----------
def wb(pat):
    import glob
    f = glob.glob(f"{SP}/{pat}/API_*.csv")[0]
    d = pd.read_csv(f, skiprows=4)
    yrs = [str(y) for y in range(Y0, Y1 + 1) if str(y) in d.columns]
    d["val"] = d[yrs].mean(axis=1)
    return d[["Country Code", "val"]].rename(columns={"Country Code": "iso3"})

gdp = wb("gdp").rename(columns={"val": "gdp_pcap"})
pop = wb("pop").rename(columns={"val": "pop"})
df = fire.merge(gdp, on="iso3", how="left").merge(pop, on="iso3", how="left")

# keep countries with real coverage + wealth data (drop tiny/no-data)
df = df[(df.ncell >= 3) & df.gdp_pcap.notna() & (df.gdp_pcap > 0) & (df.burned_frac_pct > 0)].copy()
df = df.sort_values("burned_frac_pct", ascending=False).reset_index(drop=True)

out_csv = "data_human/fire_vs_gdp_country.csv"
os.makedirs("data_human", exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"[table] {out_csv}  ({len(df)} countries)")
print(df[["admin", "gdp_pcap", "burned_frac_pct", "continent"]].head(12).to_string(index=False))

# ---------------- plot: fire vs GDP/capita, log-log, colored by continent ------
CPAL = {"Africa": "#E69F00", "South America": "#009E73", "North America": "#0072B2",
        "Asia": "#CC79A7", "Europe": "#56B4E9", "Oceania": "#D55E00",
        "Antarctica": "#999999", "Seven seas (open ocean)": "#999999"}
fig, ax = plt.subplots(figsize=(9, 6.2))
for cont, g in df.groupby("continent"):
    ax.scatter(g.gdp_pcap, g.burned_frac_pct, s=18 + 30 * np.log10(g["pop"].clip(1e5) / 1e5),
               c=CPAL.get(cont, "#777777"), edgecolor="k", linewidth=0.3, alpha=0.85, label=cont)

# fit a simple wealth-decline curve on logs: log10(fire) = a + b*log10(gdp)
x = np.log10(df.gdp_pcap.values); y = np.log10(df.burned_frac_pct.values)
b, a = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
xx = np.linspace(x.min(), x.max(), 100)
ax.plot(10**xx, 10**(a + b * xx), "k--", lw=1.8,
        label=f"fit slope={b:.2f}  r={r:.2f}")

# label a few extremes
for _, row in pd.concat([df.head(6), df.nlargest(4, "gdp_pcap")]).iterrows():
    ax.annotate(row.iso3, (row.gdp_pcap, row.burned_frac_pct), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("GDP per capita (current US$, mean 2001-2020)")
ax.set_ylabel("Burned fraction of country area (% / yr, GFED5)")
ax.set_title("Fire vs per-capita wealth, by country")
ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
ax.grid(True, which="both", alpha=0.15)
fig.tight_layout()
fig.savefig("fire_vs_gdp_country.png", dpi=150)
print(f"[fig] fire_vs_gdp_country.png   fit: log10(fire) = {a:.2f} + {b:.2f}*log10(gdp)  r={r:.2f}")
