"""Population as a human fire factor, with the collinearity caution George flagged.

Population density is the IGNITION/access axis (complement to GDP's suppression axis).
The fire literature finds a HUMPED response: fire rises with density (more ignitions),
then falls (fragmentation, suppression, paved land), so density enters as log + log^2.

Nested regression (country level, same climate/veg from ED's dump as the GDP work):
  M_clim  : fire ~ climate + veg
  M_gdp   : + log GDPpc
  M_pop   : + log popdens + (log popdens)^2      (pop beyond climate)
  M_both  : + log GDPpc + pop hump               (THE CAUTION: pop beyond climate AND GDP)
F-test on the two pop terms added to M_gdp answers "does population add skill beyond
GDP?" Also reports the GDP-vs-popdens collinearity.
"""
import glob
import numpy as np, pandas as pd, xarray as xr
import shapefile, shapely
from shapely.geometry import shape
from shapely import STRtree
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/private/tmp/claude-501/-Volumes-RICHIE---T7-FIRE-OFFLINE/c532496d-370d-40db-969d-b2a6dd880db0/scratchpad/humandata"
GFED = "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"; FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"

# grid + cell area
ds = xr.open_dataset(GFED); lat = ds.lat.values; lon = ds.lon.values
R = 6371.0; dl = np.deg2rad(0.5)
cell_area = (R**2 * dl * dl * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
frac = np.nan_to_num(ds["burntArea"].values.astype(np.float64)) / 100.0; ds.close()
nyr = frac.shape[0] // 12
ba_cell = (frac[: nyr * 12].reshape(nyr, 12, lat.size, lon.size).sum(1) * cell_area[None]).mean(0)

# climate + veg from dump
dd = xr.open_dataset(DUMP); g = lambda n: np.nan_to_num(dd[n].values.astype(np.float64))
p_ann = g("P_ann").mean(0); t_air = g("T_air").mean(0)
gpp = (np.clip(g("GPP_month_ntrl"), 0, None) * g("area_frac_ntrl")
       + np.clip(g("GPP_month_scnd"), 0, None) * g("area_frac_scnd")
       + np.clip(g("GPP_month_past"), 0, None) * g("area_frac_past")).mean(0); dd.close()

# rasterize countries
LON, LAT = np.meshgrid(lon, lat)
tree = STRtree(shapely.points(LON.ravel(), LAT.ravel()))
cidx = np.full(LAT.size, -1, np.int64)
sf = shapefile.Reader(NE); col = {f[0]: i for i, f in enumerate(sf.fields[1:])}
meta = []
for k, (rec, shp) in enumerate(zip(sf.records(), sf.shapes())):
    cidx[tree.query(shapely.make_valid(shape(shp.__geo_interface__)), predicate="contains")] = k
    meta.append((rec[col["ISO_A3_EH"]], rec[col["ADMIN"]], rec[col["CONTINENT"]]))
cidx = cidx.reshape(LAT.shape)
aw = lambda f, m: (f.ravel()[m] * cell_area.ravel()[m]).sum() / cell_area.ravel()[m].sum()

rows = []; cif = cidx.ravel()
for k, (iso3, admin, cont) in enumerate(meta):
    m = cif == k
    if m.sum() < 3: continue
    area = cell_area.ravel()[m].sum()
    rows.append(dict(iso3=iso3, admin=admin, continent=cont, area_km2=area,
                     burned_frac_pct=100.0 * ba_cell.ravel()[m].sum() / area,
                     p_ann=aw(p_ann, m), t_air=aw(t_air, m), gpp=aw(gpp, m)))
df = pd.DataFrame(rows)

# World Bank GDP + population
def wb(pat, name):
    f = glob.glob(f"{SP}/{pat}/API_*.csv")[0]; d = pd.read_csv(f, skiprows=4)
    yrs = [str(y) for y in range(2001, 2021) if str(y) in d.columns]; d[name] = d[yrs].mean(axis=1)
    return d[["Country Code", name]].rename(columns={"Country Code": "iso3"})
df = df.merge(wb("gdp", "gdp_pcap"), on="iso3", how="left").merge(wb("pop", "pop"), on="iso3", how="left")
df["popdens"] = df["pop"] / df["area_km2"]                 # people / km2
df = df[(df.burned_frac_pct > 0) & df.gdp_pcap.notna() & (df.gdp_pcap > 0)
        & df.popdens.notna() & (df.popdens > 0) & (df.p_ann > 0) & (df.gpp > 0)].copy().reset_index(drop=True)

# OLS helpers
def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    rss = float(res @ res); r2 = 1 - rss / float(((y - y.mean()) @ (y - y.mean())))
    return beta, rss, r2, res
z = lambda v: (v - v.mean()) / v.std()
y = np.log10(df.burned_frac_pct.values); one = np.ones_like(y)
lP = np.log10(df.p_ann.values); T = df.t_air.values; lG = np.log10(df.gpp.values)
lY = np.log10(df.gdp_pcap.values); lD = np.log10(df.popdens.values)
n = len(y)

Xclim = np.column_stack([one, z(lP), z(lP)**2, z(T), z(lG), z(lG)**2])
Xgdp  = np.column_stack([Xclim, z(lY)])
Xpop  = np.column_stack([Xclim, z(lD), z(lD)**2])
Xboth = np.column_stack([Xclim, z(lY), z(lD), z(lD)**2])
_, rss_clim, r2_clim, _ = ols(Xclim, y)
bg, rss_gdp, r2_gdp, res_gdp = ols(Xgdp, y)
_, rss_pop, r2_pop, _ = ols(Xpop, y)
bb, rss_both, r2_both, _ = ols(Xboth, y)

# F-test: do the 2 pop terms add skill beyond M_gdp?
df_num = 2; df_den = n - Xboth.shape[1]
F = ((rss_gdp - rss_both) / df_num) / (rss_both / df_den); pF = stats.f.sf(F, df_num, df_den)
# and beyond climate only (M_pop vs M_clim)
F0 = ((rss_clim - rss_pop) / 2) / (rss_pop / (n - Xpop.shape[1])); pF0 = stats.f.sf(F0, 2, n - Xpop.shape[1])
collin = np.corrcoef(lY, lD)[0, 1]

print(f"countries: {n}")
print(f"[R2] climate+veg={r2_clim:.3f}  +GDP={r2_gdp:.3f}  +pop(hump)={r2_pop:.3f}  +GDP+pop={r2_both:.3f}")
print(f"[pop beyond CLIMATE]     dR2={r2_pop-r2_clim:+.3f}  F={F0:.1f}  p={pF0:.1e}")
print(f"[pop beyond CLIMATE+GDP] dR2={r2_both-r2_gdp:+.3f}  F={F:.1f}  p={pF:.1e}   <-- THE CAUTION")
print(f"[collinearity] r(log GDPpc, log popdens) = {collin:+.2f}")
print("[verdict]", "population ADDS skill beyond GDP" if pF < 0.05 else
      "population does NOT add significant skill beyond GDP")
df.to_csv("data_human/fire_vs_pop_country.csv", index=False)

# figure: raw fire vs popdens (hump) + partial (pop | climate+GDP)
CPAL = {"Africa": "#E69F00", "South America": "#009E73", "North America": "#0072B2",
        "Asia": "#CC79A7", "Europe": "#56B4E9", "Oceania": "#D55E00"}
c = df.continent.map(lambda x: CPAL.get(x, "#888888"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
a1.scatter(df.popdens, df.burned_frac_pct, c=c, edgecolor="k", lw=0.3, alpha=0.85)
xx = np.linspace(lD.min(), lD.max(), 100)
bh = np.polyfit(lD, y, 2)
a1.plot(10**xx, 10**np.polyval(bh, xx), "k--", lw=1.8)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("population density (people / km2)"); a1.set_ylabel("Burned fraction (% / yr)")
a1.set_title("Raw: fire vs population density (humped)"); a1.grid(True, which="both", alpha=0.15)

# partial: residualize fire and pop-density on climate+GDP
_, _, _, res_fire_g = ols(Xgdp, y)
_, _, _, res_lD_g = ols(Xgdp, lD)
pr = np.polyfit(res_lD_g, res_fire_g, 1); prr = np.corrcoef(res_lD_g, res_fire_g)[0, 1]
a2.scatter(res_lD_g, res_fire_g, c=c, edgecolor="k", lw=0.3, alpha=0.85)
xr = np.linspace(res_lD_g.min(), res_lD_g.max(), 50)
a2.plot(xr, np.polyval(pr, xr), "k--", lw=1.8)
a2.axhline(0, color="k", lw=0.5, alpha=0.4); a2.axvline(0, color="k", lw=0.5, alpha=0.4)
a2.set_xlabel("population density  (climate + GDP removed, residual)")
a2.set_ylabel("Burned fraction  (climate + GDP removed, residual, log10)")
a2.set_title(f"Pop beyond climate+GDP: partial r={prr:+.2f}, F-test p={pF:.0e}")
a2.grid(True, alpha=0.15)
for cont, cc in CPAL.items(): a2.scatter([], [], c=cc, edgecolor="k", lw=0.3, label=cont)
a2.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig("fire_vs_pop_partial.png", dpi=150)
print("[fig] fire_vs_pop_partial.png")
