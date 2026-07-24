"""Land use as a human fire factor, with the GDP-control caution.

Land use is ED's LUH2 state, already in the dump as area_frac_{ntrl,scnd,past}
(natural / secondary / pasture), forward-runnable => coupling-legal. Pasture and
managed (secondary) land change fire directly (agricultural burning vs fragmentation),
so the land-use block is {pasture, pasture^2, secondary} (allows a non-monotonic
agricultural-burning response).

Nested regression (country level, same climate/veg from the dump as GDP/pop):
  M_clim  : fire ~ climate + veg
  M_gdp   : + log GDPpc
  M_lu    : + land-use block                    (beyond climate)
  M_both  : + log GDPpc + land-use block         (THE CAUTION: beyond climate AND GDP)
F-test on the land-use block answers "does land use add skill beyond GDP?".
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
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"; sl = slice(48, 240)

ds = xr.open_dataset(GFED); lat = ds.lat.values; lon = ds.lon.values
Rk = 6371.0; dl = np.deg2rad(0.5)
cell_area = (Rk**2 * dl * dl * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
frac = np.nan_to_num(ds["burntArea"].values.astype(np.float64)) / 100.0; ds.close()
nyr = frac.shape[0] // 12
ba_cell = (frac[: nyr * 12].reshape(nyr, 12, lat.size, lon.size).sum(1) * cell_area[None]).mean(0)

dd = xr.open_dataset(DUMP); g = lambda n: np.nan_to_num(dd[n].isel(time=sl).values.astype(np.float64))
p_ann = g("P_ann").mean(0); t_air = g("T_air").mean(0)
gpp = (np.clip(g("GPP_month_ntrl"), 0, None) * g("area_frac_ntrl")
       + np.clip(g("GPP_month_scnd"), 0, None) * g("area_frac_scnd")
       + np.clip(g("GPP_month_past"), 0, None) * g("area_frac_past")).mean(0)
f_ntrl = g("area_frac_ntrl").mean(0); f_scnd = g("area_frac_scnd").mean(0); f_past = g("area_frac_past").mean(0)
dd.close()

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
    rows.append(dict(iso3=iso3, admin=admin, continent=cont,
                     burned_frac_pct=100.0 * ba_cell.ravel()[m].sum() / area,
                     p_ann=aw(p_ann, m), t_air=aw(t_air, m), gpp=aw(gpp, m),
                     f_past=aw(f_past, m), f_scnd=aw(f_scnd, m), f_human=aw(f_scnd, m) + aw(f_past, m)))
df = pd.DataFrame(rows)

def wb(pat, name):
    f = glob.glob(f"{SP}/{pat}/API_*.csv")[0]; d = pd.read_csv(f, skiprows=4)
    yrs = [str(y) for y in range(2001, 2021) if str(y) in d.columns]; d[name] = d[yrs].mean(axis=1)
    return d[["Country Code", name]].rename(columns={"Country Code": "iso3"})
df = df.merge(wb("gdp", "gdp_pcap"), on="iso3", how="left")
df = df[(df.burned_frac_pct > 0) & df.gdp_pcap.notna() & (df.gdp_pcap > 0)
        & (df.p_ann > 0) & (df.gpp > 0)].copy().reset_index(drop=True)

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    rss = float(res @ res); r2 = 1 - rss / float(((y - y.mean()) @ (y - y.mean())))
    return beta, rss, r2, res
z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
y = np.log10(df.burned_frac_pct.values); one = np.ones_like(y)
lP = np.log10(df.p_ann.values); T = df.t_air.values; lG = np.log10(df.gpp.values); lY = np.log10(df.gdp_pcap.values)
past = df.f_past.values; scnd = df.f_scnd.values; n = len(y)

Xclim = np.column_stack([one, z(lP), z(lP)**2, z(T), z(lG), z(lG)**2])
LU = np.column_stack([z(past), z(past)**2, z(scnd)])       # land-use block (3 df)
Xgdp = np.column_stack([Xclim, z(lY)])
Xlu = np.column_stack([Xclim, LU])
Xboth = np.column_stack([Xclim, z(lY), LU])
_, rss_clim, r2_clim, _ = ols(Xclim, y)
_, rss_gdp, r2_gdp, res_gdp = ols(Xgdp, y)
_, rss_lu, r2_lu, _ = ols(Xlu, y)
_, rss_both, r2_both, _ = ols(Xboth, y)

k_lu = LU.shape[1]
F0 = ((rss_clim - rss_lu) / k_lu) / (rss_lu / (n - Xlu.shape[1])); pF0 = stats.f.sf(F0, k_lu, n - Xlu.shape[1])
F = ((rss_gdp - rss_both) / k_lu) / (rss_both / (n - Xboth.shape[1])); pF = stats.f.sf(F, k_lu, n - Xboth.shape[1])
collin = np.corrcoef(lY, past)[0, 1]

print(f"countries: {n}")
print(f"[R2] climate+veg={r2_clim:.3f}  +GDP={r2_gdp:.3f}  +landuse={r2_lu:.3f}  +GDP+landuse={r2_both:.3f}")
print(f"[landuse beyond CLIMATE]     dR2={r2_lu-r2_clim:+.3f}  F={F0:.1f}  p={pF0:.1e}")
print(f"[landuse beyond CLIMATE+GDP] dR2={r2_both-r2_gdp:+.3f}  F={F:.1f}  p={pF:.1e}   <-- THE CAUTION")
print(f"[collinearity] r(log GDPpc, pasture frac) = {collin:+.2f}")
print("[verdict]", "land use ADDS skill beyond GDP" if pF < 0.05 else "land use does NOT add significant skill beyond GDP")
df.to_csv("data_human/fire_vs_landuse_country.csv", index=False)

CPAL = {"Africa": "#E69F00", "South America": "#009E73", "North America": "#0072B2",
        "Asia": "#CC79A7", "Europe": "#56B4E9", "Oceania": "#D55E00"}
c = df.continent.map(lambda x: CPAL.get(x, "#888888"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
a1.scatter(df.f_past, df.burned_frac_pct, c=c, edgecolor="k", lw=0.3, alpha=0.85)
a1.set_yscale("log"); a1.set_xlabel("pasture fraction of country"); a1.set_ylabel("Burned fraction (% / yr)")
a1.set_title("Raw: fire vs pasture fraction"); a1.grid(True, which="both", alpha=0.15)
_, _, _, res_fire_g = ols(Xgdp, y); _, _, _, res_past_g = ols(Xgdp, past)
pr = np.polyfit(res_past_g, res_fire_g, 1); prr = np.corrcoef(res_past_g, res_fire_g)[0, 1]
a2.scatter(res_past_g, res_fire_g, c=c, edgecolor="k", lw=0.3, alpha=0.85)
xr_ = np.linspace(res_past_g.min(), res_past_g.max(), 50); a2.plot(xr_, np.polyval(pr, xr_), "k--", lw=1.8)
a2.axhline(0, color="k", lw=0.5, alpha=0.4); a2.axvline(0, color="k", lw=0.5, alpha=0.4)
a2.set_xlabel("pasture fraction (climate + GDP removed)"); a2.set_ylabel("Burned fraction (climate + GDP removed, log10)")
a2.set_title(f"Land use beyond climate+GDP: block F-test p={pF:.0e}")
a2.grid(True, alpha=0.15)
for cont, cc in CPAL.items(): a2.scatter([], [], c=cc, edgecolor="k", lw=0.3, label=cont)
a2.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig("fire_vs_landuse_partial.png", dpi=150); print("[fig] fire_vs_landuse_partial.png")
