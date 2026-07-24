"""Fire vs GDP-per-capita, CLIMATE-CONTROLLED (George's follow-up concern).

The raw fire-vs-wealth slope conflates poverty with climate, because poor
countries are disproportionately tropical savanna. This isolates the
socioeconomic signal: predict country fire from climate + vegetation only,
then test whether residual fire still falls with wealth.

Natural-fire predictors (country area-weighted means, same 0.5deg grid as GFED,
all from ED's own dump so it stays consistent with the fire model):
  precip P_ann (hump: log P and log P^2), temperature T_air, productivity/fuel GPP.
Then OLS  log10(fire) ~ climate[+veg]  -> residual, and
          log10(fire) ~ climate[+veg] + log10(GDPpc)  -> partial wealth coefficient.

Outputs the partial regression (added-variable) figure + a stats printout.
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
FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"

# ---------------- grid + cell area (shared 0.5deg) -----------------------------
ds = xr.open_dataset(GFED); lat = ds.lat.values; lon = ds.lon.values
R = 6371.0; dl = np.deg2rad(0.5)
cell_area = (R**2 * dl * dl * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))  # km2
frac = np.nan_to_num(ds["burntArea"].values.astype(np.float64)) / 100.0
ds.close()
nyr = frac.shape[0] // 12
ba_cell = (frac[: nyr * 12].reshape(nyr, 12, lat.size, lon.size).sum(1) * cell_area[None]).mean(0)  # km2/yr

# ---------------- climate + vegetation per cell (from the dump) ----------------
dd = xr.open_dataset(DUMP)
g = lambda n: np.nan_to_num(dd[n].values.astype(np.float64))
p_ann = g("P_ann").mean(0); t_air = g("T_air").mean(0)
gpp = (np.clip(g("GPP_month_ntrl"), 0, None) * g("area_frac_ntrl")
       + np.clip(g("GPP_month_scnd"), 0, None) * g("area_frac_scnd")
       + np.clip(g("GPP_month_past"), 0, None) * g("area_frac_past")).mean(0)
dd.close()

# ---------------- rasterize countries onto the grid ----------------------------
LON, LAT = np.meshgrid(lon, lat)
tree = STRtree(shapely.points(LON.ravel(), LAT.ravel()))
cidx = np.full(LAT.size, -1, np.int64)
sf = shapefile.Reader(NE); col = {f[0]: i for i, f in enumerate(sf.fields[1:])}
meta = []
for k, (rec, shp) in enumerate(zip(sf.records(), sf.shapes())):
    geom = shapely.make_valid(shape(shp.__geo_interface__))
    cidx[tree.query(geom, predicate="contains")] = k
    meta.append((rec[col["ISO_A3_EH"]], rec[col["ADMIN"]], rec[col["CONTINENT"]]))
cidx = cidx.reshape(LAT.shape)

# ---------------- area-weighted country aggregation ----------------------------
aw = lambda field, m: (field.ravel()[m] * cell_area.ravel()[m]).sum() / cell_area.ravel()[m].sum()
rows = []
cif = cidx.ravel()
for k, (iso3, admin, cont) in enumerate(meta):
    m = cif == k
    if m.sum() < 3:
        continue
    area = cell_area.ravel()[m].sum()
    rows.append(dict(iso3=iso3, admin=admin, continent=cont, ncell=int(m.sum()),
                     burned_frac_pct=100.0 * ba_cell.ravel()[m].sum() / area,
                     p_ann=aw(p_ann, m), t_air=aw(t_air, m), gpp=aw(gpp, m)))
df = pd.DataFrame(rows)

# ---------------- merge World Bank wealth --------------------------------------
def wb(pat, name):
    f = glob.glob(f"{SP}/{pat}/API_*.csv")[0]
    d = pd.read_csv(f, skiprows=4)
    yrs = [str(y) for y in range(2001, 2021) if str(y) in d.columns]
    d[name] = d[yrs].mean(axis=1)
    return d[["Country Code", name]].rename(columns={"Country Code": "iso3"})
df = df.merge(wb("gdp", "gdp_pcap"), on="iso3", how="left")

df = df[(df.burned_frac_pct > 0) & df.gdp_pcap.notna() & (df.gdp_pcap > 0)
        & (df.p_ann > 0) & (df.gpp > 0)].copy().reset_index(drop=True)

# ---------------- OLS with SEs (numpy) -----------------------------------------
def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; n, k = X.shape
    sig2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2, resid

y = np.log10(df.burned_frac_pct.values)
lP = np.log10(df.p_ann.values); T = df.t_air.values; lG = np.log10(df.gpp.values); lY = np.log10(df.gdp_pcap.values)
z = lambda v: (v - v.mean()) / v.std()
one = np.ones_like(y)

# climate + vegetation only (natural fire expectation); precip enters as a hump
Xc = np.column_stack([one, z(lP), z(lP)**2, z(T), z(lG), z(lG)**2])
bc, sec, r2c, res_fire = ols(Xc, y)                       # residual = climate-unexplained fire
_, _, _, res_gdp = ols(Xc, lY)                            # wealth orthogonalized to climate

# full model: partial coefficient on wealth, controlling climate+veg
Xf = np.column_stack([Xc, z(lY)])
bf, sef, r2f, _ = ols(Xf, y)
tval = bf[-1] / sef[-1]; pval = 2 * stats.t.sf(abs(tval), len(y) - Xf.shape[1])
# convert standardized wealth coef to "per 10x GDP"
slope_per_decade = bf[-1] / lY.std()                      # change in log10(fire) per +1 in log10(GDP)
raw_slope, raw_int = np.polyfit(lY, y, 1); raw_r = np.corrcoef(lY, y)[0, 1]
partial_r = np.corrcoef(res_gdp, res_fire)[0, 1]

print(f"countries: {len(df)}")
print(f"[raw]      log10(fire) ~ log10(GDP):  slope={raw_slope:+.2f}/decade  r={raw_r:+.2f}")
print(f"[climate]  fire ~ climate+veg only:   R2={r2c:.2f}  (natural fire explained)")
print(f"[full]     + log10(GDP):              R2={r2f:.2f}")
print(f"[WEALTH]   partial slope={slope_per_decade:+.2f}/decade  partial r={partial_r:+.2f}"
      f"  t={tval:+.1f}  p={pval:.1e}")
print(f"           wealth kept {100*slope_per_decade/raw_slope:.0f}% of the raw slope after removing climate")
df.assign(fire_resid=res_fire, gdp_resid=res_gdp).to_csv("data_human/fire_vs_gdp_partial.csv", index=False)

# ---------------- figure: raw vs climate-controlled -----------------------------
CPAL = {"Africa": "#E69F00", "South America": "#009E73", "North America": "#0072B2",
        "Asia": "#CC79A7", "Europe": "#56B4E9", "Oceania": "#D55E00"}
c = df.continent.map(lambda x: CPAL.get(x, "#888888"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

a1.scatter(df.gdp_pcap, df.burned_frac_pct, c=c, edgecolor="k", lw=0.3, alpha=0.85)
xx = np.linspace(lY.min(), lY.max(), 50)
a1.plot(10**xx, 10**(raw_int + raw_slope * xx), "k--", lw=1.8)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("GDP per capita (US$)"); a1.set_ylabel("Burned fraction (% / yr)")
a1.set_title(f"Raw: slope={raw_slope:+.2f}/decade, r={raw_r:+.2f}\n(mixes poverty with savanna climate)")
a1.grid(True, which="both", alpha=0.15)

a2.scatter(res_gdp, res_fire, c=c, edgecolor="k", lw=0.3, alpha=0.85)
xr = np.linspace(res_gdp.min(), res_gdp.max(), 50)
pr_slope = np.polyfit(res_gdp, res_fire, 1)[0]
a2.plot(xr, np.polyval(np.polyfit(res_gdp, res_fire, 1), xr), "k--", lw=1.8)
a2.axhline(0, color="k", lw=0.5, alpha=0.4); a2.axvline(0, color="k", lw=0.5, alpha=0.4)
a2.set_xlabel("GDP per capita  (climate removed, residual)")
a2.set_ylabel("Burned fraction  (climate removed, residual, log10)")
a2.set_title(f"Climate-controlled: partial slope={slope_per_decade:+.2f}/decade\n"
             f"partial r={partial_r:+.2f}, p={pval:.0e}  -> wealth signal survives")
a2.grid(True, alpha=0.15)
for cont, cc in CPAL.items():
    a2.scatter([], [], c=cc, edgecolor="k", lw=0.3, label=cont)
a2.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig("fire_vs_gdp_partial.png", dpi=150)
print("[fig] fire_vs_gdp_partial.png")
