"""Figures for the GDP -> Model F advisor deck (George's human-factor assignment).

Four beats, four figures, all from cached data (no model re-run):
  beat1_fire_vs_gdp.png  raw fire-vs-GDP + climate-controlled partial (it is real)
  beat2_controls.png     population + land use add nothing beyond GDP (right factor)
  beat3_gamma_map.png    biome-specific human term, smooth (hot Africa, cold Asia)
  beat4_regional.png     Model F matches GFED5 region by region (the payoff)

Prints every stat that lands in the speaker notes so the numbers are verified, not
recalled. Run in edfire: `python scripts/make_advisor_gdp_deck_figs.py`.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs, cartopy.feature as cfeat

REPO = Path("."); OUT = REPO / "figs_gdp_advisor"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})
INK = "#233"; FIRE = "#c1442e"; GDP = "#2a6f97"; GREY = "#9aa3ab"; GOOD = "#3d7a4e"

def ols(y, X):
    """OLS with intercept added; returns beta, per-coef p-values, R^2."""
    Xi = np.column_stack([np.ones(len(y)), X]); n, k = Xi.shape
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    resid = y - Xi @ beta; sse = resid @ resid
    sigma2 = sse / (n - k); XtXi = np.linalg.inv(Xi.T @ Xi)
    se = np.sqrt(np.diag(sigma2 * XtXi)); t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - sse / (((y - y.mean()) ** 2).sum())
    return beta, p, r2

# ============================ BEAT 1 — fire vs GDP ============================
raw = pd.read_csv("data_human/fire_vs_gdp_country.csv")
raw = raw[(raw.burned_frac_pct > 0) & (raw.gdp_pcap > 0)].copy()
lx = np.log10(raw.gdp_pcap.values); ly = np.log10(raw.burned_frac_pct.values)
sl_raw, ic_raw, r_raw, p_raw, _ = stats.linregress(lx, ly)

par = pd.read_csv("data_human/fire_vs_gdp_partial.csv")
gx = par.gdp_resid.values; fy = par.fire_resid.values
sl_par, ic_par, r_par, p_par, _ = stats.linregress(gx, fy)

print(f"[beat1] RAW      n={len(raw):3d}  slope={sl_raw:+.2f}/decade  r={r_raw:+.2f}  p={p_raw:.1e}")
print(f"[beat1] PARTIAL  n={len(par):3d}  slope={sl_par:+.2f}/decade  r={r_par:+.2f}  p={p_par:.1e}")

# named exemplars: poor African-savanna (high fire) vs wealthy (low fire).
# few + hand-placed, because rich countries all cluster near $45k / ~0% fire.
# iso -> (label, dx, dy, ha) offset in points to fan the labels apart.
EX_POOR = {"ZMB": ("Zambia", -6, 9, "right"), "SSD": ("S. Sudan", 12, 6, "left"),
           "AGO": ("Angola", 8, 3, "left"), "COD": ("DR Congo", -7, 8, "right")}
EX_RICH = {"USA": ("USA", 9, 4, "left"), "DEU": ("Germany", 9, -3, "left"),
           "NOR": ("Norway", 9, 1, "left")}
# partial panel has different near-coincident pairs -> per-iso offset overrides
OV_PAR = {"NOR": (-8, -12, "right"), "DEU": (10, 3, "left"),
          "SSD": (12, -10, "left"), "AGO": (2, 9, "left")}
rawi = raw.set_index("iso3"); pari = par.set_index("iso3")
def label_pts(ax, getxy, ex, color, ov=None):
    ov = ov or {}
    for iso, (nm, dx, dy, ha) in ex.items():
        if iso not in getxy.index: continue
        if iso in ov: dx, dy, ha = ov[iso]
        x, y = getxy.at[iso, "X"], getxy.at[iso, "Y"]
        ax.scatter([x], [y], s=46, c=[color], edgecolor="k", linewidth=.6, zorder=5)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    ha=ha, color=INK, weight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none", alpha=.8))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
# --- raw panel ---
a1.scatter(10 ** lx, raw.burned_frac_pct, s=16, c=GREY, alpha=.35, edgecolor="none")
xs = np.linspace(lx.min(), lx.max(), 50)
a1.plot(10 ** xs, 10 ** (ic_raw + sl_raw * xs), color=INK, lw=2.2)
a1.set_xscale("log"); a1.set_yscale("log")
raw_xy = rawi.assign(X=rawi.gdp_pcap, Y=rawi.burned_frac_pct)
label_pts(a1, raw_xy, EX_POOR, FIRE); label_pts(a1, raw_xy, EX_RICH, GDP)
a1.set_xlabel("GDP per capita (US$, log)"); a1.set_ylabel("burned area (% of country, log)")
a1.set_title("Fire falls as countries get richer")
a1.text(.04, .06, f"slope {sl_raw:+.2f}/decade\nr = {r_raw:+.2f}   ({len(raw)} countries)",
        transform=a1.transAxes, fontsize=11, va="bottom",
        bbox=dict(boxstyle="round,pad=.4", fc="white", ec=GREY))
# --- partial panel (same named countries, climate removed) ---
a2.scatter(gx, fy, s=16, c=GREY, alpha=.35, edgecolor="none")
xs = np.linspace(gx.min(), gx.max(), 50)
a2.plot(xs, ic_par + sl_par * xs, color=INK, lw=2.2)
a2.axhline(0, color=GREY, lw=.8, ls=":"); a2.axvline(0, color=GREY, lw=.8, ls=":")
par_xy = pari.assign(X=pari.gdp_resid, Y=pari.fire_resid)
label_pts(a2, par_xy, EX_POOR, FIRE, OV_PAR); label_pts(a2, par_xy, EX_RICH, GDP, OV_PAR)
a2.set_xlabel("GDP per capita  (climate removed)"); a2.set_ylabel("burned area  (climate removed)")
a2.set_title("Still true after removing climate")
a2.text(.04, .06, f"slope {sl_par:+.2f}/decade\nr = {r_par:+.2f}   p = {p_par:.0e}",
        transform=a2.transAxes, fontsize=11, va="bottom",
        bbox=dict(boxstyle="round,pad=.4", fc="white", ec=GREY))
# color-key for the exemplars
a1.scatter([], [], s=46, c=[FIRE], edgecolor="k", linewidth=.6, label="poorer, high fire")
a1.scatter([], [], s=46, c=[GDP], edgecolor="k", linewidth=.6, label="wealthy, low fire")
a1.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "beat1_fire_vs_gdp.png", dpi=170); plt.close(fig)

# ============================ BEAT 2 — negative controls =====================
# Reproduce the CANONICAL nested-model F-tests (fire_vs_{pop,landuse}_partial.py):
# z-scored climate+veg with a GPP hump; population as a hump (2 df), land use as a
# {pasture, pasture^2, secondary} block (3 df). Answers "beyond climate + GDP".
def z(v): return (v - v.mean()) / (v.std() + 1e-12)
def rss_of(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; return float(res @ res), res
def ftest(rss_r, rss_f, dfn, n, kf):
    F = ((rss_r - rss_f) / dfn) / (rss_f / (n - kf)); return F, float(stats.f.sf(F, dfn, n - kf))
def resid(X, y): return rss_of(X, y)[1]

pop = pd.read_csv("data_human/fire_vs_pop_country.csv")
yP = np.log10(pop.burned_frac_pct.values); one = np.ones_like(yP)
lP = np.log10(pop.p_ann.values); lG = np.log10(pop.gpp.values); lY = np.log10(pop.gdp_pcap.values)
lD = np.log10(pop.popdens.values); nP = len(yP)
XclimP = np.column_stack([one, z(lP), z(lP) ** 2, z(pop.t_air.values), z(lG), z(lG) ** 2])
XgdpP = np.column_stack([XclimP, z(lY)])
XbothP = np.column_stack([XclimP, z(lY), z(lD), z(lD) ** 2])
rss_g, _ = rss_of(XgdpP, yP); rss_b, _ = rss_of(XbothP, yP)
Fp, p_pop = ftest(rss_g, rss_b, 2, nP, XbothP.shape[1])
px_pop = resid(XgdpP, lD); fy_pop = resid(XgdpP, yP)      # added-variable coords (pop | climate+GDP)

lu = pd.read_csv("data_human/fire_vs_landuse_country.csv")
yL = np.log10(lu.burned_frac_pct.values); one = np.ones_like(yL)
lP = np.log10(lu.p_ann.values); lG = np.log10(lu.gpp.values); lY = np.log10(lu.gdp_pcap.values)
past = lu.f_past.values; scnd = lu.f_scnd.values; nL = len(yL)
XclimL = np.column_stack([one, z(lP), z(lP) ** 2, z(lu.t_air.values), z(lG), z(lG) ** 2])
LU = np.column_stack([z(past), z(past) ** 2, z(scnd)])
XgdpL = np.column_stack([XclimL, z(lY)])
XbothL = np.column_stack([XclimL, z(lY), LU])
rss_g, _ = rss_of(XgdpL, yL); rss_b, _ = rss_of(XbothL, yL)
Fl, p_lu = ftest(rss_g, rss_b, 3, nL, XbothL.shape[1])
lx_lu = resid(XgdpL, past); fy_lu = resid(XgdpL, yL)      # added-variable coords (pasture | climate+GDP)

print(f"[beat2] POPULATION beyond climate+GDP : F={Fp:.1f}  p={p_pop:.2f}  (no independent signal)")
print(f"[beat2] LAND USE   beyond climate+GDP : F={Fl:.1f}  p={p_lu:.2f}  (redundant with GDP)")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
panels = [(a1, px_pop, fy_pop, Fp, p_pop, "Population density", "no signal on its own"),
          (a2, lx_lu, fy_lu, Fl, p_lu, "Land use (pasture)", "redundant with GDP")]
for ax, xr_, yr_, Fv, pval, name, verdict in panels:
    ax.scatter(xr_, yr_, s=18, c=GREY, alpha=.55, edgecolor="none")
    s, i, r, p, _ = stats.linregress(xr_, yr_)
    xs = np.linspace(xr_.min(), xr_.max(), 50); ax.plot(xs, i + s * xs, color=INK, lw=2, ls="--")
    ax.axhline(0, color=GREY, lw=.8, ls=":"); ax.axvline(0, color=GREY, lw=.8, ls=":")
    ax.set_xlabel(f"{name}  (climate + GDP removed)"); ax.set_ylabel("burned area  (climate + GDP removed)")
    ax.set_title(f"{name}: {verdict}")
    ax.text(.04, .90, f"F = {Fv:.1f}   p = {pval:.2f}\n(not significant beyond GDP)", transform=ax.transAxes,
            fontsize=11, va="top", bbox=dict(boxstyle="round,pad=.4", fc="#f3f4f6", ec=GREY))
fig.tight_layout(); fig.savefig(OUT / "beat2_controls.png", dpi=170); plt.close(fig)

# ============================ BEAT 3 — biome gamma map =======================
gj = json.load(open("data_human/gdp_regional_gamma.json"))
gvec = gj["per_region_gamma"]; SIGMA = gj["sigma"]
REGION_BOX = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
              "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
              "Europe": (-12, 40, 36, 72), "N.America": (-168, -52, 14, 74),
              "Australia": (112, 154, -44, -10)}
lat1 = -89.5 + np.arange(180); lon1 = -179.5 + np.arange(360)
LON, LAT = np.meshgrid(lon1, lat1)
region_of = np.full((180, 360), "fb", dtype=object); assigned = np.zeros((180, 360), bool)
for r, b in REGION_BOX.items():
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]) & ~assigned
    region_of[box] = r; assigned |= box
f = np.zeros((180, 360))
for r in list(REGION_BOX) + ["fb"]:
    f[region_of == r] = gvec[r]
gfield = gaussian_filter(f, SIGMA, mode="nearest")
land = np.load("data_human/gdp_term_fields.npz")["land"]
gfield_m = np.where(land, gfield, np.nan)
print("[beat3] gamma by region: " + "  ".join(f"{r}={gvec[r]:.2f}" for r in REGION_BOX))

cmap = LinearSegmentedColormap.from_list("gamma", ["#f7f7f7", "#fdd49e", "#fc8d59", "#b30000"])
fig = plt.figure(figsize=(13, 6.2)); ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-165, 180, -58, 82], ccrs.PlateCarree())
im = ax.pcolormesh(lon1, lat1, gfield_m, cmap=cmap, vmin=0, vmax=1.6,
                   transform=ccrs.PlateCarree(), shading="auto")
ax.add_feature(cfeat.COASTLINE, lw=.5, edgecolor="#555")
ax.add_feature(cfeat.BORDERS, lw=.25, edgecolor="#999")
cb = fig.colorbar(im, ax=ax, shrink=.7, pad=.02); cb.set_label("human-term strength (gamma)")
labels = {"Africa": (18, -2, "Africa 1.6\n(strong)"), "SEAsia": (100, 8, "Asia 0.1\n(near off)"),
          "Australia": (134, -26, "Australia 0"), "S.America": (-60, -12, "S. America 0.3"),
          "Boreal": (95, 63, "Boreal 0.5"), "N.America": (-105, 42, "N. America 0.6")}
for r, (x, y, t) in labels.items():
    ax.text(x, y, t, transform=ccrs.PlateCarree(), fontsize=10, ha="center", va="center", weight="bold",
            color=INK, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=GREY, alpha=.85))
ax.set_title("The human term is biome-specific and smooth (no hard borders)", fontsize=15)
fig.tight_layout(); fig.savefig(OUT / "beat3_gamma_map.png", dpi=170); plt.close(fig)

# ============================ BEAT 4 — regional fidelity =====================
def ann_frac(path, is_pct):
    d = xr.open_dataset(path); a = np.nan_to_num(d["burntArea"].values.astype(float)); d.close()
    if is_pct: a = a / 100.0
    return a[:192].reshape(16, 12, 360, 720).sum(1).mean(0)          # annual burned fraction, 0.5deg
gf = ann_frac("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc", True)
F = ann_frac("ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/burntArea.nc", False)
lat = -89.75 + 0.5 * np.arange(360); lon = -179.75 + 0.5 * np.arange(720)
LON5, LAT5 = np.meshgrid(lon, lat)
R = 6371e3; area = (R ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, 720))
def mha(a, b):
    m = (LON5 >= b[0]) & (LON5 <= b[1]) & (LAT5 >= b[2]) & (LAT5 <= b[3])
    return float((a * area * m).sum()) / 1e10                        # m^2 -> Mha
order = ["Africa", "S.America", "SEAsia", "N.America", "Boreal", "Australia", "Europe"]
gvals = [mha(gf, REGION_BOX[r]) for r in order]; fvals = [mha(F, REGION_BOX[r]) for r in order]
print("[beat4] region        GFED5   ModelF   ratio")
for r, g, v in zip(order, gvals, fvals):
    print(f"[beat4] {r:10s} {g:7.1f} {v:7.1f}   {v/g:4.2f}x")
print(f"[beat4] GLOBAL     {sum(gvals):7.1f} {sum(fvals):7.1f}   {sum(fvals)/sum(gvals):4.2f}x")

fig, ax = plt.subplots(figsize=(12, 5.6))
x = np.arange(len(order)); w = .38
ax.bar(x - w / 2, gvals, w, label="GFED5 (satellite)", color=GREY)
ax.bar(x + w / 2, fvals, w, label="Model F", color=FIRE)
ax.set_xticks(x); ax.set_xticklabels([r.replace("S.America", "S. America") for r in order])
ax.set_ylabel("annual burned area (Mha)")
ax.set_title("Model F reproduces the regional pattern, not just the global total")
ax.legend(frameon=False, loc="upper right")
ax.text(.99, .82, f"global  {sum(fvals):.0f} vs {sum(gvals):.0f} Mha  ({sum(fvals)/sum(gvals):.2f}x)",
        transform=ax.transAxes, ha="right", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(OUT / "beat4_regional.png", dpi=170); plt.close(fig)

print("\nwrote 4 figures to", OUT.resolve())
