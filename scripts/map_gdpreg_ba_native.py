"""Native 0.5deg render of the regional-GDP model (no coarsen, no replicate) so the
map is smooth -- and previews roughly what ED produces cell-by-cell in the coupled run.

NOTE: params were tuned at 1deg, so this native-0.5 field differs slightly from the
scored 0.6783 model; global magnitude is re-pinned to GFED5 for an honest display.
"""
import json, sys
import numpy as np, xarray as xr
sys.path.insert(0, "scripts")
from reproduce_modelC import sig, supp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature

DUMP = "global_baseline_modelC_inputs_1997-2016.nc"; FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
sl = slice(48, 240); FIRE_MAX = 5.0
FB = json.load(open("models/C/params.coupledE_gdp.json"))["params"]
REG = json.load(open("data_human/gdp_regional_gamma.json")); W0 = np.log10(REG["w0_gdp"])
lat = -89.75 + 0.5 * np.arange(360); lon = -179.75 + 0.5 * np.arange(720)

g = lambda ds, n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
ds = xr.open_dataset(DUMP)
dbar = g(ds, "D_bar"); t_air = g(ds, "T_air"); p_ann = g(ds, "P_ann"); p_month = g(ds, "P_month")
gpp = (np.clip(g(ds, "GPP_month_ntrl"), 0, None) * g(ds, "area_frac_ntrl")
     + np.clip(g(ds, "GPP_month_scnd"), 0, None) * g(ds, "area_frac_scnd")
     + np.clip(g(ds, "GPP_month_past"), 0, None) * g(ds, "area_frac_past")); ds.close()
da = xr.open_dataset(FUEL); agb = np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32)); da.close()
gpp_cell = gpp.mean(0, keepdims=True)
gdp = xr.open_dataset("data_human/coupling_inputs/gdp_pcap.nc")["gdp_pcap"].values.astype(np.float32)
gam = xr.open_dataset("data_human/coupling_inputs/gdp_gamma.nc")["gdp_gamma"].values.astype(np.float32)
M = np.clip(np.power(10.0, gam * (W0 - np.log10(np.clip(gdp, 50, None)))), 0.15, 6.0)[None]

F = {k: float(FB[k]) for k in FB}
hump = lambda x, b, dec: (1 - np.exp(-np.clip(x / max(b, 1e-9), 0, 500))) * np.exp(-np.clip(x / max(dec, 1e-9), 0, 500))
base = (sig(dbar, F["k1"], F["D_low"]) * supp(dbar, F["k2"], F["D_high"])
        * (p_ann / (p_ann + F["P_half"] + 1e-12)) * (1.0 / (1.0 + p_month / (F["pre_dampen_half"] + 1e-12)))
        * hump(F["gpp_af"] * gpp, F["gpp_b"], F["gpp_d"]) * sig(t_air, F["ign_k"], F["ign_c"]))
del dbar, t_air, p_ann, p_month, gpp
trop = (np.abs(lat) < 23.5).astype(np.float32)[None, :, None]
canopy = 1.0 / (1.0 + np.power(np.clip(agb / (F["trop_agb_crit"] + 1e-12), 0, None), F["trop_k_veg"]))
prod = base * (trop * canopy + (1.0 - trop)); del base, canopy, agb
fuel = gpp_cell / (gpp_cell + F["fuel_half"] + 1e-9)
rate = np.power(np.clip(prod, 0, None), F["fire_exp"]) * (1.0 + F["fuel_k"] * fuel); del prod

# GFED5 target (native 0.5) + land mask + magnitude re-pin
dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf = np.nan_to_num(dg["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0; dg.close()
land = np.isfinite(xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0])
R = 6371e3; area = (R**2 * np.deg2rad(0.5)**2 * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, 720))
def ann(ba): return ba.reshape(16, 12, 360, 720).sum(1).mean(0)
gf_tot = float((ann(gf) * area * land).sum())
def ba_of(s): return (1.0 - np.exp(-np.minimum(s * M * rate, FIRE_MAX) / 12.0)).astype(np.float32)
lo, hi = 0.02, 8.0
for _ in range(34):
    mid = 0.5 * (lo + hi)
    if float((ann(ba_of(mid)) * area * land).sum()) < gf_tot: lo = mid
    else: hi = mid
s = 0.5 * (lo + hi); ba = ba_of(s)
print(f"[native 0.5deg] s={s:.3f}  global {float((ann(ba)*area*land).sum())/1e10:.0f} vs GFED {gf_tot/1e10:.0f} Mha")

# maps -- house style: Robinson projection, YlOrRd (0-15% linear), pcolormesh
import cartopy.feature as cfeat
mod = np.where(land, ann(ba) * 100.0, np.nan); gfed = np.where(land, ann(gf) * 100.0, np.nan)
diff = np.where(land, (ann(ba) - ann(gf)) * 100.0, np.nan)
PLON, PLAT = lon, lat
def panel(ax, data, title, cmap, vmin, vmax, clabel, diverging=False):
    ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    kw = dict(transform=ccrs.PlateCarree(), cmap=cmap, shading="auto")
    if diverging: kw["norm"] = TwoSlopeNorm(0, vmin, vmax)
    else: kw["vmin"] = vmin; kw["vmax"] = vmax
    im = ax.pcolormesh(PLON, PLAT, data, **kw)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)

fig, axes = plt.subplots(3, 1, figsize=(10, 13), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Burned area 2001-2016 mean annual (% per year) - regional-GDP model, native 0.5deg", fontsize=12)
panel(axes[0], gfed, "GFED5 observed", "YlOrRd", 0, 15, "% per year")
panel(axes[1], mod,  "ED regional-GDP model (ILAMB 0.6783)", "YlOrRd", 0, 15, "% per year")
panel(axes[2], diff, "Model - GFED5  (red = model burns more, blue = less)", "RdBu_r", -15, 15, "% per year", diverging=True)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("ba_map_gdpreg_native.png", dpi=150, bbox_inches="tight"); print("[fig] ba_map_gdpreg_native.png")
