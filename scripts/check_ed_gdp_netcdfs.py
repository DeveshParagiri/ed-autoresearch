"""Validate the ED-ready NetCDFs: apply gdp_pcap.nc + gdp_gamma.nc (read from disk,
coarsened to the 1deg the model was fit on) to the base model and confirm the BA
reproduces the 0.6783 model. Proves the NetCDF export faithfully encodes the fit.
"""
import json, os, sys
from pathlib import Path
import numpy as np, xarray as xr, cftime
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO = Path("."); YEARS = list(range(2001, 2017)); FIRE_MAX = 5.0; sl = slice(48, 240)
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"; FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
FB = json.load(open(REPO / "models/C/params.coupledE_gdp.json"))["params"]
REG = json.load(open(REPO / "data_human/gdp_regional_gamma.json"))
W0 = np.log10(REG["w0_gdp"]); S = REG["s"]; MLO, MHI = 0.15, 6.0
KEYS = ["k1","D_low","k2","D_high","fire_exp","P_half","pre_dampen_half","gpp_af","gpp_b",
        "gpp_d","ign_k","ign_c","trop_agb_crit","trop_k_veg","fuel_k","fuel_half"]

ds = xr.open_dataset(DUMP); grab = lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
d = {"dbar": coarsen(grab("D_bar")), "t_air": coarsen(grab("T_air")),
     "p_ann": coarsen(grab("P_ann")), "p_month": coarsen(grab("P_month"))}
d["gpp_monthly"] = coarsen((np.clip(grab("GPP_month_ntrl"),0,None)*grab("area_frac_ntrl")
                          + np.clip(grab("GPP_month_scnd"),0,None)*grab("area_frac_scnd")
                          + np.clip(grab("GPP_month_past"),0,None)*grab("area_frac_past")).astype(np.float32)); ds.close()
da = xr.open_dataset(FUEL); d["agb"] = coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()
gpp_cell = d["gpp_monthly"].mean(0, keepdims=True); lat1 = -89.5 + np.arange(180)

# read the NetCDFs ED will use, coarsen 0.5deg -> 1deg (same reduction as the drivers)
gdp05 = xr.open_dataset("data_human/coupling_inputs/gdp_pcap.nc")["gdp_pcap"].values.astype(np.float64)
gam05 = xr.open_dataset("data_human/coupling_inputs/gdp_gamma.nc")["gdp_gamma"].values.astype(np.float64)
gdp1 = gdp05.reshape(180,2,360,2).mean((1,3)); gam1 = gam05.reshape(180,2,360,2).mean((1,3))
w = np.log10(np.clip(gdp1, 50, None))
M = np.clip(np.power(10.0, gam1*(W0 - w)), MLO, MHI)[None]

def hump_f(x,b,dec):
    b=np.maximum(b,1e-9); dec=np.maximum(dec,1e-9)
    return (1.0-np.exp(-np.clip(x/b,0,500)))*np.exp(-np.clip(x/dec,0,500))
F={k:float(FB[k]) for k in KEYS}
onset=sig(d["dbar"],F["k1"],F["D_low"]); supr=supp(d["dbar"],F["k2"],F["D_high"])
p_flr=d["p_ann"]/(d["p_ann"]+F["P_half"]+1e-12); p_dmp=1.0/(1.0+d["p_month"]/(F["pre_dampen_half"]+1e-12))
gpp_m=hump_f(F["gpp_af"]*d["gpp_monthly"],F["gpp_b"],F["gpp_d"]); ign=sig(d["t_air"],F["ign_k"],F["ign_c"])
base=onset*supr*p_flr*p_dmp*gpp_m*ign
trop=(np.abs(lat1)<23.5).astype(float)[None,:,None]
canopy=1.0/(1.0+np.power(np.clip(d["agb"]/(F["trop_agb_crit"]+1e-12),0,None),F["trop_k_veg"]))
prod=base*(trop*canopy+(1.0-trop))
fuel=gpp_cell/(gpp_cell+F["fuel_half"]+1e-9)
rate=np.power(np.clip(prod,0,None),F["fire_exp"])*(1.0+F["fuel_k"]*fuel)
ba=(1.0-np.exp(-np.minimum(S*M*rate,FIRE_MAX)/12.0)).astype(np.float32)

can=xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land=np.isfinite(can.values[0]).reshape(180,2,360,2).any((1,3))
hd=uncoarsen(np.where(land[None],ba,np.nan).astype(np.float32))
times=[cftime.DatetimeNoLeap(y,m,15) for y in YEARS for m in range(1,13)]
dso=xr.Dataset({"burntArea":(("time","lat","lon"),hd,{"units":"1","standard_name":"burnt_area_fraction"})},
    coords={"time":times,"lat":np.arange(-89.75,90,0.5),"lon":np.arange(-179.75,180,0.5)},
    attrs={"title":"ED-ModelC-gdpreg-fromNetCDF","Conventions":"CF-1.7"})
dso=add_cf_bounds(dso); tu="days since 2001-01-01 00:00:00"
out=REPO/"ilamb/MODELS_GDP_REGIONAL_CHECK/ED-ModelC-gdpreg-nc/burntArea.nc"; out.parent.mkdir(parents=True,exist_ok=True)
for st in out.parent.glob("._*"): st.unlink()
dso.to_netcdf(out,encoding={"burntArea":{"zlib":True,"complevel":4,"_FillValue":1e20},
    "time":{"units":tu,"calendar":"noleap","dtype":"float64"},
    "time_bounds":{"units":tu,"calendar":"noleap","dtype":"float64"}},format="NETCDF4_CLASSIC")
print(f"[write] {out}  (M range [{M.min():.2f},{M.max():.2f}], w0=${10**W0:.0f}, s={S:.3f})")
