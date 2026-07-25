"""Two deliverable models from the SAME 0.6783 base (regional-GDP), curing OFF vs ON,
so the only difference is the curing knob. Writes both BA for ILAMB scoring.
  A = leaderboard model  (cure_k=0)      -> best aggregate ILAMB
  B = coupling model     (cure_k=0.15)   -> regional fidelity (steppe/Australia fixed)
"""
import json, sys, os
import numpy as np, xarray as xr, cftime
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

YEARS=list(range(2001,2017)); sl=slice(48,240)
FB=json.load(open("models/C/params.coupledE_gdp.json"))["params"]; F={k:float(FB[k]) for k in FB}   # 0.6783 base
CS=json.load(open("models/C/params.coupledE_cure.json"))["params"]                                   # curing shape
REG=json.load(open("data_human/gdp_regional_gamma.json")); GV=REG["per_region_gamma"]; W0=np.log10(REG["w0_gdp"]); SIG=REG["sigma"]
lat1=-89.5+np.arange(180); lon1=-179.5+np.arange(360); LON,LAT=np.meshgrid(lon1,lat1)
gg=lambda n,ds: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
ds=xr.open_dataset("global_baseline_modelC_inputs_1997-2016.nc")
dbar=coarsen(gg("D_bar",ds));t_air=coarsen(gg("T_air",ds));p_ann=coarsen(gg("P_ann",ds));p_month=coarsen(gg("P_month",ds))
gpp=coarsen((np.clip(gg("GPP_month_ntrl",ds),0,None)*gg("area_frac_ntrl",ds)+np.clip(gg("GPP_month_scnd",ds),0,None)*gg("area_frac_scnd",ds)+np.clip(gg("GPP_month_past",ds),0,None)*gg("area_frac_past",ds)).astype(np.float32));ds.close()
da=xr.open_dataset("global_baseline_modelCfuel_inputs_1997-2016.nc");agb=coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32)));da.close()
gpp_cell=gpp.mean(0,keepdims=True);hump=lambda x,b,dec:(1-np.exp(-np.clip(x/max(b,1e-9),0,500)))*np.exp(-np.clip(x/max(dec,1e-9),0,500))
base=(sig(dbar,F["k1"],F["D_low"])*supp(dbar,F["k2"],F["D_high"])*(p_ann/(p_ann+F["P_half"]+1e-12))*(1/(1+p_month/(F["pre_dampen_half"]+1e-12)))*hump(F["gpp_af"]*gpp,F["gpp_b"],F["gpp_d"])*sig(t_air,F["ign_k"],F["ign_c"]))
trop=(np.abs(lat1)<23.5).astype(float)[None,:,None];canopy=1/(1+np.power(np.clip(agb/(F["trop_agb_crit"]+1e-12),0,None),F["trop_k_veg"]))
R0=np.power(np.clip(base*(trop*canopy+(1-trop)),0,None),F["fire_exp"])*(1+F["fuel_k"]*(gpp_cell/(gpp_cell+F["fuel_half"]+1e-9)))
pac=p_ann.mean(0,keepdims=True);agc=agb.mean(0,keepdims=True);cured=sig(dbar,0.001,3000.0)
CURE=(1/(1+np.exp(-0.02*(pac-CS["cure_p_min"]))))*(1/(1+np.power(np.clip(agc/CS["cure_agb_crit"],0,None),3.0)))*(1/(1+gpp_cell/(CS["cure_gpp_ref"]+1e-9)))*cured
gdp=np.load("data_human/gdp_pcap_grid_1deg.npy");w=np.log10(np.clip(gdp,50,None));have=np.isfinite(gdp)&(gdp>0)
BOX={"Africa":(-20,52,-36,18),"Boreal":(40,180,48,78),"S.America":(-82,-34,-56,14),"SEAsia":(60,150,-11,30),"Europe":(-12,40,36,72),"N.America":(-168,-52,14,74),"Australia":(112,154,-44,-10)}
region=np.full((180,360),"fb",dtype=object);assigned=np.zeros((180,360),bool)
for r,b in BOX.items():
    bx=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3])&~assigned;region[bx]=r;assigned|=bx
gf=np.zeros((180,360))
for r in list(BOX)+["fb"]: gf[region==r]=GV.get(r,0.0)
M=np.clip(np.power(10.0,gaussian_filter(gf,SIG,mode="nearest")*(W0-w)),0.15,6.0);M[~have]=1.0;M=M[None]
dg=xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc");gf5=coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0);dg.close()
land=np.isfinite(xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0]).reshape(180,2,360,2).any((1,3))
R=6371e3;area=(R**2*np.deg2rad(1.0)**2*np.cos(np.deg2rad(lat1)))[:,None]*np.ones((1,360));ann=lambda b:b.reshape(16,12,180,360).sum(1).mean(0)
gtot=float((ann(gf5)*area*land).sum());ba_from=lambda rate,s:(1-np.exp(-np.minimum(s*M*rate,5.0)/12.0)).astype(np.float32)
def s_pin(rate):
    lo,hi=0.02,8.0
    for _ in range(30):
        mid=0.5*(lo+hi); lo,hi=(mid,hi) if float((ann(ba_from(rate,mid))*area*land).sum())<gtot else (lo,mid)
    return 0.5*(lo+hi)
def write(b,name):
    hd=uncoarsen(np.where(land[None],b,np.nan).astype(np.float32));times=[cftime.DatetimeNoLeap(y,m,15) for y in YEARS for m in range(1,13)]
    d=xr.Dataset({"burntArea":(("time","lat","lon"),hd,{"units":"1","standard_name":"burnt_area_fraction"})},coords={"time":times,"lat":np.arange(-89.75,90,0.5),"lon":np.arange(-179.75,180,0.5)},attrs={"Conventions":"CF-1.7"})
    d=add_cf_bounds(d);tu="days since 2001-01-01 00:00:00";os.makedirs(f"ilamb/MODELS_GEORGE/{name}",exist_ok=True)
    for s in __import__("pathlib").Path(f"ilamb/MODELS_GEORGE/{name}").glob("._*"): s.unlink()
    d.to_netcdf(f"ilamb/MODELS_GEORGE/{name}/burntArea.nc",encoding={"burntArea":{"zlib":True,"complevel":4,"_FillValue":1e20},"time":{"units":tu,"calendar":"noleap","dtype":"float64"},"time_bounds":{"units":tu,"calendar":"noleap","dtype":"float64"}},format="NETCDF4_CLASSIC")
for ck,nm in [(0.0,"ED-ModelC-A-leaderboard"),(0.15,"ED-ModelC-B-coupling")]:
    write(ba_from(R0+ck*CURE,s_pin(R0+ck*CURE)),nm); print(f"wrote {nm} (cure_k={ck})")
