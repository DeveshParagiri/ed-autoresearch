"""BEST coupling model: regional-GDP gamma (smooth, fixes Australia/Asia) + grass-curing
(fixes the temperate steppe). Fit to REGIONAL FIDELITY -- each region weighted equally by
log-ratio to GFED, so small regions (steppe, Australia) count as much as Africa -- because
the coupled carbon budget needs every region right, and aggregate ILAMB does not reward them.

Base params + curing shape from params.coupledE_cure.json; per-region gamma and cure_k
fit here by coordinate descent, global magnitude pinned to GFED5. Smooth-blended (no seams).
"""
import json, sys
import numpy as np, xarray as xr, cftime
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, uncoarsen, add_cf_bounds, sig, supp

REPO="."; YEARS=list(range(2001,2017)); FIRE_MAX=5.0; sl=slice(48,240)
DUMP="global_baseline_modelC_inputs_1997-2016.nc"; FUEL="global_baseline_modelCfuel_inputs_1997-2016.nc"
FB=json.load(open("models/C/params.coupledE_cure.json"))["params"]; F={k:float(FB[k]) for k in FB}
REG=json.load(open("data_human/gdp_regional_gamma.json")); W0=np.log10(REG["w0_gdp"]); SIG=REG["sigma"]
lat1=-89.5+np.arange(180); lon1=-179.5+np.arange(360); LON,LAT=np.meshgrid(lon1,lat1)

ds=xr.open_dataset(DUMP); grab=lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
dbar=coarsen(grab("D_bar")); t_air=coarsen(grab("T_air")); p_ann=coarsen(grab("P_ann")); p_month=coarsen(grab("P_month"))
gpp=coarsen((np.clip(grab("GPP_month_ntrl"),0,None)*grab("area_frac_ntrl")+np.clip(grab("GPP_month_scnd"),0,None)*grab("area_frac_scnd")+np.clip(grab("GPP_month_past"),0,None)*grab("area_frac_past")).astype(np.float32)); ds.close()
da=xr.open_dataset(FUEL); agb=coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()
gpp_cell=gpp.mean(0,keepdims=True); hump=lambda x,b,dec:(1-np.exp(-np.clip(x/max(b,1e-9),0,500)))*np.exp(-np.clip(x/max(dec,1e-9),0,500))

# base rate (no curing) + curing unit field (cure_k=1)
base=(sig(dbar,F["k1"],F["D_low"])*supp(dbar,F["k2"],F["D_high"])*(p_ann/(p_ann+F["P_half"]+1e-12))*(1/(1+p_month/(F["pre_dampen_half"]+1e-12)))*hump(F["gpp_af"]*gpp,F["gpp_b"],F["gpp_d"])*sig(t_air,F["ign_k"],F["ign_c"]))
trop=(np.abs(lat1)<23.5).astype(float)[None,:,None]
canopy=1/(1+np.power(np.clip(agb/(F["trop_agb_crit"]+1e-12),0,None),F["trop_k_veg"]))
prod=base*(trop*canopy+(1-trop)); fuel_term=1+F["fuel_k"]*(gpp_cell/(gpp_cell+F["fuel_half"]+1e-9))
R0=np.power(np.clip(prod,0,None),F["fire_exp"])*fuel_term
p_ann_cell=p_ann.mean(0,keepdims=True); agb_cell=agb.mean(0,keepdims=True); cured_sat=sig(dbar,0.001,3000.0)
grass_zone=1/(1+np.exp(-0.02*(p_ann_cell-F["cure_p_min"]))); grass_gate=1/(1+np.power(np.clip(agb_cell/F["cure_agb_crit"],0,None),3.0))
inv_gpp=1/(1+gpp_cell/(F["cure_gpp_ref"]+1e-9)); CURE_UNIT=grass_zone*grass_gate*inv_gpp*cured_sat

gdp=np.load("data_human/gdp_pcap_grid_1deg.npy"); w=np.log10(np.clip(gdp,50,None)); have=np.isfinite(gdp)&(gdp>0)
BOX={"Africa":(-20,52,-36,18),"Boreal":(40,180,48,78),"S.America":(-82,-34,-56,14),"SEAsia":(60,150,-11,30),"Europe":(-12,40,36,72),"N.America":(-168,-52,14,74),"Australia":(112,154,-44,-10)}
def gamma_field(gv):
    region=np.full((180,360),"fb",dtype=object); assigned=np.zeros((180,360),bool)
    for r,b in BOX.items():
        bx=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3])&~assigned; region[bx]=r; assigned|=bx
    f=np.zeros((180,360))
    for r in list(BOX)+["fb"]: f[region==r]=gv.get(r,0.0)
    return gaussian_filter(f,SIG,mode="nearest")
def M_of(gf):
    m=np.power(10.0,gf*(W0-w)); m[~have]=1.0; return np.clip(m,0.15,6.0)[None]

dg=xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf5=coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0); dg.close()
land=np.isfinite(xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0]).reshape(180,2,360,2).any((1,3))
R=6371e3; area=(R**2*np.deg2rad(1.0)**2*np.cos(np.deg2rad(lat1)))[:,None]*np.ones((1,360))
ann=lambda ba: ba.reshape(16,12,180,360).sum(1).mean(0)
gtot=float((ann(gf5)*area*land).sum())
def ba_from(rate,M,s): return (1-np.exp(-np.minimum(s*M*rate,FIRE_MAX)/12.0)).astype(np.float32)
def s_pin(rate,M):
    lo,hi=0.02,8.0
    for _ in range(30):
        mid=0.5*(lo+hi)
        if float((ann(ba_from(rate,M,mid))*area*land).sum())<gtot: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
RB={"Africa":(-20,52,-36,18),"Boreal":(40,180,48,78),"Steppe":(48,90,45,56),"Australia":(112,154,-44,-10),"S.Amer":(-82,-34,-56,14),"N.Amer":(-168,-52,14,74),"Europe":(-12,40,36,72),"India+SEA":(60,150,-11,35)}
def tot(a,b): m=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3]); return float((a*area*m).sum())/1e10
GF={r:tot(ann(gf5),b) for r,b in RB.items()}
def regloss(gv,ck,s):                       # equal-weight log-ratio over regions
    a=ann(ba_from(R0+ck*CURE_UNIT,M_of(gamma_field(gv)),s))
    return sum(np.log((tot(a,b)+1e-6)/(GF[r]+1e-6))**2 for r,b in RB.items())

gv={r:float(REG["per_region_gamma"].get(r,0.0)) for r in list(BOX)+["fb"]}; ck=0.2
ggrid=np.linspace(0,1.6,17); ckgrid=[0,0.05,0.1,0.2,0.35,0.5,0.8]
s=s_pin(R0+ck*CURE_UNIT,M_of(gamma_field(gv)))
print(f"[fit] start regloss={regloss(gv,ck,s):.3f}")
for it in range(4):
    for r in list(BOX)+["fb"]:
        gv[r]=float(min(((regloss({**gv,r:float(x)},ck,s),x) for x in ggrid))[1])
    ck=float(min(((regloss(gv,float(x),s),x) for x in ckgrid))[1])
    s=s_pin(R0+ck*CURE_UNIT,M_of(gamma_field(gv)))
    print(f"[fit] pass {it+1}: regloss={regloss(gv,ck,s):.3f} cure_k={ck:.2f}  "+" ".join(f"{r[:3]}={gv[r]:.1f}" for r in BOX))

M=M_of(gamma_field(gv)); s=s_pin(R0+ck*CURE_UNIT,M); ba=ba_from(R0+ck*CURE_UNIT,M,s)
json.dump({"per_region_gamma":gv,"cure_k":ck,"sigma":SIG,"s":s,"base":"params.coupledE_cure.json"},open("data_human/regional_cure.json","w"),indent=2)
def write_nc(b,name):
    hd=uncoarsen(np.where(land[None],b,np.nan).astype(np.float32)); times=[cftime.DatetimeNoLeap(y,m,15) for y in YEARS for m in range(1,13)]
    dso=xr.Dataset({"burntArea":(("time","lat","lon"),hd,{"units":"1","standard_name":"burnt_area_fraction"})},coords={"time":times,"lat":np.arange(-89.75,90,0.5),"lon":np.arange(-179.75,180,0.5)},attrs={"title":name,"Conventions":"CF-1.7"})
    dso=add_cf_bounds(dso); tu="days since 2001-01-01 00:00:00"
    out=f"ilamb/MODELS_REGIONAL_CURE/{name}/burntArea.nc"; import os; os.makedirs(f"ilamb/MODELS_REGIONAL_CURE/{name}",exist_ok=True)
    for stf in __import__("pathlib").Path(f"ilamb/MODELS_REGIONAL_CURE/{name}").glob("._*"): stf.unlink()
    dso.to_netcdf(out,encoding={"burntArea":{"zlib":True,"complevel":4,"_FillValue":1e20},"time":{"units":tu,"calendar":"noleap","dtype":"float64"},"time_bounds":{"units":tu,"calendar":"noleap","dtype":"float64"}},format="NETCDF4_CLASSIC"); print(f"[write] {out}")
write_nc(ba,"ED-ModelC-regcure")

print(f"\ncure_k={ck:.2f}   {'region':10s} "+" ".join(f"{r:>9s}" for r in RB))
print(f"{'':17s}{'GFED5':10s} "+" ".join(f"{GF[r]:9.0f}" for r in RB))
print(f"{'':17s}{'regcure':10s} "+" ".join(f"{tot(ann(ba),b):9.0f}" for b in RB.values()))
print(f"global regcure {float((ann(ba)*area*land).sum())/1e10:.0f} vs GFED {gtot/1e10:.0f}")
