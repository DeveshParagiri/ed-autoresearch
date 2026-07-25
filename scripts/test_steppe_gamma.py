"""How much of the Kazakh/C-Asian steppe under-prediction is the GDP term vs the base
model? Recompute the steppe with: no GDP term (M=1), current regional gamma, and gamma
forced to 0 in a steppe box. Magnitude pinned to GFED5 each time. 1deg (scored model).
"""
import json, sys
import numpy as np, xarray as xr
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, sig, supp

DUMP="global_baseline_modelC_inputs_1997-2016.nc"; FUEL="global_baseline_modelCfuel_inputs_1997-2016.nc"; sl=slice(48,240)
FB=json.load(open("models/C/params.coupledE_gdp.json"))["params"]
REG=json.load(open("data_human/gdp_regional_gamma.json")); GV=REG["per_region_gamma"]; W0=np.log10(REG["w0_gdp"]); SIG=REG["sigma"]
lat1=-89.5+np.arange(180); lon1=-179.5+np.arange(360); LON,LAT=np.meshgrid(lon1,lat1)

ds=xr.open_dataset(DUMP); grab=lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
d={"dbar":coarsen(grab("D_bar")),"t_air":coarsen(grab("T_air")),"p_ann":coarsen(grab("P_ann")),"p_month":coarsen(grab("P_month"))}
d["gpp"]=coarsen((np.clip(grab("GPP_month_ntrl"),0,None)*grab("area_frac_ntrl")+np.clip(grab("GPP_month_scnd"),0,None)*grab("area_frac_scnd")+np.clip(grab("GPP_month_past"),0,None)*grab("area_frac_past")).astype(np.float32)); ds.close()
da=xr.open_dataset(FUEL); agb=coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()
gpp_cell=d["gpp"].mean(0,keepdims=True)
gdp=np.load("data_human/gdp_pcap_grid_1deg.npy"); w=np.log10(np.clip(gdp,50,None)); have=np.isfinite(gdp)&(gdp>0)

hump=lambda x,b,dec:(1-np.exp(-np.clip(x/max(b,1e-9),0,500)))*np.exp(-np.clip(x/max(dec,1e-9),0,500))
F={k:float(FB[k]) for k in FB}
base=(sig(d["dbar"],F["k1"],F["D_low"])*supp(d["dbar"],F["k2"],F["D_high"])*(d["p_ann"]/(d["p_ann"]+F["P_half"]+1e-12))*(1/(1+d["p_month"]/(F["pre_dampen_half"]+1e-12)))*hump(F["gpp_af"]*d["gpp"],F["gpp_b"],F["gpp_d"])*sig(d["t_air"],F["ign_k"],F["ign_c"]))
trop=(np.abs(lat1)<23.5).astype(float)[None,:,None]
canopy=1/(1+np.power(np.clip(agb/(F["trop_agb_crit"]+1e-12),0,None),F["trop_k_veg"]))
prod=base*(trop*canopy+(1-trop)); fuel=gpp_cell/(gpp_cell+F["fuel_half"]+1e-9)
rate=np.power(np.clip(prod,0,None),F["fire_exp"])*(1+F["fuel_k"]*fuel)

BOX={"Africa":(-20,52,-36,18),"Boreal":(40,180,48,78),"S.America":(-82,-34,-56,14),"SEAsia":(60,150,-11,30),"Europe":(-12,40,36,72),"N.America":(-168,-52,14,74),"Australia":(112,154,-44,-10)}
STEPPE=(48,90,45,56)
def gamma_field(steppe_gamma=None):
    region=np.full((180,360),"fb",dtype=object); assigned=np.zeros((180,360),bool)
    if steppe_gamma is not None:   # steppe takes precedence
        box=(LON>=STEPPE[0])&(LON<=STEPPE[1])&(LAT>=STEPPE[2])&(LAT<=STEPPE[3]); region[box]="Steppe"; assigned|=box
    for r,b in BOX.items():
        box=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3])&~assigned; region[box]=r; assigned|=box
    f=np.zeros((180,360))
    for r in list(BOX)+["fb"]: f[region==r]=GV.get(r,0.0)
    if steppe_gamma is not None: f[region=="Steppe"]=steppe_gamma
    return gaussian_filter(f,SIG,mode="nearest")
def M_of(gf):
    if gf is None: return np.ones((1,180,360))
    m=np.power(10.0,gf*(W0-w)); m[~have]=1.0; return np.clip(m,0.15,6.0)[None]

dg=xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf5=coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0); dg.close()
land=np.isfinite(xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0]).reshape(180,2,360,2).any((1,3))
R=6371e3; area=(R**2*np.deg2rad(1.0)**2*np.cos(np.deg2rad(lat1)))[:,None]*np.ones((1,360))
ann=lambda ba: ba.reshape(16,12,180,360).sum(1).mean(0)
gtot=float((ann(gf5)*area*land).sum())
def ba_of(M,s): return (1-np.exp(-np.minimum(s*M*rate,5.0)/12.0)).astype(np.float32)
def s_pin(M):
    lo,hi=0.02,8.0
    for _ in range(32):
        mid=0.5*(lo+hi)
        if float((ann(ba_of(M,mid))*area*land).sum())<gtot: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
def steppe_mha(ba):
    m=(LON>=STEPPE[0])&(LON<=STEPPE[1])&(LAT>=STEPPE[2])&(LAT<=STEPPE[3])
    return (ann(ba)*area*m).sum()/1e10

print(f"GFED5 steppe: {steppe_mha(gf5):.1f} Mha\n")
for label,gf in [("no GDP term (M=1 everywhere)",None),("current regional gamma",gamma_field()),
                 ("gamma=0 in steppe box",gamma_field(0.0))]:
    M=M_of(gf); s=s_pin(M); print(f"{label:32s}: steppe {steppe_mha(ba_of(M,s)):.1f} Mha  (s={s:.2f})")
# what gamma does the steppe actually see (Kazakhstan near the pivot?)
m=(LON>=STEPPE[0])&(LON<=STEPPE[1])&(LAT>=STEPPE[2])&(LAT<=STEPPE[3])&have
print(f"\nsteppe GDP/cap median: ${10**np.median(w[m]):.0f}  (pivot w0=${10**W0:.0f})")
print(f"steppe current gamma (smoothed) mean: {gamma_field()[m].mean():.2f}")
print(f"steppe multiplier M mean (current): {M_of(gamma_field())[0][m].mean():.2f}")
