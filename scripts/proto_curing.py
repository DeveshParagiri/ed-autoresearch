"""Prototype: a grass-CURING fire pathway so temperate grasslands (Kazakh steppe) burn
without the high-GPP requirement that the tropical-savanna-tuned model imposes.

Diagnosis (diag_steppe_terms): steppe under-burns because low GPP -> low gpp_hump, then
fire_exp amplifies. Physically, cured dead grass burns efficiently regardless of how
productive it is. So add an ADDITIVE pathway:

  grass_fuel = gpp_cell/(gpp_cell + cure_gpp_half)     # saturates -> modest grass is enough
  cured      = onset(dbar) * p_dmp                     # dry + not recently wet -> cured
  grass_gate = 1/(1+(AGB/cure_agb_crit)^cure_agb_n)    # grassland only (forests suppressed)
  curing     = cure_k * grass_fuel * cured * ign * grass_gate * p_flr   # p_flr kills deserts
  rate       = base^fire_exp * fuel_term + curing

Reports regional Mha (magnitude pinned to GFED) for base (no curing) vs +curing, watching
that the steppe rises WITHOUT Africa/Amazon/Sahara blowing up. Sweeps cure_k.
"""
import json, sys
import numpy as np, xarray as xr
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, sig, supp

DUMP="global_baseline_modelC_inputs_1997-2016.nc"; FUEL="global_baseline_modelCfuel_inputs_1997-2016.nc"; sl=slice(48,240)
FB=json.load(open("models/C/params.coupledE_gdp.json"))["params"]; F={k:float(FB[k]) for k in FB}
REG=json.load(open("data_human/gdp_regional_gamma.json")); GV=REG["per_region_gamma"]; W0=np.log10(REG["w0_gdp"]); SIG=REG["sigma"]
lat1=-89.5+np.arange(180); lon1=-179.5+np.arange(360); LON,LAT=np.meshgrid(lon1,lat1)

ds=xr.open_dataset(DUMP); grab=lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32))
dbar=coarsen(grab("D_bar")); t_air=coarsen(grab("T_air")); p_ann=coarsen(grab("P_ann")); p_month=coarsen(grab("P_month"))
gpp=coarsen((np.clip(grab("GPP_month_ntrl"),0,None)*grab("area_frac_ntrl")+np.clip(grab("GPP_month_scnd"),0,None)*grab("area_frac_scnd")+np.clip(grab("GPP_month_past"),0,None)*grab("area_frac_past")).astype(np.float32)); ds.close()
da=xr.open_dataset(FUEL); agb=coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()
gpp_cell=gpp.mean(0,keepdims=True)
hump=lambda x,b,dec:(1-np.exp(-np.clip(x/max(b,1e-9),0,500)))*np.exp(-np.clip(x/max(dec,1e-9),0,500))

onset=sig(dbar,F["k1"],F["D_low"]); supr=supp(dbar,F["k2"],F["D_high"])
p_flr=p_ann/(p_ann+F["P_half"]+1e-12); p_dmp=1/(1+p_month/(F["pre_dampen_half"]+1e-12))
gpp_m=hump(F["gpp_af"]*gpp,F["gpp_b"],F["gpp_d"]); ign=sig(t_air,F["ign_k"],F["ign_c"])
base=onset*supr*p_flr*p_dmp*gpp_m*ign
trop=(np.abs(lat1)<23.5).astype(float)[None,:,None]
canopy=1/(1+np.power(np.clip(agb/(F["trop_agb_crit"]+1e-12),0,None),F["trop_k_veg"]))
prod=base*(trop*canopy+(1-trop))
fuel_term=1+F["fuel_k"]*(gpp_cell/(gpp_cell+F["fuel_half"]+1e-9))
rate_base=np.power(np.clip(prod,0,None),F["fire_exp"])*fuel_term

# curing pathway (prototype). KEY: FLAT across grasslands (~1 for both steppe and
# savanna) so the additive boost is EQUAL -> the smaller-base steppe gains relatively
# more after magnitude pinning. Gated: grass_carry kills deserts (need enough grass),
# grass_gate kills forests (wrong fuel), cured_sat kills wet regions.
# Identify burnable grassland by PRECIP WINDOW (not GPP -- the steppe's low GPP is
# exactly what we must not penalize): enough rain to grow grass (excludes desert),
# low biomass (excludes forest), dry enough season (excludes wet). Flat over grass.
P_GRASS_MIN=180.0; P_GRASS_K=0.02; CURE_AGB_CRIT=2.5; CURE_AGB_N=3.0; GPP_REF=0.3
p_ann_cell=p_ann.mean(0,keepdims=True)
grass_zone=1/(1+np.exp(-P_GRASS_K*(p_ann_cell-P_GRASS_MIN)))         # ~1 if enough rain, ~0 desert
grass_gate=1/(1+np.power(np.clip(agb/CURE_AGB_CRIT,0,None),CURE_AGB_N))  # ~1 grass, ~0 forest
cured_sat=sig(dbar,0.001,3000.0)                                     # cured if dry (saturates)
inv_gpp=1.0/(1.0+gpp_cell/GPP_REF)                                   # fine cured grass burns MORE per fuel
curing_unit=grass_zone*grass_gate*cured_sat*inv_gpp                 # x cure_k, boosts sparse grass
_st=((LON>=48)&(LON<=90)&(LAT>=45)&(LAT<=56)); _sv=((LON>=-18)&(LON<=40)&(LAT>=5)&(LAT<=15))
_gr=(gpp_cell[0]>0.05)&(agb.mean(0)<2)
print("[curing_unit] grassy-cell means  steppe:",round(float(curing_unit[0][_st&_gr].mean()),3),
      " savanna:",round(float(curing_unit[0][_sv&_gr].mean()),3))

# GDP multiplier (full model)
gdp=np.load("data_human/gdp_pcap_grid_1deg.npy"); w=np.log10(np.clip(gdp,50,None)); have=np.isfinite(gdp)&(gdp>0)
BOX={"Africa":(-20,52,-36,18),"Boreal":(40,180,48,78),"S.America":(-82,-34,-56,14),"SEAsia":(60,150,-11,30),"Europe":(-12,40,36,72),"N.America":(-168,-52,14,74),"Australia":(112,154,-44,-10)}
region=np.full((180,360),"fb",dtype=object); assigned=np.zeros((180,360),bool)
for r,b in BOX.items():
    bx=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3])&~assigned; region[bx]=r; assigned|=bx
gfield=np.zeros((180,360))
for r in list(BOX)+["fb"]: gfield[region==r]=GV.get(r,0.0)
gfield=gaussian_filter(gfield,SIG,mode="nearest")
M=np.clip(np.power(10.0,gfield*(W0-w)),0.15,6.0); M[~have]=1.0; M=M[None]

dg=xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf=coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0); dg.close()
land=np.isfinite(xr.open_dataset("ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"].values[0]).reshape(180,2,360,2).any((1,3))
R=6371e3; area=(R**2*np.deg2rad(1.0)**2*np.cos(np.deg2rad(lat1)))[:,None]*np.ones((1,360))
ann=lambda ba: ba.reshape(16,12,180,360).sum(1).mean(0)
gtot=float((ann(gf)*area*land).sum())
def ba_of(rate,s): return (1-np.exp(-np.minimum(s*M*rate,5.0)/12.0)).astype(np.float32)
def s_pin(rate):
    lo,hi=0.02,8.0
    for _ in range(30):
        mid=0.5*(lo+hi)
        if float((ann(ba_of(rate,mid))*area*land).sum())<gtot: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
REG_BOX={"Steppe":(48,90,45,56),"Africa":(-20,52,-36,18),"Amazon":(-75,-50,-12,5),
         "Sahara":(-10,30,18,28),"Australia":(112,154,-44,-10),"Boreal":(40,180,48,78),"India+SEA":(60,150,-11,35)}
GFED_MHA={"Steppe":23,"Africa":496,"Amazon":"low","Sahara":"~0","Australia":58,"Boreal":50,"India+SEA":68}
def mha(ba,b): m=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3]); return (ann(ba)*area*m).sum()/1e10

print(f"{'model':16s} {'GLOBAL':>7s} " + " ".join(f"{r:>9s}" for r in REG_BOX))
print(f"{'GFED5':16s} {gtot/1e10:7.0f} " + " ".join(f"{str(GFED_MHA[r]):>9s}" for r in REG_BOX))
for label,rate in [("base (no curing)",rate_base)]+[(f"+curing k={k}",rate_base+k*curing_unit) for k in (0.05,0.1,0.2,0.4)]:
    s=s_pin(rate); ba=ba_of(rate,s)
    print(f"{label:16s} {mha(ba,(-180,180,-90,90)):7.0f} " + " ".join(f"{mha(ba,b):9.0f}" for b in REG_BOX.values()))
