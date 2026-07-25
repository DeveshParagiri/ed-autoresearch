"""Term-by-term: WHY does the base model under-burn the Kazakh steppe? Compare each
factor of the fire product over steppe cells vs African-savanna cells (which the model
gets right). The factor that is much smaller in the steppe is the bottleneck.
"""
import json, sys
import numpy as np, xarray as xr
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, sig, supp

DUMP="global_baseline_modelC_inputs_1997-2016.nc"; FUEL="global_baseline_modelCfuel_inputs_1997-2016.nc"; sl=slice(48,240)
FB=json.load(open("models/C/params.coupledE_gdp.json"))["params"]; F={k:float(FB[k]) for k in FB}
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
fuel_term=1+F["fuel_k"]*(gpp_cell/(gpp_cell+F["fuel_half"]+1e-9))
rate=np.power(np.clip(base,0,None),F["fire_exp"])*fuel_term

dg=xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gf=coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0); dg.close()
gf_ann=gf.reshape(16,12,180,360).sum(1).mean(0)

def cells(box): return (LON>=box[0])&(LON<=box[1])&(LAT>=box[2])&(LAT<=box[3])&(gf_ann>0.02)
STEPPE=cells((48,90,45,56)); SAV=cells((-18,40,5,15))|cells((15,40,-20,-8))  # Sahel + S.Africa savanna
tm=lambda a: a.mean(0)   # time-mean per cell
def rmean(field2d,mask): return float(field2d[mask].mean())

print(f"active cells: steppe={STEPPE.sum()}  savanna={SAV.sum()}\n")
print(f"{'quantity':22s} {'STEPPE':>9s} {'SAVANNA':>9s} {'steppe/sav':>11s}")
rows=[("--- drivers ---",None),
      ("D_bar (dryness)",tm(dbar)),("T_air (degC)",tm(t_air)),("P_ann",tm(p_ann)),
      ("P_month",tm(p_month)),("GPP monthly",tm(gpp)),("AGB",tm(agb)),
      ("--- factors [0..1] ---",None),
      ("onset(dbar)",tm(onset)),("supr(dbar)",tm(supr)),("p_flr(P_ann)",tm(p_flr)),
      ("p_dmp(P_month)",tm(p_dmp)),("gpp_hump",tm(gpp_m)),("ign(T_air)",tm(ign)),
      ("base product",tm(base)),
      ("--- rate ---",None),
      ("base^fire_exp",tm(np.power(np.clip(base,0,None),F["fire_exp"]))),
      ("fuel_term",fuel_term[0]),("rate",tm(rate))]
for name,f in rows:
    if f is None: print(name); continue
    s,v=rmean(f,STEPPE),rmean(f,SAV); print(f"{name:22s} {s:9.4g} {v:9.4g} {s/(v+1e-12):11.2f}")
print(f"\nign params: ign_k={F['ign_k']:.4g} ign_c={F['ign_c']:.2f} degC")
print(f"GFED steppe {(gf_ann*STEPPE).sum()/STEPPE.sum()*100:.1f}%/yr  vs savanna {(gf_ann*SAV).sum()/SAV.sum()*100:.1f}%/yr")
