"""Term-by-term decomposition of the k2 coupling-ready fit in problem regions.
Shows WHICH mechanism drives the Amazon over-burn and the boreal miss.
Dump climate (ED's D_bar/T_air/P_*) + fuel-file AGB, 1-deg, 2001-2016.
"""
import json, sys
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, sig, supp, hump

REPO = Path(".")
P = json.load(open("models/C/params.coupledE.k2.json"))["params"]
DUMP = "global_baseline_modelC_inputs_1997-2016.nc"
FUEL = "global_baseline_modelCfuel_inputs_1997-2016.nc"
sl = slice(48, 240)

ds = xr.open_dataset(DUMP)
grab = lambda n: np.nan_to_num(ds[n].isel(time=sl).values.astype(np.float32), nan=0.0)
dbar = coarsen(grab("D_bar")); t_air = coarsen(grab("T_air"))
p_ann = coarsen(grab("P_ann")); p_month = coarsen(grab("P_month"))
gpp = coarsen((np.clip(grab("GPP_month_ntrl"),0,None)*grab("area_frac_ntrl")
             + np.clip(grab("GPP_month_scnd"),0,None)*grab("area_frac_scnd")
             + np.clip(grab("GPP_month_past"),0,None)*grab("area_frac_past")).astype(np.float32))
ds.close()
da = xr.open_dataset(FUEL); agb = coarsen(np.nan_to_num(da["AGB"].isel(time=sl).values.astype(np.float32))); da.close()

# GFED5 ref (0.5->1deg), fraction
dg = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
gfed = coarsen(np.nan_to_num(dg["burntArea"].isel(time=slice(0,192)).values.astype(np.float32))/100.0); dg.close()

# term fields (all 192,180,360)
onset  = sig(dbar, P["k1"], P["D_low"])
supr   = supp(dbar, P["k2"], P["D_high"])
p_flr  = p_ann/(p_ann+P["P_half"]+1e-12)
p_dmp  = 1.0/(1.0+p_month/(P["pre_dampen_half"]+1e-12))
gpp_m  = hump(P["gpp_af"]*gpp, P["gpp_b"], P["gpp_d"])
ign    = sig(t_air, P["ign_k"], P["ign_c"])
base   = onset*supr*p_flr*p_dmp*gpp_m*ign
# tropical canopy suppression (|lat|<23.5)
lat1 = -90.0+(np.arange(180)+0.5)*1.0
trop = (np.abs(lat1)<P.get("trop_lat",23.5)).astype(np.float32)[None,:,None]
ratio = np.clip(agb/(P["trop_agb_crit"]+1e-12),0,None)
canopy = 1.0/(1.0+np.power(ratio,P["trop_k_veg"]))
prod   = base*(trop*canopy+(1.0-trop))
rate_noexp = prod.copy()
rate_exp   = np.power(np.clip(prod,0,None), P["fire_exp"])
gpp_cell = gpp.mean(0, keepdims=True); fuel = gpp_cell/(gpp_cell+P.get("fuel_half",1.0)+1e-9)
amp = (1.0+P["fuel_k"]*fuel)
rate = rate_exp*amp
ba = (1.0-np.exp(-np.minimum(rate,5.0)/12.0))  # SEASONAL_TRANSFORM

lon1 = -180.0+(np.arange(360)+0.5)*1.0
LON,LAT = np.meshgrid(lon1,lat1)
land = (gfed>0).any(0)

def zone(name, lo0,lo1,la0,la1):
    m = (LON>=lo0)&(LON<=lo1)&(LAT>=la0)&(LAT<=la1)&land
    tm = lambda f: float(f.mean(0)[m].mean()) if m.any() else float("nan")   # time-mean then zone-mean
    print(f"\n== {name}  ({int(m.sum())} land cells) ==")
    print(f"  drivers:  AGB={agb.mean(0)[m].mean():6.2f}  t_air={t_air.mean(0)[m].mean():6.2f}  "
          f"dbar={dbar.mean(0)[m].mean():9.1f}  GPP={gpp.mean(0)[m].mean():6.2f}")
    print(f"  terms:    onset={tm(onset):.3f} supr={tm(supr):.3f} p_flr={tm(p_flr):.3f} "
          f"p_dmp={tm(p_dmp):.3f} gpp={tm(gpp_m):.3f} ign={tm(ign):.3f}")
    cm = float(canopy[0][m & (np.abs(LAT)<23.5)].mean()) if (m&(np.abs(LAT)<23.5)).any() else 1.0
    print(f"  suppress: canopy_mod(trop)={cm:.3f}   base_product={tm(base):.4f}")
    print(f"  amplify:  fuel_amp={float(amp[0][m].mean()):.2f}   (fire_exp={P['fire_exp']:.2f})")
    print(f"  rate:     before_exp={tm(rate_noexp):.4f} -> after_exp={tm(rate_exp):.5f} -> x amp={tm(rate):.5f}")
    print(f"  BA:       model={tm(ba)*1e3:6.3f}e-3   GFED5={gfed.mean(0)[m].mean()*1e3:6.3f}e-3   "
          f"ratio={tm(ba)/(gfed.mean(0)[m].mean()+1e-12):.2f}x")

zone("Amazon forest", -70,-50,-8,2)
zone("Cerrado / S.Brazil savanna", -60,-42,-20,-8)
zone("Boreal Eurasia", 80,140,52,68)
zone("Africa savanna (reference, works)", 15,30,5,15)
