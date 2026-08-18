"""Does ED's fire_max_disturbance_rate explain the coupled under-burn?

    C:/Users/owusu/miniforge3/envs/edfire/python.exe scripts/diag_fire_cap.py

The coupled GCB2026 S3 runs burn about 170 Mha/yr against 793 observed, and the offline
Model F they were built from burns roughly the observed amount. ED_params.defaults.cfg
line 137 sets

    fire_max_disturbance_rate = 0.2

while the offline model runs with FIRE_MAX_RATE = 5.0. The annual-to-monthly transform is

    burned_frac_month = 1 - exp(-min(rate, FIRE_MAX) / 12)

so every cell whose fire rate exceeds the cap is clipped, and savanna cells are exactly
the ones that exceed it. This rebuilds Model F's rate field offline and applies each cap
in turn. If 0.2 takes the global total down to something near 170, the cap is the answer
and no refitting is needed, only a config change.
"""
import json
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, "scripts")
from reproduce_modelC import coarsen, sig, supp

R = 6371000.0
SL = slice(48, 240)                      # 2001-2016 out of the 1997-2016 dump
P = json.load(open("models/C/params.coupledE_gdp.json"))["params"]
REG = json.load(open("data_human/gdp_regional_gamma.json"))

ds = xr.open_dataset("global_baseline_modelC_inputs_1997-2016.nc")
g = lambda n: np.nan_to_num(ds[n].isel(time=SL).values.astype(np.float32))
dbar, t_air = coarsen(g("D_bar")), coarsen(g("T_air"))
p_ann, p_month = coarsen(g("P_ann")), coarsen(g("P_month"))
gpp = coarsen((np.clip(g("GPP_month_ntrl"), 0, None) * g("area_frac_ntrl")
               + np.clip(g("GPP_month_scnd"), 0, None) * g("area_frac_scnd")
               + np.clip(g("GPP_month_past"), 0, None) * g("area_frac_past")).astype(np.float32))
ds.close()
da = xr.open_dataset("global_baseline_modelCfuel_inputs_1997-2016.nc")
agb = coarsen(np.nan_to_num(da["AGB"].isel(time=SL).values.astype(np.float32)))
da.close()

hump = lambda x, b, dec: ((1 - np.exp(-np.clip(x / max(b, 1e-9), 0, 500)))
                          * np.exp(-np.clip(x / max(dec, 1e-9), 0, 500)))

base = (sig(dbar, P["k1"], P["D_low"]) * supp(dbar, P["k2"], P["D_high"])
        * (p_ann / (p_ann + P["P_half"] + 1e-12))
        * (1.0 / (1.0 + p_month / (P["pre_dampen_half"] + 1e-12)))
        * hump(P["gpp_af"] * gpp, P["gpp_b"], P["gpp_d"])
        * sig(t_air, P["ign_k"], P["ign_c"]))

lat1 = np.arange(-89.5, 90.0, 1.0)
trop = (np.abs(lat1) < 23.5).astype(np.float64)[None, :, None]
canopy = 1.0 / (1.0 + np.power(np.clip(agb / (P["trop_agb_crit"] + 1e-12), 0, None), P["trop_k_veg"]))
prod = base * (trop * canopy + (1.0 - trop))

gpp_cell = gpp.mean(0, keepdims=True)
rate = (np.power(np.clip(prod, 0, None), P["fire_exp"])
        * (1.0 + P["fuel_k"] * (gpp_cell / (gpp_cell + P["fuel_half"] + 1e-9))))

# the GDP human term, exactly as the coupling spec hands it to ED
gdp = np.load("data_human/gdp_pcap_grid_1deg.npy").astype(np.float64)
w = np.log10(np.clip(gdp, 50.0, None))
lon1 = np.arange(-179.5, 180.0, 1.0)
LON, LAT = np.meshgrid(lon1, lat1)
gam = np.zeros((180, 360))
for name, (lo, hi, la, lb) in {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
                               "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
                               "Europe": (-12, 40, 36, 72), "N.America": (-168, -52, 14, 74),
                               "Australia": (112, 154, -44, -10)}.items():
    box = (LON >= lo) & (LON <= hi) & (LAT >= la) & (LAT <= lb)
    gam[box] = REG["per_region_gamma"].get(name, 0.0)
M = np.clip(np.power(10.0, gam * (np.log10(REG["w0_gdp"]) - w)), 0.15, 6.0)
rate = rate * REG.get("scale", 1.0659) * M[None, :, :]

lat_e = np.deg2rad(lat1 + 0.5)
lat_w = np.deg2rad(lat1 - 0.5)
area = (R ** 2) * np.deg2rad(1.0) * (np.sin(lat_e) - np.sin(lat_w))
AREA = np.abs(area)[:, None] * np.ones((1, 360))

print("Model F offline, global burned area under each cap")
print("  GFED5 observed                          793 Mha/yr")
for cap in (5.0, 2.0, 1.0, 0.5, 0.2):
    frac = 1.0 - np.exp(-np.minimum(rate, cap) / 12.0)
    mha = float((frac.reshape(16, 12, 180, 360).sum(1).mean(0) * AREA).sum() / 1e10)
    tag = "  <- offline setting" if cap == 5.0 else ("  <- ED_params.defaults.cfg" if cap == 0.2 else "")
    print(f"  fire_max_disturbance_rate = {cap:<4} -> {mha:6.0f} Mha/yr{tag}")

print(f"\n  cells whose rate exceeds 0.2/yr: {float((rate > 0.2).mean()) * 100:.2f} percent of all cells")
print(f"  those cells hold {float((rate * (rate > 0.2)).sum() / rate.sum()) * 100:.1f} percent of the total fire rate")
