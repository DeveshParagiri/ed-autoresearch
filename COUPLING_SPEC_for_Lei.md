# Coupling spec — Model C/E fire + GDP human term into GlobalED

For Lei. This is the recipe to run our latest offline fire model **online inside ED**,
coupling-consistent. It references the ED source in `ED_Source_Code/GlobalED/` by
file:line so you can see exactly where each piece goes.

The offline model is authoritative in Python: `scripts/reproduce_modelC.py`
(`fire_C`, `sig`, `supp`, `hump`) + `scripts/add_gdp_regional.py` (the GDP term and the
regional gamma). Everything below mirrors those.

## 0. What this model is
- Official ILAMB vs GFED5 burned area: **Overall 0.6783** (above paper Model E 0.6646
  and CLM6 0.6562), on ED's own dump climate, regionally faithful (every region
  0.52-1.14x GFED5), no hard seams, forward-runnable.
- It is Model C/E plus a physical human-suppression term keyed to GDP per capita,
  with the suppression strength varying smoothly by region (biome).

## 1. The fire equation (per cell, per month)
All inputs are ED-internal state (Section 2), except GDP + gamma (Section 3).

```
onset  = 1 / (1 + exp(-k1*(Dbar   - D_low )))     # fire onset with dryness
supr   = 1 / (1 + exp( k2*(Dbar   - D_high)))     # suppression at extreme dryness
p_flr  = P_ann  / (P_ann  + P_half)               # annual-precip floor (enough veg to burn)
p_dmp  = 1      / (1 + P_month/pre_dampen_half)    # wet-month damping
gpp_m  = (1 - exp(-gpp_af*GPP_m / gpp_b)) * exp(-gpp_af*GPP_m / gpp_d)   # GPP hump (fuel growth)
ign    = 1 / (1 + exp(-ign_k*(T_air - ign_c)))    # temperature ignition
base   = onset * supr * p_flr * p_dmp * gpp_m * ign

# tropical closed-canopy suppression, ONLY where |lat| < 23.5 (kills Amazon/Congo false fire):
canopy = 1 / (1 + (AGB/trop_agb_crit)^trop_k_veg)
prod   = base * canopy      if |lat| < 23.5   else   base

# fuel-capacity amplitude (period-mean GPP):
fuel   = GPP_mean / (GPP_mean + fuel_half)
rate   = prod^fire_exp * (1 + fuel_k*fuel)

# --- NEW: human GDP suppression term ---
M      = clip( 10^( gamma(x,y) * (w0 - log10(GDPpc)) ),  0.15,  6.0 )
rate   = s * M * rate

# ED disturbance rate (monthly burned fraction), same transform as the offline model:
burned_frac_month = 1 - exp( -min(rate, FIRE_MAX) / 12 )
```

`sig(x,k,c)=1/(1+exp(-k(x-c)))`, `supp(x,k,c)=1/(1+exp(k(x-c)))`. `gamma(x,y)` is the
smooth regional field (Section 3). This replaces the current native fire line
`ignition_rate = fuel*fp1*pow(dryness/30000,10)` at `fire.cc:216` (note that `pow(,10)`
is the over-concentration we diagnosed; our `fire_exp`=2.15 is the tamed version, and
the tropical `canopy` term is the fix for the Amazon over-burn the code flags at
`fire.cc:219`).

## 2. Driver -> ED live-state mapping (this is what makes it coupling-consistent)
Every driver is already computed by ED, so nothing is a static map that goes stale.

| offline driver | ED live variable | where |
|---|---|---|
| `Dbar` (dryness) | `cs->sdata->dryness_index_avg` | `read_site_data.h:51`, set in `update_fuel` via `calcSiteDrynessIndex` (`fire.cc:181`) |
| `T_air` | `cs->sdata->temp_average` | used at `fire.cc:68` |
| `P_ann`,`P_month` | `cs->sdata->precip_*` | `fire.cc:68`, `sdata->precip[time_period]` |
| `GPP_m`, `GPP_mean` | ED live GPP (photosynthesis.cc) — same source as the dump's `GPP_month_*` | per-patch/cohort |
| `AGB` | `cs->total_ag_biomass[landuse]` (live) | referenced `fire.cc:227` |
| `lat`, grid index | `sdata->lat_`, `globX_`, `globY_` | `read_site_data.h:16-20` |

IMPORTANT consistency check: the offline params were fit on the DUMP's values of these
(D_bar reaches ~5e6, `D_high`=2.2e6 sits inside that). Since the dump was produced BY
ED, ED's live `dryness_index_avg` / GPP should be on the same scale. Confirm that before
trusting the thresholds; if ED's live dryness is on a different scale than the dump, the
sigmoid thresholds (`D_low`,`D_high`,`ign_c`) need rescaling, not the whole model.

## 3. Two NEW gridded inputs (load them exactly like `gfed_bf`)
ED already loads a `[12][360][720]` NetCDF field indexed by `[globY_][globX_]`
(`gfed_bf`, declared `edmodels.h:675`, loaded in `load_GFED` `read_site_data.cc:1602-1632`,
used `fire.cc:45`). Add two fields the same way:

- `data->gdp_pcap[360][720]`  — GDP per capita (current US$), a NetCDF on the 0.5deg
  grid. Static (present-day) for the historical run; **time-varying for forward runs**
  (SSP scenarios provide gridded GDP per capita to 2100, so switch files by
  `mechanism_year` exactly like GFED at `read_site_data.cc:1581`).
- `data->gdp_gamma[360][720]` — the smooth per-region suppression strength (static map).

Loader stub (mirror `load_GFED`):
```c
nc_open(gdp_file, NC_NOWRITE, &ncid);
nc_inq_varid(ncid, "gdp_pcap", &varid);
nc_get_vara_double(ncid, varid, index, count, &(data->gdp_pcap[0][0]));
// same for gdp_gamma
```
We will hand you both NetCDFs on the 0.5deg grid (or tell us ED's exact grid/orientation
and we regrid). Constants: `w0 = 9242.0` (US$/cap pivot), `s = 1.0659` (global scale),
`FIRE_MAX = 5.0`, clip `[0.15, 6.0]`, gamma smoothing already baked into the field.

## 4. Where the GDP multiplier goes in `fire.cc`
The old `fire_suppression` block (`fire.cc:78-127`, currently `#if DOES_COMPILE` = off)
already did region-based human suppression with hardcoded floors (Africa 0.43,
N.America 0.02, ...). Our term is its physical, data-driven replacement — put it in the
same place, after `fireterm` is computed:
```c
double w   = log10(fmax(data->gdp_pcap[gY][gX], 50.0));
double M   = pow(10.0, data->gdp_gamma[gY][gX] * (9242.0_log10 - w));   // = w0 term
M          = fmin(fmax(M, 0.15), 6.0);
fireterm  *= data->gdp_scale * M;     // gdp_scale = s = 1.0659
```
(No fire on cropland stays as-is, `fire.cc:33`.)

## 5. Parameter values (fitted; `models/C/params.coupledE_gdp.json`)
```
k1=0.00211  D_low=119.8   k2=0.03417  D_high=2.2134e6  fire_exp=2.1451
P_half=259.70  pre_dampen_half=17.927  gpp_af=0.03655  gpp_b=3.0e-5  gpp_d=463.54
ign_k=0.00345  ign_c=19.360  trop_agb_crit=11.235  trop_k_veg=0.8348
fuel_k=0.02957  fuel_half=64.244
```
Per-region gamma (smooth-blended, Gaussian sigma=4 deg; `data_human/gdp_regional_gamma.json`):
```
Africa 1.60   Europe 0.70   N.America 0.60   Boreal 0.50
S.America 0.30   SEAsia 0.10   Australia 0.00   elsewhere 0.00
```

## 6. The one reconciliation task
The offline model uses **GPP** for fuel; ED's *native* fire used cohort **biomass**
(`cp->fuel`). ED computes GPP live, so use ED's live GPP for `GPP_m`/`GPP_mean` (not
biomass) to match how the model was tuned. The `AGB` term already maps to ED's live
`total_ag_biomass`, which closes the fire->biomass feedback (the "biomass map goes
inconsistent" point from the 07/23 meeting). Verify units/scale of ED live GPP vs the
dump's `GPP_month_*` before the production run.

## 7. Known limitation to communicate
- Boreal is under-predicted (0.52x GFED5): fuel-limited, and GDP suppresses wealthy
  boreal. That is a base-model fuel issue, not the human term. Not a coupling blocker.
- `gdp_gamma` is a static lat/lon map, so it does not migrate with vegetation in a
  prognostic run. Fine for GCB near-term; the migration-safe version keys gamma to a
  live vegetation-state variable (tree cover / PFT) instead of region boxes.

## 8. Deliverable files (the "latest coupled-consistent model")
- BA output (scored 0.6783): `ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/burntArea.nc`
- Base params: `models/C/params.coupledE_gdp.json`
- Regional gamma + constants: `data_human/gdp_regional_gamma.json`
- Gridded GDP driver (1deg npy now; NetCDF 0.5deg on request): `data_human/gdp_pcap_grid_1deg.npy`
- Build + reproduce: `scripts/add_gdp_regional.py`, `scripts/optimize_modelC_coupled.py`
  (`GDP_TERM=1`), `scripts/reproduce_modelC.py`
- Full method + all human-factor tests: `GDP_HUMAN_TERM_FINDINGS.md`
