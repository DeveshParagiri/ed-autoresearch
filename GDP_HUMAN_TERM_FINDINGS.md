# Human suppression term (GDP per capita) — findings (2026-07-24)

Prompted by the 2026-07-23 advisor meeting. George's explicit assignment: "go pull
the GDP and the population and the fire frequency for every country and build this
plot", then bolt a socioeconomic term onto the fire model. His rule: make humans
*socioeconomically* dependent, not latitude dependent (poor countries light more
fire, wealthy countries suppress it), added as a multiplier onto the existing model.

## Data (all offline-cached, reproducible)
- Fire: GFED5 burned area 0.5deg monthly 2001-2020 (`ilamb_ref_official`), aggregated
  to mean annual burned *fraction* per country.
- Country polygons: Natural Earth 50m admin_0 (`ISO_A3_EH`, `CONTINENT`).
- Wealth: World Bank GDP per capita current US$ (`NY.GDP.PCAP.CD`) + population
  (`SP.POP.TOTL`), mean 2001-2020. NOTE: World Bank JSON API is blocked here; the
  CSV download endpoint (`.../indicator/NY.GDP.PCAP.CD?downloadformat=csv`) works.

## Step 1 — the raw plot (`fire_vs_gdp_country.py`)
Fire falls with per-capita wealth across 164 countries:
**log10(fire) = 3.47 - 0.92 log10(GDPpc),  r = -0.55.** Poor African savanna nations
pin the top (South Sudan, Guinea, Togo, CAR, Ghana burn 50-65%/yr at GDPpc < $1.5k);
wealthy countries collapse to the bottom (Norway, Switzerland, Qatar < 0.1%/yr).
Fig: `fire_vs_gdp_country.png`. Table: `data_human/fire_vs_gdp_country.csv`.

## Step 2 — climate control (`fire_vs_gdp_partial.py`)
The raw slope conflates poverty with savanna climate (poor countries are
disproportionately tropical). Regress country fire on climate + vegetation only
(precip hump, temperature, GPP; all from ED's own dump), then test the residual:

| step | slope /decade | note |
|---|---|---|
| raw | -0.92 (r -0.55) | mixes poverty with climate |
| climate + veg only | — | explains R2 = 0.48 |
| + wealth | — | R2 -> 0.60 |
| **partial wealth** | **-0.70 (r -0.47), t -6.7, p 4e-10** | survives climate removal |

**Wealth keeps 76% of its raw slope after removing climate.** Two countries with the
same precip/temp/productivity still differ by wealth: the poorer burns more. The
socioeconomic signal is real, not a climate artifact. Fig `fire_vs_gdp_partial.png`
(right panel = the added-variable plot, climate removed). Table
`data_human/fire_vs_gdp_partial.csv`.

## Step 3 — bolt onto the model + score (`add_gdp_term.py`)
Base = `params.coupledE.k2.json` (best single-global dump fit, one global form, no
regional hard-coding, so the human term's gain is cleanly attributable). Term:

    M(cell) = clip( 10^( gamma * (w0 - log10 GDPpc) ),  0.15, 6 )      w0 = median wealth

applied to the fire RATE (where `fuel_k` acts), with a single global rescale so the
global total is pinned to the base (the gain is *pattern*, not a magnitude re-tune).
GDP gridded to 1deg, every land cell filled from nearest country (`gdp_pcap_grid_1deg.npy`).

Official ILAMB (GFED5, burned area), gamma sweep:

| model | Overall | Bias | RMSE | Seasonal | Spatial |
|---|---|---|---|---|---|
| base (no human) | 0.6547 | 0.6952 | 0.4863 | 0.8004 | 0.8054 |
| **+GDP g=0.15** | **0.6603** | 0.7020 | 0.4888 | 0.8004 | **0.8213** |
| +GDP g=0.30 | 0.6602 | 0.7017 | 0.4891 | 0.8004 | 0.8207 |
| +GDP g=0.50 | 0.6534 | | | | 0.7980 |
| +GDP g=0.70 | 0.6498 | | | | 0.7832 |

**+0.0056 Overall (0.6547 -> 0.6603), the entire gain in the Spatial score
(0.805 -> 0.821); Seasonal untouched** (GDP redistributes in space, not time). The
model prefers a MILD term (g ~ 0.15-0.3); past 0.3 it over-suppresses and the score
falls. Regional burned area (Mha/yr):

| | Africa | S.Amer | N.Amer | India+SEA | Europe | Boreal |
|---|---|---|---|---|---|---|
| GFED5 | 496 | 65 | 22 | 68 | 17 | 50 |
| base | 395 | 153 | 39 | 93 | 16 | 4 |
| +GDP | 451 | 129 | 29 | 100 | 12 | 3 |

Amplifies poor Africa toward GFED5, suppresses the Amazon over-burn and N.America.
Fig: `gdp_term_figure.png` (`fig_gdp_term.py`).

## Reconciliation of the two gammas
Country regression wants g ~ 0.70; the gridded model wants g ~ 0.15-0.3. Not a
contradiction: the model already carries explicit climate + productivity drivers
that do most of what the country regression lumped into "climate," so the residual
wealth correction the model needs is gentler. The data picks a mild human term.

## Caveats (stated honestly)
- Cannot fix the boreal gap (2-4 vs GFED5 50): that is fuel/climate limitation, and
  boreal countries are wealthy so the term suppresses there, if anything.
- Slightly over-suppresses Europe (16 -> 12 vs 17).
- Static GDP grid (mean 2001-2020) for this test; the WB series is time-varying back
  to 1960, so the coupled version can use the real time series.

## Why this matters
- Scale: the whole Model C -> E form change was ~ +0.015 Overall. This single human
  variable recovers ~ a third of that, orthogonally (spatial only), so it can stack
  on the form change.
- Coupling: GDP per capita is forward-runnable (WB back to 1960, gridded pop exists
  for millennia) and uses ED's own dump climate, so it satisfies Lei's requirement
  (forward-runnable, no remote-sensing-only inputs). Unlike roads / remote sensing,
  this term can go into the global carbon budget version.

## Files
- Scripts: `fire_vs_gdp_country.py`, `fire_vs_gdp_partial.py`, `add_gdp_term.py`, `fig_gdp_term.py`.
- Figures: `fire_vs_gdp_country.png`, `fire_vs_gdp_partial.png`, `gdp_term_figure.png`.
- Data: `data_human/fire_vs_gdp_country.csv`, `fire_vs_gdp_partial.csv`, `gdp_pcap_grid_1deg.npy`.
- Scored BA (gitignored, regenerable): `ilamb/MODELS_GDP/ED-ModelC-{base,gdp,gdp30,gdp50,gdp70}/`.

## Step 4 — JOINT re-fit (2026-07-24): gamma fit WITH all params
Added `GDP_TERM=1` to `optimize_modelC_coupled.py` (fits `gdp_gamma` jointly, applied
as a rate multiplier in `predict()`). Two matched TPE runs from the k2 warm start,
`DUMP_CLIMATE=1 SEASONAL_TRANSFORM=1 FUEL_AMP=1 MAG_BAND=1.3`, 1200 trials each:
`params.coupledE_gdp.json` (with) and `params.coupledE_nogdp.json` (control).

Official ILAMB (GFED5 burned area):

| model | Overall | Bias | RMSE(2x) | Seasonal | Spatial |
|---|---|---|---|---|---|
| k2 base (no human) | 0.6547 | 0.6952 | 0.4863 | 0.8004 | 0.8054 |
| bolt-on GDP g=0.15 | 0.6603 | 0.7020 | 0.4888 | 0.8004 | 0.8213 |
| refit, no GDP (control) | 0.6567 | 0.7093 | 0.4903 | 0.7882 | 0.8055 |
| **refit + GDP (joint)** | **0.6695** | 0.7418 | 0.5065 | 0.7744 | 0.8183 |

**+0.0148 over base to 0.6695** (above paper Model E 0.6646 and the BA leaderboard
top CLM6 0.6562, on coupling-ready dump climate). CLEAN ISOLATION: extra optimization
alone (control) buys +0.0020; the GDP term buys +0.0128 on top. The joint fit dropped
**fire_exp 7.09 -> 2.15** (retired the boreal-crushing concentration hack) and set
**gdp_gamma = 1.36**, i.e. the human term SUBSTITUTES for the ad-hoc concentration.

BUT regional check (Mha/yr vs GFED5) shows the aggregate gain hides a new problem:

| | Africa | S.Amer | N.Amer | Boreal | Europe | India+SEA | GLOBAL |
|---|---|---|---|---|---|---|---|
| GFED5 | 496 | 65 | 22 | 50 | 17 | 68 | 793 |
| k2 base | 393 | 153 | 39 | 4 | 16 | 93 | 766 |
| refit+GDP | 517 | 71 | 18 | 29 | 20 | **222** | 932 |

Fixes Africa (1.04x), the Amazon (1.09x), N.America (0.82x), lifts boreal (4->29), but
a single GLOBAL gdp_gamma=1.36 OVER-AMPLIFIES poor monsoon Asia (India+SEA 3.3x) and
pushes global to 1.18x. Same compensating-error lesson: one global wealth coefficient
cannot serve both African savanna (needs amplification) and poor monsoon Asia (must
not be amplified) — the cerrado-vs-Africa indistinguishability, now in the human term.
=> NOT coupling-ready as-is. The fix is a BIOME/REGION-specific gdp_gamma (George's
biome-specific direction, landing on the human predictor).

## Step 5 — REGIONAL (biome) gdp_gamma (2026-07-24): coupling-ready, best score
`scripts/add_gdp_regional.py`: gdp_gamma varies by region, SMOOTH-blended (Gaussian
SIGMA=4, no hard seams), fit by coordinate descent to minimize global area-weighted
RMSE with the global magnitude PINNED to GFED5. Base = the joint refit params
(`params.coupledE_gdp.json`, scalar gamma stripped). Fitted gamma:

    Africa 1.60   Europe 0.70   N.America 0.60   Boreal 0.50
    S.America 0.30   SEAsia 0.10   Australia 0.00   elsewhere 0.00

i.e. strong amplification in poor African savanna, ~none in poor monsoon Asia -- the
biome discrimination a single global coefficient could not make.

Official ILAMB (GFED5 burned area):

| model | Overall | Bias | RMSE(2x) | Seasonal | Spatial |
|---|---|---|---|---|---|
| scalar global gamma | 0.6695 | 0.7418 | 0.5065 | 0.7744 | 0.8183 |
| **REGIONAL gamma** | **0.6783** | 0.7531 | 0.5120 | 0.7745 | 0.8400 |

**0.6783, +0.0088 over the scalar, and NOT compensating error** -- every region within
0.52-1.14x of GFED5, India+SEA fixed (222 -> 58, 0.85x), global magnitude 1.0x. Best
model to date, above paper Model E (0.6646) and CLM6 (0.6562), on coupling-ready dump
climate with no seams. Regional Mha:

| | Africa | S.Amer | N.Amer | Boreal | Europe | India+SEA | GLOBAL |
|---|---|---|---|---|---|---|---|
| GFED5 | 496 | 65 | 22 | 50 | 17 | 68 | 793 |
| regional | 567 | 54 | 25 | 26 | 17 | 58 | 793 |

This is George's biome-specific direction realized on the human predictor, and it
satisfies Lei simultaneously (dump climate, no seams, forward-runnable GDP, regionally
faithful). Weak spot: boreal 0.52x (fuel-limited + GDP suppresses wealthy boreal).
Fitted gamma saved to `data_human/gdp_regional_gamma.json`.

## Step 6 — Population factor TESTED, does NOT help (2026-07-24)
`scripts/fire_vs_pop_partial.py`. Population density (people/km2, the ignition axis) as
a country term, with George's collinearity caution built in (nested regression + F-test):

| model | R2 |
|---|---|
| climate+veg | 0.480 |
| +GDP | 0.595 |
| +pop(hump) | 0.482 |
| +GDP+pop | 0.596 |

Population adds NOTHING beyond climate (dR2=+0.002, F=0.3, p=0.74) or beyond climate+GDP
(dR2=0.000, F=0.1, p=0.93). r(GDP, popdens) = -0.05 -> UNCORRELATED, so it is not
redundancy with GDP; national-average population density simply carries no country-level
burned-fraction signal. DECISION: no population term (would add noise). Caveat: this is
NATIONAL-AVERAGE density; the literature's humped pop-fire effect is grid-cell/local and
averaging washes it out, so a gridded/local pop field is a separate untested question.
Fig `fire_vs_pop_partial.png`, table `data_human/fire_vs_pop_country.csv`.

## Step 7 — Land use TESTED, real but redundant with GDP (2026-07-24)
`scripts/fire_vs_landuse_partial.py`. Land use = ED's LUH2 tiles in the dump
(area_frac_{ntrl,scnd,past}), coupling-legal. Block = {pasture, pasture^2, secondary}:

| model | R2 |
|---|---|
| climate+veg | 0.483 |
| +GDP | 0.596 |
| +landuse | 0.517 |
| +GDP+landuse | 0.610 |

Beyond CLIMATE: dR2=+0.034, F=3.6, p=0.014 (SIGNIFICANT -- unlike population, land use
carries a real pro-fire signal). Beyond CLIMATE+GDP: dR2=+0.015, F=1.9, p=0.13 (NOT
significant). r(GDP, pasture)=-0.18: poor countries have more pasture, so GDP already
proxies much of it. SECOND redundancy: the model's GPP driver is the area-weighted sum
over the ntrl/scnd/past tiles, so the fire model already ingests land use via GPP.
DECISION: no separate land-use term (p=0.13 + GPP double-counting). Fig
`fire_vs_landuse_partial.png`, table `data_human/fire_vs_landuse_country.csv`.

## Step 8 — fFire emissions on the regional-GDP BA (2026-07-24)
Retuned the combustion betas (`tune_combustion_params.py --ba-model ED-ModelC-gdpreg`,
`models/hurtt-betas-gdpreg/betas.gfed5.json`) on the new 0.6783 burned area, computed
fFire, scored official ILAMB vs 10 external TRENDY models:

| | old canonical | regional-GDP |
|---|---|---|
| Burned area | 0.6473 (#3) | 0.6783 (would be #1) |
| **Fire emissions** | 0.6534 (#4) | **0.6676 (#3, tied #2 w/ ELM-FATES; only CLM6 0.6913 ahead)** |

Both sides improved: fix where things burn, and the carbon follows.

MAGNITUDE FALSE ALARM (important, documented so nobody re-chases it): compute_emissions
printed "global total 3.4 PgC/yr (GFED ref ~2.0)" and I nearly magnitude-constrained it.
But **GFED5 fFire IN THIS BENCHMARK integrates to 3.38 PgC/yr** with the pipeline's area
method (not the literature ~2 -- likely a units/definition difference in the regridded
ref). The model's 3.41 PgC/yr is a **1.01x match to the reference it is scored against**.
No magnitude problem. Added an optional `FF_MAG_BAND` env to the tuner anyway (off by
default) and corrected the misleading "~2.0" comment in compute_emissions.py.

## HUMAN-FACTOR SURVEY COMPLETE
| factor | beyond climate+GDP | in model |
|---|---|---|
| GDP (economics) | -- (it is the reference) | YES: biome gamma, ILAMB 0.6783 |
| Land use (LUH2) | p=0.13, redundant w/ GDP + GPP | no |
| Population (ignition) | p=0.93, no signal | no |
| Roads / remote sensing | no long time series | Tier-2, paper-only |
GDP is the only human forcing with independent skill. The others were tested and
rejected with statistics (the rigor George asked for).

## NOT DONE / NEXT
- Boreal still low (0.52x): fuel-limited base + GDP suppresses wealthy boreal. Needs a
  base-model fuel fix, not a human term. Could protect it with a per-region RMSE weight.
- Combustion/fFire retune on the new BA, and a proper coupling handoff of this model to Lei.
- Consider keying gamma to a continuous vegetation/biome variable (true PFT) rather than
  region boxes -- the fully migration-safe form for prognostic runs.
- Per-biome split (George's 2nd sketch): fit the wealth term separately in savanna vs
  forest, check the human effect changes by vegetation type.
- Stack onto Model E / the smooth coupling model, not just the single-global base.
- Nothing promoted to canonical; this is an additive-term experiment.

## Step 9 — temperate-grassland (steppe) gap + grass-curing term (2026-07-25)
GFED burns the Kazakh/C-Asian steppe at 9.3%/yr; the model captured only 0.26x. Diagnosis
(`diag_steppe_terms.py`): NOT temperature (ign flat 0.96) and NOT the GDP term (steppe GDP
~$7.5k sits at the pivot, M=1.04; `test_steppe_gamma.py`: gamma 0.5->0 moves steppe 6.2->6.1).
It is the FUEL coupling -- the model ties fire to GPP (calibrated on high-GPP African savanna),
so low-productivity grass is under-fired; fire_exp amplifies it.

FIX: a grass-CURING pathway (`proto_curing.py`, wired into `optimize_modelC_coupled.py` as
`CURING=1`): cured fine grass burns efficiently regardless of productivity. Identify grassland
by PRECIP (not GPP -- the steppe's low GPP is what we must not penalize), gate out forest (AGB)
and desert (precip) and wet (dryness), boost where fuel is fine/sparse (inverse-GPP). Additive
to the rate. Physically fixes the steppe (2->11) and Australia (11->45).

KEY FINDING (third demonstration of the paper's thesis): curing vs aggregate ILAMB, gammas
fixed, `sweep_regcure.py`:

| cure_k | steppe | Australia | Africa | ILAMB |
|---|---|---|---|---|
| 0.00 | 2 | 11 | 612 (1.23x) | 0.6902 |
| 0.10 | 8.5 | 36 | 593 | 0.6708 |
| 0.20 | 10 | 42 | 588 | 0.6576 |
| 0.35 | 11 | 45 | 585 | 0.6490 |

Every increment of curing that improves regional fidelity LOWERS aggregate ILAMB,
monotonically. ILAMB scores the compensating-error model (cure_k=0: Africa 1.23x, steppe
dead) HIGHEST. The optimizer, maximizing aggregate ILAMB, chose a weak cure_k=0.045
(`params.coupledE_cure.json`, joint fit 0.6769) that barely moved the steppe -- because the
aggregate metric gives no credit for small regions.

DECISION (two models, two purposes):
- Paper / ILAMB leaderboard: the regional-GDP model (0.6783), balanced and honest.
- Coupled run (regional carbon must be right): moderate curing cure_k~0.10-0.15 (steppe/
  Australia much better, ILAMB ~0.68), a documented and defensible cost.
Scripts: `diag_steppe_terms.py`, `test_steppe_gamma.py`, `proto_curing.py`,
`assemble_regional_cure.py`, `sweep_regcure.py`, `map_gdpreg_ba*.py`.
