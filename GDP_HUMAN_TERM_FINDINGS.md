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

## NOT DONE / NEXT
- Joint re-fit: let the optimizer fit gamma WITH all params (not bolt-on to a frozen
  base) — usually finds a bit more. Add `gdp_gamma` to `optimize_modelC_coupled.py`.
- Per-biome split (George's 2nd sketch): fit the wealth term separately in savanna vs
  forest, check the human effect changes by vegetation type.
- Stack onto Model E / the smooth coupling model, not just the single-global base.
- Nothing promoted to canonical; this is an additive-term experiment.
