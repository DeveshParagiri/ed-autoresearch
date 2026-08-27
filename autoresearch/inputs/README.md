# Inputs

All predictors cover January 2001 through December 2016 on the same south-to-north global 1-degree grid for the model to potentially consume.

## Coupled-use constraint

An offline predictor is not automatically valid for an 1850-start coupled ED run. The core precipitation, temperature, and `dryness` fields may be supplied by the coupled meteorological forcing; the `ed.nc` state variables come from ED itself; LUH2 supplies historical land use; and the lightning field may only be used as a fixed climatology. Until a complete 1850 forcing is installed and validated, `model.py` must not use `wind_speed_mean`, `vapor_pressure_deficit_mean`, `maximum_consecutive_dry_days`, or `wet_day_fraction`. The present wind and VPD fields come from TerraClimate, while dry-spell length and wet-day fraction come from NOAA CPC daily precipitation, so none of the installed fields covers 1850. The static GPW `population_density` field is also not a historical 1850 population forcing and must not be used in a coupled-ready candidate.

CRUJRA can supply wind and the temperature, humidity, pressure, and precipitation needed to derive VPD and dry-spell diagnostics consistently, but its record begins in 1901. It therefore offers a 1901 pathway, not an 1850 solution. Any replacement must be provenance-checked for temporal coverage, units, aggregation, regridding, and behavior against the modern prepared fields before those predictors become coupled-valid.

NOAA 20CRv3 is a viable pre-1901 bridge for VPD: its daily 2-m temperature and relative-humidity files are available in 1850, and the physical derivation `VPD = es(T) * (1 - RH/100)` produced finite daily values for all 365 days in a downloaded 1850 smoke test. In a 2001 overlap check, monthly 20CR VPD had area-weighted spatiotemporal correlation 0.959 with the installed TerraClimate field and monthly spatial correlations from 0.953 to 0.973 without empirical scaling. This validates the source route, not the installed input: the complete 1850-2016 bridge and transition still need to be prepared and checked before `vapor_pressure_deficit_mean` may enter `model.py`.

Daily fire-weather duration is a more informative candidate than monthly mean VPD: over 2001-2015, the fraction of days above 1 kPa retains partial correlation 0.211 with fire residuals after controlling for the active model and monthly VPD. NOAA 20CRv3 provides this diagnostic through 2015; NCEP/NCAR Reanalysis 1 supplies 2016 from the same daily temperature-and-humidity derivation. Their 2001-2015 overlap has area-weighted correlation 0.876, with every yearly correlation between 0.864 and 0.888. One global overlap calibration, `20CR-like fraction = 0.081331 + 0.968217 * NCEP fraction`, makes the 2016 transition finite and mean-consistent. This validates a procurement route only. The complete 1850-2016 field is not installed, so neither daily VPD duration nor any derivative may enter `model.py` yet.

Daily precipitation-event diagnostics also have a validated historical route but remain uninstalled. NOAA 20CRv3 supplies daily precipitation from 1850 through 2015 and NCEP/NCAR Reanalysis 1 supplies 2016. In the 2001 overlap, 20CR versus CPC correlations are 0.764 for wet-day fraction and 0.808 for maximum consecutive dry days; 20CR versus NCEP correlations are 0.856 and 0.842. A whole-cell held-out online model reaches only 0.7316 with these fields, so full procurement is currently not justified. Until an 1850-2016 bridge is actually prepared and validated, neither event diagnostic may enter `model.py`.

The packaged `annual_precipitation` is the completed calendar-year total repeated in every month, so it exposes later months to an earlier timestep. It must not enter a prefix-causal model directly. Derive any annual-scale water state from current and preceding `monthly_precipitation`, as the canonical model does with a causal twelve-month EMA.

The prepared LUH2 aggregates are not one mutually exclusive compositional vector. On the fixed evaluator land mask, `luh2_secondary_fraction` has mean 0.947 and is at least 0.5 in 94.8% of cells, while the median sum of the six LUH2 aggregates is 1.72. The fields may be used as independently defined land-use signals, but must not be summed or interpreted as exclusive cover shares; in particular, `luh2_secondary_fraction` must not define a dominant-secondary or total-natural partition without reprocessing the source states.

The frozen non-GPP fields in `ed.nc` come from the published EDv3 transient that prescribed GFED4 burned area, so their offline values are descendants of an observed fire product. GPP comes from a separate coupled-ED dump whose fire configuration and restart ancestry are unresolved. These are legitimate prognostic state variables when produced online by a clean coupled ED run, but the packaged fields do not provide leakage-free validation. That requires exact source, configuration and restart provenance with endogenous fire, plus pre-fire fuel, moisture and snow-state exports.

| File            | Variables                                                                                                                                                                          | Source and treatment                                             |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `climate.nc`    | `dryness`, `annual_precipitation`, `monthly_precipitation`, `air_temperature`, `vapor_pressure_deficit_mean`, `wind_speed_mean`, `wet_day_fraction`, `maximum_consecutive_dry_days` | Existing CRUJRA v3.5 fields; TerraClimate VPD and wind; wet-day fraction and within-month dry-spell length from NOAA CPC daily precipitation |
| `ed.nc`         | `gpp`, `aboveground_biomass`, `soil_carbon`, `leaf_area_index`, `natural_canopy_height`, `secondary_canopy_height`, `natural_vegetation_fraction`, `secondary_vegetation_fraction` | GPP from Lei's coupled-ED dump; other fields from frozen EDv3    |
| `luh2.nc`       | `luh2_cropland_fraction`, `luh2_pasture_fraction`, `luh2_rangeland_fraction`, `luh2_primary_fraction`, `luh2_secondary_fraction`, `luh2_urban_fraction`                            | LUH2-GCB2026; latitude corrected; annual values repeated monthly |
| `population.nc` | `population_density`                                                                                                                                                               | GPW v4.11; latest prior census epoch repeated monthly            |
| `lightning.nc`  | `lightning_flash_rate`                                                                                                                                                             | LIS/OTD HRMC v2.3 monthly climatology repeated yearly            |

Set `INPUTS` in `model.py` to the exact variable names the model uses:

```python
INPUTS = ("dryness", "gpp")
```

Add/remove a name to load/unload a predictor.
