# Inputs

All predictors cover January 2001 through December 2016 on the same south-to-north global 1-degree grid for the model to potentially consume.

## Coupled-use constraint

An offline predictor is not automatically valid for an 1850-start coupled ED run. The core precipitation, temperature, and `dryness` fields may be supplied by the coupled meteorological forcing; the `ed.nc` state variables come from ED itself; LUH2 supplies historical land use; and the lightning field may only be used as a fixed climatology. Until a complete 1850 forcing is installed and validated, `model.py` must not use `wind_speed_mean`, `vapor_pressure_deficit_mean`, `maximum_consecutive_dry_days`, or `wet_day_fraction`. The present wind and VPD fields come from TerraClimate, while dry-spell length and wet-day fraction come from NOAA CPC daily precipitation, so none of the installed fields covers 1850. The static GPW `population_density` field is also not a historical 1850 population forcing and must not be used in a coupled-ready candidate.

CRUJRA can supply wind and the temperature, humidity, pressure, and precipitation needed to derive VPD and dry-spell diagnostics consistently, but its record begins in 1901. It therefore offers a 1901 pathway, not an 1850 solution. Any replacement must be provenance-checked for temporal coverage, units, aggregation, regridding, and behavior against the modern prepared fields before those predictors become coupled-valid.

NOAA 20CRv3 is a viable pre-1901 bridge for VPD: its daily 2-m temperature and relative-humidity files are available in 1850, and the physical derivation `VPD = es(T) * (1 - RH/100)` produced finite daily values for all 365 days in a downloaded 1850 smoke test. In a 2001 overlap check, monthly 20CR VPD had area-weighted spatiotemporal correlation 0.959 with the installed TerraClimate field and monthly spatial correlations from 0.953 to 0.973 without empirical scaling. This validates the source route, not the installed input: the complete 1850-2016 bridge and transition still need to be prepared and checked before `vapor_pressure_deficit_mean` may enter `model.py`.

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
