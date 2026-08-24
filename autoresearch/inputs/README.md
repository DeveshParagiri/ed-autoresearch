# Inputs

All predictors cover January 2001 through December 2016 on the same south-to-north global 1-degree grid for the model to potentially consume.

| File            | Variables                                                                                                                                                                          | Source and treatment                                             |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `climate.nc`    | `dryness`, `annual_precipitation`, `monthly_precipitation`, `air_temperature`                                                                                                      | CRUJRA v3.5                                                      |
| `ed.nc`         | `gpp`, `aboveground_biomass`, `soil_carbon`, `leaf_area_index`, `natural_canopy_height`, `secondary_canopy_height`, `natural_vegetation_fraction`, `secondary_vegetation_fraction` | GPP from Lei's coupled-ED dump; other fields from frozen EDv3    |
| `luh2.nc`       | `luh2_cropland_fraction`, `luh2_pasture_fraction`, `luh2_rangeland_fraction`, `luh2_primary_fraction`, `luh2_secondary_fraction`, `luh2_urban_fraction`                            | LUH2-GCB2026; latitude corrected; annual values repeated monthly |
| `population.nc` | `population_density`                                                                                                                                                               | GPW v4.11; latest prior census epoch repeated monthly            |
| `lightning.nc`  | `lightning_flash_rate`                                                                                                                                                             | LIS/OTD HRMC v2.3 monthly climatology repeated yearly            |

Set `INPUTS` in `model.py` to the exact variable names the model uses:

```python
INPUTS = ("dryness", "gpp")
```

Add/remove a name to load/unload a predictor.
