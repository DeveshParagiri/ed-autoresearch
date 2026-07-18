# ED coupling

The offline model interprets its output as an annual fire disturbance rate, then applies ED’s saturating monthly transform so the NetCDF is comparable to what coupled ED would write to TRENDY-format burned area.

## For bit-exact transfer into coupled ED

1. Implement the Model C / E driver response in ED’s fire update (or an equivalent).  
2. Raise `fire_max_disturbance_rate` if savanna rates exceed the default cap.  
3. Use monthly patch dynamics where that is intended (`PATCH_FREQ`).  
4. For Model E’s continental branch, tag sites by region.

Coupled GPP–fire feedback will require recalibration; the structure is meant to transfer. No in-repo C++ patch is maintained; coupling is a separate ED-side port once offline params are frozen.
