"""Regrid Lei's HYDE population density onto the fire model's grid.

    C:/Users/owusu/miniforge3/envs/edfire/python.exe scripts/make_population_driver.py

Reads  D:/population_density_0p5_1700_2025.nc
Writes data_human/pop_density_1deg_2001_2016.npy   (192, 180, 360) float32, capita per km2
       data_human/pop_density_1deg_meta.json

Why this file and not the GDP grid. George disallowed GDP for the coupled runs because it
does not reach 1850. This does, back to 1700, it is gridded rather than national, and TRENDY
already uses it, so its provenance is not arguable. It is the natural replacement.

It also lets us redo a test we previously got wrong. Population was tested in July and showed
no signal, p equals 0.93, but that used NATIONAL AVERAGE density. Our own note recorded at the
time that the humped population-fire relationship in the literature is local and that
averaging it over a country destroys it. This is the gridded version.

Three transformations are needed and each one can silently ruin the field.
  latitude runs north to south in the source and south to north in the model grid
  the source is 0.5 degree and the model computes on 1 degree
  the source is annual and the model steps monthly

Density is per unit area, so coarsening averages it with the cosine weight of each subcell
rather than summing. Summing would turn a density into something with no meaning.
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC = Path("D:/population_density_0p5_1700_2025.nc")
OUT = REPO / "data_human" / "pop_density_1deg_2001_2016.npy"
META = REPO / "data_human" / "pop_density_1deg_meta.json"
Y0, YF = 2001, 2016

ds = xr.open_dataset(SRC)
years = ds.year.values.astype(int)
sel = np.where((years >= Y0) & (years <= YF))[0]
assert len(sel) == YF - Y0 + 1, f"expected {YF - Y0 + 1} years, found {len(sel)}"

a = np.nan_to_num(ds.population_density.isel(time=sel).values.astype(np.float64))
lat = ds.lat.values.astype(np.float64)
ds.close()

# the source runs 89.75 down to -89.75, the model grid runs south to north
if lat[0] > lat[-1]:
    a = a[:, ::-1, :]
    lat = lat[::-1]
assert abs(lat[0] + 89.75) < 1e-6, f"unexpected first latitude {lat[0]}"

# 0.5 -> 1 degree. Density is an intensive quantity, so the coarse value is the area-weighted
# MEAN of the four subcells, not their sum.
w = np.cos(np.deg2rad(lat)).reshape(180, 2)
blocks = a.reshape(len(sel), 180, 2, 360, 2)
num = (blocks * w[None, :, :, None, None]).sum(axis=(2, 4))
den = (w.sum(axis=1)[None, :, None] * 2.0)
ann = (num / den).astype(np.float32)

# annual to monthly, held constant within a year. Population does not have a seasonal cycle
# and interpolating one in would be inventing signal.
monthly = np.repeat(ann, 12, axis=0)
assert monthly.shape == (12 * len(sel), 180, 360), monthly.shape

OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, monthly)
land = monthly[0] > 0
META.write_text(json.dumps({
    "source": str(SRC),
    "source_institution": "PBL Netherlands Environmental Assessment Agency and Utrecht University",
    "dataset": "HYDE, as supplied for GCB2026",
    "units": "capita per km2",
    "years": [Y0, YF],
    "shape": list(monthly.shape),
    "grid": "1 degree, lat ascending -89.5 to 89.5, lon -179.5 to 179.5",
    "note": "annual values held constant within each year",
}, indent=1), encoding="utf-8")

print(f"wrote {OUT.name}  {monthly.shape}")
print(f"  populated cells {int(land.sum())} of {land.size}")
for y in (2001, 2008, 2016):
    i = (y - Y0) * 12
    v = monthly[i][land]
    print(f"  {y}  median {np.median(v):7.2f}   p99 {np.percentile(v, 99):9.1f}   "
          f"max {v.max():9.1f}  capita/km2")
