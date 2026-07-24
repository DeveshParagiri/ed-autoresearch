"""Produce the two gridded NetCDF inputs ED loads for the GDP human term:
  gdp_pcap.nc   -- GDP per capita (current US$), 0.5deg
  gdp_gamma.nc  -- smooth per-region suppression strength, 0.5deg

Grid matches ED's dump (which ED itself wrote): 0.5deg, lat ASCENDING -89.75..89.75,
lon -179.75..179.75, so array[globY_][globX_] lines up with read_site_data.cc:1657-1661.
Coordinate variables are included so orientation is self-describing -- if ED's globY_=0
is at +90 instead of -90, flip lat (and the data rows) and nothing else changes.

Values reproduce the scored 0.6783 model's fields at ED's native 0.5deg (the fit ran at
1deg with gamma smoothed at 4 deg; here gamma is smoothed at 4 deg = 8 cells on 0.5deg).
Constants (unchanged): w0=9242 US$/cap pivot, s=1.0659 scale, clip [0.15,6.0], applied
inside ED per COUPLING_SPEC_for_Lei.md.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import shapefile, shapely
from shapely.geometry import shape
from shapely import STRtree
from scipy.ndimage import distance_transform_edt, gaussian_filter

REPO = Path("."); OUT = REPO / "data_human" / "coupling_inputs"; OUT.mkdir(parents=True, exist_ok=True)
SP = "/private/tmp/claude-501/-Volumes-RICHIE---T7-FIRE-OFFLINE/c532496d-370d-40db-969d-b2a6dd880db0/scratchpad/humandata"
NE = f"{SP}/ne50/ne_50m_admin_0_countries.shp"

# ED / dump grid (0.5deg, lat ascending)
lat = -89.75 + 0.5 * np.arange(360)
lon = -179.75 + 0.5 * np.arange(720)
LON, LAT = np.meshgrid(lon, lat)

REG = json.load(open(REPO / "data_human" / "gdp_regional_gamma.json"))
GVEC = REG["per_region_gamma"]; SIGMA_DEG = REG["sigma"]           # 4.0
SIGMA_CELLS = SIGMA_DEG / 0.5                                       # 8 cells on 0.5deg
REGION_BOX = {"Africa": (-20, 52, -36, 18), "Boreal": (40, 180, 48, 78),
              "S.America": (-82, -34, -56, 14), "SEAsia": (60, 150, -11, 30),
              "Europe": (-12, 40, 36, 72), "N.America": (-168, -52, 14, 74),
              "Australia": (112, 154, -44, -10)}

# ---- rasterize countries -> per-cell GDP per capita, fill nearest ----
tree = STRtree(shapely.points(LON.ravel(), LAT.ravel()))
cidx = np.full(LAT.size, -1, np.int64)
sf = shapefile.Reader(NE); col = {f[0]: i for i, f in enumerate(sf.fields[1:])}
isos = []
for k, (rec, shp) in enumerate(zip(sf.records(), sf.shapes())):
    cidx[tree.query(shapely.make_valid(shape(shp.__geo_interface__)), predicate="contains")] = k
    isos.append(rec[col["ISO_A3_EH"]])
cidx = cidx.reshape(LAT.shape)

f = glob.glob(f"{SP}/gdp/API_*.csv")[0]; wbd = pd.read_csv(f, skiprows=4)
yrs = [str(y) for y in range(2001, 2021) if str(y) in wbd.columns]
wbd["g"] = wbd[yrs].mean(axis=1); g_of = dict(zip(wbd["Country Code"], wbd["g"]))
gdp = np.full(LAT.shape, np.nan)
for k, iso in enumerate(isos):
    v = g_of.get(iso)
    if v is not None and np.isfinite(v) and v > 0:
        gdp[cidx == k] = v
miss = ~np.isfinite(gdp)                                            # fill every cell from nearest country
gdp = gdp[tuple(distance_transform_edt(miss, return_distances=False, return_indices=True))]
gdp = np.clip(gdp, 50.0, None).astype(np.float32)

# ---- per-region gamma field, smooth-blended ----
region_of = np.full(LAT.shape, "fb", dtype=object); assigned = np.zeros(LAT.shape, bool)
for r, b in REGION_BOX.items():
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3]) & ~assigned
    region_of[box] = r; assigned |= box
gfield = np.zeros(LAT.shape)
for r in list(REGION_BOX) + ["fb"]:
    gfield[region_of == r] = GVEC.get(r, 0.0)
gamma = gaussian_filter(gfield, SIGMA_CELLS, mode="nearest").astype(np.float32)

# ---- write NetCDFs (CF, coord vars for self-describing orientation) ----
def write(arr, var, units, longname, fname, extra):
    ds = xr.Dataset({var: (("lat", "lon"), arr, {"units": units, "long_name": longname})},
                    coords={"lat": ("lat", lat.astype(np.float64), {"units": "degrees_north"}),
                            "lon": ("lon", lon.astype(np.float64), {"units": "degrees_east"})},
                    attrs={"Conventions": "CF-1.7", "title": longname,
                           "grid": "0.5deg, lat ascending -89.75..89.75, lon -179.75..179.75 "
                                   "(ED dump grid; array[globY_][globX_])", **extra})
    p = OUT / fname
    ds.to_netcdf(p, encoding={var: {"zlib": True, "complevel": 4, "_FillValue": 1e20}},
                 format="NETCDF4_CLASSIC")
    print(f"[write] {p}  {var} range [{float(arr.min()):.3g}, {float(arr.max()):.3g}]")

write(gdp, "gdp_pcap", "US$", "GDP per capita current US$ (World Bank mean 2001-2020)",
      "gdp_pcap.nc", {"pivot_w0_usd": 9242.0, "note": "static present-day; use SSP time series for forward runs"})
write(gamma, "gdp_gamma", "1", "GDP suppression strength gamma (smooth-blended by region)",
      "gdp_gamma.nc", {"sigma_deg": SIGMA_DEG, "per_region": json.dumps(GVEC),
                       "apply": "M=clip(10^(gamma*(log10(9242)-log10(gdp_pcap))),0.15,6); rate*=1.0659*M"})
print("[done] ED-ready GDP inputs in", OUT)
