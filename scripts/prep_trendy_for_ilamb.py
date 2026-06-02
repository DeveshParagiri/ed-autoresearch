"""
prep_trendy_for_ilamb.py
Normalize every TRENDY v14 burntArea NetCDF into MODELS/<NAME>/burntArea.nc so
that ilamb-run can compare them against GFED4.1s alongside our ED-Model A/B/C.

Output NC conventions (what ILAMB expects):
  - variable: burntArea (fraction, 0..1)
  - dims:     (time, lat, lon) with lat S->N, lon -180..180
  - time:     cftime NoLeap monthly mid-month for 2001-01 through 2016-12 (192)
  - units on burntArea: "1"   (fraction)

Annual-only TRENDY models are repeated to monthly.
1-D/region-summed files (CARDAMOM) are skipped.

Run:
    python scripts/prep_trendy_for_ilamb.py
"""
import warnings
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

REPO      = Path(__file__).resolve().parents[1]
TRENDY_DIR= Path(r"C:\Users\owusu\OneDrive\Documents\UMD STUFF\ED AUTORESEARCH\fire_autoresearch\data\raw\trendy")
MODELS_DIR= REPO / "models"

YEARS    = list(range(2001, 2017))
N_MONTHS = 192
TIMES    = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
TIME_UNITS = "days since 2001-01-01 00:00:00"

# Monthly time bounds (first day of month -> first day of next month)
def _build_time_bounds():
    bnds_lo = [cftime.DatetimeNoLeap(y, m, 1) for y in YEARS for m in range(1, 13)]
    bnds_hi = []
    for y in YEARS:
        for m in range(1, 13):
            ny = y + (1 if m == 12 else 0)
            nm = 1 if m == 12 else m + 1
            bnds_hi.append(cftime.DatetimeNoLeap(ny, nm, 1))
    return np.array(list(zip(bnds_lo, bnds_hi)))
TIME_BNDS = _build_time_bounds()

# filename -> friendly model folder name
MODELS = {
    "CLASSIC_S3_burntArea.nc":   "CLASSIC",
    "EDv3_S3_burntArea.nc":      "EDv3",
    "CLM6.0_S3_burntArea.nc":    "CLM6",
    "VISIT_S3_burntArea.nc":     "VISIT",
    "CLM-FATES_S3_burntArea.nc": "CLM-FATES",
    "ELM-FATES_S3_burntArea.nc": "ELM-FATES",
    "E3SM_S3_burntArea.nc":      "E3SM",
    "JSBACH_S3_burntArea.nc":    "JSBACH",
    "SDGVM_S3_burntArea.nc":     "SDGVM",
}

def decode_time_index(ds):
    """Return the 192 indices (2001-01..2016-12) into the file's time axis.
       Handles monthly, annual, and mixed calendars."""
    t   = ds["time"] if "time" in ds.variables or "time" in ds.coords else ds["Time"]
    u   = t.attrs.get("units", "") or ""
    cal = t.attrs.get("calendar", "") or "noleap"
    vals= np.asarray(t.values, dtype=np.float64)
    n   = len(vals)

    # Decimal-year format (e.g. ELM-FATES: 1701.0, 1701.0833, ...)
    if not u and vals.min() > 1500 and vals.max() < 2200:
        years  = np.floor(vals).astype(int)
        months = np.clip(np.round((vals - years) * 12).astype(int) + 1, 1, 12)
    else:
        # Try CF decoding
        try:
            dts = cftime.num2date(vals, u, calendar=cal)
            years  = np.array([d.year  for d in dts])
            months = np.array([d.month for d in dts])
        except Exception:
            # Fallback heuristics
            if "months since" in u:
                base_year = int(u.split("since")[1].strip()[:4])
                years  = base_year + (vals.astype(int) // 12)
                months = (vals.astype(int) % 12) + 1
            else:
                return None, None

    # Detect annual cadence (unique year counts match n, or step ~365d)
    n_years = len(np.unique(years))
    is_annual = (n_years == n)

    if is_annual:
        # Map year -> index, repeat 12×
        yr2idx = {int(y): i for i, y in enumerate(years)}
        idx = []
        for y in YEARS:
            if y not in yr2idx:
                return None, None
            idx.extend([yr2idx[y]] * 12)
        return np.array(idx), "annual"

    # Monthly: find each (Y, M); if missing, fall back to nearest month in file
    ym_val = years * 12 + (months - 1)
    idx = []
    missing = 0
    for y in YEARS:
        for m in range(1, 13):
            hit = np.where((years == y) & (months == m))[0]
            if len(hit) == 0:
                target = y * 12 + (m - 1)
                near  = int(np.argmin(np.abs(ym_val - target)))
                idx.append(near)
                missing += 1
            else:
                idx.append(int(hit[0]))
    if missing:
        print(f"    (filled {missing} missing months with nearest available)")
    return np.array(idx), "monthly"


def normalize_latlon(da):
    """Ensure lat/lon names, lat S->N, lon -180..180."""
    rename = {}
    for cand in ["latitude", "LAT", "Lat"]:
        if cand in da.dims or cand in da.coords: rename[cand] = "lat"
    for cand in ["longitude", "LON", "Lon"]:
        if cand in da.dims or cand in da.coords: rename[cand] = "lon"
    if rename: da = da.rename(rename)

    # Some E3SM-style files have lat/lon as non-dim coords
    if "lat" not in da.dims and "latitude" in da.coords:
        da = da.rename({"latitude": "lat"})
    if "lon" not in da.dims and "longitude" in da.coords:
        da = da.rename({"longitude": "lon"})

    # Flip lat if N->S
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))

    # Roll lon 0..360 → -180..180
    lon = da["lon"].values
    if lon.max() > 180.5:
        da = da.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    return da


def convert_to_fraction(arr, units):
    u = (units or "").strip().lower()
    if u in ("%", "percent"):           return arr / 100.0
    if u in ("fraction", "1", "-", ""): return arr
    if u in ("m2/m2", "m^2/m^2"):       return arr
    return arr   # hopeful default


def process(fname, outname):
    fp = TRENDY_DIR / fname
    if not fp.exists():
        print(f"  [skip] {fname} missing"); return False

    ds = xr.open_dataset(fp, decode_times=False)
    if "burntArea" not in ds.data_vars:
        print(f"  [skip] {fname}: no burntArea variable"); ds.close(); return False

    da = ds["burntArea"]
    if da.ndim != 3:
        print(f"  [skip] {fname}: not 3-D (shape {da.shape})"); ds.close(); return False

    idx, cadence = decode_time_index(ds)
    if idx is None:
        print(f"  [skip] {fname}: could not decode time"); ds.close(); return False

    da = da.isel(time=idx)
    da = normalize_latlon(da)

    arr = da.values.astype(np.float32)
    arr = np.where(np.isfinite(arr) & (arr >= 0), arr, np.nan)
    arr = convert_to_fraction(arr, da.attrs.get("units", ""))

    lat = da["lat"].values
    lon = da["lon"].values

    out_dir = MODELS_DIR / outname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / "burntArea.nc"

    out = xr.Dataset(
        {"burntArea":    (("time", "lat", "lon"), arr,
                          {"units": "1", "long_name": "Burned Area Fraction",
                           "standard_name": "burned_area_fraction"}),
         "time_bounds":  (("time", "nv"), TIME_BNDS)},
        coords={"time": ("time", TIMES,
                          {"bounds": "time_bounds", "standard_name": "time", "axis": "T"}),
                "lat":  ("lat",  lat.astype(np.float64)),
                "lon":  ("lon",  lon.astype(np.float64))},
        attrs={"title": f"TRENDY v14 {outname} burntArea (2001-2016, remapped for ILAMB)",
               "source_file": fname,
               "cadence": cadence,
               "Conventions": "CF-1.7"})
    enc = {"burntArea":   {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":        {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = out_nc.with_suffix(".nc.tmp")
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    import os; os.replace(tmp, out_nc)
    ds.close()
    print(f"  [ok]   {outname:10s}  {arr.shape}  cadence={cadence}  grid={lat.size}x{lon.size}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("prep_trendy_for_ilamb.py")
    print("=" * 60)
    ok = 0
    for src, name in MODELS.items():
        if process(src, name):
            ok += 1
    print(f"\n{ok}/{len(MODELS)} TRENDY models normalized into {MODELS_DIR}")
