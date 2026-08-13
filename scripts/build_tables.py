"""Build the paper's tables from the scored outputs, numbered as the topic-sentence outline requires.

    python scripts/build_tables.py

Writes paper_gmd/TABLES.md with
  Table 1        global performance, versions down, burned area and fire carbon emissions
                 against GFED5, with the ILAMB statistics
  Tables 2 to 8  the same table for each of the seven fitting regions, same row headings and
                 same column headings as Table 1
  Table 9        the model versions and the attributes that define them, authored not computed
  Table 10       performance in the coupled ED model, which does not exist yet

Everything except Table 9 is computed from the model output and the official ILAMB databases, so the
tables cannot drift from the figures. Emissions are reported wherever burned area is reported, because
the paper's claim that correcting burned area does not correct emissions can only be shown that way.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper_gmd" / "TABLES.md"
BA_DIR = REPO / "paper_gmd" / "models"
FF_DIR = REPO / "paper_gmd" / "models_ffire_paper"
BA_REF = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
FF_REF = REPO / "ilamb_ref_official/DATA/fFire/GFED5/fFire.nc"
BA_DB = REPO / "paper_gmd/scoring/ba_regions/scalar_database.csv"
FF_DB = REPO / "paper_gmd/scoring/ffire_regions/scalar_database.csv"

VERSIONS = [("ED-stock", "Original formulation"), ("C", "C"), ("D", "D"), ("E-clean", "E"),
            ("F", "F *"), ("G7", "G"), ("H", "H"), ("I", "I")]

# the seven regions the model is fitted on, which are the paper's regions
REGIONS = [("africa", "Africa"), ("samerica", "South America"), ("namerica", "North America"),
           ("boreal", "Boreal Eurasia"), ("seasia", "Southeast Asia"),
           ("australia", "Australia"), ("europe", "Europe")]
BOX = {"africa": (-20, 52, -36, 18), "samerica": (-82, -34, -56, 14),
       "namerica": (-168, -52, 14, 74), "boreal": (40, 180, 48, 78),
       "seasia": (60, 150, -11, 30), "australia": (112, 154, -44, -10),
       "europe": (-12, 40, 36, 72)}

R = 6.371e6
SEC_PER_YR = 3.1536e7
F1_THRESH = 0.001            # 0.1 percent of cell area per year

HEADER = ("| Version | Burned area (Mha yr-1) | GFED5 | Bias | RMSE | Seasonal | Spatial | Overall | "
          "F1 | Emissions (Pg C yr-1) | GFED5 | Overall |")
RULE = "|---" * 12 + "|"

# Table 9 is authored. Its columns are an editorial choice and its cells are definitions rather than
# measurements. Region counts are deliberately omitted, per Richard 08/13, so the column reads
# Continental and the detail belongs in the methods.
TABLE9 = """## Table 9. Model versions and the attributes that define them

| Model | Target | Statistic optimized | Spatial resolution of parameterization | Temporal resolution of optimization | Temporal resolution of evaluation | Human factor | Vegetation dependence | PFT |
|---|---|---|---|---|---|---|---|---|
| Original formulation | GFED5 | not fit by us | Global | n/a | monthly + annual | - | ED native scheme | - |
| A | GFED4.1s | S_overall | Global | Monthly | monthly + annual | - | none | - |
| B | GFED4.1s | S_overall | Global | Monthly | monthly + annual | - | none | - |
| C | GFED5 | S_overall | Global | Monthly | monthly + annual | - | none | - |
| D | GFED5 | spatial-Taylor | Global | Annual | monthly + annual | - | none | - |
| E | GFED5 | spatial-Taylor | Continental | Annual | monthly + annual | - | biomass gate + fuel amplitude | - |
| F * | GFED5 | S_overall | Global base + regional GDP coefficient | Annual | monthly + annual | GDP per capita | biomass gate + fuel amplitude | - |
| G | GFED5 | S_overall | Continental | Monthly | monthly + annual | - | none | - |
| H | GFED5 | S_overall | Global | Monthly | monthly + annual | GDP per capita | none | - |
| I | GFED5 | S_overall | Continental | Monthly | monthly + annual | - | biomass gate | - |
| J | GFED5 | S_overall | [pending] | Monthly | monthly + annual | - | [pending] | PFT-specific |

A letter, once assigned, is fixed to the attributes it names, and any new combination takes a new
letter. Model J is committed and not yet run, so its row is reserved rather than filled.

Models A and B were fitted against GFED4.1s on a driver pipeline later found to be faulty and their
code has been removed, so they carry no entry in Tables 1 to 8. They are shown here as development
history.
"""


def cell_area(lat, lon):
    dlon = np.deg2rad(abs(lon[1] - lon[0]))
    h = abs(lat[1] - lat[0]) / 2.0
    al = (R ** 2) * dlon * (np.sin(np.deg2rad(lat + h)) - np.sin(np.deg2rad(lat - h)))
    return np.abs(al)[:, None] * np.ones((1, len(lon)))


def annual_ba(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    a = np.nan_to_num(da.values[sel].astype(np.float64))
    if da.attrs.get("units", "1") in ("%", "percent"):
        a = a / 100.0
    ann = a.reshape(sel.sum() // 12, 12, *a.shape[1:]).sum(1).mean(0)
    lat, lon = da.lat.values, da.lon.values
    da.close()
    return ann, lat, lon


def annual_ff(path):
    d = xr.open_dataset(path)
    a = d["fFire"]
    lat = a["lat"].values if "lat" in a.coords else a["latitude"].values
    lon = a["lon"].values if "lon" in a.coords else a["longitude"].values
    yrs = np.array([int(str(t)[:4]) for t in a["time"].values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    x = np.nan_to_num(a.values[sel].astype(np.float64)).mean(0)
    d.close()
    return x, lat, lon


def mask_of(key, lat, lon):
    lo0, lo1, la0, la1 = BOX[key]
    return ((lon >= lo0) & (lon <= lo1))[None, :] & ((lat >= la0) & (lat <= la1))[:, None]


def scores(db):
    t = defaultdict(dict)
    with open(db) as f:
        for r in csv.DictReader(f):
            try:
                t[(r["Model"], r["Region"])][r["ScalarName"]] = float(r["Data"])
            except ValueError:
                pass
    return t


def main():
    ba_s, ff_s = scores(BA_DB), scores(FF_DB)

    ref_ba, lat, lon = annual_ba(BA_REF)
    area = cell_area(lat, lon)
    ref_ff, flat, flon = annual_ff(FF_REF)
    farea = cell_area(flat, flon)
    ref_burn = ref_ba > F1_THRESH

    ba, ff = {}, {}
    for key, _ in VERSIONS:
        ba[key] = annual_ba(BA_DIR / key / "burntArea.nc")[0]
        ff[key] = annual_ff(FF_DIR / key / "fFire.nc")[0]

    def mha(f, m=None):
        return float((f * (area if m is None else area * m)).sum()) / 1e10

    def pgc(f, m=None):
        return float((f * (farea if m is None else farea * m)).sum()) * SEC_PER_YR / 1e12

    def f1_of(key, m=None):
        mb = ba[key] > F1_THRESH
        rb = ref_burn
        if m is not None:
            mb, rb = mb & m, rb & m
        tp = int((mb & rb).sum())
        fp = int((mb & ~rb).sum())
        fn = int((~mb & rb).sum())
        return 2 * tp / (2 * tp + fp + fn) if tp else float("nan")

    def block(region_key, m, fm):
        """one table, identical columns everywhere"""
        rows = [HEADER, RULE]
        o_ba, o_ff = mha(ref_ba, m), pgc(ref_ff, fm)
        for key, name in VERSIONS:
            b = ba_s.get((key, region_key), {})
            e = ff_s.get((key, region_key), {})
            g = lambda d, k: f"{d[k]:.4f}" if k in d else "n/a"
            f1 = f1_of(key, m)
            rows.append(
                f"| {name} | {mha(ba[key], m):.0f} | {o_ba:.0f} | "
                f"{g(b,'Bias Score')} | {g(b,'RMSE Score')} | {g(b,'Seasonal Cycle Score')} | "
                f"{g(b,'Spatial Distribution Score')} | {g(b,'Overall Score')} | "
                f"{'n/a' if np.isnan(f1) else format(f1, '.3f')} | "
                f"{pgc(ff[key], fm):.3f} | {o_ff:.3f} | {g(e,'Overall Score')} |")
        return rows

    L = ["# Tables", "",
         "Built by `scripts/build_tables.py` from the model output and the official ILAMB databases,",
         "`paper_gmd/scoring/ba_regions/` and `paper_gmd/scoring/ffire_regions/`. Regenerate rather than",
         "edit by hand. Table 9 is the exception and is authored inside the script.", "",
         "**Naming.** The unoptimized baseline is the original formulation, ED's native fire scheme.",
         "Optimized versions carry letters, and a letter is fixed to the attributes it names.", "",
         "**Model F is marked with an asterisk throughout.** It was fitted on the coupled model's own",
         "climate rather than on reanalysis, with its global total pinned to the observed value, so its",
         "bias score reflects a constraint we imposed rather than skill. Every other version uses CRUJRA",
         "climate with its magnitude free.", "",
         "**Every table has the same rows and the same columns.** Burned area and fire carbon emissions",
         "are reported together everywhere, because the paper's claim that correcting one does not",
         "correct the other can only be read from tables that carry both.", "",
         "## Table 1. Global performance", ""]
    L += block("global", None, None)

    for i, (key, name) in enumerate(REGIONS, start=2):
        m, fm = mask_of(key, lat, lon), mask_of(key, flat, flon)
        L += ["", f"## Table {i}. {name}", ""]
        L += block(key, m, fm)

    L += ["",
          "Fire presence F1 uses a threshold of 0.1 percent of cell area per year on the 0.5 degree",
          "grid, over 2001 to 2016. Regional totals use the boxes the model is fitted on, defined in",
          "`paper_gmd/regions_7.txt`. They do not tile the land surface, so land outside all seven",
          "appears in Table 1 only.", "",
          TABLE9, "",
          "## Table 10. Performance in the coupled ED model", "",
          "The coupled run does not exist. The table will carry the reintegrated version scored inside",
          "the coupled model against its own offline result, with the same columns as Table 1. It needs",
          "the coupled run and it needs the reported version to be chosen.", ""]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  Table 1 global, Tables 2 to {1+len(REGIONS)} regional, Table 9 attributes, Table 10 pending")
    print(f"  observed {mha(ref_ba):.0f} Mha/yr and {pgc(ref_ff):.2f} Pg C/yr")


if __name__ == "__main__":
    main()
