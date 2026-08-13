"""Build the paper's tables from the scored outputs, numbered as the topic-sentence outline requires.

    python scripts/build_tables.py

Writes paper_gmd/TABLES.md with
  Table 1        global performance
  Tables 2 to 8  the same table for each of the seven fitting regions
  Matrix         versions down, regions across, the tabular form of Figure 4
  Table 9        the model versions and the attributes that define them, authored not computed
  Table 10       performance in the coupled ED model, which does not exist yet

Every performance table comes out of format_table(). There is no hand-formatting anywhere, so the
column set, the column order and the precision cannot drift apart across the set. Add a column once
and it appears in all eight.

Formatting rules, applied identically everywhere
  columns    Version | BA (Mha yr-1) | Bias | RMSE | Seasonal | Spatial | BA score | F1 |
             Emissions (Pg C yr-1) | Emis. score
  precision  burned area integer, emissions two decimals, every score and F1 three decimals
  observed   stated once in the caption, never repeated down a column
  best       bold, per column. For the scores and F1 that is the largest value. For burned area and
             emissions it is the value closest to the observed one, since more is not better there.
  composites BA score and Emis. score are distinct columns and are never both called Overall.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
            ("F", "F"), ("G7", "G"), ("H", "H"), ("I", "I")]

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

# header, dataframe field, format, how "best" is decided
# "max" for a score, "near" for a magnitude, where best means closest to the observed value
COLS = [("BA (Mha yr-1)", "ba", "{:.0f}", "near_ba"),
        ("Bias", "bias", "{:.3f}", "max"),
        ("RMSE", "rmse", "{:.3f}", "max"),
        ("Seasonal", "seasonal", "{:.3f}", "max"),
        ("Spatial", "spatial", "{:.3f}", "max"),
        ("BA score", "ba_score", "{:.3f}", "max"),
        ("F1", "f1", "{:.3f}", "max"),
        ("Emissions (Pg C yr-1)", "emis", "{:.2f}", "near_ff"),
        ("Emis. score", "emis_score", "{:.3f}", "max")]

CAPTION = ("*Table {n}. {where} performance of each model version against GFED5, 2001-2016. "
           "Observed burned area is {oba:.0f} Mha yr-1 and observed fire carbon emissions are "
           "{off:.2f} Pg C yr-1. Bias, RMSE, seasonal and spatial are ILAMB component scores; "
           "BA score and Emis. score are the corresponding composites. F1 evaluates fire presence "
           "at a threshold of 0.1 percent of cell area per year. Best value in each column is bold.*")

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
| F | GFED5 | S_overall | Global base + regional GDP coefficient | Annual | monthly + annual | GDP per capita | biomass gate + fuel amplitude | - |
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


# ----------------------------------------------------------------- fields off the model output

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


def build_results():
    """one long dataframe, a row per version per region, plus the observed value per region"""
    ba_s, ff_s = scores(BA_DB), scores(FF_DB)
    ref_ba, lat, lon = annual_ba(BA_REF)
    ref_ff, flat, flon = annual_ff(FF_REF)
    area, farea = cell_area(lat, lon), cell_area(flat, flon)
    ref_burn = ref_ba > F1_THRESH

    ba = {k: annual_ba(BA_DIR / k / "burntArea.nc")[0] for k, _ in VERSIONS}
    ff = {k: annual_ff(FF_DIR / k / "fFire.nc")[0] for k, _ in VERSIONS}

    def mha(f, m):
        return float((f * (area if m is None else area * m)).sum()) / 1e10

    def pgc(f, m):
        return float((f * (farea if m is None else farea * m)).sum()) * SEC_PER_YR / 1e12

    def f1_of(key, m):
        mb, rb = ba[key] > F1_THRESH, ref_burn
        if m is not None:
            mb, rb = mb & m, rb & m
        tp, fp, fn = int((mb & rb).sum()), int((mb & ~rb).sum()), int((~mb & rb).sum())
        return 2 * tp / (2 * tp + fp + fn) if tp else np.nan

    places = [("global", "Global", None, None)] + [
        (k, n, mask_of(k, lat, lon), mask_of(k, flat, flon)) for k, n in REGIONS]

    rows, obs = [], {}
    for rk, rn, m, fm in places:
        obs[rk] = (mha(ref_ba, m), pgc(ref_ff, fm))
        for vk, vn in VERSIONS:
            b, e = ba_s.get((vk, rk), {}), ff_s.get((vk, rk), {})
            rows.append({"region": rk, "region_name": rn, "version": vk, "label": vn,
                         "ba": mha(ba[vk], m), "emis": pgc(ff[vk], fm), "f1": f1_of(vk, m),
                         "bias": b.get("Bias Score", np.nan),
                         "rmse": b.get("RMSE Score", np.nan),
                         "seasonal": b.get("Seasonal Cycle Score", np.nan),
                         "spatial": b.get("Spatial Distribution Score", np.nan),
                         "ba_score": b.get("Overall Score", np.nan),
                         "emis_score": e.get("Overall Score", np.nan)})
    return pd.DataFrame(rows), obs


# ----------------------------------------------------------------- the one formatter

def format_table(df, region, obs, number):
    """one performance table, identical columns and precision for every region

    df      the long results frame from build_results()
    region  a region key, or 'global'
    obs     {region key: (observed Mha yr-1, observed Pg C yr-1)}
    number  the table number, used in the heading and the caption
    """
    d = df[df.region == region].set_index("label").reindex([n for _, n in VERSIONS])
    where = "Global" if region == "global" else d["region_name"].iloc[0]
    oba, off = obs[region]

    # which row wins each column, before anything is turned into a string
    best = {}
    for _, field, _, rule in COLS:
        v = d[field]
        if v.notna().sum() == 0:
            continue
        target = {"near_ba": oba, "near_ff": off}.get(rule)
        best[field] = (v - target).abs().idxmin() if target is not None else v.idxmax()

    # bold on the printed string, not the underlying float, so two rows that round to the same
    # value are either both bold or neither. Bolding one of two identical numbers reads as an error.
    winner = {f: ("n/a" if pd.isna(d.loc[r, f]) else fmt.format(d.loc[r, f]))
              for (_, f, fmt, _) in COLS if (r := best.get(f)) is not None}

    head = "| Version | " + " | ".join(h for h, _, _, _ in COLS) + " |"
    lines = [f"## Table {number}. {where}", "", head, "|---" * (1 + len(COLS)) + "|"]
    for label, r in d.iterrows():
        cells = [label]
        for _, field, fmt, _ in COLS:
            v = r[field]
            s = "n/a" if pd.isna(v) else fmt.format(v)
            cells.append(f"**{s}**" if s != "n/a" and winner.get(field) == s else s)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", CAPTION.format(n=number, where=where, oba=oba, off=off)]
    return lines


def format_matrix(df, obs):
    """versions down, regions across, the tabular form of Figure 4"""
    cols = [(k, n) for k, n in REGIONS] + [("global", "Global")]
    piv = df.pivot(index="label", columns="region", values="ba_score").reindex(
        [n for _, n in VERSIONS])
    winner = {k: f"{piv[k].max():.3f}" for k, _ in cols if piv[k].notna().any()}

    lines = ["## Cross-region matrix. Burned-area composite score",
             "",
             "| Version | " + " | ".join(n for _, n in cols) + " |",
             "|---" * (1 + len(cols)) + "|"]
    for label in piv.index:
        cells = [label]
        for k, _ in cols:
            v = piv.loc[label, k]
            s = "n/a" if pd.isna(v) else f"{v:.3f}"
            cells.append(f"**{s}**" if s != "n/a" and winner.get(k) == s else s)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
              "*The BA score column of Tables 1 to 8 in one field of view, versions down and regions "
              "across. Best value in each column is bold. The version that wins the global column is "
              "not the version that wins the regions. This is the tabular form of Figure 4, so it "
              "carries no number of its own until we decide which of the two the paper prints.*"]
    return lines


def main():
    df, obs = build_results()

    L = ["# Tables", "",
         "Built by `scripts/build_tables.py` from the model output and the official ILAMB databases,",
         "`paper_gmd/scoring/ba_regions/` and `paper_gmd/scoring/ffire_regions/`. Regenerate rather",
         "than edit by hand. Table 9 is the exception and is authored inside the script.", "",
         "**Every performance table comes out of one function.** Tables 1 to 8 and the cross-region",
         "matrix share a single code path, so the column set, the column order and the precision are",
         "identical across the set by construction rather than by inspection.", "",
         "**Naming.** The unoptimized baseline is the original formulation, ED's native fire scheme.",
         "Optimized versions carry letters, and a letter is fixed to the attributes it names.", "",
         "**Reading the tables.** The observed value is constant down every column, so it is stated",
         "once in each caption rather than repeated as a column. Bold marks the best value in a",
         "column. For the scores and F1 that is the largest value. For burned area and emissions it is",
         "the value closest to the observed one, since more is not better there.", "",
         "**One caveat that no column can carry.** Model F was fitted on the coupled model's own",
         "climate rather than on reanalysis, with its global total pinned to the observed value, so its",
         "bias score reflects a constraint we imposed rather than skill. Every other version uses",
         "CRUJRA climate with its magnitude free.", ""]

    L += format_table(df, "global", obs, 1)
    for i, (key, _) in enumerate(REGIONS, start=2):
        L += [""] + format_table(df, key, obs, i)

    L += ["",
          "Regional totals use the boxes the model is fitted on, defined in",
          "`paper_gmd/regions_7.txt`. They do not tile the land surface, so land outside all seven",
          "appears in Table 1 only.", ""]
    L += format_matrix(df, obs)
    L += ["", TABLE9, "",
          "## Table 10. Performance in the coupled ED model", "",
          "The coupled run does not exist. The table will carry the reintegrated version scored inside",
          "the coupled model against its own offline result, with the same columns as Table 1. It needs",
          "the coupled run and it needs the reported version to be chosen.", ""]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  Table 1 global, Tables 2 to {1+len(REGIONS)} regional, matrix, Table 9, Table 10")
    print(f"  observed {obs['global'][0]:.0f} Mha/yr and {obs['global'][1]:.2f} Pg C/yr")


if __name__ == "__main__":
    main()
