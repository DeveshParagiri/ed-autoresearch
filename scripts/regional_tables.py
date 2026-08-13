"""Build George's per-region results tables from an ILAMB run scored over a set of regions.

Usage:  python scripts/regional_tables.py [build_dir] [out.md] [fit7|gfed14]
Default build dir is paper_gmd/scoring/ba_regions, and the region set is inferred from its name.

  fit7    the seven boxes the optimizer fits on, paper_gmd/regions_7.txt
  gfed14  the fourteen standard GFED regions shipped with ILAMB, which we neither chose nor fitted on

One table per region plus the global table, identical row headings and column headings in every one,
which is the form George asked for on 2026-08-06. Rows are model versions in paper order, columns are
the modelled value, the benchmark value and the ILAMB component and overall scores.
"""

import csv
import os
import sys
from collections import defaultdict

BUILD = sys.argv[1] if len(sys.argv) > 1 else "paper_gmd/scoring/ba_regions"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_gmd/REGIONAL_TABLES.md"

# ILAMB model directory -> the name the paper uses. Order is the paper's order.
ROWS = [
    ("ED-stock", "Original formulation"),
    ("C", "C"),
    ("D", "D"),
    ("E-clean", "E"),
    ("F", "F *"),
    ("G7", "G"),
    ("H", "H"),
    ("I", "I"),
]
# Scored but kept out of the main tables, with the reason.
ASIDE = {
    "G": "the five-region assembly, must never be shown as Model G",
    "G6": "G7 minus Australia, a development check",
    "Ibest": "keep-best-per-region variant of I",
}

REGION_SETS = {
    # the boxes the optimizer fits on
    "fit7": [
        ("global", "Global"),
        ("africa", "Africa"),
        ("samerica", "South America"),
        ("namerica", "North America"),
        ("boreal", "Boreal Eurasia"),
        ("seasia", "Southeast Asia"),
        ("australia", "Australia"),
        ("europe", "Europe"),
    ],
    # the fourteen standard GFED regions, shipped with ILAMB, chosen by the fire community
    "gfed14": [
        ("global", "Global"),
        ("bona", "Boreal North America"),
        ("tena", "Temperate North America"),
        ("ceam", "Central America"),
        ("nhsa", "Northern Hemisphere South America"),
        ("shsa", "Southern Hemisphere South America"),
        ("euro", "Europe"),
        ("mide", "Middle East"),
        ("nhaf", "Northern Hemisphere Africa"),
        ("shaf", "Southern Hemisphere Africa"),
        ("boas", "Boreal Asia"),
        ("ceas", "Central Asia"),
        ("seas", "Southeast Asia"),
        ("eqas", "Equatorial Asia"),
        ("aust", "Australia"),
    ],
}
SETKEY = sys.argv[3] if len(sys.argv) > 3 else ("gfed14" if "gfed14" in BUILD else "fit7")
REGIONS = REGION_SETS[SETKEY]

COLS = [
    ("Model Period Mean (intersection)", "Model"),
    ("Benchmark Period Mean (intersection)", "GFED5"),
    ("Bias Score", "Bias"),
    ("RMSE Score", "RMSE"),
    ("Seasonal Cycle Score", "Seasonal"),
    ("Spatial Distribution Score", "Spatial"),
    ("Overall Score", "Overall"),
]


def load(build):
    path = os.path.join(build, "scalar_database.csv")
    data = defaultdict(dict)      # (model, region) -> {scalar: value}
    units = {}
    for r in csv.DictReader(open(path)):
        try:
            v = float(r["Data"])
        except (TypeError, ValueError):
            continue
        data[(r["Model"], r["Region"])][r["ScalarName"]] = v
        units.setdefault(r["ScalarName"], r["Units"])
    return data, units


def fmt(v, scalar):
    if v is None:
        return "n/a"
    return f"{v:.4f}" if "Score" in scalar else f"{v:.3g}"


def main():
    data, units = load(BUILD)
    present = {m for m, _ in data}
    rows = [(k, n) for k, n in ROWS if k in present]
    missing = [k for k, _ in ROWS if k not in present]

    unit = units.get("Model Period Mean (intersection)", "")
    lines = [
        "# Regional results tables",
        "",
        f"Built from `{BUILD}/scalar_database.csv` by `scripts/regional_tables.py`. Official ILAMB, one",
        "run, every version scored against GFED5 over the same regions, so every number in every table is",
        "strictly comparable.",
        "",
        (
            "The regions are the boxes the optimizer fits on, defined in `paper_gmd/regions_7.txt` and "
            "identical to `_REGION_BOX` in `scripts/assemble_continental.py`."
            if SETKEY == "fit7" else
            "The regions are the fourteen standard GFED regions shipped with ILAMB. We did not choose "
            "them and did not fit on them, which is what makes them the stronger test."
        ),
        "Model and GFED5 columns are the",
        f"period mean burned area in {unit}. The four score columns are the ILAMB component scores and",
        "Overall is their weighted combination, (Bias + 2 RMSE + Seasonal + Spatial) / 5.",
        "",
        (
            "**The seven regions do not tile the land surface.** Cells outside all of them run on the "
            "global parameter set and appear in the global table only."
            if SETKEY == "fit7" else
            "**These regions cut across the boxes the model was fitted on**, so no region here matches a "
            "parameter set one for one."
        ),
        "",
        "**Model F is marked with an asterisk** because it was fitted on the coupled model's own climate",
        "rather than on reanalysis, with its global total pinned to the observed value, so its bias score",
        "reflects a constraint we imposed. It is shown because excluding the fourth-highest-scoring version",
        "would hide evidence, and its regional behaviour matters to the argument.",
        "",
        "The GFED5 column varies slightly between versions because ILAMB compares on the cells where both",
        "the version and the reference report data, and the original formulation carries a different mask.",
        "",
    ]

    # summary, the point of the whole exercise
    reg_keys = [k for k, _ in REGIONS if k != "global"]
    stat = []
    for mkey, mname in rows:
        vals = [data.get((mkey, k), {}).get("Overall Score") for k in reg_keys]
        if any(v is None for v in vals):
            continue
        wins = sum(
            1 for i, k in enumerate(reg_keys)
            if vals[i] >= max(data.get((m, k), {}).get("Overall Score", -1) for m, _ in rows)
        )
        stat.append((mname, data.get((mkey, "global"), {}).get("Overall Score"),
                     sum(vals) / len(vals), wins))

    if stat:
        lines += [
            "## Global ranking against regional ranking",
            "",
            f"| Version | Global Overall | Mean of the {len(reg_keys)} regions | Regions won |",
            "|---|---|---|---|",
        ]
        for name, g, rm, w in stat:
            lines.append(f"| {name} | {g:.4f} | {rm:.4f} | {w} |")
        by_g = [s[0] for s in sorted(stat, key=lambda s: -s[1])]
        by_r = [s[0] for s in sorted(stat, key=lambda s: -s[2])]
        lines += [
            "",
            f"Ranked by the global score, {', '.join(by_g)}.",
            "",
            f"Ranked by the mean of the {len(reg_keys)} regions, {', '.join(by_r)}.",
            "",
        ]

    for key, name in REGIONS:
        if not any((m, key) in data for m, _ in rows):
            continue
        lines += [f"## {name}", "", "| Version | " + " | ".join(c[1] for c in COLS) + " |",
                  "|---" * (len(COLS) + 1) + "|"]
        for mkey, mname in rows:
            d = data.get((mkey, key), {})
            cells = [fmt(d.get(s), s) for s, _ in COLS]
            lines.append(f"| {mname} | " + " | ".join(cells) + " |")
        lines.append("")

    if ASIDE:
        lines += ["## Versions scored but held out of the tables", ""]
        for k, why in ASIDE.items():
            if k in present:
                d = data.get((k, "global"), {})
                o = fmt(d.get("Overall Score"), "Overall Score")
                lines.append(f"- **{k}**, global Overall {o}. {why}.")
        lines.append("")

    if missing:
        lines += ["## Missing from this run", "", f"- {', '.join(missing)}", ""]

    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {OUT}")
    print(f"{len(rows)} versions x {len(REGIONS)} regions")
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
