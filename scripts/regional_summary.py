"""Condense a regional ILAMB run into the two main-text tables.

Usage:  python scripts/regional_summary.py [build_dir] [out.md] [fit7|gfed14]

The per-region tables in REGIONAL_TABLES*.md are the supplement. These are what goes in the paper.
Regions are rows and versions are columns, so the whole comparison sits on one page.
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regional_tables import ROWS, REGION_SETS  # noqa: E402

BUILD = sys.argv[1] if len(sys.argv) > 1 else "paper_gmd/scoring/ba_gfed14"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_gmd/REGIONAL_SUMMARY.md"
SETKEY = sys.argv[3] if len(sys.argv) > 3 else ("gfed14" if "gfed14" in BUILD else "fit7")
REGIONS = REGION_SETS[SETKEY]


def load():
    data = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(BUILD, "scalar_database.csv"))):
        try:
            data[(r["Model"], r["Region"])][r["ScalarName"]] = float(r["Data"])
        except (TypeError, ValueError):
            pass
    return data


def main():
    data = load()
    rows = [(k, n) for k, n in ROWS if any((k, g) in data for g, _ in REGIONS)]
    regs = [(k, n) for k, n in REGIONS if k != "global"]
    names = [n for _, n in rows]

    L = [
        "# Regional results, the main-text tables",
        "",
        f"Condensed from `{BUILD}/scalar_database.csv` by `scripts/regional_summary.py`. The full",
        "per-region tables, with every component score, are the supplement.",
        "",
        "## Table 3. Overall benchmark score by region",
        "",
        "Bold is the best version in that region.",
        "",
        "| Region | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]

    wins = defaultdict(int)
    for gk, gn in regs:
        vals = [data.get((k, gk), {}).get("Overall Score") for k, _ in rows]
        best = max(v for v in vals if v is not None)
        cells = []
        for (k, n), v in zip(rows, vals):
            if v is None:
                cells.append("n/a")
                continue
            if v == best:
                wins[n] += 1
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        L.append(f"| {gn} | " + " | ".join(cells) + " |")

    def agg(fn, label):
        vals = []
        for k, n in rows:
            v = [data[(k, g)]["Overall Score"] for g, _ in regs if (k, g) in data]
            vals.append(fn(v))
        return f"| **{label}** | " + " | ".join(f"{v:.4f}" for v in vals) + " |"

    L += [
        "|" + "---|" * (len(names) + 1),
        "| **Global** | " + " | ".join(
            f"{data[(k, 'global')]['Overall Score']:.4f}" for k, _ in rows) + " |",
        agg(lambda v: sum(v) / len(v), f"Mean of the {len(regs)}"),
        agg(min, "Worst region"),
        "| **Regions won** | " + " | ".join(str(wins[n]) for n in names) + " |",
        "",
        "## Table 4. Mean annual burned fraction by region, percent of land area",
        "",
        "| Region | GFED5 | " + " | ".join(names) + " |",
        "|---" * (len(names) + 2) + "|",
    ]

    for gk, gn in regs + [("global", "Global")]:
        obs = [data[(k, gk)]["Benchmark Period Mean (intersection)"]
               for k, _ in rows if (k, gk) in data
               and "Benchmark Period Mean (intersection)" in data[(k, gk)]]
        o = f"{sum(obs) / len(obs):.3g}" if obs else "n/a"
        cells = []
        for k, _ in rows:
            v = data.get((k, gk), {}).get("Model Period Mean (intersection)")
            cells.append("n/a" if v is None else f"{v:.3g}")
        L.append(f"| {gn} | {o} | " + " | ".join(cells) + " |")

    L += [
        "",
        "The GFED5 column is the mean over versions, which differ in the third digit because ILAMB",
        "compares each version on the cells where it and the reference both report data.",
        "",
    ]

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {OUT}, {len(regs)} regions x {len(rows)} versions")


if __name__ == "__main__":
    main()
