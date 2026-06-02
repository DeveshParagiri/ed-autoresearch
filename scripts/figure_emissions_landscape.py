"""Figure: TRENDY + ED Model C variants on BA and fFire (GFED5).
Uses the renamed leaderboard variants (ED-ModelC-Emissions, ED-ModelC-EmpiricalEmit).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "EMISSIONS"
OUT.mkdir(parents=True, exist_ok=True)

BA_CSV = REPO / "ilamb_out_ba_renamed" / "scores.csv"
FF_CSV = REPO / "ilamb_out_ffire_renamed" / "scores.csv"


def score(csv, model):
    df = pd.read_csv(csv)
    hdr = list(df.columns); vals = df.iloc[0].tolist()
    for h, v in zip(hdr[1:], vals[1:]):
        if h == model:
            try: return float(v)
            except: return np.nan
    return np.nan


TRENDY = ["CLM6", "ELM-FATES", "CLASSIC", "CLM-FATES", "VISIT", "E3SM", "JSBACH", "SDGVM"]
ED_VARIANTS = ["ED-ModelC-GFED5", "ED-ModelC-Emissions", "ED-ModelC-EmpiricalEmit"]

rows = []
for m in TRENDY + ED_VARIANTS:
    b = score(BA_CSV, m); f = score(FF_CSV, m)
    if not (np.isnan(b) or np.isnan(f)):
        rows.append((m, b, f))
rows.sort(key=lambda r: -(r[1] + r[2]) / 2)

fig, ax = plt.subplots(figsize=(10, 7))
colors = []
for m, b, f in rows:
    if m == "ED-ModelC-Emissions":     colors.append("#1f77b4")
    elif m == "ED-ModelC-EmpiricalEmit": colors.append("#9467bd")
    elif m == "ED-ModelC-GFED5":       colors.append("#d62728")
    else:                              colors.append("#888")

xs = [r[1] for r in rows]; ys = [r[2] for r in rows]; names = [r[0] for r in rows]
ax.scatter(xs, ys, c=colors, s=110, edgecolor="k", linewidth=0.6, zorder=5)
for x, y, n in zip(xs, ys, names):
    ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points",
                fontsize=8.5, alpha=0.9)
ax.set_xlabel("Burned area (GFED5) ILAMB Overall", fontsize=11)
ax.set_ylabel("Fire carbon emissions (GFED5) ILAMB Overall", fontsize=11)
ax.set_title("ED Model C variants vs TRENDY peers on GFED5", fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(score(FF_CSV, "CLASSIC"), color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
ax.axvline(score(BA_CSV, "CLASSIC"), color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

from matplotlib.patches import Patch
legend = [
    Patch(facecolor="#888", edgecolor="k", label="TRENDY peers"),
    Patch(facecolor="#d62728", edgecolor="k", label="ED-ModelC-GFED5 (BA only)"),
    Patch(facecolor="#1f77b4", edgecolor="k", label="ED-ModelC-Emissions (process-based fFire)"),
    Patch(facecolor="#9467bd", edgecolor="k", label="ED-ModelC-EmpiricalEmit (empirical EF)"),
]
ax.legend(handles=legend, loc="lower left", fontsize=9)
fig.tight_layout()
out_p = OUT / "FIG_landscape_BA_vs_fFire.png"
fig.savefig(out_p, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_p}")

df_out = pd.DataFrame(rows, columns=["Model", "BA_Overall_GFED5", "fFire_Overall_GFED5"])
df_out.to_csv(OUT / "scores_table.csv", index=False)
print(f"wrote {OUT / 'scores_table.csv'}")
print(df_out.to_string(index=False))
