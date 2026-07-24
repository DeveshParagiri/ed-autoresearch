"""Advisor figure for the GDP human-suppression term.

A: GDP per capita (the driver)         sequential
B: human multiplier M (amplify poor / suppress wealthy)   diverging about 1x
C: change in burned area (+GDP - base) diverging about 0   (red = more fire)
D: ILAMB Overall vs term strength gamma (the measured gain)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

z = np.load("data_human/gdp_term_fields.npz")
lat, lon = z["lat1"], z["lon1"]; land = z["land"]; gam = float(z["gam"])
gdp = np.where(land, z["gdp1"], np.nan)
M = np.where(land, z["M"], np.nan)
dba = np.where(land, (z["gdp_ann"] - z["base_ann"]) * 100.0, np.nan)   # % / yr
PC = ccrs.PlateCarree(); ext = [-180, 180, -60, 84]
imshow_kw = dict(transform=PC, extent=[-180, 180, -90, 90], origin="lower")

# ILAMB Overall vs gamma (from the scoring run)
gammas = np.array([0.0, 0.15, 0.30, 0.50, 0.70])
overall = np.array([0.6547, 0.6603, 0.6602, 0.6534, 0.6498])

fig = plt.figure(figsize=(13.5, 8.2))
def mapax(pos):
    ax = fig.add_subplot(pos, projection=PC); ax.set_extent(ext, PC)
    ax.add_feature(cfeature.COASTLINE, lw=0.35, edgecolor="#444")
    ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="#888")
    return ax

# --- A: GDP per capita, sequential (cividis, CVD-safe) ---
axA = mapax(221)
imA = axA.imshow(np.log10(gdp), cmap="cividis", vmin=2.4, vmax=4.9, **imshow_kw)
axA.set_title("A. GDP per capita (the driver)", fontsize=11, loc="left")
cbA = fig.colorbar(imA, ax=axA, orientation="horizontal", pad=0.03, shrink=0.85, aspect=32)
cbA.set_ticks([np.log10(v) for v in (500, 2000, 10000, 60000)]); cbA.set_ticklabels(["$500", "$2k", "$10k", "$60k"])

# --- B: human multiplier M, diverging about 1x (log10 M, red=amplify) ---
axB = mapax(222)
imB = axB.imshow(np.log10(M), cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.45, 0.45), **imshow_kw)
axB.set_title(f"B. Human term (gamma={gam:.2f}):  amplify where poor, suppress where wealthy",
              fontsize=10.5, loc="left")
cbB = fig.colorbar(imB, ax=axB, orientation="horizontal", pad=0.03, shrink=0.85, aspect=32)
cbB.set_ticks([np.log10(v) for v in (0.4, 0.7, 1, 1.5, 2.5)]); cbB.set_ticklabels(["0.4x", "0.7x", "1x", "1.5x", "2.5x"])

# --- C: change in burned area, diverging about 0 ---
axC = mapax(223)
v = float(np.nanpercentile(np.abs(dba), 98))
imC = axC.imshow(dba, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), **imshow_kw)
axC.set_title("C. Change in burned area (+GDP - base):  red added, blue removed",
              fontsize=10.5, loc="left")
cbC = fig.colorbar(imC, ax=axC, orientation="horizontal", pad=0.03, shrink=0.85, aspect=32)
cbC.set_label("burned fraction change (% / yr)", fontsize=9)

# --- D: ILAMB Overall vs gamma ---
axD = fig.add_subplot(224)
axD.plot(gammas, overall, "-o", color="#0072B2", lw=2, ms=7, zorder=3)
ib = int(np.argmax(overall))
axD.scatter([gammas[ib]], [overall[ib]], s=140, facecolor="none", edgecolor="#C0392B", lw=2, zorder=4)
axD.axhline(overall[0], color="#888", ls="--", lw=1)
axD.annotate("base (no human term)", (0.0, overall[0]), textcoords="offset points",
             xytext=(6, -13), fontsize=9, color="#555")
axD.annotate(f"{overall[ib]-overall[0]:+.4f} Overall\n(all in spatial score)",
             (gammas[ib], overall[ib]), textcoords="offset points", xytext=(10, 6),
             fontsize=9.5, color="#C0392B")
axD.set_xlabel("human term strength  gamma"); axD.set_ylabel("ILAMB Overall Score")
axD.set_title("D. Measured skill gain vs term strength", fontsize=11, loc="left")
axD.grid(True, alpha=0.18); axD.set_ylim(0.648, 0.663)
for sp in ("top", "right"): axD.spines[sp].set_visible(False)

fig.suptitle("A physical human predictor (GDP per capita) added to the single-global fire model",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("gdp_term_figure.png", dpi=150)
print("[fig] gdp_term_figure.png")
