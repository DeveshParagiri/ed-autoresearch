"""
Where does the 0.039 per-cell burned-fraction ceiling come from?

Imports the REAL fire_C + drivers from reproduce_modelC and decomposes the
transform rate -> annual_frac -> monthly_frac to find what caps the top end.
"""
import importlib.util, sys, types, numpy as np
from pathlib import Path

sys.modules.setdefault("h5py", types.ModuleType("h5py"))  # unused loader only

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)

import json
params = json.load(open(REPO / "models" / "C" / "params.json"))["params"]
d = rmc.load_drivers()

with np.errstate(over="ignore", invalid="ignore"):
    rate = rmc.fire_C(d, params)            # annual fire rate yr^-1, (192,180,360)

FIRE_MAX = rmc.FIRE_MAX_RATE
rate_capped = np.minimum(rate, FIRE_MAX)
annual_frac = 1.0 - np.exp(-rate_capped)
monthly     = annual_frac / 12.0            # what gets written

# period-mean monthly fraction per cell (matches the 1:1 plot quantity)
pm = monthly.mean(axis=0)

print("="*70)
print("STRUCTURAL CEILINGS (independent of drivers)")
print("="*70)
print(f"  monthly_frac = (1 - exp(-rate))/12  ->  HARD MAX = 1/12 = {1/12:.4f}")
print(f"  GFED5 per-cell period-mean reaches            0.1044  (ABOVE 1/12!)")
print()
print("="*70)
print("REALIZED RATE FIELD (on real drivers, all months/cells)")
print("="*70)
flat = rate.ravel()
print(f"  rate yr^-1:   max {flat.max():.3f}   99.9pct {np.percentile(flat,99.9):.3f}"
      f"   99pct {np.percentile(flat,99):.3f}")
n_at_cap = int((rate >= FIRE_MAX).sum())
print(f"  cells-months hitting FIRE_MAX={FIRE_MAX}:  {n_at_cap}  "
      f"({100*n_at_cap/rate.size:.4f}% of all)")
print(f"  monthly_frac: max {monthly.max():.4f}  (theoretical max {1/12:.4f})")
print(f"  period-mean monthly_frac per cell: max {pm.max():.4f}")
print()

# Where are the highest-burning cells, and what's their rate?
top = pm >= np.percentile(pm[pm>0], 99.9)
print("="*70)
print("TOP-BURNING CELLS (period-mean >= 99.9pct)")
print("="*70)
ridx = rate[:, top]                          # (192, Ncells)
print(f"  n cells: {int(top.sum())}")
print(f"  their per-month rate yr^-1: mean {ridx.mean():.3f}  max {ridx.max():.3f}")
print(f"  their annual_frac: mean {(1-np.exp(-np.minimum(ridx,FIRE_MAX))).mean():.3f}")
print(f"  -> period-mean monthly tops out at ~{pm[top].mean():.4f}")
print()

# Counterfactual: if the SAME annual fire were concentrated in a 3-month dry
# window instead of spread over 12, what monthly fraction would the top cells show?
print("="*70)
print("COUNTERFACTUAL: same annual fire, concentrated in 3-month season")
print("="*70)
af_top = (1 - np.exp(-np.minimum(rate[:, top], FIRE_MAX)))  # per-month annual_frac
# treat each month's annual_frac as the year's burn, spread /3 not /12
print(f"  current spread /12 -> peak monthly {(af_top/12).max():.4f}")
print(f"  spread /3 (dry season) -> peak monthly {(af_top/3).max():.4f}")
print(f"  no-divide (single burn pulse) -> peak monthly {af_top.max():.4f}")
