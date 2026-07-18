# Methods notes (offline setup)

## What is predicted

Monthly burned-area fraction on a global grid, 2001–2016. The model is closed-form in its drivers (no prognostic vegetation state of its own) and is run **offline**.

## Drivers

| Field | Source |
|---|---|
| Dryness \(D\), air temperature, annual and monthly precip | CRUJRA → `data/crujra/` |
| GPP (fuel proxy) | Lei coupled ED dump (`global_baseline_modelC_inputs_*.nc`) |
| AGB (emissions / optional veg terms) | `global_baseline_modelCfuel_inputs_*.nc` |

Hybrid sourcing is intentional: observed climate for fair skill vs GFED5; model GPP for transfer into the coupled run.

## Grid and scoring

- Compute on **1°**, expand to **0.5°** for ILAMB  
- Reference: **GFED5** via official **ILAMB** `ConfBurntArea`  
- Base formula: [`models/formula.md`](../models/formula.md)  
- Paper ladder: Models C → D → E (see root README)

## Emissions

Burned area × pool combustion fractions (dryness-gated) → fFire. Per-continent betas live under `models/combustion/continental/`.
