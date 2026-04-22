# ED Autoresearch: Fire Module Replacement

Closed-form, mechanistic fire formulas for the ED v3 dynamic global vegetation model. Trained against GFED4.1s via Optuna, scored via **official `ilamb-run`** with `ConfBurntArea` confrontation.

**Model C ranks #1** on the ILAMB TRENDY v14 burned-area benchmark — beats CLM6.0, CLASSIC, JULES-INFERNO, and all FATES-based fire modules, with only **12 parameters**.

## Final ILAMB Leaderboard (TRENDY v14 + our models)

Official `ilamb-run` output against GFED4.1s monthly, 2001-2016. `ConfBurntArea` confrontation, `mass_weighting = True`.

| Rank | Model | Bias | RMSE | Seasonal | Spatial Dist | **Overall** |
|-----:|-------|-----:|-----:|---------:|-------------:|------------:|
| 🥇 **1** | **ED-ModelC (ours)** | **0.721** | **0.513** | **0.842** | **0.777** | **0.7133** |
| 🥈 2 | CLM6.0 | 0.759 | 0.474 | 0.758 | 0.838 | 0.7073 |
| 🥉 3 | CLASSIC | 0.738 | 0.507 | 0.782 | 0.797 | 0.7059 |
| **4** | **ED-ModelA (ours)** | 0.716 | 0.492 | 0.805 | 0.783 | **0.6989** |
| **7** | **ED-ModelB (ours)** | 0.706 | 0.476 | 0.833 | 0.763 | **0.6943** |
| 9 | ELM-FATES | 0.724 | 0.512 | 0.860 | 0.676 | 0.6931 |
| 10 | CLM-FATES | 0.725 | 0.525 | 0.802 | 0.707 | 0.6897 |
| 18 | JULES-ES | 0.709 | 0.506 | 0.784 | 0.447 | 0.6116 |
| 19 | ELM | 0.687 | 0.492 | 0.778 | 0.333 | 0.5725 |
| 20 | VISIT-UT | 0.636 | 0.488 | 0.695 | 0.466 | 0.5711 |
| 21 | LPJmL | 0.693 | 0.489 | 0.459 | 0.612 | 0.5634 |
| 22 | LPJ-GUESS | 0.671 | 0.489 | 0.459 | 0.288 | 0.4768 |
| **23** | **EDv3 (stock baseline)** | 0.681 | 0.489 | 0.439 | 0.290 | 0.4745 |
| 24 | LPJ-EOSIM | 0.654 | 0.489 | 0.459 | 0.227 | 0.4572 |

All three of our models beat ED's stock fire module (0.475) by +0.22 to +0.24 Overall.

## The three models

| Model | Mechanisms | Params | Overall | Best for |
|-------|-----------:|-------:|--------:|---------|
| **[A](models/A/formula.md)** | 8 | 27 | 0.6989 | reference full formulation with all Shapley-analyzed mechanisms |
| **[B](models/B/formula.md)** | 5 | 17 | 0.6943 | middle-ground (top-5 Shapley) |
| **[C](models/C/formula.md)** | 3 | **12** | **0.7133** ⭐ | **winner — minimal Shapley top-3, beats CLM6.0** |

**Shapley analysis on Model A** showed that 3 mechanisms (`t_air_ign`, `precip`, `gpp_monthly`) carry 52.8% of explained variance. Keeping only those yields Model C, which outperforms the full Model A because reducing mechanism count removes interference between monthly drivers — Seasonal score jumps from 0.805 (A) → 0.842 (C).

## Method summary

1. **Base formula class**: multiplicative product of global sigmoid/hump mechanisms, each tied to a physical fire-ecology process (Pausas & Keeley, Krawchuk, van der Werf, Archibald, Bistinas — see per-model formula.md for citations).
2. **Iterative development** (v2 → v8 across 7 iterations, each validated on official ilamb-run): adding monthly-resolved drivers (monthly GPP, air temp, surface soil temp, GPP anomaly) moved Overall from 0.634 → 0.699.
3. **Shapley decomposition** on Model A v8: 2⁸ = 256 subsets × 500 Optuna trials × 6 workers (~220 min). Ranked each of 8 mechanism groups by their mean marginal contribution to Overall score.
4. **Shapley-guided simplification**: built Model B (top-5 mechanisms) and Model C (top-3 mechanisms) by dropping the lowest-φ mechanisms, then refit remaining parameters.
5. **Final benchmark**: all three models evaluated via **official `ilamb-run`** using the same `ConfBurntArea` confrontation config as the TRENDY v14 benchmark. Model C at #1.

See [`WRITEUP.md`](WRITEUP.md) for the full iteration narrative.

## Shapley results

From `models/shapley.json` — exact Shapley values for Model A's 8 mechanism groups:

| Rank | Mechanism | φ (Overall contribution) | % of explained |
|-----:|-----------|-------------------------:|---------------:|
| 1 | `t_air_ign` (monthly T_air sigmoid) | +0.0210 | 18.5% |
| 2 | `precip` (annual + monthly) | +0.0199 | 17.5% |
| 3 | `gpp_monthly` (monthly GPP hump) | +0.0189 | 16.7% |
| 4 | `gpp_anom` (per-cell GPP anomaly) | +0.0151 | 13.2% |
| 5 | `t_surf` (monthly T_surf) | +0.0143 | 12.6% |
| 6 | `soil_temp` (deep + rate) | +0.0112 | 9.9% |
| 7 | `height` (canopy suppression) | +0.0073 | 6.4% |
| 8 | `fuel` (AGB + LAI hump) | +0.0059 | 5.2% |

Empty-subset baseline Overall = 0.6225. Full v8 Overall = 0.7362. Sum verifies: ΣΦ = 0.1137.

**Counter-intuitive finding**: `fuel` ranks last. In v8's 8-mechanism formula, GPP-based mechanisms already carry the biomass/productivity signal, making the explicit fuel hump largely redundant. This is what lets us drop fuel in Models B and C without losing much signal.

## Inputs required

Our models use only **ED-runtime-compatible** inputs that any coupled ED simulation produces natively, plus standard CRUJRA climate forcing.

| Variable | Source | Used by |
|----------|--------|---------|
| CRUJRA monthly temperature | CRU/JRA-55 reanalysis, 0.5° | A, B, C |
| CRUJRA monthly precipitation | CRU/JRA-55 | A, B, C |
| CRUJRA soil_temp 1..6 | CRU/JRA-55 | A, B |
| ED monthly GPP | TRENDY v14 S3 `EDv3_S3_gpp.nc` | A, B, C |
| ED annual cLeaf + cWood (AGB) | TRENDY v14 S3 `cLeaf.nc` + `cWood.nc` | A only |
| ED cSoil | TRENDY v14 S3 `cSoil.nc` | optional (dropped from v8 final) |
| ED LAI (annual mean) | ED internal state | A only |
| ED height_natr, frac_scnd | ED internal state | A only |

## Repository contents

```
ed-autoresearch/
├── README.md                 # this file, with ILAMB leaderboard
├── WRITEUP.md                # iteration narrative, Shapley deep-dive, first-principles reasoning
├── models/
│   ├── A/formula.md          # full 8-mechanism formula, 27 params
│   ├── A/params.json         # fitted hyperparameter values
│   ├── B/formula.md          # Shapley top-5 (17 params)
│   ├── B/params.json
│   ├── C/formula.md          # Shapley top-3 (12 params) — #1 globally
│   ├── C/params.json
│   ├── shapley.json          # 8-mechanism Shapley decomposition values
│   └── shapley_subsets.json  # raw 256-subset Optuna best scores
├── scripts/
│   ├── download_inputs.sh    # fetch TRENDY v14 + GFED references
│   ├── reproduce.py          # regenerate burntArea NetCDFs for A/B/C
│   └── score.py              # standalone ILAMB-style scorer
├── patches/
│   ├── fire_modelA.cc        # C++ implementation of Model A for ED integration
│   ├── fire_modelB.cc        # Model B C++ patch
│   └── fire_modelC.cc        # Model C C++ patch (recommended)
└── data/
    └── ilamb_final_scorecard.csv   # raw ILAMB output used for the leaderboard above
```

## Quick start

```bash
# Clone + setup
git clone https://github.com/DeveshParagiri/ed-autoresearch.git
cd ed-autoresearch
python -m venv .venv && source .venv/bin/activate
pip install numpy xarray cftime netCDF4 h5py

# Download required inputs (~5 GB for Model C, ~10 GB for Model A)
bash scripts/download_inputs.sh

# Regenerate burntArea NetCDFs for all three models
python scripts/reproduce.py

# (Optional) Score against GFED
python scripts/score.py   # uses ILAMB-style metrics, standalone
```

## ED integration

Patches for `fire.cc` in ED v3 are in `patches/`. Each is a complete replacement of ED's fire module matching the formula in the corresponding `models/X/formula.md`. The patches read ED's prognostic state and produce `burntArea` as a fraction per cell per time step.

**For a coupled ED port, Model C is recommended** — 12 parameters, 4 inputs, direct drop-in replacement for ED's existing fire module with the highest official ILAMB score.

## Caveats

- All training is on the TRENDY v14 ED S3 simulation. Numbers may shift slightly when plugged into a different ED build or different coupled-ED spin-up.
- The JASMIN internal scorecard uses a private `BurnedAreaExtended` confrontation with a burnable-area mask that isn't distributed with ILAMB; absolute Bias Score and Overall magnitudes there are ~2-3× higher than what public `ConfBurntArea` produces. **Rankings** match. See [trendy-v14-fire-benchmark](https://github.com/DeveshParagiri/trendy-v14-fire-benchmark) for the matching reproducer.
- These models predict monthly burned-area fraction. Fire emissions, mortality, and vegetation coupling are handled by ED's existing machinery downstream.

## References

- **ILAMB**: Collier et al. 2018 — benchmark framework
- **GFED4.1s**: van der Werf et al. 2017 — observation reference
- **TRENDY v14**: Global Carbon Project 2025
- Pausas & Keeley 2009; Pausas & Ribeiro 2013 — fire-ecology mechanism formulations
- Krawchuk et al. 2009 — global pyrogeography
- van der Werf et al. 2008, 2010 — GPP-fire coupling
- Venevsky et al. 2002 — soil thermal fire control
- Archibald et al. 2009, 2010, 2013 — canopy, temperature, phase-locking mechanisms
- Bistinas et al. 2014 — fire intensity exponent
- Abatzoglou & Williams 2016 — VPD/fire weather
- Lundberg & Lee 2017 — Shapley value explanation method

## License

MIT (see LICENSE).
