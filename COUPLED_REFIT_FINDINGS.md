# Coupling-ready Model E refit — findings (2026-07-22)

Prompted by Lei's email (thread "Model E code question"). Lei's two requirements for
using Model E inside coupled ED:
1. Use ED's own dryness (`D_bar` from the dump) so ED has ONE dryness definition, not two.
2. No per-continent hard seams (blocky patterns propagate into coupled vegetation/carbon).

## What was built
- New `DUMP_CLIMATE=1` mode in `scripts/optimize_modelC_coupled.py` (default off = canonical
  unchanged): pulls D_bar/T_air/P_ann/P_month from the dump instead of CRUJRA. Widened the
  dryness sigmoid ranges (ED D_bar reaches ~4.8e6 vs CRUJRA ~7e4). Env-configurable fire_exp
  bound (`FIRE_EXP_HI`).
- Single-global dump fit (TAG=coupledE, coupledE_fx with fire_exp<=2.5).
- 7 per-continent dump fits: `models/C/params.coupledE_{af,bor,sam,sea,eur,nam,aus}.json`
  (DUMP_CLIMATE=1 REGION=<r> FIRE_EXP_HI=2.5 FUEL_AMP=1 SPATIAL_OBJ=1 SEAS_W=0 nsga2, 1500 trials).
- Smooth assembly: `scripts/assemble_smooth_coupledE.py` blends the 5 regional + global-dump
  fallback parameter fields (Gaussian, SIGMA=4 deg, log-space) -> NO hard seams.
  Output: `ilamb/MODELS_SMOOTH_COUPLED/ED-ModelC-{hard,smooth}/burntArea.nc`.

## Results (official ILAMB, GFED5; burned area Mha/yr)

| model | ILAMB Overall | Boreal (GFED5 50) | S.Amer (GFED5 65) | seams | ED D_bar |
|---|---|---|---|---|---|
| Paper E (CRUJRA, per-continent) | 0.6646 | 54 | 77 | yes | no |
| Single-global dump (k2) | 0.6532 | 2 | 196 | no | yes |
| **Coupling-ready SMOOTH (dump, FINAL)** | **0.6426** | **63** | **76** | **no** | **yes** |

FINAL build = all 7 continents fit on dump climate (af, bor, sam, sea, eur, nam, aus) +
global-dump fallback elsewhere, smooth-blended (SIGMA=4). Every region within 0.88-1.38x
of GFED5, global 1.05x:

| region | GFED5 | smooth | ratio |   | region | GFED5 | smooth | ratio |
|---|---|---|---|---|---|---|---|---|
| Africa | 496 | 437 | 0.88x |  | S.Amer | 65 | 76 | 1.17x |
| Boreal | 50 | 63 | 1.28x |  | N.Amer | 22 | 29 | 1.33x |
| India+SEA | 68 | 94 | 1.38x |  | Australia | 58 | 53 | 0.91x |
| **GLOBAL** | **793** | **832** | **1.05x** |  | | | | |

Adding N.America + Australia fits moved N.Amer 40 -> 29 (GFED5 22) and cost -0.0008 on
ILAMB Overall (0.6434 -> 0.6426) -- the same "aggregate metric does not reward regional
fidelity" pattern. Kept, because regional carbon fluxes are what the coupled run needs.

## The key finding (this IS the paper's thesis, confirmed on ED's drivers)
The single-global dump model scores HIGHER on ILAMB Overall (0.6532) than the smooth-regional
model (0.6434), yet it is the one with the dead boreal (2 vs 50) and 3x Amazon over-burn
(196 vs 65). ILAMB Overall rewards the compensating-error model because its inflated
variance/contrast pleases the Taylor spatial term. ILAMB Overall does NOT reward regional
fidelity and can rank a physically worse model higher.

Diagnostic (`scripts/diag_coupledE.py`) traced the single-global failure to (a) fire_exp~7
crushing marginal (boreal) cells and over-concentrating savanna, and (b) the structural fact
that cerrado and African savanna have near-identical drivers (base_product 0.56 vs 0.53) but
6x different GFED5 fire — a single global form cannot separate them. Constraining fire_exp
(coupledE_fx) did NOT fix the regional compensating error -> single global form is out.

## The recommended deliverable
Coupling-ready SMOOTH model (`ED-ModelC-smooth` above). Satisfies BOTH Lei requirements
(ED's D_bar, no seams) AND keeps regional carbon fluxes right (boreal alive, Amazon tamed).
Its slightly lower aggregate ILAMB is ILAMB's blindness, not a real deficiency; for a carbon
budget, regional fidelity is what matters.

## OPEN QUESTION for Lei (gates everything)
Does Lei object to the HARD SEAMS (then smooth-regional is the answer) or to ANY spatial
parameter variation at all (then only single-global survives, with its regional bias)? His
"blocky patterns" wording points to seams. CONFIRM before finalizing.

## Open improvement items (not blocking)
- DONE: N.America + Australia now fit (N.Amer 40 -> 29). Worst remaining region is India+SEA 1.38x.
- Seasonal component weak (0.737) — regional fits were pure-spatial (SEAS_W=0); a small
  seasonal blend could recover it (trades spatial).
- Regional fits used 1500 trials; more may help. Score cost vs paper E (0.6646 -> 0.6434) is
  mostly the dump-vs-CRUJRA dryness (~0.02), accepted for coupling consistency.

## Files
- Maps: `coupling_ready_maps.png`, `option3_smooth_proto.png`, `coupledE_ba_maps.png`
  (also in `NEW MAPS/coupledE/`).
- Params: `models/C/params.coupledE_{af,bor,sam,sea,eur}.json`, `params.coupledE_fx.json`
  (global dump fallback).
- Scripts: `optimize_modelC_coupled.py` (DUMP_CLIMATE), `assemble_smooth_coupledE.py`,
  `diag_coupledE.py`, `map_coupledE.py`, `prototype_smooth_regional.py`.
- BA nc: `ilamb/MODELS_SMOOTH_COUPLED/ED-ModelC-smooth/burntArea.nc`.

## NOT DONE / NEXT
- Gmail reply to Lei is DRAFTED but on hold (waiting until Richard sends). Needs the
  seams-vs-any-variation question + these results.
- Nothing promoted to canonical. This is a coupled-model candidate, separate from the paper E.
