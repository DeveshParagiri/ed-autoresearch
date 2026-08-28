# Pointwise land-cover interface result at `121c83c`

The mechanism is rejected. Neither the direct pairwise interface nor the Simpson-weighted mosaic cleared the four-fold annual-log, normalized-allocation, and raw-cycle gate. The exact canonical score and ecological ratios were therefore deliberately not computed.

The run loaded model blob `b82c285259f35f0f942ddc8a78663d8d14dd36b1`, reproduced the `0.719892388` incumbent, and evaluated all 18,316 non-benchmark land-mask cells in four disjoint spatial whole-cell folds containing 4,954, 4,573, 4,349, and 4,440 cells. The pointwise future-perturbation check returned prefix `max_abs=0` through month 96.

| Formulation and fixed bracket | Annual-log result | Allocation/raw-cycle result | Gate |
| --- | --- | --- | --- |
| Pairwise access only, `a=.20, b=0` | Improved all four folds by `+.01987,+.01579,+.00738,+.01043` | Allocation failed folds 2–3; raw cycle failed fold 1 | FAIL |
| Pairwise combined, `a=.20, b=.10` | Improved all four folds by `+.01170,+.00468,+.00129,+.00082` | Allocation failed folds 2–3; raw cycle failed folds 1 and 3 | FAIL |
| Simpson access only, `a=.20, b=0` | Improved all four folds by `+.01137,+.00985,+.00464,+.00718` | Allocation failed folds 2–3; raw cycle failed fold 1 | FAIL |
| Simpson combined, `a=.20, b=.10` | Improved all four folds by `+.00833,+.00627,+.00226,+.00311` | Allocation failed folds 2–3; raw cycle failed fold 1 | FAIL |
| Pairwise spread brake only, `a=0, b=.20` | Degraded all four folds | Mostly degraded allocation and raw cycle | FAIL |
| Simpson spread brake only, `a=0, b=.20` | Degraded all four folds | Mixed, with no all-fold gain | FAIL |

The result is informative rather than null. Natural-managed co-occurrence contains a stable missing annual-magnitude signal: both access-only formulations improve annual-log loss in every block. That signal does not supply the missing seasonal allocation, and the independent crop/urban mosaic brake points in the wrong annual direction. The candidate therefore cannot be promoted as one coupled mechanism. A future annual-map mechanism could revisit the access signal only if it is paired with a separately validated temporal allocator; this experiment provides no license to add a local fragmentation coefficient to the canonical model.

The implementation is `land_cover_interface_mechanism_121c83c.py`; complete fixed-bracket evidence is in `land_cover_interface_mechanism_121c83c.txt`. The overlap audit is `land_cover_interface_audit_121c83c.md`. No neighbour, coordinate, region, target, or completed-record statistic enters prediction.
