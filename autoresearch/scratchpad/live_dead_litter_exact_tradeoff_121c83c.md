# Exact Overall-first replay of live-to-dead litter forms

This scratch-only replay evaluates six fixed forms from the held litter experiment against canonical `121c83c`. The current canonical blob is `b82c285259f35f0f942ddc8a78663d8d14dd36b1`. The incumbent exact proxy is Overall `0.719892388`, bias `0.757875412`, RMSE `0.548437882`, seasonal `0.860504847`, and spatial `0.884205917`, with global burned-area ratio `1.158499082`.

The revised decision rule permits limited held timing reversals when exact Overall improves and no severe ecological pathology appears. A severe pathology is a nonfinite ratio, a ratio outside 0.25 to 4.0, or a stratum ratio changing more than 25% relative to the incumbent. GFED, regions, ecological masks, and Congo masks enter only post-prediction audits.

The exact live/dead equations were evaluated over independent spatial chunks without changing their pointwise state. Maximum relative litter-mass closure error over all full-grid chunks is `7.68858285095e-16`; the 64-cell prefix audit closes to `3.62741940422e-16`. This closure applies to live/dead litter input, turnover, decomposition, combustion, and terminal pools. Direct replacement is a capacity substitution and relative allocation is a timing factor, so neither is claimed to conserve incumbent hazard itself. Reversing and perturbing all inputs after month 96 changes every candidate before month 96 by exactly `0`.

## Exact scores

| Candidate | Overall | Delta | Bias delta | RMSE delta | Seasonal delta | Spatial delta | Area ratio | Area-ratio delta | Severe ecology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Direct load, w 0.10 | 0.719942836 | +0.000050448 | +0.000356181 | +0.000162693 | +0.000143688 | -0.000573012 | 1.150978265 | -0.007520817 | none |
| Direct load, w 0.25 | 0.719775211 | -0.000117177 | +0.000482745 | +0.000257895 | +0.000327089 | -0.001911507 | 1.139642893 | -0.018856189 | none |
| Direct load, w 0.50 | 0.718999499 | -0.000892889 | -0.000623555 | +0.000010573 | +0.001578952 | -0.005440990 | 1.120605486 | -0.037893596 | none |
| Relative allocation, w 0.10 | 0.719864614 | -0.000027774 | -0.000578550 | -0.000049901 | +0.000023674 | +0.000515810 | 1.176402508 | +0.017903426 | none |
| Relative allocation, w 0.25 | 0.719209919 | -0.000682469 | -0.002276055 | -0.000645667 | +0.000231825 | -0.000076784 | 1.203057624 | +0.044558542 | none |
| Relative allocation, w 0.50 | 0.716551324 | -0.003341064 | -0.006683458 | -0.002860066 | +0.000008999 | -0.004310729 | 1.246959838 | +0.088460756 | none |

Direct load replacement at blend 0.10 is the sole candidate with positive exact Overall. It improves bias, RMSE, seasonal skill, and global area calibration, while accepting a `-0.000573012` spatial component tradeoff. Its exact gain is `+0.000050448`, which is positive but does not change the score rounded to three decimals.

## Regional Overall deltas

| Candidate | BONA | TENA | CEAM | NHSA | SHSA | EURO | MIDE | NHAF | SHAF | BOAS | CEAS | SEAS | EQAS | AUST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 0.10 | +0.001564783 | +0.000271397 | +0.001111225 | +0.001058823 | +0.002050483 | -0.000498663 | -0.000011539 | -0.000844397 | +0.000077404 | +0.000812728 | +0.000335385 | +0.000210384 | +0.000141848 | -0.000421110 |
| Direct 0.25 | +0.002833000 | +0.000527341 | +0.003073998 | +0.002692631 | +0.005052260 | -0.000257335 | -0.000025882 | -0.002390119 | -0.000484335 | +0.002205953 | +0.001927456 | +0.000420179 | +0.000351170 | -0.001309347 |
| Direct 0.50 | +0.005068123 | +0.001543042 | +0.005926403 | +0.005528714 | +0.010370532 | +0.001018454 | -0.000063952 | -0.005076040 | -0.002785868 | +0.005546335 | +0.003406395 | +0.000523180 | +0.000465284 | -0.002530640 |
| Relative 0.10 | +0.001103630 | -0.001372188 | -0.002318082 | -0.003731880 | -0.004205431 | -0.001660389 | -0.001722492 | +0.002338293 | -0.000226760 | +0.000704406 | +0.000525306 | -0.001016862 | -0.000382202 | +0.000745835 |
| Relative 0.25 | +0.002557775 | -0.003357680 | -0.005717278 | -0.009138742 | -0.010172033 | -0.006081996 | -0.004303364 | +0.005246158 | -0.001911012 | +0.001606049 | +0.001992317 | -0.002834921 | -0.001216800 | +0.001468303 |
| Relative 0.50 | +0.004934836 | -0.006728648 | -0.011178064 | -0.017837054 | -0.019556008 | -0.010845307 | -0.008355703 | +0.007917694 | -0.006968655 | +0.003188127 | +0.003703975 | -0.006708520 | -0.002396419 | +0.001227983 |

The accepted direct 0.10 form improves ten of fourteen regional Overall scores. Its losses are limited to Europe, the Middle East, northern Africa, and Australia, with the largest being northern Africa at `-0.000844397`.

## Full ecology and Congo audit

| Stratum | Incumbent | Direct 0.10 | Direct 0.25 | Direct 0.50 | Relative 0.10 | Relative 0.25 | Relative 0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Intact tropical closed | 0.980541934 | 0.975267888 | 0.967355921 | 0.954166938 | 0.989697135 | 1.003422307 | 1.026277312 |
| Temperate closed | 1.002383541 | 0.998077933 | 0.991618869 | 0.980852156 | 1.009686478 | 1.020635416 | 1.038869091 |
| Boreal | 1.052314143 | 1.058971184 | 1.068949553 | 1.085560951 | 1.062367675 | 1.077440188 | 1.102540233 |
| Tropical open | 1.062626253 | 1.053549134 | 1.039917047 | 1.017153007 | 1.076786958 | 1.097956945 | 1.133052111 |
| Productive rangeland | 0.979995892 | 0.983896282 | 0.989743417 | 0.999479396 | 0.988045893 | 1.000107657 | 1.020175610 |
| Cropland | 0.941778648 | 0.950200649 | 0.962808499 | 0.983754654 | 0.957261792 | 0.980430971 | 1.018899011 |
| Arid low fuel | 1.256490060 | 1.255465815 | 1.253928148 | 1.251361913 | 1.257678159 | 1.259458357 | 1.262420225 |
| Democratic Republic of the Congo | 0.825776419 | 0.813180765 | 0.794181581 | 0.762231138 | 0.846280847 | 0.876614736 | 0.926074732 |
| Republic of the Congo | 0.672925302 | 0.668147496 | 0.660976421 | 0.649012878 | 0.683778419 | 0.700016549 | 0.726969906 |
| Congo combined | 0.815930314 | 0.803838257 | 0.785600996 | 0.754938036 | 0.835813041 | 0.865238932 | 0.913249134 |

No candidate crosses the severe-ecology threshold. For the accepted direct 0.10 form, tropical-open, productive-rangeland, cropland, and arid ratios move closer to one; boreal rises slightly from `1.05231` to `1.05897`, and Congo combined falls modestly from `0.81593` to `0.80384`.

The final Overall-first decision is `accept=1` for direct litter-load replacement at blend 0.10. It is a valid but very small exact improvement and should be treated as a marginal canonical candidate, not evidence that the broader litter family resolved the remaining waveform limitation. The replay is reproducible with `autoresearch/scratchpad/live_dead_litter_exact_tradeoff_121c83c.py`; no canonical artifact or official results ledger was changed.
