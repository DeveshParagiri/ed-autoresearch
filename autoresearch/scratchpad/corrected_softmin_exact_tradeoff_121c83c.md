# Exact Overall-first replay of corrected limiting-factor candidates

This scratch-only replay evaluates the four preregistered held-screen survivors against canonical `121c83c`. The current canonical blob is `b82c285259f35f0f942ddc8a78663d8d14dd36b1`. The incumbent exact proxy is Overall `0.719892388`, bias `0.757875412`, RMSE `0.548437882`, seasonal `0.860504847`, and spatial `0.884205917`, with global burned-area ratio `1.158499082`.

The revised decision rule accepts a mild timing tradeoff only when exact Overall improves and no severe ecological pathology appears. A severe ecological pathology means a new nonfinite ratio, a ratio outside 0.25 to 4.0, or a change exceeding 25% relative to the incumbent stratum ratio. GFED, regional labels, ecological masks, and country masks enter only post-prediction audits.

## Exact scores

| Candidate | Overall | Delta | Bias delta | RMSE delta | Seasonal delta | Spatial delta | Area ratio | Area-ratio delta | Prefix max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmin beta 25, w 1 | 0.713521545 | -0.006370843 | -0.003310144 | -0.004308349 | -0.007765430 | -0.012161942 | 1.525271533 | +0.366772450 | 0 |
| softmin beta 8, w 1 | 0.702426865 | -0.017465523 | -0.007851703 | -0.007636064 | -0.012219838 | -0.051983944 | 2.076306081 | +0.917806999 | 0 |
| hard minimum, w 0.10 | 0.719697620 | -0.000194768 | -0.000057731 | -0.000187748 | -0.000397400 | -0.000143210 | 1.167259812 | +0.008760729 | 0 |
| harmonic, w 0.10 | 0.719649762 | -0.000242626 | -0.000133121 | -0.000397754 | +0.000207412 | -0.000491914 | 1.189142966 | +0.030643884 | 0 |

The full softmins reverse their sampled held annual gains at exact global scale and strongly overinflate total burned area. The two weak formulations stay close to the incumbent, but their small timing or regional benefits do not overcome exact bias, RMSE, and spatial losses. Since exact Overall is negative for every candidate, none is acceptable under the user's rule.

## Regional Overall deltas

For softmin beta 25 at full replacement, regional deltas are `bona -0.115243790`, `tena -0.053563014`, `ceam -0.043777519`, `nhsa -0.030383036`, `shsa -0.021108434`, `euro -0.035566584`, `mide -0.017431587`, `nhaf +0.004263272`, `shaf -0.001810983`, `boas +0.062435132`, `ceas +0.021662426`, `seas -0.011763598`, `eqas -0.019865126`, and `aust -0.005961871`.

For softmin beta 8 at full replacement, regional deltas are `bona -0.294490313`, `tena -0.086124377`, `ceam -0.066158322`, `nhsa -0.049514985`, `shsa -0.036253674`, `euro -0.060362518`, `mide -0.034462863`, `nhaf +0.006658368`, `shaf -0.002973539`, `boas +0.037001016`, `ceas +0.049346560`, `seas -0.023532377`, `eqas -0.063394222`, and `aust +0.002870116`.

For hard minimum at blend 0.10, regional deltas are `bona +0.002406005`, `tena -0.002383316`, `ceam -0.001980555`, `nhsa -0.001151674`, `shsa -0.001024758`, `euro -0.002105716`, `mide -0.000986671`, `nhaf +0.000175791`, `shaf -0.000128405`, `boas +0.001230421`, `ceas +0.000977045`, `seas -0.000336199`, `eqas -0.000293759`, and `aust -0.000226722`.

For harmonic limiting at blend 0.10, regional deltas are `bona +0.005673192`, `tena -0.008683769`, `ceam -0.006251944`, `nhsa -0.005226499`, `shsa -0.003403797`, `euro -0.005180470`, `mide -0.002821083`, `nhaf +0.001155799`, `shaf +0.000401919`, `boas +0.002980524`, `ceas +0.002801987`, `seas -0.001413875`, `eqas -0.002300112`, and `aust +0.000376595`.

## Full ecology and Congo audit

| Stratum | Incumbent | Beta 25, w 1 | Beta 8, w 1 | Hard, w 0.10 | Harmonic, w 0.10 |
|---|---:|---:|---:|---:|---:|
| Intact tropical closed | 0.980541934 | 1.330488649 | 1.661352346 | 0.992065420 | 1.028344838 |
| Temperate closed | 1.002383541 | 2.885716832 | 5.047481083 | 1.043842935 | 1.097657675 |
| Boreal | 1.052314143 | 5.397578808 | 16.665041979 | 1.081133908 | 1.124230765 |
| Tropical open | 1.062626253 | 1.179577033 | 1.279017569 | 1.066788683 | 1.084473145 |
| Productive rangeland | 0.979995892 | 1.178034724 | 1.432178185 | 0.986506061 | 1.001452451 |
| Cropland | 0.941778648 | 1.142870513 | 1.317570408 | 0.949338467 | 0.971094659 |
| Arid low fuel | 1.256490060 | 5.371365571 | 15.284334147 | 1.278751502 | 1.317511028 |
| Democratic Republic of the Congo | 0.825776419 | 0.915028295 | 0.995223120 | 0.828937369 | 0.840752294 |
| Republic of the Congo | 0.672925302 | 0.825371703 | 0.952602056 | 0.678766042 | 0.692949286 |
| Congo combined | 0.815930314 | 0.909252948 | 0.992477629 | 0.819263886 | 0.831231370 |

Softmin beta 25 creates severe new intact-tropical, temperate, boreal, and arid failures. Softmin beta 8 is more pathological, especially boreal `16.6650` and arid low-fuel `15.2843`. Hard-min and harmonic blends produce no severe ecological pathology, including in both Congo country masks, but exact Overall still decreases.

The final decision is `accept=0`: no named candidate has positive exact Overall. The complete replay is reproducible with `autoresearch/scratchpad/corrected_softmin_exact_tradeoff_121c83c.py`; no canonical artifact or official evaluator ledger was changed.
