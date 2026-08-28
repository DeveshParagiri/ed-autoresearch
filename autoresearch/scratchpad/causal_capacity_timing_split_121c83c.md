# Causal capacity/timing split at `121c83c`

The live-to-dead litter correction was decomposed in log-hazard space as
`log(F_t) = C_t + W_t`, where `C_t = EMA_tau(log(F_t))` is slow fuel capacity
and `W_t` is the current timing anomaly. The tested prediction used all of
`C_t` and either zero or one quarter of `W_t`. Six-, twelve-, and twenty-four-
month globally fixed time constants and litter blends 0.10 and 0.25 were
declared before scoring. The construction is pointwise and future-prefix
mutation was exactly zero for every bracket.

Only `tau6_b0.10_w0.25` improved annual-log loss in all four held whole-cell
folds while retaining positive aggregate held gain. It still worsened raw
cycle loss in two folds. Exact replay scored **0.719846352**, a
**-0.000046036** change from 0.719892388. Bias improved by 0.000347346, RMSE by
0.000039018, and seasonal skill by 0.000000422, but spatial skill fell by
0.000655983. Global burned-area ratio improved from 1.158499082 to 1.149620439.

Eleven of fourteen regions improved, with the largest gains in SHSA, CEAM,
NHSA, BONA, and BOAS. NHAF fell by 0.001094304, Australia by 0.000433049,
SHAF by 0.000125291, and MIDE by 0.000028270. No severe ecological pathology
was introduced; arid low-fuel changed from 1.256490060 to 1.255206429 times
observed and combined Congo from 0.815930314 to 0.802708601.

The factorization is therefore falsified as a standalone score improvement.
It partially separates the held annual and allocation effects, but slow
filtering cannot recover the missing annual spatial ordering and the exact
spatial cost dominates. No canonical edit, official evaluation, or Optuna
run follows.
