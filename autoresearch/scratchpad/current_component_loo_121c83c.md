# Current component removal audit at 121c83c

The exact incumbent is 0.719892388. This operating-point diagnostic removes one
declared physical component at a time with every parameter fixed. It is directly
useful for pruning, but it is not a Shapley table because interactions are not
averaged over all 32,768 subsets.

| Removed component | Overall without | Full minus without |
| --- | ---: | ---: |
| arrival order | 0.719819767 | +0.000072621 |
| dead fuel pool | 0.719756369 | +0.000136019 |
| phenology | 0.719328003 | +0.000564385 |
| secondary open footprint | 0.719169816 | +0.000722572 |
| rare ignition | 0.716754253 | +0.003138135 |
| curing | 0.714056236 | +0.005836152 |
| annual regime closure | 0.712929314 | +0.006963074 |
| pathway hazards | 0.712643633 | +0.007248755 |
| fuel | 0.711912504 | +0.007979884 |
| surface opportunity bank | 0.711397728 | +0.008494660 |
| dryness | 0.709309654 | +0.010582734 |
| regime capacity | 0.708745691 | +0.011146697 |
| precipitation | 0.708309892 | +0.011582496 |
| cropland | 0.708077108 | +0.011815280 |
| temperature | 0.651260603 | +0.068631785 |

Every component is positive at the current operating point. The three weakest
terms earn mainly through seasonal or spatial structure rather than magnitude:
dead fuel contributes +0.000965622 seasonal while slightly costing bias, RMSE,
and spatial skill; phenology contributes +0.002924280 seasonal while slightly
costing bias and RMSE; arrival order contributes only +0.000072621 Overall. No
single-component prune is eligible. Targeted pair removal among the weakest
terms is the remaining simplification test.

The exhaustive `ar ablate` command was stopped after 93 of 32,768 subsets. At
the observed throughput it would require roughly a day and would interfere with
the higher-value held mechanism tests. No partial Shapley value is reported.
