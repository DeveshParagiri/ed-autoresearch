# Exact interface-access proxy at `121c83c`

Under the specified Overall-first rule, Simpson-weighted access at `a=.05` is accepted as a scratch proxy. It raises exact Overall from `0.719892388` to `0.719911197`, a gain of `+0.000018809`, and no ecological ratio meets the predeclared severe-pathology threshold. This is a very thin tradeoff win, not a robust promotion result.

| Candidate | Overall | Δ Overall | Bias | RMSE | Seasonal | Spatial | Global area ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 0.719892388 | — | 0.757875412 | 0.548437882 | 0.860504847 | 0.884205917 | 1.158499082 |
| pairwise `.05` | 0.719869011 | -0.000023377 | 0.757486525 | 0.548348654 | 0.860490478 | 0.884670744 | 1.178747800 |
| pairwise `.10` | 0.719570646 | -0.000321742 | 0.756726305 | 0.548123054 | 0.860492576 | 0.884388244 | 1.199546657 |
| pairwise `.20` | 0.718144016 | -0.001748372 | 0.754308662 | 0.547225801 | 0.860502884 | 0.881456930 | 1.242855849 |
| Simpson `.05` | 0.719911197 | +0.000018809 | 0.757658539 | 0.548396583 | 0.860490478 | 0.884613802 | 1.173023102 |
| Simpson `.10` | 0.719779901 | -0.000112487 | 0.757222675 | 0.548278373 | 0.860490478 | 0.884629602 | 1.187853928 |
| Simpson `.20` | 0.719058688 | -0.000833700 | 0.755786620 | 0.547790150 | 0.860507685 | 0.883418834 | 1.218462349 |

The accepted proxy earns only through Spatial `+0.000407885`; Bias changes `-0.000216873`, RMSE `-0.000041299`, and Seasonal `-0.000014369`. Its global area ratio rises by `+0.014524019`, from `1.158499082` to `1.173023102`. Prefix causality is bit-exact with `max_abs=0` after perturbing every input after month 96.

| GFED region | Δ Overall, Simpson `.05` | GFED region | Δ Overall, Simpson `.05` |
| --- | ---: | --- | ---: |
| BONA | +0.000097325 | TENA | -0.001390998 |
| CEAM | -0.001871391 | NHSA | -0.002946062 |
| SHSA | -0.003240125 | EURO | -0.000984365 |
| MIDE | -0.000926154 | NHAF | +0.001613888 |
| SHAF | +0.000020702 | BOAS | +0.000148190 |
| CEAS | +0.000580946 | SEAS | -0.000174216 |
| EQAS | -0.000124496 | AUST | +0.000278696 |

Six of fourteen regions improve. The losses are concentrated in the Americas and the gains chiefly in northern Africa, boreal Asia, central Asia, and Australia; that breadth is weak and should remain visible beside the global win.

| Ecological audit | Incumbent ratio | Simpson `.05` ratio | Change |
| --- | ---: | ---: | ---: |
| intact tropical closed | 0.980541934 | 0.983372643 | +0.002830709 |
| temperate closed | 1.002383541 | 1.013299830 | +0.010916289 |
| boreal | 1.052314143 | 1.054094449 | +0.001780306 |
| tropical open | 1.062626253 | 1.073099889 | +0.010473636 |
| productive rangeland | 0.979995892 | 0.999813460 | +0.019817568 |
| cropland | 0.941778648 | 0.944377228 | +0.002598580 |
| arid low fuel | 1.256490060 | 1.268958169 | +0.012468109 |
| Democratic Republic of the Congo | 0.825776419 | 0.832661127 | +0.006884708 |
| Republic of the Congo | 0.672925302 | 0.685991847 | +0.013066545 |
| combined Congo | 0.815930314 | 0.823213233 | +0.007282919 |

The access signal improves the existing underburn in intact tropical forest, productive rangeland, cropland, and both Congo countries. It also increases already-high temperate, tropical-open, and arid ratios, but the largest relative movement is about two percent and no ratio crosses the pathology bounds of `[0.25,4]` or changes by more than 25 percent. The exact evidence therefore passes the user's rule, while its tiny gain, eight regional losses, component tradeoffs, and higher global overburn argue against canonical promotion without a stronger independent reason.
