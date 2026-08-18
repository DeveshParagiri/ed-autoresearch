# Research memory

Updated: 2026-08-18

`experiment.stock-baseline` is complete. Three runs evaluated the pinned TRENDY v14 EDv3 S3 burned-area output, and the two matched runs at revision `d1518e88f03c371303bd06d9ff767d6ea1ee3358` reproduced the candidate artifact, complete metric vector, ILAMB score products, and five canonical figures byte for byte. The GFED5 overall score is 0.437410 and the GFED4.1s overall score is 0.477503. The stock field underestimates global mean burned area under both products, and its weakest scalar component is spatial distribution, especially against GFED5.

All declared data resolve. The active burned-area contract remains unchanged and locks the native ED output, GFED5, GFED4.1s, both ILAMB configurations, ILAMB 2.7.3, the 2001-2016 period, eight reporting regions, metric fields, plot scales, and five canonical figures. GFED5 emissions are installed but remain outside this contract because the native baseline has no `fFire` output.

The blocker is now the model implementation, not the evaluation system. This workspace still lacks the ED source revision, build configuration, input deck, and producing command behind the pinned artifact. The immediate action is to obtain and version that material or establish a clean mechanistic implementation with the same `burntArea.nc` interface, then declare one interpretable descendant experiment aimed at a specific spatial or seasonal failure. Do not introduce Optuna until that direct candidate and its development objective reproduce without search.

Model A through E parameters, outputs, masks, and expected scores remain excluded from the clean research line.
