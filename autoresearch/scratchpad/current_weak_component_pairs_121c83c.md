# Weak component pair audit at 121c83c

The exact incumbent is 0.719892388. Every pair among the four weakest
operating-point components was removed, then each was paired with rare ignition
as the next weakest annual source. All canonical parameters remained fixed.

| Removed pair | Overall without | Delta from full | Inclusion interaction |
| --- | ---: | ---: | ---: |
| arrival order + dead fuel | 0.719653881 | -0.000238507 | -0.000029866 |
| arrival order + phenology | 0.719023339 | -0.000869049 | -0.000232042 |
| arrival order + secondary footprint | 0.719014665 | -0.000877723 | -0.000082530 |
| dead fuel + phenology | 0.718901162 | -0.000991226 | -0.000290821 |
| dead fuel + secondary footprint | 0.718916939 | -0.000975449 | -0.000116858 |
| phenology + secondary footprint | 0.718030169 | -0.001862219 | -0.000575261 |
| arrival order + rare ignition | 0.716656341 | -0.003236047 | -0.000025291 |
| dead fuel + rare ignition | 0.716564364 | -0.003328024 | -0.000053870 |
| phenology + rare ignition | 0.716352328 | -0.003540060 | +0.000162460 |
| secondary footprint + rare ignition | 0.715959763 | -0.003932625 | -0.000071918 |

No pair removal improves Overall. Most inclusion interactions are negative,
which means the weak terms are complementary at this operating point rather
than redundant. The only positive interaction, phenology with rare ignition,
is much smaller than their combined contribution and still loses 0.003540060.
The current stack has no safe weak-component prune.
