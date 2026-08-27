# Half-strength green-up brake exact audit

The pinned official model blob was `7a8511b761e83788a5af3a824389099761e06432`. The dry-phase allocator remained at `fire_season_w=0.3`; only `greenup_brake` changed from 2.0 to 1.0. This was a scratch proxy audit, not an official evaluation.

The incumbent exact score was 0.717563738 with bias 0.754840048, RMSE 0.546217671, seasonality 0.861309117, and spatial distribution 0.879234184. The candidate scored 0.717396438 with bias 0.754864415, RMSE 0.546206087, seasonality 0.860515494, and spatial distribution 0.879190106. Candidate-minus-incumbent deltas were -0.000167301 Overall, +0.000024367 bias, -0.000011585 RMSE, -0.000793623 seasonality, and -0.000044078 spatial distribution.

Regional Overall deltas were Australia -0.000485554, boreal Asia -0.000602151, boreal North America -0.000267028, Central America +0.000304382, central Asia -0.000750714, equatorial Asia +0.000030975, Europe +0.002214149, Middle East +0.000892714, northern Africa -0.000181646, northern South America +0.000257656, southeast Asia +0.000083329, southern Africa -0.000439038, southern South America +0.000286083, and temperate North America +0.000834906.

Ecological model-to-observation ratios changed as follows: intact tropical closed canopy 0.901891447 to 0.902449390, temperate closed canopy 1.118133119 to 1.117678824, boreal forest 1.113922955 to 1.106166649, tropical open woodland 1.035449941 to 1.035273264, productive rangeland 1.037088826 to 1.036535122, cropland 0.925133417 to 0.922659079, and arid low fuel 1.242530218 to 1.239050756.

Halving every input after month 96 changed the first 96 candidate predictions by exactly zero. The complete three-prediction process took 42.548 seconds and reached a peak resident set of 6,137,053,184 bytes, or 5.716 GiB on macOS.

Decision: reject weakening the green-up brake. It makes several overburn ratios safer and slightly recovers intact tropical fire, but deepens cropland underburn and loses seasonality broadly enough to reduce Overall. The incumbent strengths remain preferable; a replacement needs a new wet-season production and standing-dead release clock rather than a weaker generic green-up response.
