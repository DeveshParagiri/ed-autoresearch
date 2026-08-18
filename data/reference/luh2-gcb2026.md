# LUH2-GCB2026 source manifest

The land-use inputs are from the official Zenodo record [Land-Use Harmonization 2 for Global Carbon Budget 2026](https://zenodo.org/records/20932027). They are stored outside Git at `$ED_FIRE_SOURCE_ROOT/data/luh2/GCB2026` and exposed to this workspace through `data/inputs/candidate-drivers/luh2`.

| File | Bytes | Official MD5 |
| --- | ---: | --- |
| `states.nc` | 6,080,810,028 | `8a74fa6273d38307a14780bc654e0028` |
| `transitions.nc` | 16,842,976,839 | `d1a6c7d5fae587beb186cb1af1cfa032` |
| `management.nc` | 1,534,118,739 | `d20b4dfd9813384203aef57593c5609c` |

Run `python scripts/download_luh2_gcb2026.py --output-dir "$ED_FIRE_SOURCE_ROOT/data/luh2/GCB2026"` to retrieve or verify the files. The all-data wrapper invokes the same downloader when `--fetch-public` is set and the three files are absent.
