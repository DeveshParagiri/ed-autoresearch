# Build, configure Africa, run

Only start this once `DISCOVERED.md` exists and Richard has agreed to proceed.

Work in **Richard's own directory**, never in Lei's. Copy the source out first.

```bash
mkdir -p ~/gED_africa && cd ~/gED_africa
cp -r <the ED source dir found in discovery> ./src
cd src
```

## 1. Fix the Makefile if the toolchain moved

`reference/Makefile` hardcodes

```
INC = -I/apps/netcdf/4.1.3/include -I/apps/IntelTBB/2017U3/include -I/apps/BerkeleyDB/4.6.21NC/include
LIB = -L/apps/netcdf/4.1.3/lib -L/apps/IntelTBB/2017U3/lib/intel64/gcc4.4 -L/apps/BerkeleyDB/4.6.21NC/lib
```

If discovery found those paths intact, change nothing. If not, load the modern modules and
repoint both lines at what `module show <name>` reports. Note that `CXX = gcc` builds C++
with the C driver, which works but may need `-lstdc++` at link time if it fails.

Build, and expect warnings. Only errors matter.

```bash
make clean && make 2>&1 | tail -30
```

If it fails, fix one error at a time and record what you changed in `DISCOVERED.md`. Do not
silently patch the science code. Build errors are toolchain problems, not model problems.

## 2. Set the domain to Africa

ED supports regional runs natively. In `MLU_io.defaults.cfg`,

```
is_site = 0
region  = "AFRICA"
```

and the AFRICA block already exists, do not redefine it.

```
AFRICA = { LATMIN = -40; LATMAX = 40; LONMIN = -20; LONMAX = 60; };
```

That box covers the whole continent, which is more than we strictly need but it matches the
paper's Africa fitting region closely enough and it is already defined and tested.

**Sanity check the cost before launching.** Africa at 0.5 degrees is roughly 160 by 160
cells, so about 26000 grid cells before the land mask, against roughly 260000 globally. So
expect very roughly a tenth of a global run. If a global run is known to take days, Africa is
hours. If you cannot find out what a global run costs, run **one year first** and multiply.

## 3. Decide the run length, and be honest about spin-up

ED needs equilibrated vegetation before a transient run means anything. Three cases.

**Case A, a usable restart exists** in `/gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/Result/`.
Copy it out, set in `ED_params.defaults.cfg`

```
restart          = 1
old_restart_read = 1
restart_dir      = "<Richard's own copy of the restart dir>"
```

and run only the transient period. This is the good case.

**Case B, a global restart exists but not an Africa one.** ED reads restarts per site, so a
global restart usually still works for a subdomain. Try it. If it fails, say so rather than
forcing it.

**Case C, no restart.** A cold Africa spin-up is needed. Do not launch it without telling
Richard the estimated wall time first. Get that estimate by running a single year and
scaling. If it is longer than overnight, the right move is probably to wait for Lei.

## 4. The experiment, and this is the whole point

Run the **same** Africa configuration twice, changing exactly one line.

```
run A   ED_params.defaults.cfg:137   fire_max_disturbance_rate = 0.2      the current value
run B   ED_params.defaults.cfg:137   fire_max_disturbance_rate = 1.0      what we think it should be
```

Everything else identical. Keep the outputs in separate directories.

`00_FINDINGS.md` predicts that offline, going from 0.2 to 1.0 takes burned area from 494 to
793 Mha globally, a rise of about 60 percent. If the Africa runs show a rise of that rough
size, the cap is confirmed as a real cause in the coupled model and it is a one-line fix.
If burned area barely moves, the cap is not the binding constraint and the dryness
hypothesis H1 moves to the front.

Either answer is a result. Record the Africa-total burned area for both.

## 5. The dump that matters most

Whatever happens with the cap, get ED to write its **live** driver state for the Africa run.

```
dryness_index_avg        the D_bar the fire equation actually sees
GPP                      live, per cell per month
AGB                      total above-ground biomass
```

`read_site_data.h:51` holds `cs->sdata->dryness_index_avg`, and `fire.cc:181` sets it via
`calcSiteDrynessIndex`. Find where the outputter writes monthly fields, `outputter.cc` and
`print_output.cc`, and add these three alongside burned area.

This is the single most valuable output of the trip. With it, the offline model can be
driven by ED's own live state and the remaining 326 Mha gap becomes diagnosable instead of
speculative. Specifically, compare the live `dryness_index_avg` distribution against
`D_high = 2.2134e6` and report what fraction of Africa cells exceed it. Offline that
fraction is 0.48 percent. If in the coupled model it is much larger, H1 is confirmed.

## 6. Bring it home

Copy the burned area output and the driver dump back, then follow `03_SCORE.md`. Keep the
files small by slicing to 2001-2016 on the HPC before transferring.
