# First hour on the HPC. Find out what is actually there

Nothing below assumes anything about the machine. Work through it in order and record every
answer in `DISCOVERED.md`, which you create as you go. Richard moves between machines and
that file is the only thing that survives the trip.

Do not build or run anything yet. This step is only about finding out what exists.

## 1. Get on the machine and establish the basics

The group HPC hosts the paths seen in the configs, `/gpfs/data1/hurttgp/`. Richard should
already have an account. If he needs to log in interactively, tell him to type

```
! ssh <user>@<host>
```

in the prompt, since the `!` prefix runs it in his session and the output lands here.

Record: hostname, username, home directory, and the answer to each of these.

```bash
whoami; hostname; pwd
df -h ~ | tail -1                      # how much space in home
echo $SCRATCH; ls -d /scratch/$USER 2>/dev/null   # where large output can go
which sbatch qsub                      # SLURM or PBS, this decides how jobs are submitted
module avail 2>&1 | head -40           # what module system and what is available
```

## 2. Can Richard see Lei's data at all

Everything hinges on this. If these are unreadable the trip changes shape completely.

```bash
ls -ld /gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/
ls    /gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/Data/Climate/CRUJRA/v2.4/ | head
ls    /gpfs/data1/hurttgp/gel1/leima/AssignTask/gED/Result/ | head -30
```

The third one is `restart_dir` from `ED_params.defaults.cfg` line 44. **That is where spun-up
restart files live and it is the single most valuable thing on this machine.** A usable
restart is the difference between an afternoon and a fortnight.

If you find restarts, record for each one the filename, the size, the date, and anything in
the name that indicates domain or year.

## 3. Find the actual ED source and run directory

`reference/` holds copies from the external drive, but the HPC copy is authoritative and may
have drifted.

```bash
find /gpfs/data1/hurttgp -maxdepth 5 -name "fire.cc" 2>/dev/null | head
find /gpfs/data1/hurttgp -maxdepth 5 -name "ED_params*.cfg" 2>/dev/null | head
find ~ -maxdepth 4 -name "fire.cc" 2>/dev/null | head        # Richard's own copy, if any
```

When you find them, **diff against `reference/`** and record every difference. Two lines
matter most.

```bash
grep -n "fire_max_disturbance_rate" <found>/ED_params.defaults.cfg     # expect 0.2
grep -n "fire_suppression " <found>/ED_params.defaults.cfg             # expect 0
```

## 4. Richard mentions he did a spin-up on this machine before, not Africa

Find it. It may be reusable, and even if the domain is wrong it tells us how he ran things.

```bash
ls -lt ~ | head -20
find ~ -maxdepth 3 -type d \( -name "*ED*" -o -name "*gED*" -o -name "*spin*" \) 2>/dev/null | head
find ~ -maxdepth 4 -name "*.restart*" -o -maxdepth 4 -name "restart*" 2>/dev/null | head
grep -rl "region" ~ --include="*.cfg" 2>/dev/null | head
```

For anything found, record the region it was configured for, the years, and whether it
completed. Old job scripts are gold, they show the queue, the walltime and the module loads
that actually worked on this machine.

```bash
find ~ -maxdepth 3 \( -name "*.sh" -o -name "*.sbatch" -o -name "*.pbs" \) -newer /etc/hostname 2>/dev/null | head -20
```

## 5. Check the build toolchain exists

The Makefile in `reference/` expects three libraries at fixed paths.

```bash
ls -d /apps/netcdf/4.1.3 /apps/IntelTBB/2017U3 /apps/BerkeleyDB/4.6.21NC 2>&1
module avail netcdf 2>&1 | head
gcc --version | head -1
```

Those paths are old. If they are gone, the Makefile needs its `INC` and `LIB` lines
repointed at whatever the current modules provide. Record what is actually available.

## 6. Write DISCOVERED.md

Before doing anything else, write `DISCOVERED.md` with a section per item above. Then stop
and show Richard a short summary with a clear recommendation on whether to proceed to
`02_RUN_AFRICA.md` or whether something is missing that only Lei can supply. If the latter,
add it to `QUESTIONS_FOR_LEI.md`.

The honest possibilities to name for him, if they turn out to be true.

- No readable restart anywhere means a cold Africa spin-up, which is a long job, and it is
  worth asking whether waiting for Lei is faster than starting one.
- No readable climate data means nothing can run and the trip is a question list, not a run.
- A working old spin-up of his own, even for the wrong region, means the build and the job
  submission are already solved and only the domain has to change, which is the best case.
