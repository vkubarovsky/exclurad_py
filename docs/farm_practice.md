# Running a campaign on the JLab farm

Everything here is measured on the 2026-09-28 pi0 reference campaign —
800 array tasks, 2 M generated events, 4 M through GEMC, zero FATAL — and
on what went wrong getting there. Numbers are from that run unless said
otherwise.

Companion: `docs/benchmarking.md` answers *how to measure*. This answers
*how to run*.

## Use these scripts, do not write your own

```
production/farm/build_tables.sh    eta table + one majorant PER MODE
production/farm/gen_array.sbatch   stage A, generation
production/farm/sim_array.sbatch   stage B, GEMC + background + recon + ntuple + DST
production/farm/post_array.sbatch  catch-up when B dies after GEMC
```

All four take `PROD_DIR`. This is not tidiness: every trap below is already
handled in them, and the only reliable way these lessons travel is in a file
someone copies. A hand-rolled chain starts from zero and rediscovers them
one run at a time.

```bash
export PROD_DIR=/volatile/clas12/vpk/<campaign>
mkdir -p $PROD_DIR/cards
cp ~/exclurad_py/cards/born_pi0_rga.txt $PROD_DIR/cards/rga_born.txt
cp ~/exclurad_py/cards/rad_pi0_rga.txt  $PROD_DIR/cards/rga_rad.txt
bash ~/exclurad_py/production/farm/build_tables.sh
```

## Before you submit 200 jobs: ten minutes

1. **`proposal-weighted bound` in `tables/build.log`** — about 1.95 on the
   RG-A box. This is the only number in that file worth reading. Ignore
   `max B`: it is one probe times the margin and came out 12.72 / 9.79 /
   9.87 on three runs of the same box. Ignore the median over cells: it
   counts the empty corner of the box the same as the cell every event
   lands in. A third of the grid getting zero probes is geometry — ten
   times the probes bought back 18 of 150 cells — not a budget problem.
2. **Smoke-run one task at NEV=50**, both modes. Check `setup 0s` (the
   majorant loaded rather than rebuilt), `health: clean`, no `#` in the
   LUND, `sigma_nb` present in `config.json`, and `rc_backend`.
3. **Seeds.** `SEEDBASE` must be passed per campaign. It used to be baked
   in, which would have made two campaigns share a random stream — the
   events differ if the box differs, but samples whose streams coincide
   cannot be combined or compared as independent.
4. **Background on disk.** See the tape trap below. `jcache get` takes a
   while; start it before you need it.

## Chaining the stages

Keep generation and simulation as separate arrays. One job doing both wants
twelve hours; the farm backfills and the queue is full of 2–5 hour jobs, so
a long request waits for a rare gap. It also decouples the failures: a
crash in reconstruction must not throw away three hours of GEMC.

To run unattended, chain with **`aftercorr`**, not `afterok`:

```bash
GEN=$(sbatch --parsable --array=1-200 gen_array.sbatch rad)
sbatch --dependency=aftercorr:$GEN --array=1-200 sim_array.sbatch rad inb
sbatch --dependency=aftercorr:$GEN --array=1-200 sim_array.sbatch rad outb
```

`aftercorr` starts task N of the second array when task N of the first
finishes — exactly the chunk-to-chunk mapping. `afterok` waits for the
whole array and one failed task blocks everything; `afterany` starts even
after a failure, which means feeding GEMC a missing LUND.

Note the two simulations share one generation: the generator knows nothing
about the torus, so one LUND feeds both polarities. That is deliberate —
inbending and outbending then differ only in acceptance, not in statistics.

## Sizing, measured over 800 tasks of 5000 events

| step | min | median | 95 % | max |
|---|---:|---:|---:|---:|
| generation, rad | 2.55 h | 3.61 h | 3.81 h | 4.00 h |
| generation, born | 0.03 h | 0.03 h | 0.03 h | 0.03 h |
| GEMC | 2.77 h | 4.90 h | 5.96 h | 10.17 h |
| bg-merger | 0.3 min | 0.5 min | 0.8 min | 2.3 min |
| recon-util | 1.15 h | 1.97 h | 2.49 h | 3.86 h |
| ntuple + DST | 0.3 min | 0.4 min | 0.5 min | 1.1 min |
| **whole simulation** | **4.01 h** | **6.93 h** | **8.55 h** | **9.80 h** |

Generation times are from before the 3.85x speedup; divide by about four.

**5000 events per task was wrong in both directions at once.** Too long to
backfill well — a 12 h request sat behind 4 h jobs — and still too short
for the tail: 3 of 800 died on the limit anyway. The spread across nodes is
a factor 3.7, so the limit is set by the tail and not by the median.

**Aim for 2–3 hours per task.** About 1000–1500 events for the full
simulation chain. Jobs are free; waiting is not. We share one account, so a
150-task array with a 10 h limit starves everyone else — my own two-chunk
catch-up sat PENDING on (Priority) for a full day.

## Waiting for a job, without waiting forever

Use `production/wait_for.sh`. Do not write `until grep -q WORD log; do sleep`.

    bash production/wait_for.sh slurm <name-or-jobid> [timeout_s] [poll_s]
    bash production/wait_for.sh ssh   <host> <cmd>    [timeout_s]
    bash production/wait_for.sh pid   <pid>           [timeout_s]
    bash production/wait_for.sh file  <sentinel>      [timeout_s]

A watcher must key on something the work writes **unconditionally** — its
own exit — and must carry a deadline that says why it gave up. A word that
appears only on success turns every failure into an infinite sleep, and
silence is the commonest failure mode there is. Four processes were found
on the mini on 2026-10-05 that had been sleeping six and seven days; one
was watching for `RESULT|ERROR` in a log whose job had died with
`AttributeError`, which is neither word.

For a job you launch yourself, write the sentinel from a trap rather than
from the success path, so it fires on a crash and on a kill too:

    trap 'echo "__DONE rc=$?" >> "$LOG"' EXIT

My own watchers during the reference campaign were half right and worth
contrasting. The one on the campaign finishing asked the scheduler and
exited on either outcome:

    [ "$n" -ge 200 ] && break                        # done
    [ -n "$n" ] && [ "$q" -eq 0 ] && break           # job gone, still short

That is the correct shape. Others in the same run were `until grep -q
"wrote " $LOG`, which survived only because the jobs happened to succeed.

## One more failure mode to recognise

A `exclurad_py.run` that has written `etatable.npz` and then sits at 100 %
CPU with an unreaped pool worker will not recover — kill it. Observed three
times, 41 hours each, while twins with the same card finished in 48 s; and
eleven orphaned multiprocessing workers (PPID 1) idle for seven days after
a `build_bound.py` that had exited normally. Same place both times, leaving
the multiprocessing block after the table is written. Cause not found.

## Traps

**The background is on tape.** `/cache` is tape-backed and the 50 nA
directories were EMPTY when I came to use them, including the one the
previous production had used. 100 files x 152 MB per polarity:

```bash
jcache get /mss/clas12/rg-a/production/background/rga_fall2018/tor-1.00_sol-1.00/50nA_10604MeV/10k/*.hipo
```

The old guard was `[ -f "$BG" ]`, which an evicted stub passes. The job then
spent three hours in GEMC and reported a *merge* failure for what was a
cold cache.

**Size guards the failure mode walks straight through.** `[ -s merged.hipo ]`
passes a 336 kB header-only file. Absolute thresholds are worse: 10 MB is a
5000-event number, and a 100-event test chunk legitimately merges to 3.5 MB
and was called fatal. Use relative ones — merged > gemc, because bg-merger
*adds* hits; recon > merged/20. Note reconstruction GROWS the file,
3.3 -> 7.3 MB measured.

**`ln -sf` succeeds against a missing target, and `ls | wc -l` counts
dangling symlinks exactly like files.** A fresh production directory with no
`lund_osg/` let `cp` fail, created 50 dangling links, reported "50 of 50
chunks" from an empty directory, printed DONE, and `rm -rf $W` deleted
200 k events. Make the output directories first, and never print DONE
without the thing it claims.

**The majorant cache falls through SILENTLY on a key mismatch and rebuilds
inside every job. The eta table raises.** So one failure mode tells you and
the other does not. Rebuild when model, ebeam, box, vcut, ihel, ptarg,
t_nucl or fermi changes — not for nev, seed, theta_e, y, xB, vz.

One majorant **per mode**: `_maj_key` carries `born_only`, so a born job
cannot use the rad majorant. Measured cost of getting that wrong: 25 s in
each of 200 jobs, silently.

**Drop `ETA_PROPOSAL_CAP` and two things break at once:** `eta_hi` goes
5.0 -> 36.5 so stage-1 acceptance craters, AND the majorant silently
rebuilds because `eta_hi` is in its key. The production set is

```
TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
```

Assert `maj_from_cache` and `setup < 1 s` in any harness and this class of
error cannot reach a number you report. An environment guard proves what you
*asked for*; only the artefact proves what you *got*.

## Reading a finished campaign

**The provenance copy is renamed, and the same search gives opposite answers
in one tree.** `gen_array` copies the run's `inputs/config.json` to
`$R/$MODE/prov/<mode>_<NNN>.config.json`, so a search by basename finds
nothing — but only in campaigns made after the rename:

```
prod_pi0_rga_t25_20260928   find -name config.json          ->   0
                            find -path '*prov*' -name '*.config.json' -> 400

night_amp2612_20261003      find -name config.json          -> 320
                            prov/*.config.json              ->   0
```

Anyone checking a mixed set with one search gets a confident negative on half
of it, which is worse than silence because it stops the search. Search
`*.config.json`, both ways, and count the files before concluding anything.

**What that file answers.** The whole `report` block: `sigma_nb` (the
normalisation), and `b_viol` / `max_b_viol` / `accepted` — the stage-2 numbers
that say whether the sample is exactly unweighted. Since 96555d9 the count is
also in the `.info` sidecar as `run.bound.*`, which survives any pipeline, but
older campaigns have it only in `prov/`.

**Do not read `health: clean` as zero violations.** `health()` warns above
0.1 % of accepts and said nothing below it until 96555d9. For scale, measured
over whole campaigns this way:

| campaign | b_viol / accepted | rate | worst |
|---|---:|---:|---:|
| reference million (amp2609) | — | 0.0044 % | x1.87 |
| night amp2613, 160 jobs | 316 / 2 145 731 | 0.0147 % | x5.41 |
| night amp2614, 160 jobs | 46 / 1 201 080 | 0.0038 % | x1.69 |

**`sf_usage.out_of_range_frac` in the sidecar is the model's own confession,
and on the current box it is about half.** The reference campaign reports
0.55-0.59 of structure-function CALLS outside `sf.validity`. Per EVENT, from
the generated LUND, it splits almost entirely one way:

| edge | born | rad |
|---|---:|---:|
| Q2 < 1.15 (validity floor) | 47.5 % | 49.1 % |
| \|t\| > 1.8 | 4.0 % | 4.0 % |
| W2 > 16.0 | 0.02 % | 0.06 % |
| any | 50.3 % | 52.0 % |

So the box's Q2 floor of 0.80 is where half the sample lives relative to the
model, not the `t` ceiling of 2.50 that looks like the aggressive edge. That
is a deliberate choice -- outbending acceptance reaches down to ~0.9 -- but it
is a choice to re-make before a 100 M run, not a detail. The t-SHAPE out there
is not pathological: the generated \|t\| spectrum below Q2 = 1 has
b = 1.3-1.95 GeV^-2 over windows from 0.10 to 1.20, which is the range
measured in photoproduction.

**A guard that can hang costs more than it saves.** `hipo-utils -dump` to
count events in a gemc.hipo is not header-only — it dumps every event and
does not return; `| head -1` does not save it. Two catch-up jobs burned six
hours each and produced empty logs. `-info` does it in 0.136 s. Put a
`timeout` on anything inside a guard.

**The job log filter must include warnings.** `gen_array` greps the run log
down to a few patterns; an eta-interpolation warning the 2026-09-28 million
actually emitted was invisible in all 200 job logs and surfaced a day later
only because someone ran the same tables by hand. The filter now takes
`WARNING` and `***`, and `interp` is recorded in `config.json`.

**The hadron pid.** `hipo_to_ntuple.groovy` defaults to 2212. On a neutron
target the LUND carries the struck neutron AND the spectator proton, so the
default silently picks the spectator: `badGen` stays 0, the ntuple is full,
every row is the wrong particle. `sim_array` now derives it from the card's
model.

**`/tmp` is noexec on the compute nodes**, so sqlite-jdbc cannot load its
native library and both `bg-merger` and `recon-util` die. The scripts set
`JDK_JAVA_OPTIONS` to put the JVM temp directory on `/volatile`. It does
**not** reproduce on a login node — test with a short sbatch job.

## The shared checkout, and where work happens

**`~/exclurad_py` is the shared checkout and has no owner.** It is the copy on
`production` that everyone reads and merges into, and it is what `gen_array`
runs from. Work happens in per-agent worktrees beside it:

```
~/exclurad_py               shared, production, no owner -- do not leave it dirty
~/exclurad_py_speed         speed-numba
~/exclurad_py_fermi         fermi-flux
~/exclurad_py_htilde        htilde-model
~/exclurad_py_jpsi_fermi    jpsi-fermi
~/exclurad_py_ref           detached, for byte comparisons
```

`git worktree add ~/exclurad_py_<name> -b <branch>` makes another, and
`git worktree list` shows what exists.

Two separate reasons, and the second is the one that nearly bit on
2026-10-07:

- A feature branch merged into the shared checkout to "try something" leaves a
  branch named `production` that is actually something else, and production
  arrays then get the wrong scripts with nothing to say so.
- **Git cannot tell whose uncommitted file is whose.** A `git stash`,
  `git checkout <branch>` or `git reset --hard` by anyone in that directory
  takes EVERY uncommitted change with it, including another session's. Two
  sessions were editing different files there ten minutes apart; it stayed
  harmless only because each staged its own path and committed within minutes.

So: a change that takes longer than a few minutes goes in your own worktree. A
one-line fix in a file you own may stay here, but then commit it now, not at
the end of the conversation.

## The interpreter is not the code checkout

`gen_array` has both, and they are different things:

```
V=${EXRU:-$HOME/exclurad_py}                  the code
PY=${EXRU_PY:-/work/clas12/vpk/venv-prod-numba}   the interpreter
```

The venv is the production stack with numba added and nothing else changed
— `--system-site-packages`, so python 3.9.25, numpy 1.23.5 and scipy 1.9.3
are the system ones. That matters: every byte-identity result was measured
against numpy 1.23.5, and a venv that pulled its own numpy 2.x would
invalidate all of them silently. `RC_NUMBA=1` is set so a job reaching an
interpreter without numba dies instead of quietly running 2.3x slower, and
`NUMBA_CACHE_DIR=$W` keeps the compile cache private to the job.

## After it finishes

```bash
grep -l FATAL $PROD_DIR/*/logs/*.log | wc -l        # expect 0
python3 production/gemc_check/gen_vs_rec.py <ntuple.f32> <tag> <outdir> [--log]
```

and from the `report` block of each `config.json`: `b_viol` and
`max_b_viol` (the 2026-09-28 million: 0.0044 %, worst x1.87), the spread of
`sigma_nb` across chunks (1.4 %, matching 1/sqrt(5000)), and `rc_backend`.

Then add a row to `docs/PRODUCTIONS.md`. A sample nobody can date, locate
and attribute is not a result.
