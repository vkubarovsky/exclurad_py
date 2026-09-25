# Fermi motion for a deuteron target — working brief

Branch `fermi`, forked from `production` at `16412f8`.
Author of this brief: the session that fixed the stage-2 sampling bug on
2026-09-25. Ask it (or VPK) before deviating from anything marked **must**.

---

## 1. The task

RG-B runs a deuteron target. Give the struck nucleon a Fermi momentum drawn
from a deuteron wave function, emit the spectator nucleon as a real final-state
particle, and put the whole thing behind a card flag. It applies to all four
combinations — pi0/p, pi0/n, eta/p, eta/n — because a proton measurement on a
deuteron target is also planned.

**Phase 1 scope:** PWIA, unpolarised deuteron, spectator on-shell, struck
nucleon off-shell by the spectator prescription, no final-state interactions.

**Out of scope for now:** FSI, tensor polarisation of the deuteron (`ptarg` is
a longitudinal *nucleon* polarisation and means something else for a deuteron),
coherent production on the deuteron (gamma* d -> pi0 d), and any change to the
proton-target physics.

---

## 2. Where to work

| machine | path | notes |
|---|---|---|
| mac mini (primary) | `/Users/vpk/exclurad_py_fermi` | git worktree on branch `fermi`. **This is your directory.** |
| mac mini | `/Users/vpk/exclurad_py` | **NOT yours.** Branch `production`, actively used. |
| ifarm2402 | make your own: `git clone git@github.com:vkubarovsky/exclurad_py.git /volatile/clas12/vpk/exclurad_py_fermi && cd $_ && git checkout fermi` | git reaches GitHub from ifarm. |
| ifarm2402 | `/volatile/clas12/vpk/exclurad_py` | **NOT yours.** Production checkout. |
| clonfarm11 | — | **Do not use today.** See §9. |

`/Users/vpk/exclurad_py_fermi` is a git *worktree*, not a clone: it shares the
object database with the main checkout but has its own HEAD, index and files.
Two consequences:

- Git will refuse to check out `production` there. That is the safety property;
  do not work around it.
- The stash stack is **shared**. Never use bare `git stash` / `git stash pop` —
  you could pop someone else's work. Use a temporary WIP commit instead.

Rebase onto `production`; do not merge back and forth. The diff has to stay
readable for review.

---

## 3. Read before writing any code

| what | where | why |
|---|---|---|
| `_draw_vertex()` | `exclurad_py/generator/events.py:608` | the hook — see §4 |
| the four `rc_factor(..., Ebeam=self._E, ...)` calls | events.py 735, 904, 1021, 1062 | per-event energy already flows through |
| `self._mT = float(getattr(self.sf, "mT", _MP_T))` | events.py:163 | the target mass you are about to make dynamic |
| `_maj_key()` / `RCTable._key()` | events.py:527, tables.py:84 | both key on the box, so stale caches cannot be silently reused |
| `FIELDS`, `N_REQUIRED`, `DEFAULT_OPT` | `exclurad_py/runcard.py:39-66` | the card-extension mechanism you will use |
| commit `225d4d9` and its message | `git show 225d4d9` | the sampler bug fixed the same day; §5 depends on understanding it |

---

## 4. Architecture — the good news

`_draw_vertex()` runs **on every trial, before `_propose()`**, and already
varies the beam energy per event (target straggling, `itarg`). Downstream:

    eta table :  self.table(Q2, W2, t, phi)        <- no energy; key at nominal Ebeam
    rc_factor :  rc_factor(..., Ebeam=self._E, ...) <- the event's real energy

So the generator already lives with a table built at one energy and exact
radiative corrections computed at another. The table is only a **proposal**;
exactness comes from the second acceptance stage, which recomputes `rc_factor`
at the event's own kinematics.

Fermi motion fits the same shape:

1. `_draw_fermi()` next to `_draw_vertex()`, drawn per trial.
2. Generate in the struck nucleon's rest frame with the existing machinery.
3. Boost every final-state particle back to the lab.
4. Append the spectator.

**Must not: do not build a five-dimensional eta table indexed by energy.** It
is the first idea that comes to mind and it is days of wasted work — stage 2
already absorbs the table's error.

---

## 5. Architecture — the trap

`B_cell`, the stage-2 bound, is calibrated per cell of the (Q2, W2, t, phi)
grid. Fermi motion adds spread that the cell cannot see, so
`r = sigma_exact / (born * eta_table)` widens inside each cell and `B` grows.

After `225d4d9` the **proposal carries `B_cell`** (the cell is drawn
proportional to `M * B_cell`), so a larger `B` costs trials, not correctness.
The temptation will be to "fix the efficiency" by touching the cdf or going
back to a single global `B`. That is exactly the code that was broken until
this morning — it divided the sample's shape by `B_cell` and produced 22 %
steps at the cell edges.

**Must not touch without asking:** `_propose`, `calibrate`, `_b_full`,
`_b_cell`, `_cell_cdf`, `sigma_total`. If the task appears to require it,
stop and ask.

---

## 6. Physics decisions to make explicitly and write down

**Off-shellness.** The spectator is on-shell; the struck nucleon is not:
`E_N = M_d - E_s`, so its effective mass `m* != M_N`. Either evaluate the cross
section at `M_N` (standard PWIA with on-shell extrapolation — amp2609 knows
nothing else) or at `m*` (kinematically consistent, but the model was never fit
there). Recommended: `M_N` in the cross section and in the radiative
corrections, off-shellness only in the boost — but **the choice must be stated
in the note, not left implicit.** It matters at the percent level.

**`self._mT` becomes per-event** if you go the `m*` route. Audit every use
of it, including inside the majorant. This is the most likely source of silent
errors.

**Flux.** The virtual-photon flux convention in this code is settled and hard
won (core uses the `|q|` flux; there are adapters for Hand). Do not re-derive
it. Read the existing flux notes first and say which convention the
moving-frame flux follows.

**Cross section per deuteron vs per nucleon.** State which one `sigma_nb`
reports after the change, and make sure the wave function's normalisation is
consistent with it.

**A consequence worth knowing in advance:** Fermi smearing moves the *observed*
W (reconstructed assuming a nucleon at rest) by roughly +-100-150 MeV. An event
generated at W > 1.8 can land below 1.8 in the lab. The W box edge stops being
a sharp edge of the sample. Do not treat that as a bug.

---

## 7. The card flag

`runcard.py` already supports optional trailing fields with defaults
(`N_REQUIRED = 13`, `DEFAULT_OPT`), so **append** after `ptarg` as field 21.
Cards without it keep parsing unchanged — that is the whole point of the
mechanism; do not insert in the middle.

    none                               ! fermi   none | hulthen | paris | cdbonn

A string rather than an integer: the wave function is a physics input that
should be readable in the card and visible in the LUND header, not decoded from
a `2`. Precedent: the `iacc  none` field.

The channel already says which nucleon is struck (`pi0.amp2609` vs the neutron
models), so no separate target field is needed. The flag means: *the struck
nucleon is bound in a deuteron — give it Fermi motion and emit the spectator.*

The setting **must** reach the `.info` sidecar and `config.json`, along with the
wave function's name. A sample that does not describe itself is worse than no
sample.

---

## 8. Acceptance criteria

1. **With `fermi = none`, the output is byte-identical to `production` for the
   same seed** — sha256 of the LUND, not "agrees within errors". This implies
   the Fermi momentum is drawn *only* when the flag is on, so the RNG stream is
   untouched otherwise. Ship the check as a script in the branch and run it
   before every commit. This single test is what keeps the branch from
   interfering with the running campaign.
2. The sampled `p_F` reproduces the analytic `|psi(p)|^2`; report the mean and
   rms.
3. **Per-event 4-momentum conservation in the LUND record:**
   `beam + d = e' + meson + N' + N_spectator (+ gamma)`, residual at machine
   precision. This catches boost errors immediately.
4. `B` and ms/event measured with the flag on and off
   (`production/bound_map.py`). A larger `B` is acceptable; lost accuracy is
   not.
5. W with and without Fermi motion: the smearing should be ~+-100-150 MeV and
   should populate the region below 1.8.
6. `production/shape_steps.py` on a born/rad pair generated with the flag on:
   the cell edges must be no more significant than the control points. Its
   header explains why a difference of means is the wrong estimator — read it
   before trusting a number it prints.

Arrays for 3, 5 and 6 come from `production/lund_kinematics.py`.

---

## 9. Machines and how to use them

**mac mini — free, use it for development.** 14 cores, 64 GB, load ~3. Enough
for correctness work and small runs. Fastest per core of anything available.

**clonfarm11 — DO NOT USE TODAY.** 128 threads, and 101 of them are running the
production campaign until roughly 19:20. Ask before starting anything there.
When it is free: output goes to `/data/vpk` (`/home` is full), there is no
cvmfs and no gfortran, and the two T4 GPUs have no driver.

**ifarm2402 — 256 cores and mostly idle.** Measured 2026-09-25: 80-84 % CPU
idle, 8 running processes, 1355 GB of 1510 GB memory available, despite ~158
logged-in users. **Read the `%Cpu(s) ... id` line from `top`, not the load
average** — the load average lags badly on this node (it read 71 while the
machine was 80 % idle, which is how the first version of this brief got it
wrong).

**There is a hard per-user CPU quota: 25.6 cores.** The idle cores are not
yours to take:

    $ systemctl show user-$(id -u).slice -p CPUQuotaPerSecUSec -p MemoryMax
    CPUQuotaPerSecUSec=25.600000s
    MemoryMax=137438953472            # 128 GB

    $ cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cpu.max
    2560000 100000                    # 25.6 cores

Starting more than ~24 workers does not raise throughput, it only divides the
same 25.6 cores among more processes and makes every one of them slower. This
was measured the hard way on 2026-09-25: 100 workers drew exactly 2543 % CPU
against the 2560 % ceiling, i.e. 25 % each.

**Use 24 workers.** Note also that one worker shows up as THREE processes
(`bash -c` -> `bash` -> `python3`), so 24 workers means ~72 entries in `ps` —
do not be alarmed by the count, and do not let someone else be: only a third
of them consume anything.

A second trap, hit twice in one day: `pkill -f <pattern>` matches the command
line of the ssh session running it, so `ssh host 'pkill -f foo.sh'` kills its
own shell before the next command runs. Put the kill in its own call, or match
a pattern that cannot appear in your own command line.

Always set:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

(the generator is single-threaded per worker; without this every worker
spawns a BLAS thread pool and the node grinds).

The pi0 production does NOT use this node — with a 25.6-core quota it could
not help against clonfarm11's 100 cores, so ifarm is free for you.

**The farm (slurm) — for big runs only.** Submit from ifarm2402. The template
is `production/farm/gen_array.sbatch`; note its `R=` path is hardcoded to a
2026-09-24 production directory and must be pointed at your own. Reality check
before you plan around it: `PriorityWeightAge = 0`, so waiting buys nothing,
our priority is 3976 (fairshare 2310 + partition 1667), and the farm has been
~98 % allocated with tens of thousands of jobs pending. Jobs can sit for many
hours. Use `--partition=production`; `--partition=priority` has a 4-hour wall
clock that has already killed a set of our jobs at 04:00:20.

Output on ifarm goes to `/volatile/clas12/vpk/...` (petabytes free). `/work` is
9.5 GB and effectively unusable.

---

## 10. What to hand back

A branch that passes §8, plus a short note recording: the off-shell treatment
chosen and why, the wave function and its normalisation, where the moving-frame
flux factor comes from, and the measured cost in `B` and ms/event.

Do not merge to `production` yourself.
