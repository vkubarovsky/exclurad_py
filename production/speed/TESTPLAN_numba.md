# Accepting numba: what has been tested, and what is left

Opt-in via `RC_NUMBA=1`.  Unset, the numpy path runs and is byte-identical
to production, so **merging the branch changes nothing for anyone who does
not set the flag.**  That is the safety property the whole plan rests on.

## Already passed

| # | test | result |
|---|---|---|
| 1 | `selftest`, 8 physics checks, kernel ON | all pass |
| 2 | `sigma_F` over the production box, **x86** | **1.9e-13** worst |
| 3 | same, mini (ARM) | 1.1e-9 median, 1.7e-7 worst |
| 4 | **40 paired slurm tasks, compute nodes** | **39/40 LUND byte-identical**; 40th = 1 ulp in 1 number of 162 lines |
| 5 | 4000 events, both backends, same seed | same events, same order; worst number 6.3e-08 |
| 6 | **`sigma_nb` (campaign normalisation)** | **bit-identical** |
| 7 | all channels: pi0, eta, phi, rho, J/psi | 1e-8 … 1e-11 |
| 8 | polarised target (lam=0.8) | **byte-identical** — numpy fallback verified to trigger |
| 9 | numpy path vs production | byte-identical, 228/228 |
| 10 | production stack: py 3.9.25 + numpy 1.23.5 | **runs**, no stack change |
| 11 | 20 concurrent cold-cache jobs, one shared tree | 20/20 ok, one distinct result (no cache race) |
| 12 | read-only source tree (container case) | works; compiles in memory, +5.3 s once per job = 0.2 % of a chunk |
| 13 | degenerate-denominator guard | never fires; raises loudly if it ever does |

Items 4, 5 and 6 are the ones that matter: the sample is the *same sample*,
and the number the campaign normalises on is *exactly* equal.

## Left to do, in order

**A. Production-shaped closure — the real acceptance test.**  20 chunks x
5000 events, both backends, identical seeds, on compute nodes.  Compare
per chunk: `sigma_nb`, `b_viol`, event count, and the merged MX^2 / v /
phi / `<sin phi>`.  Acceptance: `sigma_nb` equal to <1e-12 per chunk, no
new `b_viol`, distributions compatible.  ~2 hours of farm time.  **I can
run this.**

**B. One chunk through the full chain**, GEMC + recon, both backends,
compared at the reconstructed level.  Almost certainly redundant given
(4)-(5) — the generator output is the same events — but it is cheap
insurance and it is the level the collaboration reads.  **I can run this
if you want it.**

**C. The OSG container.**  Build `clas12-mcgen` with numba + llvmlite and
run one chunk inside it.  **This is the actual gate and it is not mine** —
it needs whoever owns the container.  Everything above says the code is
ready; nothing above proves the container is.

**D. A cache policy decision, once, before 200 jobs.**  The compiled cache
lands in `exclurad_py/core/__pycache__`.  Test 11 says concurrent writes
are safe and test 12 says a read-only tree still works, so any of these is
fine — pre-warm it once after deploy, set `NUMBA_CACHE_DIR` per job, or do
nothing and pay 5.3 s per job.  Pick one deliberately rather than
discover it.

## What would make me say no

* `sigma_nb` differing at all between backends on any chunk in (A)
* the kernel's guard firing in (A) — it means degenerate kinematics reach
  the kernel in production, and the numpy path was masking them silently
* any new `b_viol` that the numpy path does not also produce

## What this does NOT test

The numba path's agreement is measured *against our own numpy path*.  Both
share the same physics, the same conventions and the same quadrature, so
this says the port is faithful — it says nothing new about whether the
underlying RC is right.  That is what `selftest`, the Born closure and the
comparisons against published data are for, and none of them move.
