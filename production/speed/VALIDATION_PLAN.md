# Validating the speed work — plan

Two things need validating, and they need different plans, because they
make different claims.

    PART 1  merged in production   claim: the output is UNCHANGED
    PART 2  branch speed-numba     claim: the output is EQUIVALENT

Part 1 can be proved.  Part 2 can only be measured.  The stages below are
ordered so that the cheapest thing that could falsify the work comes
first, and nothing expensive runs until the cheap checks pass.

---

## Stage 0 — finish proving Part 1.  Half a day, mine.

Part 1 claims byte-identity, which is the strongest claim available, and
byte-identity was verified on the pi0 rad card, `--jobs`, 40 farm nodes and
two numpy versions.  **But the changes are in shared `core/` code, and
these paths were never byte-tested end to end:**

| path | why it matters |
|---|---|
| **Fermi / deuteron (`fermi = av18`)** | on production, goes through the same core, never tested |
| **born mode** (`iborn = 1`) | different code path; born twins normalise the rad samples |
| **eta card** | different model family |
| **a vector-meson card** (phi or rho) | rho also draws a Breit-Wigner mass |
| **polarised target** (`ptarg != 0`) | exercises the POL_KEYS branch of the hadronic tensor |
| **helicity twins** (`ihel = +1 / -1`) | BSA samples are generated this way |

Method: for each, generate a short sample on `production` and on the
pre-speed commit `64a6906` with the same seed and `cmp` the LUND files.

**Acceptance: byte-identical. Anything else stops everything else** until
it is understood — Part 1's whole claim is that there is nothing to
discuss.

---

## Stage 1 — numba functional equivalence at production scale.  ~2 h farm, mine.

20 chunks x 5000 events, both backends, identical seeds, on compute nodes,
production card and environment.

Compare **per chunk**:

| quantity | acceptance |
|---|---|
| `sigma_nb` (what the campaign normalises on) | equal to < 1e-12 |
| `b_viol`, `max_b_viol` | no violation numpy does not also produce |
| events written | identical |
| the kernel's degenerate-row guard | **never fires** |
| merged MX^2, v, phi, `<sin phi>` | compatible within the samples' statistics |

Rationale for the scale: 100 000 events is 25x the largest sample compared
so far, and it is the first test with enough events for a rare kinematic
corner to appear.  The guard is the point: if degenerate rows reach the
kernel in production, the numpy path has been masking them silently and we
want to know that regardless of numba.

---

## Stage 2 — one chunk through the full chain.  ~1 h farm, mine.

GEMC + recon on one chunk from each backend, compared at the reconstructed
level.  Expected to be redundant: Stage 1 should show the generator emits
the *same events*.  It is worth doing once anyway, because the
reconstructed level is what the collaboration reads, and "we checked" is
cheaper than "we reasoned".

---

## Stage 3 — deployment.  Not mine.

* Build `clas12-mcgen` / the OSG container with `numba` + `llvmlite`, run
  one chunk inside it, and compare against the same chunk outside.
* Decide the cache policy once: pre-warm after deploy, `NUMBA_CACHE_DIR`
  per job, or accept 5.3 s per job.  All three are safe (tested); the point
  is to choose rather than discover.

**This is the real gate.** Everything above says the code is ready. None of
it says the container is.

---

## Stage 4 — canary, on the first real campaign.

Do not switch 100 % of a campaign at once.  Run the first production
campaign with `RC_NUMBA=1` on a **fraction of chunks** — 10 % is enough —
and compare those chunks against the numpy chunks of the same campaign:
`sigma_nb`, `b_viol`, and the analysis-level distributions.

This costs nothing (the chunks are needed anyway), and it is the only test
that runs on the real box, the real statistics and the real downstream
analysis.  `RC_NUMBA` being an environment variable is what makes it
possible: the two halves differ by one line in the sbatch.

---

## What would make me stop

* Stage 0 not byte-identical anywhere — Part 1's claim is false and the
  merged work needs re-examining before anything else happens.
* `sigma_nb` differing between backends on any chunk.
* The kernel guard firing.
* A `b_viol` the numpy path does not also produce.

## Keeping it true afterwards

The harnesses are in the repo, not in my head:

    production/speed/sigf_grid.py       sigma_F over the production box, bitwise
    production/speed/channels_grid.py   all channels + the polarised branch
    production/speed/compare_dist.py    two LUND samples, statistically
    production/speed/bench_sigf.py      timing, with the load printed next to it

Worth wiring `sigf_grid` and `channels_grid` into whatever runs `selftest`,
so the next person who touches `core/` finds out immediately rather than
in a campaign.
