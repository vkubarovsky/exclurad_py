# What this generator's time is made of

Read this before optimising anything here.  It is short, it is measured, and
it would have saved three wrong diagnoses on 2026-09-29.

## The model

    cost  =  (number of integrand points)  x  (cost per point)

and nothing else matters.  Everything expensive in rad mode -- the exact
acceptance, the photon tail sampler, the Born trials -- evaluates the *same*
integrand: `rad_L`, `hadronic_tensor`, `contractLH` (`core/spinors.py`,
`core/hadronic.py`).

Counted directly, production card and settings, per **written** event:

| | integrand points | share |
|---|---:|---:|
| tail sampler (`TailSampler._build_grid`) | 118 620 | 51 % |
| `sigma_F` (exact RC, 423 calls x 13 440) | 113 702 | 49 % |
| **total** | **232 323** | |

The tail sampler decomposes exactly, and this is the number people are
surprised by:

    n_v 20 x n_x 32 x n_phi 12 x n_sub 12 x 2 channels = 184 320 per build
    x 0.64 builds per written event                    = 118 620

against `sigma_F`'s 20 x (12+16) x 12 x 2 = 13 440 per call.  **One tail
build costs 13.7 sigma_F.**  The arithmetic predicts the clock: 184 320
points at 1.86 us/point is 343 ms, against 368 ms measured on two machines.

`_density_batch` does **not** call `sigma_F` -- it evaluates the integrand on
the CellGrid's own stratified nodes.  The two halves are different point
sets, not two resolutions of the same points, so there is nothing to
deduplicate.

## What follows, including what does NOT work

There are exactly two factors, so there are exactly two kinds of lever.

**Cost per point.**  Attacked and banked: the 2026-09-29 `speed` work
(batching the v grid, sparse `fslash`, three-operand `contractLH`) gives
**1.657x**, byte-identical, no new dependency.  Further gains here mean
leaving numpy -- numba and LLVM in the OSG container.

**Point count.**  Everything left, and every route to it is an accuracy or
efficiency trade, not an engineering decision:

| lever | what it changes | cost |
|---|---|---|
| `n_sub` (12), `n_x` (32) in `TailSampler` | tail grid, 184 320/build | degrades `sigma_hard` AND the per-cell majorant |
| `RC_QUAD` | 13 440/`sigma_F` call | physics accuracy of the RC |
| stage-2 acceptance | the *call count* | **already minimised** (cached stratified bound, 2026-09-27) |

`n_sub` deserves a specific warning: it is stratified sub-sampling for the
per-cell integral *and* the per-cell majorant, so it is two accuracy knobs
wearing one name, and the second surfaces as overweight events rather than
as a visible bias.  Measure it as a curve -- cost against `n_overweight` and
`max_overweight` -- before quoting a number.

## Three hypotheses this model kills, all of them tested

Each looked reasonable and each failed for a reason the model predicts:

* **"Refine the eta table."**  A better table changes neither factor.
  Measured: halving the interpolation error (rms 1.332 -> 0.668) bought
  **3 %** for 144 s of build.
* **"Tighten the majorant (`MAJ_SHAPE`)."**  Stage-1 Born is a small share
  of the points: 221.6 us per `born_xsec` at ~93 trials is ~20 ms against
  ~500 ms per event.  A perfect majorant buys a few percent.
* **"More stage-2 bound probes (`B_PROBES`)."**  Acceptance changes the
  call *count*, and production had already minimised it.  A scan that
  appeared to show 26 % was measuring against a baseline whose majorant had
  no stored bound.  `B_PROBES` is read only when the bound is not cached
  (`events.py:1140`).

## A guard that follows from the third one

**An environment guard proves what you asked for; only the artefact proves
what you got.**  Assert on the provenance line, not on the variable that was
supposed to produce it:

    "stage-2 bounds: 300 probes, global B = 4.91"      <- in-job lottery, NOT production
    "stage-2 bound from cache: shape (4,4,4,6), ..."   <- the stratified bound

This is the same failure as the majorant silently rebuilding when
`ETA_PROPOSAL_CAP` is missing.  See `docs/benchmarking.md`.
