# LUND production optimization — brief

Your worktree: `/Users/vpk/exclurad_py_speed`  ·  your branch: `speed`
Written 2026-09-29 by the session running the pi0 reference campaign.

## The one-sentence task

Make **rad-mode LUND generation** faster without changing a single number it
produces. Nothing else in the chain is yours.

## Why this and nothing else

Measured on the current campaign, seconds per event:

| step | whose code | s/event | yours? |
|---|---|---:|---|
| tables | ours | — (once per box, 15 min total) | no — already amortised to nothing |
| LUND born | ours | 0.02 | no — already free |
| **LUND rad** | **ours** | **2.60** | **YES** |
| GEMC | not ours | 3.35 | no |
| coatjava recon | not ours | 1.28 | no |

At 100 M events on OSG, rad generation is **72 000 core-hours**, 36 % of what
GEMC costs. Getting to 1 s/event saves ~50 000 core-hours per campaign. That
is the entire justification; if the 100 M campaign is cancelled this work has
no urgency.

**Target: under 1 s/event on a farm COMPUTE node.** Not on the mini, not on a
login node. See "How to measure" — this is where people go wrong.

## Where the time goes — measured today, do not trust older numbers

`cProfile` on the production card (`cards/rad_pi0_rga.txt`), 25 events, mac
mini, 946 ms/event:

| | calls (25 events) | own time | share |
|---|---:|---:|---:|
| `c_einsum` | **355 319** | 8.73 s | **36.9 %** |
| `rad_L` (excl. einsum) | 7 322 | 3.55 s | 15.0 % |
| `hadronic_tensor` (excl. einsum) | 10 454 | 3.07 s | 13.0 % |
| `structure_arr` — **the physics model** | 10 454 | 1.05 s | **4.5 %** |
| `op`, `_boost_matrix`, `photon_basis` | ~84 000 | 1.78 s | 7.5 % |
| `normalize_axis_tuple`, `_broadcast_to`, `norm` | ~377 000 | 0.60 s | 2.5 % |

Read it this way: **the cost is numpy overhead, not arithmetic.** 355 k einsum
calls at ~25 µs each, on arrays of 336 elements, is dispatch and temporary
allocation. `normalize_axis_tuple` is called 218 000 times per 25 events and
contains no physics whatsoever. The actual structure-function model is 4.5 %.

**A STALE NUMBER YOU WILL FIND AND MUST IGNORE.** Older notes and
`docs/` say "sigma_F is 73 % of the cost". That was measured *before* the
2026-09-24 quadrature change ((30,16,24,16) → (20,12,16,12)) and is no longer
true. If you see it quoted, it is out of date; the table above replaces it.

## The shape of the work

**Phase 0 — half a day, no numba, do this FIRST.** It pays for itself and it
tests the diagnosis: if half a day gives 2x, the cost really is dispatch and
the numba estimate below is trustworthy. If it gives nothing, stop and
re-profile before spending a week.

- `np.einsum` without `optimize=True` runs a naive C loop. Try it on the hot
  calls in `core/spinors.py:rad_L` and `core/hadronic.py`.
- Several einsums contract against **constant** gamma matrices (`GAMMA`,
  `SIGN`) and are recomputed ~14 000 times per event. Precompute them once.
- Simple contractions are faster as `@` / `np.tensordot` / plain broadcasting.
- `photon_basis` and `_boost_matrix` do work that may not depend on the
  angular point — check what can be hoisted out of the loop.

**Phase 1 — 4–5 days, numba.** Estimated 10–30x on the jitted part (~75 % of
the cost), so 3–4x overall: 2.6 → 0.7–0.9 s/event.

| | |
|---|---|
| port `rad_L` | 1 day — Dirac algebra; a sign error here is SILENT |
| port `hadronic_tensor`, `photon_basis`, `_boost_matrix` | 1 day |
| split the model call out of the kernel | 0.5 day — `structure_arr` is a Python object and cannot be jitted, but it is already called once per (v, channel) block, so the cut is clean |
| **validation** | **1 day, not optional** |
| build, `parallel`/`cache`/`fastmath`, farm + container install | 1 day |

Neither `np.einsum` nor `np.linalg.norm(axis=, keepdims=)` exists in numba
nopython mode. Every one becomes an explicit loop. That is the bulk of the work.

## CHECK THIS BEFORE PORTING ANYTHING

**numba is installed nowhere.** Not on the mini, not on ifarm:

```
mini    : ModuleNotFoundError: No module named 'numba'
ifarm   : ModuleNotFoundError: No module named 'numba'
          Python 3.9.25, numpy 1.23.5
```

The mini runs Python 3.14 with numpy 2.x. So:

1. Establish that numba installs and runs **on ifarm** (Python 3.9, numpy
   1.23) before porting a single function. If it cannot, a hand-written C
   extension is the fallback — more days, but no LLVM dependency.
2. Whatever you write must run on **both** numpy 1.23 and 2.x. Test on ifarm,
   not only on the mini.
3. numba pulls LLVM. For the OSG container and for `clas12-mcgen` that is a
   real dependency question, not a detail. Find out early whether it is
   acceptable — the answer can kill the numba route and you want to know on
   day one.

## The correctness bar — this is the point of the job

You are not allowed to change any number. The bar, in order:

1. **Byte-identical LUND from the same seed.** If your change does not alter
   how much randomness is consumed, then `exclurad_py.run` on the same card
   and seed must produce a byte-identical file. This is the strongest test
   available and it is cheap — aim to keep it true, and when you cannot, say
   exactly why.
2. `sigma_F` agreement to ~1e-12 against the current numpy path, on a grid of
   points spanning the production box, not one point.
3. `eta` over the box — the quantity the generator actually uses.
4. Physics distributions on a real sample: MX², v, phi, and `<sin phi>`.
   `production/validate_channel.py` and the `gemc_check` tools do this.

A silent 1e-3 error in the Dirac algebra will pass a casual eye and ruin a
100 M campaign. Two separate bugs this month — a shape divided by the wrong
bound, and a photon-pair selection — produced perfectly healthy-looking logs.

## How to measure, because this is where today's confusion came from

**Machines differ by a factor of three and a single run means nothing.**
Measured today, the same card, same commit, rad:

| where | ms/event |
|---|---:|
| mac mini (M4 Pro) | **946** |
| ifarm2402 login node | 1815 |
| farm compute node, median of 200 tasks | **2586** (range 1824–2870) |

A colleague quoted 1.74 s/event this morning; it was a single 100-event run on
the login node, and the production number is 2.60. That is a 50 % error on the
largest single cost in the campaign.

Rules:
- Always say which machine a number came from.
- Never compare a number from one machine against a number from another.
- The headline number is a **median over many slurm tasks on compute nodes**.
  Develop and profile on the mini; re-measure on the farm before claiming anything.
- Profile with `cProfile` sorted by `tottime`, and report call counts, not just
  times — 355 000 calls of 25 µs is a different problem from 100 calls of 90 ms.

## Staying out of everyone's way

- **Work only in `/Users/vpk/exclurad_py_speed`, only on branch `speed`.** Do
  not check out `production`; it is checked out in the main worktree and git
  will refuse anyway.
- **Do not push to `production`.** When you are ready, push `speed` and say
  so; merging is coordinated, because other branches are in flight
  (`fermi-flux`, `jpsi-fermi`).
- **The farm is busy with the reference campaign** — 400 simulation tasks in
  `/volatile/clas12/vpk/prod_pi0_rga_t25_20260928`, plus 100 tasks from the
  deuteron session. Do not touch that directory. Your own slurm work is
  welcome but keep it modest while the campaign runs, and use your own
  `/volatile/clas12/vpk/<something>_speed` area.
- **The mini is shared.** A benchmark on a loaded machine is worthless, and
  you will load it for others. Before timing anything on the mini, check
  `uptime` and `ps`; if something of mine is running, either wait or say so in
  the result. Message the session named "Launch generator" to coordinate — it
  is running the campaign and can tell you when the mini is quiet.
- The laptop is a different CPU. Fine for editing and for correctness tests;
  useless for any number you intend to compare with a mini number.

## Useful starting points

```
exclurad_py/core/radiative.py    sigma_F, quad_default        (205 lines)
exclurad_py/core/spinors.py      rad_L, contractLH, F_IR      (146 lines)
exclurad_py/core/hadronic.py     hadronic_tensor, photon_basis (177 lines)
cards/rad_pi0_rga.txt            the production card
```

Tables for the production box already exist and take 15 minutes to rebuild if
you need your own:

```
/Volumes/wd_14tb/prod_pi0_rga_fa18_t25_20260928/tables/   (mini)
/volatile/clas12/vpk/prod_pi0_rga_t25_20260928/tables/    (ifarm — read only, not yours)
```

Run the generator with `ETA_TABLE` and `MAJ_TABLE` pointing at them, plus
`TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0`.
These are the production settings; a benchmark without them measures something
else.

## What "done" looks like

A merge request on `speed` with: the measured speedup on farm compute nodes
(median over tasks, with the spread), the four validation results above, an
honest statement of what you could not make byte-identical and why, and a
one-line answer on whether numba is acceptable for the OSG container.

Partial is fine. Phase 0 alone, measured and validated, is a real result.
