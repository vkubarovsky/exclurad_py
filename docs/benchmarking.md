# Measuring speed in this project

Written 2026-09-29, from a day in which three separate timing numbers were
quoted, all of them honestly measured and all of them meaning different
things. Everything here is measured on our machines, not general advice.

## The machines differ by a factor of three

Same card, same commit, rad mode, `cards/rad_pi0_rga.txt`:

| where | ms/event |
|---|---:|
| mac mini (M4 Pro) | 946 |
| ifarm2402 **login node** | 1815 |
| farm **compute node**, median of 200 slurm tasks | **2586** (range 1824–2870) |

The production number is the last one. A login-node measurement reads ~30 %
fast, and the login node's load is whatever 180 other users are doing —
one measurement there recorded 12.4 s/event at load average 859.

## A RATIO is machine-dependent too — this is the non-obvious one

The natural defence against the table above is "quote a speedup, not an
absolute time; the machine cancels". **It does not cancel.** Measured on the
same code change, `production` against `speed`:

| where | speedup |
|---|---:|
| mac mini | 1.94x |
| farm compute nodes, median of 40 paired tasks | **1.657x** |

A 17 % overstatement, and numpy is not the cause — the same mini gives 1.90
and 1.98 on numpy 1.23.5 and 2.4.6. Different CPU, different cache
behaviour. It bites hardest for exactly the kind of change that is worth
making: an optimisation that removes framework overhead from around the
arithmetic changes the overhead *fraction*, and that fraction is a property
of the machine.

**So: develop and profile on the mini, but quote neither an absolute time
nor a speedup from it.**

## How to take a number that means something

- **Pair the arms inside one slurm task**, both timed back to back on the
  same node, and take the median of the per-task ratios. The node spread
  (1824–2870 ms/event, ×1.57) then cancels where it matters instead of
  being averaged over.
- **Report the spread**, not just the median. The reference measurement was
  40 tasks on 20 nodes: ratio median 1.657, 10–90 % 1.592–1.755.
- **One pair is not a measurement.** A single 60-event pair read 2.00x for a
  change worth 1.94x on that machine, and a single pair under a different
  numpy read 1.60x — nearly reported as "the speedup is numpy-dependent",
  which it is not. Interleave at least three.
- **Check the baseline against production.** The reference harness measured
  2388 ms/event for the unmodified code against the campaign's 2586. That
  agreement is what says the harness is timing the production generator and
  not something adjacent to it.
- **Guard the environment.** Production is
  `TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0`.
  Drop `ETA_PROPOSAL_CAP` and `eta_hi` goes 5.0 → 36.5, stage-1 acceptance
  craters, and — because `eta_hi` is in `_maj_key` — the majorant silently
  rebuilds instead of loading. Drop `MAJ_TABLE` and every task rebuilds it.
  Neither says anything in the log. Assert `maj_from_cache is True` and
  `setup < 1 s` in any harness, and this whole class of error cannot reach a
  number you report.
- **State the stack.** An interactive ifarm shell gets numpy 2.3.3 from the
  venv in `~/.bashrc`; a slurm job gets the system python 3.9.25 with numpy
  1.23.5. Those are different major versions of numpy.

## Byte-identity is also per CPU GENERATION on this farm

Measured 2026-10-03: the same code, the same seed, the same tables, the
same `RC_NUMBA=0`, 300 events of the production rad card.

| machine | CPU | md5 |
|---|---|---|
| ifarm2401 | AMD EPYC 9554 (Zen 4) | `f8d0e0137a78a934` |
| ifarm2402 | AMD EPYC 9554 (Zen 4) | `f8d0e0137a78a934` |
| farm2517 | AMD EPYC 9354 (Zen 4) | `f8d0e0137a78a934` |
| farm180222 | Intel Xeon Gold 6148 (Skylake-SP) | `f8d0e0137a78a934` |
| **farm1959** | **AMD EPYC 7502 (Zen 2)** | **`ce1416234decc5ab`** |

Four of five agree, and they agree with the 2026-09-28 reference campaign.
The odd one out differs by **one unit in the last place** of one photon
momentum component in the second event, with the first event identical —
so it is arithmetic, not a shifted random stream. The one machine that
differs is also the only one without AVX-512, which is a plausible cause
and is NOT verified: saying so would need the dispatched flexiblas kernel
to be checked.

**`sigma_nb` is bit-identical on all five**, because it comes from the
accept counters and the cached majorant and never touches the photon
kinematics. So the physics reproduces everywhere and the bytes do not.

This cost a day. A canary comparing the merged generator against the
banked campaign came back "0 of 20 chunks byte-identical", which was read
as a code change — twice, by two people, in opposite directions — before
anyone looked at which nodes the two sides had run on.

### So

* **Byte comparisons: ifarm2401 or ifarm2402.** Both EPYC 9554, verified
  identical to each other and to the campaign. `production/pick_ifarm.sh`
  picks whichever is less loaded; measured the same minute, 2401 was at
  5 % and 2402 at 66 %, and the same run took 4m29 against 6m10.
  300 events is plenty — a difference shows up by the second event. Do not
  run a 5000-event chunk on a login node.
* **Anything that must run on compute nodes and be compared byte for byte
  needs `--constraint`**, or the scheduler will spread the arms over
  different generations and you will be comparing CPUs. `farm25` matches
  the login nodes; `farm19` does not.
* **Timing stays on compute nodes** regardless, in paired arms, as above.

## Byte-identity is the cheapest correctness bar, and it is per stack

If a change does not alter how much randomness is consumed, the same card
and seed must produce a byte-identical LUND. That is stronger than any
tolerance and costs one `cmp`. It held for the `speed` work on four stacks —
mini/numpy 2.4.6, mini/numpy 1.23.5, 40 farm compute nodes, and the
`--jobs 2` parallel path.

But it is a **within-stack** statement. The same unmodified code gives
`sigma_F` values differing by up to 2.3e-07 between numpy 2.4.6 and 1.23.5
(168 of 228 box points), most likely from `roots_legendre` differing between
scipy versions, i.e. slightly different quadrature nodes. Physically
irrelevant; fatal to a naive cross-stack `cmp`.

Related, and operational: **`.npz` tables do not survive a numpy major
version.** A table written by numpy 2.x cannot be read by 1.23 —
`ModuleNotFoundError: No module named 'numpy._core'` out of
`format.read_array`. It fails loudly, which is the good case, but it means
tables cannot be shared between the mini and ifarm at all.
