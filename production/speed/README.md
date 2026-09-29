# Rad-mode generation: 2x, byte-identical

Branch `speed`.  Phase 0 of the LUND production optimisation (`SPEED_BRIEF.md`),
pure numpy, no new dependency.

## Result

| | |
|---|---|
| **speedup on farm COMPUTE NODES** | **1.657x**, median of 40 paired slurm tasks (10-90 %: 1.592-1.755) |
| LUND output | **byte-identical**, including 40/40 independent checks on 20 compute nodes |
| new dependencies | **none** |
| stacks verified bit-identical | mini ARM numpy 2.4.6 and 1.23.5; ifarm x86 numpy 1.23.5; 40 farm nodes |
| **does it meet the brief's <1 s/event target?** | **No.** 2.60 -> ~1.56 s/event |

Quote the ratio, not the milliseconds.  The mini was swapping for most of the
day (load average 1 -> 175 with ~87 % of the CPU idle), so absolute numbers off
it mean little; interleaved pairs still measure the ratio well because both
arms see the same machine within seconds of each other.

### The number that counts: farm compute nodes

40 slurm tasks, each timing BOTH arms back to back **on the same node**, so
the node-to-node spread cancels in the per-task ratio.  Production card,
production settings, the campaign's own tables read-only, 20 distinct nodes,
`OPENBLAS_NUM_THREADS=1` as production.

| | median | min | max |
|---|---:|---:|---:|
| production (`origin/production`) | 2388.2 | 1503.4 | 3979.2 ms/event |
| `speed` | 1427.5 | 951.0 | 2200.9 ms/event |
| **per-task ratio** | **1.657** | 1.527 | 1.808 |

10-90 % of the per-task ratio: **1.592-1.755**.  Guard failures: 0.  LUND
byte-comparison inside each task: **40 of 40 identical**.

The baseline median, 2388 ms/event, sits just under the campaign's own 2586,
which is the check that this harness measures the production generator and
not something else.  Applying the ratio to the campaign's number: **2586 ->
about 1560 ms/event**.

### The mini overstates the farm gain -- ratios do not transfer either

| what | stack | ratio |
|---|---|---:|
| **farm compute nodes, 40 paired tasks** | **ifarm x86, numpy 1.23.5** | **1.657x** |
| 60-event run, 3 interleaved pairs | mini ARM, numpy 2.4.6 | 1.94x |
| `sigma_F`, 5 interleaved reps | mini ARM, numpy 2.4.6 | 1.98x |
| `sigma_F`, 5 interleaved reps | mini ARM, numpy 1.23.5 | 1.90x |

The brief says never to compare an absolute number across machines.  This
says the same is true of a **ratio**: the mini reads 1.94x for a change that
is worth 1.66x where production runs, a 17 % overstatement, and the numpy
version is not the cause (1.90 vs 1.98 on the same box).  Different CPU,
different cache and memory behaviour, and this change is largely about how
much numpy overhead sits on top of the arithmetic.  Quote the farm number.

One more methodological note, because it nearly went into this report as a
finding: a **single** 60-event pair on numpy 1.23 read 1.60x and looked like
evidence that the speedup was numpy-version dependent.  Three interleaved
pairs and the `sigma_F` A/B both said ~1.9x on that same box.  It was load
noise on a swapping machine.  One pair is not a measurement.

## Where the time actually went

The brief's profile was right that the cost is numpy overhead rather than
arithmetic, but the mechanism is sharper than "dispatch", and the sharper
version is what made the fix easy:

**einsum has no specialised kernel past three operands.**  A four-operand
`einsum` falls into a generic nested loop over every index.  The two worst
call sites in the profile were exactly the two with the most operands:

| call site | operands | was | now |
|---|---|---|---|
| `fslash` -- `'...m,m,mij->...ij'` | 3 | 86 us | 20 us |
| `contractLH` -- `'...mn,m,n,...mn->...'` | 4 | 343 us | 265 us |

`fslash` alone was **60 % of `rad_L`**.

The two-operand einsums are already efficient: every hand-rolled numpy
replacement tried for them (`@`, `tensordot`, broadcast+`sum`) was **slower**,
at both 336 and 6720 rows.  There is nothing left to win there.

## The four changes

1. **Batch the v grid** (the bulk of the 2x).  `sigma_F` evaluated its 20-node
   v grid one node at a time, on arrays of 336 elements -- hundreds of numpy
   calls each doing a few microseconds of work.  Nothing in that loop couples
   angular nodes to each other, so the v-nodes are now stacked into one set of
   array calls.  Function calls per `sigma_F` fell **10.6x**.
2. **`fslash` via the sparsity of gamma^mu** (16 nonzeros of 64): the
   contraction is 16 scaled adds.  4-6x.
3. **`contractLH`**: fold the two metric factors into one constant so it is a
   three-operand einsum.  1.3x.
4. **`photon_basis` / `photon_pols`**: `np.cross` and `np.linalg.norm` inlined
   -- most of their cost was `normalize_axis_tuple` / `moveaxis` /
   `broadcast_to`, ~220 000 calls per 25 events containing no physics.  The
   inverse boost is derived from the forward one by negating its two mixed
   blocks instead of being rebuilt.

## Why it is bit-identical, and how that was established

Bit-identity was not hoped for and checked afterwards; it was the design
constraint, and each candidate rewrite was **tested bitwise against the
expression it replaced** before being adopted.  That test rejected most of
them.  The surviving rules:

* Folding a factor of exactly `+-1` into a constant is exact.
* Skipping structural zeros cannot change a sum that starts at `+0.0`.
* Negating a float is exact, and `(-b)*(-b) == b*b`.
* Batching is exact for work that is **per row**; the reduction is the only
  place order matters, so it is still done per v and accumulated in the
  original v-outer / channel-inner order.

Reductions are where intuition fails: numpy's sum over a **last** axis uses
pairwise/SIMD accumulation and does **not** reproduce einsum's sequential
order, while a sum over a middle axis does.  This is why the rewrites had to
be measured rather than reasoned about.

**Deliberately BLAS-free.** `np.tensordot` makes `fslash` another 2x and was
bit-identical in 300/300 trials -- but it routes through BLAS, which einsum
never did, so results would start depending on the BLAS build (Accelerate on
the mini, OpenBLAS on ifarm) and on its threading. For ~3 % of `sigma_F` that
is a bad trade in a 100 M-event campaign.

**One thing deliberately left on the table.** `photon_pols` depends only on
the photon *direction* and so is v-independent in exact arithmetic; caching it
across the v grid would be worth a few more percent.  It is **not** bit-
invariant (normalising `om*nhat` differs from normalising `nhat` by 1-2 ulp),
so it was left alone.  Byte-identity is worth more than a few percent.

## Validation

In the brief's own order:

1. **Byte-identical LUND from the same seed** -- yes.  60-event rad run,
   production card and settings, same md5.  This subsumes (4): identical bytes
   means identical MX^2, v, phi and `<sin phi>` by construction.
2. **`sigma_F` over the production box** -- 228 points (W2 3.24-16, Q2 0.8-10,
   |t| 0.001-2.5, three phi): **0 of 228 bits differ**.  Not "agrees to 1e-12".
3. **eta over the box** -- covered by (1): the eta table is built from
   `sigma_F`, and the run reproduces the byte-identical LUND through it.
4. **Physics distributions** -- see (1).

Plus: `--jobs 3` (the multi-process path the production wrapper uses) gives
three byte-identical part files; the eight-check `selftest` prints every
number identical to the last digit, and `sigma_F` is bit-identical at **every batch size** from 336 to 6720
rows, which is independent evidence that the batching is order-preserving.

Reproduce:

    python3 -m production.speed.sigf_grid dump ref.npz     # on production
    python3 -m production.speed.sigf_grid dump new.npz     # on speed
    python3 -m production.speed.sigf_grid cmp ref.npz new.npz
    python3 -m production.speed.bench_sigf 9

## Two findings that are not mine but matter

* **`.npz` tables do not cross numpy major versions.** A table written by
  numpy 2.x cannot be read by 1.23: `ModuleNotFoundError: numpy._core`. It
  fails loudly, which is the good case, but a shared `ETA_TABLE` between the
  mini and ifarm will not work.
* **The same code is not bit-reproducible across the two stacks.** Baseline
  `sigma_F`, numpy 2.4.6 vs 1.23.5: 168 of 228 points differ, max relative
  2.3e-07. That is far above ulp level and is most likely `roots_legendre`
  differing between scipy versions, i.e. slightly different quadrature nodes.
  Physically irrelevant, but it means "byte-identical" is a within-stack
  statement -- for anyone's change, not just this one.

## Does this finish the job?  No -- and the arithmetic

The brief's target was **under 1 s/event on a compute node**.  At 1.657x the
campaign's 2.60 s/event becomes about **1.56 s/event**.  That is a real
saving and it is not the target.

Per 100 M events, taking the brief's 72 000 core-hours at 2.60 s/event:

| | core-hours per 100 M | saved |
|---|---:|---:|
| production today | 72 000 | -- |
| `speed` (this branch) | ~43 500 | **~28 500** |
| the brief's 1 s/event target | ~27 700 | a further ~15 800 |

So Phase 0 captures roughly two thirds of what was on the table, with no new
dependency and with the output provably unchanged.

## numba, in one line

**numba is not needed for what is on this branch, and the remaining gap is a
weaker case for it than it looked this morning.**  Whether LLVM is acceptable
in the OSG container and in `clas12-mcgen` is a packaging policy question
that is not mine to answer -- but it now only has to be answered if ~15 800
core-hours per 100 M campaign is worth it, since the other ~28 500 are
already available without asking anyone.  It installs and runs on both
machines (tested by the campaign session: numba 0.67.0 on the mini's 3.14 and
on ifarm's 3.13, though production slurm runs 3.9).

If Phase 1 is taken up, the profile says where: after this work the remaining
time is genuine array arithmetic in `rad_L` and the tensor assembly, and the
structure-function model -- the actual physics -- is ~6 % of it.  Nothing
further is available from pure numpy without giving up bit-identity.

## What is NOT done

* **The `table_shape` lever is untouched.** The eta interpolation over this
  box is poor (rms(rel) 0.292), so stage 1 proposes badly and stage 2 throws
  work away; a finer table might buy acceptance for a one-off build cost.  It
  is deliberately left out of this branch because it is a **different kind of
  change**: a different table means different random draws, so byte-identity
  is gone by construction and the bar drops to distributions plus `b_viol`.
  It also has a measured precedent of losing (1008 -> 1035 ms/event at
  TABLE_SHAPE x4, on a narrower box) because a finer table is a slower
  lookup per event.  Separate branch, separate argument.
* **x86 bit-identity rests on `sigma_F` and on 40 end-to-end LUND
  comparisons**, both on ifarm.  That is strong, but the 228-point box grid
  was run on the login node, not on a compute node.
