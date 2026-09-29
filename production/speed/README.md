# Rad-mode generation: 2x, byte-identical

Branch `speed`.  Phase 0 of the LUND production optimisation (`SPEED_BRIEF.md`),
pure numpy, no new dependency.

## Result

| | |
|---|---|
| speedup, rad generation | **2.00x** (mac mini, back-to-back pair, production card + settings) |
| LUND output | **byte-identical**, md5 `4b8c5b617aec452275f6416bd566e590` |
| new dependencies | **none** |
| numpy versions verified | 2.4.6 and **1.23.5** (what slurm runs) |
| farm compute-node number | **not yet measured** -- see "What is not done" |

Quote the ratio, not the milliseconds.  The mini was swapping for most of the
day (load average 1 -> 154 with 87 % of the CPU idle), so absolute numbers off
it mean little; a back-to-back pair still measures the ratio well because both
halves see the same machine.

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

Plus: the eight-check `selftest` prints every number identical to the last
digit, and `sigma_F` is bit-identical at **every batch size** from 336 to 6720
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

## What is not done

* **No farm compute-node measurement.** The headline number the brief asks for
  is a median over many slurm tasks, and ssh to ifarm hung all session
  (ProxyJump through login.jlab.org; a valid Kerberos ticket was present and
  direct connection to ifarm2402:22 times out from 129.57.79.26). The 2.60
  s/event production figure is therefore **not yet confirmed to have become
  ~1.30**; that claim needs the farm.
* **numpy 1.23 was verified on ARM, not x86.** Same numpy and scipy versions
  as production, different CPU and SIMD.
* **numba / Phase 1 not started**, and on this evidence it should not be
  started before the farm re-measure. 2x with no new dependency, bit-identical
  on both numpy versions, does not need the OSG container question answered;
  3-4x with LLVM does, and that question is not ours to decide.
