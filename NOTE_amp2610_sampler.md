# amp2610 and the sampler: what I got wrong, and what is actually true

**Rewritten 2026-10-03 after VPK pushed back twice, correctly both times.**
The first version of this file called the slow t-dependence of `<Etilde>` a
blocker and asked VPK to choose between narrowing the card, refitting, or
damping the model.  That was premature on every count.

## The measurements that stand

300 rad events, pi0 RG-A card, ifarm2401, identical conditions:

| model | backend | per event | health |
|---|---|---|---|
| amp2609 | numpy | 901 ms | clean |
| amp2609 | numba | 464 ms | clean |
| amp2610 | numpy | 9813 ms | stage-2 bound exceeded |
| amp2610 | numba | 5256 ms | 89/53497 accepts, max x239 |

The slowdown and the stage-2 violations are real and specific to amp2610.

## What I got wrong

**I never turned the knobs before blaming the model.**  `events.py` says in its
own comment that the default `(5,5,5,8)` is too coarse for the wide RG-A box and
that production has to pin it.  That was the first thing to try and I did it
last.

**I leaned on a number the code warns is misleading.**  The eta interpolation
`rms(rel)` is computed from **25 uniformly drawn probe points**, and
`check_interpolation` documents the trap: *"a corner where no event ever lands
dominates an unweighted absolute rms... the absolute rms is 0.9-2.3 while the
sample itself is sound."*  I read rms(rel) = 94.2 as a catastrophe.

Refining the table settles it — the number is noise:

| TABLE_SHAPE | setup | rms(rel) |
|---|---|---|
| 5,5,5,8 | 51 s | 94.2 |
| 8,8,8,12 | 71 s | 364.9 |
| 10,10,10,16 | 120 s | 85.5 |
| 12,12,12,20 | — | 221.8 |

No trend, a factor four either way with no monotonicity.  On this box the
quantity measures the probe draw, not the table.  It should not be used as a
gate for amp2610, and arguably not for anything on a box this wide.

**I named the wrong knob.**  `b_viol` is the stage-2 bound, controlled by
`B_SHAPE` and `B_PROBES`.  The eta table is the proposal.  I proposed three
model-level remedies without touching either.

## What is actually true

The only hard evidence is `b_viol`, measured on real accepted events rather than
on uniform probes.  And it responds to the knob that owns it:

| setting | accepts | violations | max excess |
|---|---|---|---|
| default | 53497 | 89 | **x239** |
| `B_SHAPE=6,6,6,8`, `B_PROBES=1500` | 12710 | 80 | **x28.3** |

A factor eight off the worst excess, and four times fewer stage-2 tests for the
same 300 events, i.e. a bound that fits closer.  Violations are not gone.

## What is still open

Whether more probes drive `b_viol` to zero, and what that costs in time.  If it
does, this was a sampler that had been tuned for one model and needed retuning
for another, which is ordinary, and the right outcome is a pinned
`B_SHAPE`/`B_PROBES`/`TABLE_SHAPE` in the production wrapper -- not a change to
the model or the card.

## On the Etilde slope, for the record

`<Etilde>`'s t-slope is 0.77 at xB = 0.25 against 1.79 for the block it
replaced, and its `b'` is -0.30 against -1.0 to -1.3 for every other block.  Past
the fitted |t| = 1.8 the longitudinal part therefore stops falling and comes to
dominate (L/T = 0.44 at |t| = 2.5, 1.10 at 3.5).

VPK's position is that a slow t-dependence is not in itself a problem, and the
card is explicit that beyond |t| = 1.8 the sample is *"the model EXTRAPOLATED,
kept as feed-in across the cut, not as a prediction"*.  Nobody claims the ratio
is right there; it only has to be samplable.  That is a sampler requirement, and
the sampler is where it should be met.

Worth knowing, not worth acting on by itself: the Etilde slope is constrained by
almost nothing.  It carries 0.2 % of sigma_L at -t' = 0.01 and 57 % at 1.0, so
it is invisible where the data are dense and dominant where they are sparse;
only 24.5 % of cross-section points sit above |t| = 1.0 and 9.9 % above 1.5.
The observable that would pin it best, sigma_LT, is the one the experiments
disagree about.  A profile in that slope would say what returning it to ~1.8
costs in chi2.  Cheap, and worth having before anyone argues about it.
