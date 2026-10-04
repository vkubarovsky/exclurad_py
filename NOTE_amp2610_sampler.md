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

---

# Second pass, 2026-10-03: eight hypotheses, one measured fact

Stopped here at VPK's request.  What follows is what was measured, not what was
concluded.

## The measured fact

At the table node W2 = 16, Q2 = 1.15, t = -0.3, h = +1, swapping ONE structure
function of amp2611 for amp2609's:

    sigma_L   ->    -6992        <-- collapses by six orders
    sigma_T   ->  5.52e+09
    sigma_TT  ->  5.58e+09
    sigma_LT  ->  5.60e+09
    the other five        unchanged at 5.63e+09

**sigma_L carries the blow-up, and only sigma_L.**  The interference terms do
not.  This is a bisection, not an argument.

sigma_born agrees to 5 % across all three models, so the denominator is not the
issue: the whole factor 1e9 is in sigma_obs, i.e. inside the radiative integral.

## What was excluded, each by measurement

| suspected region | test | result |
|---|---|---|
| large \|t\| | tie b'(Etilde) to b'(Htilde): 9 orders more suppression at \|t\| = 14.6 | eta went UP, 2.6e9 -> 5.4e9 |
| low Q2 | damping [Q2/(Q2+m_rho^2)]^1.156 on the longitudinal sector | only x17 |
| near the W2 threshold | missing-mass cut vcut = 0.2 instead of 0 | only x38 |
| coarse eta table | TABLE_SHAPE 5,5,5,8 -> 12,12,12,20 | no trend: 94, 365, 86, 222 (25-probe noise) |
| stage-2 bound | B_SHAPE, B_PROBES | max excess x239 -> x12, violations persist |
| eta proposal cap | ETA_PROPOSAL_CAP 5 -> 50 -> 500 | B x cap = const; the cap only moves the factor |

## Where the models agree

At W2 = 8, Q2 >= 2.5 all three give eta = 0.889 to four digits.  The quiet part
of the kinematics is fine.  It breaks at HIGH W2, where the radiative tail in
omega is long -- and it is long by design, because the card runs vcut = 0 so the
MX^2 cut is applied offline.

## The probe bug, for the record

Every eta I quoted before this section for amp2609 was computed at h = 0 while
the table is built at h = +1 (the RCTable default).  At this node that is the
difference between 1.03e6 and 10.61.  The h = +1 numbers reproduce the stored
table exactly and are the ones above.  The earlier claim that "amp2609 is also
pathological, it just got lucky" came from that bug and is withdrawn.

## What is NOT known

Which part of the radiative tail carries the weight.  The points were ranked by
the RATIO between models, which is not the same as their contribution to the
integral, and the integrand weights are not visible from a model wrapper.
Answering it means instrumenting sigma_F, which should not be done blind.

## Status of the tags

* `amp2609` -- unchanged, byte-identity of its output verified on this branch.
* `amp2610` -- H~, free Etilde slope.  Correct against the fit to 2.1e-15.  NOT
  usable in the generator.
* `amp2611` -- H~ with b'(Etilde) tied to b'(Htilde).  Correct to 5.4e-16.  Also
  NOT usable: the tie cost +31.3 chi2 and did not fix the RC.

Nothing has been generated with either.  `~/exclurad_py` is untouched; all of
this lives in the `htilde-model` worktree.
