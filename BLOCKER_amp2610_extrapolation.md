# amp2610 is not safe outside the fitted |t|, and the generator notices

**Found 2026-10-03 while timing the new model.  The 100 k sample was NOT
generated: it would not have been exactly unweighted.**

## What the generator said

300 rad events, pi0 RG-A card, ifarm2401, same conditions for both:

| model | backend | generate | per event |
|---|---|---|---|
| amp2609 | numpy | 270 s | 901 ms |
| amp2609 | numba | 139 s | 464 ms |
| amp2610 | numpy | 2944 s | 9813 ms |
| amp2610 | numba | 1577 s | 5256 ms |

**11x slower**, and worse than slow:

    amp2609   eta interpolation rms(rel) 0.292    health: clean
    amp2610   eta interpolation rms(rel) 94.208   health: stage-2 bound
              exceeded in 89/53497 accepts (max x239) -- the sample is NOT
              exactly unweighted

The 0.292 for amp2609 matches the value documented in
`production/speed/README.md`, so the setup is comparable and the regression is
real.

## It is not the port

The port is exact.  `validation/amplitude_closure.py` agrees with the fit
repository to 0.00e+00 on 606 points in four channels for both vector lengths,
on the mini and on ifarm, and amp2609's LUND output is byte-identical across
this branch (pi0 rad, pi0 born, eta rad, pi0 neutron).

It is also not the model costing more per call: that is only 24 % (77.9 ->
96.7 ms per 200 k points).

## It is the t-slope of the longitudinal sector

At x_B = 0.123, the t-slope `b + b' ln x` of the block feeding T00m:

    amp2609  L block    0.0000 + (-1.2839)(-2.094) = +2.689
    amp2610  Etilde     0.3512 + (-0.2995)(-2.094) = +0.978
    amp2610  Htilde     0.1631 + (-1.0127)(-2.094) = +2.284

`<Etilde>` in the new fit is nearly **three times flatter in t** than the block
it replaced.  Inside the fitted region this is invisible.  Outside it, sigma_L
stops falling and takes over (W2 = 8, Q2 = 1, pi0 p):

| \|t\| | sigma_L/sigma_T amp2609 | amp2610 |
|---|---|---|
| 0.30 | 0.093 | 0.122 |
| 1.00 | 0.043 | 0.126 |
| **1.80 — fit validity ends** | 0.0056 | **0.237** |
| 2.50 | 0.00084 | **0.443** |
| 3.50 | 0.000056 | **1.104** |
| 5.00 | 0.0000010 | **4.414** |

The RG-A card generates to |t| = 2.5 on purpose -- *"generate WIDER than you
analyse, so resolution migration across the analysis cut at |t| = 2 is
simulated"* -- and the radiative integral's shifted `tt` reaches further still.
So the eta table is built over a region where the two models do not merely
differ in normalisation: they differ in which cross section dominates.  That is
why interpolating eta fails by a factor 320 and the stage-2 bound is exceeded.

Positivity is NOT violated anywhere -- that was checked separately and B >= 0 is
structural.  The extrapolation is unphysical-looking, not inconsistent.

## Nothing in the fit prevented this

The fit penalises a NEGATIVE slope at x_B = 0.05, 0.35, 1.00.  `<Etilde>`'s
slope is +0.978, so it passed.  Nothing asked it to be steep enough to stay
sensible past |t| = 1.8, and no data exists there to ask.

## The decision is VPK's

1. **Narrow the card to the validity limit** (|t| <= 1.8).  Cheapest, but it
   gives up the feed-in band that simulates resolution migration across the
   |t| = 2 analysis cut -- which is why the band was widened to 2.5 on
   2026-09-28 in the first place.
2. **Refit with a floor on the Etilde t-slope.**  Honest, since the slope is
   unconstrained by data; costs a refit and changes amp2610.
3. **Damp the model outside its validity** inside the generator.  Keeps the
   band, but the samples there stop being "the model extrapolated" and become
   "the model plus a cutoff", which has to be said out loud wherever they are
   used.

I have not chosen.  Option 2 is the one I would argue for, because the flat
slope is a property of the fit that nobody inspected, not a generator problem,
and fixing it where it arose keeps the model and the generator telling the same
story.

## Also measured, separately

The numba speedup on amp2609 at these settings is **1.94x** on generate
(901 -> 464 ms/event), on the ifarm2401 login node.  Not the 4.13x claimed for
the speed branch -- but that figure was measured on compute nodes and as a total
including other changes, so this is not a contradiction, only a different
number under different conditions.  Worth reconciling before anyone quotes 4x.
