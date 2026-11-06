# Extracting the Born cross section for ep -> e'p'J/psi

Written 2026-11-06 at VPK's request.  What to do with the three things we
have -- data binned in (W, Q2, t), a radiative MC and a Born MC -- to get a
Born cross section out, which inputs are still missing, and which traps in
*our* samples will bite.

Companion notes, which this one does not repeat:

* `docs/RC_METHOD_FOR_JPSI.md` -- how the pi0 RC-method study was done and
  what to change for J/psi.
* `docs/RC_METHOD_COMPARISON.md` -- MC-ratio versus `rc_factor`, and what
  the difference costs.

This note is about the **cross-section extraction**, one step further down.


## 1. The inputs

VPK's list stopped at three.  The full set:

| # | Input | Where it comes from |
|---|---|---|
| 1 | Data yield in (W, Q2, t) bins | analysis ntuple |
| 2 | Radiative MC through GEMC | `osg_mariana_lund` arm `rad` |
| 3 | **borndecay** MC through GEMC | `osg_mariana_lund` arm `borndecay`, Sec. 5.2 |
| 4 | Integrated luminosity L | for exactly the run/trigger selection that made the yield |
| 5 | Signal extraction prescription | M(ee) window + background shape |
| 6 | BR(J/psi -> e+e-) = (5.971 +- 0.032) % | PDG.  **Not in the generator** -- see Sec. 5 |
| 7 | Data/MC efficiency scale factors | FT electron, lepton and proton tracking, trigger |
| 8 | Bin volumes, centroids, bin-averaging rule | chosen once and written down |
| 9 | One cut list, applied to data and BOTH MC arms | including the exclusivity cut |
| 10 | A model to iterate with | Sec. 3, step 7 |

Items 4-10 are not optional refinements.  Each one is a factor or a bias of
order the effect being measured.


## 2. The master formula

Per-event weight of each arm, from what the generator already reports:

    w_arm = sigma_gen(arm) / N_gen(arm)      [nb per event]
    sigma_gen = report.sigma_nb  in <job>.config.json

`n_i` = **reconstructed** MC events of that arm landing in bin i after the
identical analysis chain.  `m_i` = **generated** (truth) events in bin i with
no cuts at all.

### Form A -- ratio to the radiative MC.  This is the one to compute.

R_i IS A RATIO OF TWO RADIATIVE-LEVEL NUMBERS.  Both the numerator and the
denominator are reconstructed, after cuts, with radiation present.  The data
are radiative, so the comparison is like with like.  There is NO Born
quantity in R_i at all: the only Born object in Form A is the model factor
that multiplies it.

    sigma_Born(i) = R_i * sigma_Born_model_binavg(i)

    R_i = N_data(i) / ( L * BR * w_rad * n_rad(i) )

### Form B -- explicit factors.  This is the one to tabulate systematics on.

    sigma_Born_binavg(i) = N_data(i)
                           -------------------------------------------------
                           L * BR * dOmega(i) * eps_Born(i) * C_rad(i)

    eps_Born(i) = n_bdec(i) / m_bdec(i)

    C_rad(i)    = ( w_rad * n_rad(i) ) / ( w_bdec * n_bdec(i) )

where `bdec` is the **borndecay** arm -- Born production, decay radiation ON.
Not the `born` arm.  Sec. 5.2 is the reason, and it is the one thing in this
note that cannot be got wrong without spoiling the answer.

### What Form A assumes, and what it does not

Write the whole chain -- radiation, resolution, migration, acceptance, cuts --
as one operator K that takes a BORN cross section in and gives a
RECONSTRUCTED RADIATIVE yield out.  Then

    N_data(i)                = L * BR * (K sigma_Born_true )(i)
    L * BR * w_rad * n_rad(i) = L * BR * (K sigma_Born_model)(i)

so R_i is the ratio of two OUTPUTS of K,

    R_i = (K sigma_true)(i) / (K sigma_model)(i)
        = [ sigma_Born_true / sigma_Born_model ]  x  [ C_rad_true / C_rad_model ]

and therefore R_i equals the Born ratio we want ONLY IF
C_rad_true = C_rad_model.  That is the assumption of the method, and it is
not free: it is the physics content of the RC engine, tested elsewhere (the
two-method comparison of docs/rc_method_jpsi/, and the generator-level
closure to 0.1 %), not by this formula.

Formally, Form A is exact if and only if sigma_true/sigma_model is CONSTANT
over the region that feeds bin i -- the bin plus the migration tails plus the
radiative smearing.  Then K is linear on that region, (K c sigma)(i) =
c (K sigma)(i), so R_i = c exactly.  Step 7 is what makes that true: after
reweighting, the ratio is locally constant by construction.

What is safe and what is not:

    overall normalisation of the model      cancels EXACTLY (the identity above)
    shape of the model inside the bin       does NOT cancel -> step 7
    correctness of C_rad                    does NOT cancel -> tested separately

The two are algebraically the same once the bin-averaging is done
consistently, and "consistently" has an exact meaning.  Substituting eps and
C_rad into Form B, n_bdec(i) CANCELS:

    eps_Born(i) * C_rad(i) = w_rad * n_rad(i) / ( w_bdec * m_bdec(i) )

    sigma_Born_binavg(i) = N_data(i) * w_bdec * m_bdec(i)
                           --------------------------------------
                           L * BR * dOmega(i) * w_rad * n_rad(i)

which is Form A with

    sigma_Born_model_binavg(i) = w_bdec * m_bdec(i) / dOmega(i)

i.e. the model's bin-averaged Born cross section taken from the TRUTH counts
of the Born MC -- generated events in the bin, times the cross section per
event, per unit bin volume -- rather than from an integral of the model.  Use
that and the two forms agree numerically, not merely algebraically.  **Compute A.**  Three reasons:

1. Acceptance, efficiency, resolution smearing, bin migration and radiation
   all sit inside one measured number, `n_rad(i)`, produced by running the
   same analysis code over reconstructed MC.  Nothing is factorised that
   does not factorise.

2. The M(ee) window efficiency cancels.  The same window is applied to the
   data and to `n_rad(i)`, so it never has to be computed -- and the
   radiative lineshape is simulated rather than assumed.

3. **A wrong model normalisation cancels exactly.**  If the model is off by
   a factor f, then `n_rad(i)` scales by f, `R_i` by 1/f, and
   `sigma_Born_model` by f.  The product does not move.

Point 3 matters here and is not hypothetical.  `jpsi.diffrad_vm` carries
sigma_T ~ Q^-3.11, which double-counts the 1/Q2 already in the flux, and the
model sits 4-6x below the data at FT kinematics.  Form A is immune to that.
What survives is only the **shape inside the bin** -- which is what step 7
iterates away.


## 3. Procedure

**Step 1 -- one cut list.**  Write the cuts once; apply them to data, rad MC
and Born MC from the same code.  Include the exclusivity cut.  Both cards
carry `vcut = 0`, so nothing was cut at generation and the whole radiative
phase space is in the files; the exclusivity cut is applied **offline**, and
that is the only way migration through resolution gets simulated.

**Step 2 -- signal yield per bin.**  Count in a window around
M(ee) = 3.0969 GeV.  Do **not** fit a free lineshape: a free Gaussian width
absorbs the radiative tail and biases the yield.  Window counting reduces
that model bias by about a factor 5.  Subtract background from sidebands or
a fitted continuum and propagate the subtraction error into `N_data(i)`.

**Step 3 -- the same window on the MC.**  Identical M(ee) window on the
reconstructed rad MC gives `n_rad(i)`.  Both arms used here run `idecay = 2`,
so the decay FSR is simulated and the lineshape is never assumed -- that is
what makes the window efficiency cancel between data and `n_rad(i)`, and what
keeps the decay radiation OUT of `C_rad`, where it would be double-counted
against BR (Sec. 5.2).

**Step 4 -- normalise.**  `Y_rad(i) = L * BR * w_rad * n_rad(i)`.  Average
`sigma_nb` over jobs; all production jobs share the card, so `w_rad` is
common to them.

**Step 5 -- form R_i and multiply by the model.**  Bin-average the model's
Born cross section over the bin with the model's own density, **or** quote at
the bin centroid with a bin-centring correction.  State which.  They differ
wherever the t-slope is steep, which near threshold is everywhere.

**Step 6 -- compute C_rad(i) anyway.**  Form A does not need it, but it must
be reported, and its bin-to-bin behaviour is the diagnostic that says whether
the correction is sane.  A C_rad flat in W, or above 1, means something is
wrong upstream.  Note that the published 0.80 / 0.76 / 0.62 by W bin was
measured with the `born` arm as denominator, so it carries the decay-lepton
factor as well; the extraction's C_rad = rad/borndecay will sit closer to 1.

The third arm is not a free check on F(K), and this note said so in an
earlier revision.  Measured over 300 jobs per arm,

    born       <sigma_nb> 1.948358e-04 +- 1.8e-06   preset kills 0.4182
    borndecay  <sigma_nb> 1.926512e-04 +- 1.8e-06   preset kills 0.4243
    borndecay / born = 0.98879 +- 0.01296   (-1.1 %, 0.9 sigma from 1)

Since int(MB) is identical for the two arms and `written` = 5000 in both, that
ratio is purely the effect of decay FSR on the GENERATION daughter window
(|p| 1-9 GeV, theta 5-40 deg).  The analysis cut |p| > 1.95 GeV is tighter, so
it is F(K) for a window the analysis does not use -- and at 1.1 +- 1.3 % it is
consistent with nothing happening.  The decay-radiation model is checked
against the PDG instead, Sec. 5.3.

**Step 7 -- iterate.**  Reweight the MC event by event by
`sigma_extracted / sigma_model` as a function of (W, Q2, t); recompute
`n_rad(i)` and `R_i`.  Two or three passes converge.  This is required, not
cosmetic: the model is fitted over Q2 = 0.015-0.1 GeV^2 while the generation
box runs 0.001-0.25, so without iteration the bin-averaging is
model-dominated at both Q2 edges.

**Step 8 -- efficiency scale factors.**  GEMC does not reproduce the FT
electron or the tracking efficiencies.  Apply the measured data/MC ratios to
`Y_rad(i)`, per bin wherever they are kinematics-dependent.

**Step 9 -- quote it.**  For `ep -> e'p'J/psi` no flux factor is needed.  If
converting to sigma(gamma*p -> J/psi p), state the flux convention (Hand or
Gilman) explicitly: it moves the number.


## 4. Statistical error

From Form A, with the three counts independent:

    (dR/R)^2 = 1/N_data(i) + 1/n_rad(i) + (background subtraction term)

The Born arm enters the statistical error **only** through `C_rad` and
`eps_Born`, i.e. only in Form B and in the reported diagnostics.  That is
not a reason to generate less Born: see Sec. 5, the arms must match.


## 5. Facts checked in the actual samples, 2026-11-06

### BR is not in the generator

    grep -rniE "branching|0.0597|5.97" --include=*.py exclurad_py/  ->  nothing

`report.sigma_nb` is the **full J/psi production cross section**.  Every
event is decayed to e+e- with probability 1 (`idecay = 2`).  The data yield
is an e+e- yield.  So BR multiplies the MC, once.  Forget it and the answer
is wrong by 16.7.

### The Born arm must be `borndecay`, because BR is the full width

**This is the crux of the whole procedure.**  VPK, 2026-11-06.

The normalisation uses BR(J/psi -> e+e-) = (5.971 +- 0.032) % from the PDG.
That number is the **radiation-inclusive leptonic width**: Gamma(J/psi ->
e+e-) is measured inclusively, so J/psi -> e+e-gamma is already counted
inside it.

Radiation of the decay leptons is therefore **not** a correction to be
removed.  It is part of the physical final state that BR counts.  Its only
effect on the measurement is on the ACCEPTANCE -- through the M(ee) window and
through the lepton momenta -- so it must be simulated in every arm that is
used, so that it cancels in `C_rad` and is folded correctly into `eps`.  Put
it into `C_rad` instead and it is counted twice: once in BR, once in the
correction.

So the Born arm for an extraction is the one with Born PRODUCTION and decay
radiation ON.  The `osg_mariana_lund` campaign has exactly that, as a third
arm.  Card fields verified on ifarm, 2026-11-06:

    field        meaning                     rad   born   borndecay
    -----------------------------------------------------------------
     1  model                          jpsi.vmd2026_pdet  (all three)
     4  iborn    vertex radiation            0      1        1
     8  vcut                                0.0    0.0      0.0
    12  idecay   decay-lepton radiation      2      1        2
    18  itarg    pre-vertex radiator         1      0        0

    use:  C_rad = rad / borndecay   -> production radiation only
                                       (vertex + pre-vertex straggling);
                                       decay radiation cancels

Statistics: 2000 LUND files x 5000 events per arm per period
(fa18 / sp18 / sp19), and all three arms are linked into each of the nine OSG
dataset directories, so born, borndecay and rad are all submitted.

The `born` arm (`idecay = 1`) is NOT wrong.  It was built for the RC-method
comparison in `docs/rc_method_jpsi/`, where method 2 multiplies the analytic
decay-lepton factor F(K) explicitly, so having the decay radiation inside
`C_rad` was deliberate and consistent there.  It is the wrong denominator for
a cross-section extraction, which is a different question.

Note the model differs between campaigns: `osg_mariana_lund` uses
`jpsi.vmd2026_pdet`, `prod_jpsi_f18in_20261009` uses `jpsi.diffrad_vm_pdet`.
Form A is invariant under the model normalisation, so that costs nothing
there; only the shape inside the bin matters, and step 7 iterates it away.

### The decay-radiation model agrees with the PDG radiative branching fraction

The premise of Sec. 5.2 -- that BR is radiation-inclusive -- is in the PDG
listing itself, and it also gives a number the generator can be tested on.
PDG 2025, J/psi(1S), with `e+ e- gamma` INDENTED under `e+ e-`, which in PDG
notation marks a submode, i.e. a subset of the line above:

    Gamma_5   e+ e-            ( 5.971 +- 0.032 ) %
    Gamma_6     e+ e- gamma    [a]  ( 8.8 +- 1.4 ) x 10^-3
    Gamma_7   mu+ mu-          ( 5.961 +- 0.033 ) %

    [a] For E_gamma > 100 MeV.

So BR(e+e-) is the full width and the radiative part is quoted as a fraction
of it:

    Gamma_6 / Gamma_5 = 8.8e-3 / 5.971e-2 = 14.74 +- 2.35 %   for E_gamma > 100 MeV

Since M_ee^2 = M^2 - 2 M k*, that threshold is M(ee) < 2.99523 GeV.  Our own
decay model, called directly with no production and no acceptance cut:

    exclurad_py DecayFSR (idecay = 2, exact matrix element), 300 000 decays

      E_gamma > 100 MeV :  14.38 +- 0.06 %
      PDG               :  14.74 +- 2.35 %
      difference -0.36 %  =  0.15 sigma

      E_gamma >  20 MeV : 23.95 %     E_gamma > 200 MeV : 10.12 %
      E_gamma >  50 MeV : 18.63 %     E_gamma > 400 MeV :  6.06 %
      E_gamma > 100 MeV : 14.38 %     E_gamma > 800 MeV :  2.56 %
      mean E_gamma 72.0 MeV, max 1548 MeV

This is the first independent check on `generator/decay_rad.py`.  It also
explains the card's own "~15 % of decays end below M(ee) = 3.0 GeV": 3.0 GeV
corresponds to k* = 95.4 MeV, essentially the PDG threshold, which is why the
two numbers coincide.

Reproduce with `python3 -c` on `DecayFSR(3.0969)` and `decay_meson("jpsi", ...)`;
PDG source: rpp2025-list-J-psi-1S.pdf.

### The card header of prod_jpsi_f18in_20261009 is false

That campaign's rad card says "Born twin: IDENTICAL except iborn (field 4)"
and offers a `diff` command to prove it.  The command prints five
differences: fields 4, 11 (seed), 12 (`idecay`), 13 (`outroot`) and 18
(`itarg`).  The danger is not the confusion but the repair: somebody who
trusts the comment may "fix" field 12 or 18 to make the diff clean, and a
Born arm whose `itarg` or `idecay` has been flipped silently changes what
`C_rad` means.  It is the comment that is wrong, not the cards.

The `itarg = 0` point-vertex bug is **not** in this sample: the fix
`bbe6628` (2026-10-22 11:25) is an ancestor of the sample's recorded commit
`8ac2fc0` (2026-10-22 12:36), so the Born arm has the proper 5 cm vertex
spread rather than vz = 0.

### Do not take C_rad from sigma_nb

    born  report.sigma_nb = 3.494470e-04
    rad   report.sigma_nb = 3.368744e-04       ratio rad/born = 0.9640

`sigma_nb` counts events **before** the acceptance preset, so a rad/Born
ratio built from it is about 1.33 % high -- and in any case it is a box
average, not a bin correction.  `C_rad(i)` must come from reconstructed
yields inside the bin.

### The prod_jpsi_f18in_20261009 arms are not at equal statistics

    born out_inb   1120 f32        rad out_inb   1167 f32
    born out_outb   852 f32        rad out_outb  1180 f32

The Born outbending arm is 28 % short.  That limits the precision of both
`C_rad` and `eps_Born`.  Finish it before quoting a systematic on either.

### iacc = jpsi is a hard generation cut, not a flag

`events.py` does `continue` on failure: the event is never written.  The
windows travel in the `.lund.info` sidecar:

    proton     |p| 0.2-4.5 GeV    theta  3-45 deg
    daughter   |p| 1.0-9.0 GeV    theta  5-40 deg
    electron   |p| 0.1-2.5 GeV    theta  2-20 deg

If any analysis cut is **looser** than the corresponding preset edge, the MC
has exactly zero events out there and the acceptance is wrong, not merely
imprecise.  Measured in a 2000-event sample, the live edges are the proton
theta floor (min 3.013 deg, zero events below) and the daughter theta
ceiling (max 39.96 deg).  The proton |p| window never bites: the kinematic
minimum is 0.7156 GeV/c against a 0.2 GeV/c floor.

### t from the proton, and decide t or t'

The proton t estimator has RMS 0.047 GeV^2 against 0.230 for the J/psi one,
5-6x better.  The existing analysis ntuple carries `t_prime`, not `t`.
Since |t|_min runs from 2.10 GeV^2 at threshold to 0.4536 at the top of the
W range, binning in t' and in t map onto quite different regions.  Fix the
choice once and use it in the data, in both MC arms, and in the model
bin-averaging.

For reference, the forward kinematic edge at the ceiling
W2 = M^2 + 2*M*E = 20.7717 GeV^2 (W = 4.5576):

    |t|_min = 0.4536 GeV^2    |p_proton| = 0.7156 GeV/c    theta_p = 0.67 deg

so at the forward edge the recoil proton goes down the beamline and is
undetectable; the 3 deg preset floor removes it.  Over the whole box the
recoil proton never exceeds theta = 25.3 deg -- it is a Forward Detector
particle everywhere, never CD.


## 6. Systematics

| Source | How to estimate |
|---|---|
| Model shape inside the bin | spread over iterations; vary the Q2 slope within its fit error |
| Exclusivity cut | vary the MM2 window; `C_rad` is what moves |
| Signal extraction | window width; sideband versus fitted background |
| MC statistics | `n_rad`, `n_born` separately |
| Efficiency scale factors | their own uncertainties, bin-dependent |
| Luminosity | from the data side |
| BR | 0.54 % relative |
| Bin centring | centroid versus bin-averaged model |


## 7. Ntuple branches the chain needs

The reader in `production/analysis/campaign_plots.py` uses

    E_photon, Q2, W, t_prime, invariantMass,
    scattered_theta, scattered_p, proton_theta, proton_p,
    positron_theta, positron_p, electron_theta, electron_p,
    time_ee, x_FT, y_FT

Data and both MC arms must expose the same set, filled by the same code, or
Step 1 cannot be honoured.


## 8. How this worktree was made, and the rule behind it

    git -C ~/exclurad_py worktree add -b jpsi-born-xsec-note \
        ~/exclurad_py_bornxsec origin/production

A worktree, not a branch switch in `~/exclurad_py`: that checkout is shared
with running jobs and other sessions, so its branch must not move.  The base
is **`origin/production`**, not the local `production`, which at the time of
writing carried two unpushed commits (`e1dc8cc`, `ffd26d8`) -- branching
from the published tip keeps this note's branch from dragging unrelated
unpushed work onto GitHub.
