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
| 2 | Radiative MC through GEMC | `prod_jpsi_f18in_20261009/rad` |
| 3 | Born MC through GEMC | `prod_jpsi_f18in_20261009/born` |
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

    sigma_Born(i) = R_i * sigma_Born_model_binavg(i)

    R_i = N_data(i) / ( L * BR * w_rad * n_rad(i) )

### Form B -- explicit factors.  This is the one to tabulate systematics on.

    sigma_Born_binavg(i) = N_data(i)
                           -------------------------------------------------
                           L * BR * dOmega(i) * eps_Born(i) * C_rad(i)

    eps_Born(i) = n_born(i) / m_born(i)

    C_rad(i)    = ( w_rad  * n_rad(i)  ) / ( w_born * n_born(i) )

The two are algebraically the same once the bin-averaging is done
consistently.  **Compute A.**  Three reasons:

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
reconstructed rad MC gives `n_rad(i)`.  Both arms run `idecay = 2`, so the
decay FSR is in the simulation.

**Step 4 -- normalise.**  `Y_rad(i) = L * BR * w_rad * n_rad(i)`.  Average
`sigma_nb` over jobs; all production jobs share the card, so `w_rad` is
common to them.

**Step 5 -- form R_i and multiply by the model.**  Bin-average the model's
Born cross section over the bin with the model's own density, **or** quote at
the bin centroid with a bin-centring correction.  State which.  They differ
wherever the t-slope is steep, which near threshold is everywhere.

**Step 6 -- compute C_rad(i) anyway.**  Form A does not need it, but it must
be reported, and its bin-to-bin behaviour is the diagnostic that says
whether the correction is sane.  On this MC we measured 0.80 / 0.76 / 0.62
by W bin -- falling as W rises.  A C_rad flat in W, or above 1, means
something is wrong upstream.

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

### The twins differ in two fields, not one

From `prod_jpsi_f18in_20261009/{born,rad}/prov/*_043.card.txt`:

    field  4  iborn:   born 1    rad 0
    field 18  itarg:   born 0    rad 1

This is **correct**: Born means no radiation of any kind, internal or
external, so `C_rad` from these twins includes target straggling *and*
vertex radiation, which is what corrects data to Born.  But the card header
claims "IDENTICAL except iborn (field 4)", which is false, and anyone
diffing the two cards on that promise will think the sample is broken.
The header comment should be fixed.

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

### The arms are not yet at equal statistics

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
