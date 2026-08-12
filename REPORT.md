# mesonrc — build and validation report

Built overnight 2026-08-11 → 12 on branch `feature/mesonrc-package`
(worktree `~/mesonrc-build`, repo `~/mesonrc`).

This report says what was built, what was verified, **what was found to be
wrong and fixed**, and — importantly — what is *not* yet validated. Read
the "Not yet validated" section before using any number from this code in
a physics result.

---

## 1. What was built

### 1.1 The package

Three codes collapse into one. `mesonrc/core/` is the RC engine, imported
unchanged from the `vmrad` delivered on 2026-08-11 and now **frozen**: it
knows nothing about channels, models, or topologies. Everything that
varies is one of three axes, and the layout mirrors exactly those:

| axis | where it lives | why it is separate |
|---|---|---|
| channel (rho, phi, J/psi, pi0, eta) | `(m_V, m_U)` on the model | the only thing the engine needs |
| model vintage | `mesonrc/models/` registry | where the physics content lives and grows |
| detection topology | `mesonrc/topology.py` | silently wrong if implicit — see §1.3 |

The only physics difference between the vector and pseudoscalar copies of
the original `vmrad` was six lines about `m_U`. Confirmed by diff; there
was never a reason for two packages.

### 1.2 Model registry — the transparency mechanism

Ten models registered, keyed `<channel>.<vintage>`:

```
eta.refit2013   eta.tableB1   eta.vpk2021
pi0.vpk2014     pi0.vpk2021
jpsi.diffrad    jpsi.diffrad_vm   jpsi.roberts2026
rho.demo        phi.demo
```

The registry **refuses a model with no provenance** — that is enforced in
`register()`, not a convention. Every model records where it came from,
its reference, and a `validity` range; every output file is stamped with
model name, version, git commit and branch.

The pi0/eta parameter columns were transcribed directly from the Fortran
in front of me:

* `eta.tableB1` — the eta column of `~/eta-rc-2026/model2013note/pi0eta.par`,
  i.e. analysis note CLAS-2016-101 Appendix B Table B.1: `sigT=204.99`,
  `sigL=1392.1`, `sigTT=-3083.0`, `sigLT=271.6`, `alphaTT=1.101`,
  common `/(Q2+1)^2`. **This is the published production model.**
* `eta.refit2013` — the repo-default eta column (`sigT=42.56526`,
  `alphaT=0.06235287`, `sigL=1743.521`, `n=2.165608`). A *later* refit,
  explicitly marked NOT the published model. I initially wrote these
  parameters from the README summary and then replaced them with the
  actual `pi0eta.par` values — the README rounds them.
* `pi0.vpk2014`, and the 2021 GT fits for both channels, including
  EXCLURAD's `dvmpw` wrapper factor 2 on `sigma_LT`, which belongs to the
  wrapper and is applied in the model class, not in the functional form.

`validity` is enforced and **counted**: out-of-range calls are clamped and
`sf.usage()` reports the fraction. At the standard eta test point 0.15% of
the tail's SF evaluations land outside the fit's grid — small, but it is
now visible instead of silent.

### 1.3 Topology, made explicit

The trap that motivated a whole module: for pi0/eta, EXCLURAD **detects
the proton and misses the meson**. So `mV = m_p`, `mU = m_meson`,
`v = MX^2 - m_meson^2`, the SFs are functions of the nucleon transfer
`t_N` (not the kinematic `t` built from the detected proton), and because
the hadron-plane x-axis is defined by the detected hadron, **`sigma_LT` and
`sigma_LT'` change sign**. All four of those facts now live on a
`Topology` object and are stamped into output. Previously they were buried
inside an adapter class in a comparison script.

### 1.4 Generator

Unweighted, LUND output, "measured-first" structure — sample the
**observed** kinematics from `sigma_obs`, then decide whether the event was
radiative — so the generator inherits the exact RC rather than applying it
as an afterthought. This is the structure of exclugen and `diffrad_vm` v2,
and the v2 phase-space construction (no ISR/FSR kinematic branch; the two
collinear peaks are importance-sampling channels only) is what the photon
sampler implements.

Output is unweighted because GEMC does not propagate weights and CLAS12
analysis counts events. `write_lund` **refuses** to write an event whose
weight differs from 1 unless explicitly overridden.

---

## 2. Four real errors found and fixed

All were mine, in the generator, and all would have produced
plausible-looking wrong output.

### 2.1 Re-exponentiating `delta_inf` at the slicing point (physics)

To sample the radiative tail you need a positive-definite density, which
the Bardin–Shumeiko subtracted integrand is not. The natural move is to
slice at a small `v_min` and write

```
sigma_soft = sigma_Born * exp(delta_inf(v_min)) * (1 + delta_VR + delta_vac)   # WRONG
```

This is wrong, and badly. Exponentiation resums soft emission up to the
**physical** cut `v_max`; re-exponentiating at an arbitrary slicing point
resums to the wrong place. At the standard eta point
`delta_inf(v_max) = -0.26` but `delta_inf(10^-6 v_max) = -1.13`, so `exp()`
is nowhere near linear and the cancellation that should be exact fails:

| `v_min` | soft | hard | `sigma_obs` | eta |
|---:|---:|---:|---:|---:|
| 2e-5 | 0.01091 | 0.01343 | 0.02435 | 1.036 |
| 2e-6 | 0.00944 | 0.01683 | 0.02627 | 1.118 |
| 2e-7 | 0.00817 | 0.02023 | 0.02839 | 1.209 |
| 2e-8 | 0.00706 | 0.02363 | 0.03069 | 1.306 |

A 23% dependence on a quantity that must not appear in the answer, against
the correct `eta = 0.8235`. **Diagnosis:** the IR identity itself was
verified sound — the subtraction term integrates to
`sigma_Born * d(delta_inf)` to 0.06% — so the error was in my bookkeeping,
not in `core`. **Fix:** keep the exponentiation at `v_max` and define the
soft branch as the remainder, `sigma_soft = sigma_obs - sigma_hard`, which
reproduces `sigma_obs` identically for any `v_min` by construction. The
residual constraint is that `sigma_soft` must stay positive, which bounds
`v_max/v_min` to about 1e5 at CLAS12 kinematics; `choose_vmin` enforces it
and generation refuses to proceed if no valid slicing point exists.

### 2.2 Per-cell majorant over a peaked density (sampling)

The first sampler used one majorant per `v`-cell while the density varies
by orders of magnitude across the collinear peaks, so acceptance collapsed;
it also would have discretized photon directions onto quadrature nodes,
which is simply not a generator. Replaced with a proper 3D cell grid in
`(ln v, x, phi_k)` where `x = ln(a - cos theta_c)` flattens the peaks —
the same substitution the RC quadrature uses — with continuous sampling
inside each cell.

Tuning the majorant (`n_sub`, `n_x`, `safety`) against measured overweight
rate:

| n_sub | n_x | safety | µs/photon | efficiency | overweight frac | max overweight |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 24 | 1.5 | 5.8 | 0.50 | 1.7e-3 | 2.57 |
| 6 | 24 | 2.0 | 6.5 | 0.44 | 5.3e-5 | 1.57 |
| 8 | 32 | 2.5 | 10.0 | 0.30 | 1.2e-5 | 1.22 |
| **12** | **32** | **3.0** | **11.8** | **0.25** | **0** | **1.00** |

Defaults are the zero-overweight row. Overweight events are counted and
reported, never clipped.

### 2.3 Cell-centre photon broke four-momentum conservation (kinematics)

Reusing one `TailSampler` per coarse cell (to avoid a 0.4 s grid build per
event) is an approximation in the photon *distribution*. It became an
outright error because the photon four-vector was taken from the sampler
too: the energy `omega = v / (2 z)` with `z = Lam0 - vec(Lam).nhat` is a
property of the **event**, not of the cell. **Essentially every radiative
event failed four-momentum closure** -- one pre-fix run recorded 921
closure failures against 922 hard events, i.e. the soft branch was fine and
the radiative branch was entirely broken. (A second pre-fix run showed this
as "65% of accepted events", which understated it: the surviving fraction
was just the soft branch.) Fixed by `tail.photon_at()`, which rebuilds the
photon and the detected hadron at the event's own kinematics from the
sampled `(v, nhat)`. Closure residual is now 6.6e-15 and the failure count
is zero. The residual approximation is confined to the *distribution* of
`(v, nhat)` — which is the documented gap 3, and is now benign.

### 2.4 LUND wrote the wrong PIDs (bookkeeping)

`ev["meson"]` is the *detected* hadron, and in the EXCLURAD pi0/eta
topology the detected hadron is the **proton**. The writer labelled that
slot with the meson PID, producing "eta" particles carrying the proton's
momentum and a recoil misidentified as a neutron. Caught by reading the
written file back and checking PID against mass. PID assignment is now
topology-driven. The same read-back check found the output precision
(`%14.6e`) was too coarse — 18 of 20000 events violated four-momentum at
1e-6 when re-read; widened to `%18.10e`, now exact to 1e-8.

This one is worth noting as a process point: the internal closure check
passed at 1e-15 the whole time. Only reading the *file* back, as a
consumer would, exposed it.

### 2.5 The performance lesson (not an error, but the design driver)

A naive per-photon sampler took **>100 s for 200 photons**; the vectorized
one does **11.8 µs/photon**. The cause is not the physics — numpy is fast
on large arrays and ruinous on scalar-sized ones, and the naive loop built
an `EventKin` object and a 4×4 tensor per candidate. The rule the whole
generator is built around: **vectorize over events, never loop over them.**
`_meson_at_v` and `_density_batch` inline the v-dependent kinematics so a
whole block costs one array operation.

The same logic drove the RC-table design: `rc_factor` costs ~0.11 s/point
and cannot be called per event, so `sigma_obs = sigma_Born * eta` with
`sigma_Born` exact per event (0.19 ms) and `eta` interpolated from a
precomputed table whose interpolation error is **measured, not assumed**
(coarse 3×3×3×6 table: mean −5e-5, RMS 1.1e-3, max 2.9e-3 on eta).

---

## 3. Validation status

### 3.1 Internal consistency — all passing

`python3 -m mesonrc.validation.selftest`

| check | residual | tolerance |
|---|---:|---:|
| Born closure (all five SF coefficients) | 1.6e-8 | 1e-6 |
| current conservation of `L_R` | 4.0e-11 | 1e-8 |
| soft factorization `L_R -> 4 F_IR L_0` | 4.3e-4 | 1e-3 |
| IR bookkeeping (subtraction == `d delta_inf`) | 9.9e-5 | 3e-4 |
| slicing form vs Bardin–Shumeiko | 6.1e-4 | 2e-3 |
| `sigma_F` grid convergence | 6.6e-4 | 2e-3 |
| sampler closure (chi2/dof of the `ln v` spectrum) | 1.11 | 3.0 |
| TwoSF: RC shift from the interference SFs | 4.3e-3 | (informational) |

The quadrature-limited entries (IR, slicing, grid) all sit at a few times
the `sigma_F` grid error; tightening them means refining the quadrature,
not fixing physics. The IR check initially "failed" at 5.7e-4 purely
because I ran it at default quadrature — at fine quadrature it is 9.9e-5,
confirming it was the grid and not the identity.

Four-momentum closure of assembled events is exact to 1e-16, with beam
along +z, target at rest, and the photon landing in the ISR peak as
expected.

### 3.1b End-to-end generator run (eta, `eta.tableB1`)

`Ebeam=10.6`, `Q2 in (1.0, 2.5)`, `W2 in (4.5, 7.0)`, `t in (-0.5, -0.15)`,
`MX^2 < 0.20 + m_eta^2`, helicity averaged, 20 000 unweighted events:

| quantity | value |
|---|---|
| setup (eta table 162 RC points + majorant) | 22 s (3.5 s with cached table) |
| generation | **2.11 ms/event** (10^6 events ~ 35 min) |
| unweighting efficiency | 0.469 |
| radiative fraction | 0.651 |
| majorant overweight events | **0** |
| four-momentum closure failures | **0** (max residual 6.6e-15) |
| eta interpolation error (coarse table) | mean -5e-5, RMS 1.1e-3, max 2.9e-3 |

Independent read-back of the written LUND file: 20 000 events, **0**
violating four-momentum at 1e-8, PIDs and masses correct
(11/0.00051, 221/0.54786, 2212/0.93827, 22/0.0).

### 3.1c Vector channels — smoke test and a cross-check against the
earlier study

All three J/psi models run at the near-threshold point of
`2026_RC_test_comparison` (E=10.6, Q2=0.5505, W2=17.64, t=-1.2):

| model | eta (phi=1.0) | v_max |
|---|---:|---:|
| `jpsi.diffrad` | 0.8861 | 0.07114 |
| `jpsi.roberts2026` | 0.8600 | 0.07114 |
| `jpsi.diffrad_vm` | 0.8497 | 0.07114 |

Two independent consistency indicators here. `v_max(t) = 0.07114` matches
the value recorded in that study to five digits. And the eta values sit
next to its phi-integrated numbers — 0.890 for the DIFFRAD SF and 0.852
for the diffrad_vm model — which is what one expects for a single phi
point versus the phi average, not a validation but a sign nothing has
drifted in the port.

The spread across the three models, **0.85 to 0.89 at one kinematic
point**, is the SF-model dependence of the RC that motivates the whole
exercise.

`rho.demo` and `phi.demo` run; their interference SFs are worth only 0.0-0.1%
on eta at the demo point, but those are invented parameterizations and the
number means nothing physically — it exercises the code path, no more.

### 3.2 Against the PUBLISHED eta table — **PASSED, 0.76 % mean**

Target: first (Q2, xB, t) bin of `radcoreta_published.vec` (PRC 95 035202),
20 phi points, E = 5.75 GeV (CLAS6), vcut = 0.20, model `eta.tableB1`.
Reproduce with `validation/eta_published_direct/run_direct.py`.

**Port verified first** against the double-precision Fortran (`sfdrv`
driver) over a 48-point grid: sigma_T and sigma_L to 2.2e-16, sigma_TT to
5.6e-14, sigma_LT to 2.8e-14, tminq to 6.6e-15 — machine level, so any RC
difference is the RC code, not the model.

**The bin definition — this was the whole story.** In the CLAS pi0/eta
analyses **t is measured from the recoil proton**, always:

    t = (p - p')^2 = 2 M^2 - 2 M E_p'(lab)

so fixing t fixes the proton's LAB ENERGY, and as v runs across the
missing-mass window it is the proton's ANGLE that moves.  `EventKin` now
takes `t_kind`, and `baryon_detected` selects `t_kind="target"`.

| | value |
|---|---:|
| points compared | 20 |
| mean residual | **+0.00721 (+0.757 %)** |
| rms residual | **0.00304 (0.328 %)** |
| max abs residual | 0.01160 (1.237 %) |

No window matching, no remapping of t: mesonrc is run in the analysis's own
bin definition and compared straight to the published numbers.

The residual has a **shape**: ~+0.002 at the phi endpoints, ~+0.011
mid-phi.  `~/eta-rc-2026/README.md` records the *same shape* for the
Fortran's own reproduction of this table (endpoints +-0.001, mid-phi
+0.007) and flags it as unexplained — probably a slightly different
2010-era code or model iteration.  mesonrc sits about +0.004 above the
Fortran's reproduction, consistently across phi.  So the disagreement with
the published table is largely a feature the reference code shares, not
something mesonrc introduces.

**Retracted: an earlier "4 % fixed-t effect", and the claim that EXCLURAD
uses a different t.**  Both were wrong, and it is worth being precise about
how.

A first pass ran with `t_kind="photon"` — holding (q - p')^2 fixed instead
of (p - p')^2.  That invariant has a spurious kinematic limit v_max = 0.087
at this bin (vs 1.18 for the correct one), truncating the 0.20 window and
shifting eta by about -0.04.  That was a plain bug.  It was then reported
as a physics finding about "fixed-t versus fixed-angle" binning, which
compounded the error with a wrong story.

EXCLURAD converts t to cos(theta) at Born (main loop, iphy=11 block) and
freezes the ANGLE through the v integration, while mesonrc's `T_TARGET`
holds t itself.  Those are genuinely different prescriptions, and the
kinematics do drift: at this bin, freezing cos(theta) lets t move from
-0.12000 to -0.12665 by v = 0.199, i.e. 5.5 % of |t|.  **But the effect on
the RC is 1e-5 absolute (0.001 %)** — see the table below.  The two agree
to five decimals at every phi.

| convention | eta vs published: mean | rms |
|---|---:|---:|
| `t_kind="target"` (t held fixed) | +0.757 % | 0.00304 |
| `t_kind="cm_angle"` (EXCLURAD) | +0.757 % | 0.00305 |

Two reasons it does not propagate: the drift is small in absolute terms,
and in this topology `ProtonDetectedModel` evaluates the structure
functions at the *supplied* t_N with shifted (W2~, Q2~), so the model never
sees the drift — only the phase space does.

So the published-table agreement above is **independent of the angle-vs-t
choice**, which strengthens it.  `t_kind` is kept because the real
distinction — WHICH PARTICLE t is measured from, `"photon"` vs `"target"` —
does matter and was the actual bug.  `"cm_angle"` is retained only as a
verification of EXCLURAD's convention.

An intermediate matched-window comparison against EXCLURAD (both codes
forced to v < 0.08, where the conventions nearly coincide) gave mean
+0.199 %, rms 0.047 %, and remains valid as a check of the RC machinery
alone; it lives in `validation/eta_matched/`.  It is superseded by the
direct comparison above.

### 3.2b Full-grid validation, both channels — **PASSED**

Run over the complete published grids in EXCLURAD's own convention
(`t_kind="cm_angle"`), E = 5.75 GeV:

| channel | bins | mean | rms | max |
|---|---:|---:|---:|---:|
| pi0 vs PRC 90 025205 (`~/excl2013/radcor_all.vec`, vcut 0.17) | 3104 | **-0.20 %** | **0.47 %** | 3.9 % |
| eta vs PRC 95 035202 (`radcoreta_published.vec`, vcut 0.20) | 1814 | +1.07 % | 0.58 % | 2.5 % |

Model ports verified against their own Fortran first: `pi0.vpk2013` to
14-16 significant digits against the 2013 source, `eta.tableB1` to 13-16
against the double-precision driver.

**The eta offset is not mesonrc.**  Running EXCLURAD (`model2013note`) at
the production vcut over 180 usable bins and decomposing:

| comparison | mean | rms |
|---|---:|---:|
| mesonrc vs Fortran | **+0.148 %** | 0.202 % |
| Fortran vs published | **+0.520 %** | 0.408 % |
| mesonrc vs published | +0.669 % | 0.392 % |

mesonrc reproduces EXCLURAD to 0.15 %, the same level as pi0.  About 78 %
of the eta discrepancy is the Fortran's own failure to reproduce the
published table from Table B.1 — which `~/eta-rc-2026/README.md` already
flags as unexplained.  Two independent codes now agree with each other and
both disagree with the published numbers, so **the published eta RC table
is probably not exactly reproducible from the documented model**, unlike
pi0 whose 2013 production is byte-reproducible.  That is a result for the
eta revisiting work, not a code defect.

### 3.2c The bin-definition question — RESOLVED, no bias

**EXCLURAD uses the standard definition of t.**  Traced end to end:
`t0`/`tt`/`tldtm` are (q - p')^2, an internal Bardin-Shumeiko variable;
`sffun` converts it to cos(theta) and `dvmpw/CHECK_KINE` converts back to
`del2` = (p - p')^2 before XSIGMA_* is called.  Verified against the mass
arrays (`amhad(3)`=0.938 detected proton, `amunm(3)`=0.5479 undetected
eta) and by re-deriving both conversions.  There is no redefinition.

**What differs** is which variable is frozen during the photon
integration.  EXCLURAD is written for (W, Q2, cos(theta*), phi) binning —
its native 2002 parameterization — and the DVMP wrapper maps the supplied
t onto cos(theta*) once, at Born.  Freezing the angle lets the physical t
of tail events drift.

**Size of the drift.**  ~0.007 GeV^2 at the top of the window, against
analysis t bins 0.06-0.5 GeV^2 wide.  NOTE (VPK): the RC is evaluated at a
KINEMATICAL POINT, not integrated over a bin, so bin width does not absorb
this in the calculation itself; it only matters when the point value is
applied to binned data.  An earlier draft used bin width as the resolution
and was wrong to.

**Status of the claimed 13-17 % effect: REAL AT A POINT, but its
significance is unsettled.**  It comes from bins where the fixed-t window
truncates, and those are exactly the bins sitting on the kinematic edge:

| | median \|t\| - \|t_min\| | n |
|---|---:|---:|
| truncated (v_max < v_cut) | **0.019** | 302 |
| untruncated | 0.416 | 2802 |

Every bin within 0.02 GeV^2 of t_min is truncated (186/186).  At t = t_min,
v_max -> 0 IDENTICALLY: the Born configuration is already collinear, so any
energy loss is kinematically forbidden at that t.  This is not an artifact
of the evaluation — it is the correct fixed-t limit.  In the worst bin
(Q2=2.733, xB=0.384, t_min=-0.1915, bin t=-0.195) v_max recovers
0.018 -> 0.227 over Delta|t| = 0.045 and eta recovers 0.778 -> 0.898.

**What is settled and what is not.**
* Settled: EXCLURAD uses the standard definition of t; there is no
  redefinition (above).
* Settled: at a fixed kinematical point, v_max(t) is a genuine limit, and
  EXCLURAD integrates past it for bins near t_min.
* NOT settled: whether that biases the published results.  The RC is
  evaluated at a point but APPLIED to a bin of data whose events span a t
  range where v_max runs from ~0 to past the cut.  The strictly-fixed-t
  point value may be a worse estimator of the bin's correction than
  EXCLURAD's.  Deciding this needs a t-integrated (Born-weighted) RC across
  the edge bins, compared against both point prescriptions.  Neither code
  does that today.  Until it is done, no claim of bias is supported and
  none is made here.

### 3.3 A process failure worth recording

The first attempt at this validation ran as a background agent and
produced nothing. Its processes were killed by an over-broad
`pkill -f "mesonrc"` issued while debugging unrelated slow generator runs
— the pattern matched the agent's own processes, since it was working
under `~/mesonrc/validation/`. It had got as far as porting the model and
building the Fortran driver; that work (`sfdrv.f`) was reused here.
Lesson: scope process kills to a PID or a distinctive pattern, never to
the project name.

---

## 4. Not yet validated / known gaps

Listed plainly because these are what would bite.

1. **Bins adjacent to t_min need bin-integration, not point evaluation**
   (§3.2c). Not specific to this code; ~10 % of the pi0 grid sits within
   0.05 GeV^2 of t_min where v_max(t) varies fast across the bin.
2. **The eta published table may not be reproducible from Table B.1**
   (§3.2b): mesonrc and EXCLURAD agree to 0.15 % and both sit ~0.5 % above
   the published numbers. Worth settling before the revisiting work leans
   on that table.
3. **The photon spectrum uses the majorant cell centre, not the exact
   event kinematics.** The soft/hard *branching* uses the event's own
   `sigma_obs` and is exact; the photon *direction and energy* come from a
   `TailSampler` built at the centre of the majorant cell and reused, to
   avoid a ~0.4 s grid build per event. Within a cell the density is
   smooth, but this is an approximation and it is not quantified. Fixing
   it properly means a final accept/reject against the event's true
   density, batched over a block — the machinery (`_density_batch`) already
   exists.
4. **`rho.demo` and `phi.demo` are invented parameterizations**, flagged
   as such in their provenance. No rho/phi number from this code means
   anything until a real fit is registered.
5. **All three J/psi models set `sigma_TT = sigma_LT = sigma_LT' = 0`**,
   because none of the underlying models provides them. The engine does
   not require this — that is the entire point — but the vector-meson gap
   is only *closable*, not yet *closed*, and needs an SDME fit.
6. **The `delta_VR` Li2 mass discrepancy is unresolved.** mesonrc uses the
   undetected mass `m_U`, EXCLURAD uses the target mass; ~0.1% on eta. The
   covariant derivation says `m_U`. Not "fixed" in either code — flagged.
7. **Vector-channel RC runs (§3.1c) but vector *generation* is untested
   end to end.** The generator was exercised on eta only.
8. **No LUND file has been through GEMC.** The format is written to the
   CLAS12 spec but has not been read back by the actual simulation chain.

---

## 5. How to pick this up

```bash
cd ~/mesonrc-build
python3 -m mesonrc.validation.selftest          # should print "all checks passed"
python3 -c "from mesonrc import models; print(models.list_models())"
```

The branch is `feature/mesonrc-package`; `main` still holds the original
`vmrad` exactly as delivered. Nothing is pushed anywhere. The eta-grid
validation lives on the separate branch `validate/eta-published-grid` in
the main worktree `~/mesonrc`, so the two lines of work do not collide.

Suggested order once the eta verdict is in:

1. Read the eta validation verdict; if it fails, stop and fix `core`.
2. pi0 against PRC 90 025205 — reuse `validation/legacy.py`.
3. Make the photon exact (gap 3) — this is bounded, known work.
4. Register a real rho or phi fit and run the first full vector generation.
