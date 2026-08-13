# 100k generation and comparison against diffrad_vm — 2026-08-12

## What was produced

100 000 **unweighted** events with explicit hard photons, LUND format,
Ebeam 10.6 GeV, for four channels:

| channel | model | ms/event | unweight. eff | overweight | closure fails |
|---|---|---:|---:|---:|---:|
| pi0 | `pi0.vpk2013` | 1.59 | 0.281 | 0 | 0 |
| eta | `eta.tableB1` | 1.54 | 0.301 | 0 | 0 |
| phi | `phi.diffrad_vm` | 6.56 | 0.046 | 0 | 0 |
| jpsi | `jpsi.diffrad_vm` | 4.22 | 0.064 | 75 | 0 |

Files: `production/{pi0,eta,phi,jpsi}_mesonrc_100k.lund` (+ `_kin.npz`).
Every LUND file carries a provenance header (model, version, git commit,
box, cut).  Four-momentum closure was verified for every event.

`phi.diffrad_vm` was added for this run — transcribed from `sigma_T_phi` /
`sigma_L_phi` of `diffrad_vm.f90` with the tuned defaults (alf1=400,
alf2=2.0, alf3=0.2, nuT=2.253, bt=1.326, cR=1.0), i.e. the model behind
`samples_v2/phi_*_100k`.  `jpsi.diffrad_vm` was re-verified against the
current source (nuT=3.114, mg2=2.637, cR=0.4).

## Comparison against `~/MC_vector_mesons/samples_v2` — **DISAGREES**

**CORRECTION (VPK, 2026-08-13): the reference output is UNWEIGHTED.**  None
of these generators produce weighted LUND.  Header column 10 is the event
cross section `sigev`, written for information (`gen_lib.f90:386`:
`npart,1,1,0,ihel, 11,ebeam,2212,1,sigev, xb,y,w2,q2,nu`) — not a weight.
Column 5 is the beam HELICITY, and columns 11-15 are xB, y, W2, Q2, nu,
which can be read directly instead of recomputing invariants.  An earlier
draft weight-filled the reference histograms with column 10; every number
below has been recomputed unweighted.  The diffrad_vm discrepancy gets
slightly WORSE: phi frac(E_gamma>0.15) is 0.177 unweighted (not 0.236), so
the ratio is 2.8 rather than 2.1; jpsi 0.051, ratio 5.3.

One thing that is not directly comparable and was handled:
* the **radiative fraction is not an observable** — it is set by where the
  soft/hard split v_min sits.  The comparable quantity is the photon
  spectrum above a common threshold.  (exclugen states its split
  explicitly: `vmin = 1e-4` in the run card.)

### What agrees

* **v_max is identical.**  diffrad_vm's `vmax_kin` (a fixed-t expression)
  and `EventKin.vmax_kin` agree to 1.000 at every point tested, both
  channels.  Same inelasticity window.
* **Photon spectrum shape.**  Median E_gamma above 0.15 GeV: phi 0.594 vs
  0.575 (+3 %), jpsi 0.292 vs 0.256 (+14 %).
* **jpsi |t| distribution** tracks the reference well (fig. panel e).

### What does not

| | mesonrc | diffrad_vm (weighted) | ratio |
|---|---:|---:|---:|
| phi, frac E_gamma > 0.15 GeV | 0.494 | 0.236 | **2.1** |
| jpsi, frac E_gamma > 0.15 GeV | 0.273 | 0.051 | **5.4** |

The ratio is not flat in threshold (phi 1.7 → 2.1 → 0.7 from 0.02 to
2 GeV), so it is a shape difference, not a normalization.  The Q2
distributions also differ, severely for jpsi (panel d): the reference has
almost nothing above Q2 = 0.1 while mesonrc extends across the box.

**This is unresolved.  The generator is NOT validated against diffrad_vm.**

### Ruled out

* Reference weights (weighted and unweighted fractions differ by <10 %).
* The reference's own kinematic cuts (y, xB, and for jpsi the e'/proton
  momentum and angle cuts) — the reference keeps 100 % of its events under
  them, mesonrc 90 %, and the ratios barely move.
* v_max definition (identical, above).

### One concrete defect found

The **phi eta-table is too coarse**.  Over that box eta spans 0.76–3.09,
and a 4x4x4x6 interpolation table gives rms error **0.143**, max 0.30 — vs
0.001 on the narrow eta box used for the RC validation.  Because the
branching is `soft = sigma_obs - sigma_hard`, an underestimated sigma_obs
inflates the hard fraction directly.  This plausibly accounts for much of
the phi excess.  It does **not** explain jpsi, whose table is fine
(rms 0.033).

### Next steps to close this

1. Refit the phi run with a much finer eta table (or split the box), and
   re-check `check_interpolation` before generating.
2. For jpsi, compare Born-level distributions first (no RC): if dN/dQ2
   already disagrees, the problem is the Born cross section or the flux
   convention, not the radiative tail.  That is the single most
   discriminating test and should be done before anything else.
3. Check whether diffrad_vm's own soft/hard split is being compared
   like-for-like — the spectrum agreement in shape but not rate is the
   signature of a branching-probability difference rather than a tail
   shape difference.
