# The amplitude model, state of 2026-09-05

For whoever installs the next `amp` version into `exclurad_py`.

The model is fitted in `~/rc_iter1`, now public at
https://github.com/vkubarovsky/rc_iter1 — clone it rather than copying files by
hand, so the provenance of a parameter vector stays attached to it.

---

## 1. The one thing that can go wrong silently

`~/exclurad_py/exclurad_py/models/` carries **two copies of the amplitude code**:

| file | what it understands |
|---|---|
| `_amplitude_fit.py` | the old 27-parameter layout: `R_ET` with `ET_d = R * ET_u * exp(db t)`, and the `ln 0.15` offset in the slope |
| `_amplitude_fit2.py` | the current layout: independent `Ebar_T^d` block (`p[13]`, `p[14]`, `p[12]`, `p[27]`), the optional xB shape, and the flavour phases `p[34..36]` |

The 2026-09-05 fit is a **37-element vector** with an independent `Ebar_T^d` and
three flavour phases.  It is computed correctly **only by `_amplitude_fit2.py`**.

Registered through the old file it does not fail — `amplitudes._flavour` simply
takes the `len(p) > 27` branch as false and falls back to the legacy `R_ET`
parametrisation, reading `p[13]` and `p[14]` as a ratio and a slope shift when
they are now a normalisation and a slope.  The generator then runs, produces
events, and the cross sections are wrong with nothing in the log.

Of the four installed vectors, only `amp2026n_par.npy` is 37 long; `amp2021`,
`amp2026` and `amp2026s` are all 27.  **Check which of the two files
`amp2026n`'s registration binds `_AF` to before trusting anything it produced.**

Related: `install_amp2026s.sh` in `~/rc_iter1` converts with `reparam.to_old()`
because the old file wants the `ln 0.15` convention.  For a 37-parameter model
that conversion must **not** be applied — it would flatten the new blocks.  A
new install script is needed, not a reuse of that one.

---

## 2. Which vector to install

Three candidates, all under `~/rc_iter1/runs/`, all fitted to the same database:

| run | sets fitted | points | chi2/ndf | n/p |
|---|---|---|---|---|
| `90_base` | 9, no Hall-A 2021 | 1360 | 1.2268 | 0.98 |
| `B00_base` | 10, with y21 | 1501 | 1.3713 | 1.06 |
| `C0_with_compass` | 11, + COMPASS | 1527 | 1.3918 | 1.04 |

Recommended: **`C0_with_compass`**.  It has the widest kinematic support, and
Hall-A 2021 is the only data above Q^2 = 4.3 — it reaches 8.31, everything else
stops at 4.33.  Without it the whole upper half of the Q^2 range a 10.6 GeV
generator will sample is extrapolation, and the model was overshooting sigma_U
there by a factor 1.8 before y21 was fitted.

The lower chi2/ndf of `90_base` is not a reason to prefer it: it is lower
because the hardest data are not in it.

Copy the `.npy` into the generator tree with its md5 and its run tag recorded.
A generator must not depend on a directory that keeps being refitted.

**Do not apply the normalisation nuisances.**  `summary.json` has a `norms`
block — 1.059 for De Masi, 1.056 for Hall-A 2016, 0.978 for the neutron.  Those
are the *experiments'* normalisations, fitted with the Gaussian priors their
papers quote.  The model as it stands is the cross section; multiplying by them
in a generator would be wrong.

---

## 3. The interface

```python
import amplitudes as amp
s = amp.structure(p, ch, t, xB, Q2)     # t is the physical Mandelstam t, NEGATIVE
e = amp.epsilon(xB, Q2, Ebeam)
```

`ch` is `"pi0p"`, `"etap"` or `"pi0n"`.  Returns `None` below the meson
production threshold — handle that explicitly rather than letting it become a
silent zero weight.

`s` holds nine structure functions in nb/GeV^2.  The five for an unpolarised
target are `T`, `L`, `TT`, `LT`, `LTp`; the other four (`Tp`, `LTUL`, `TTUL`,
`LTpLL`) are for a polarised target, if that ever comes up.

What to sample from:

    d2sigma/dt dphi = (1/2pi) [ sigma_T + eps sigma_L
                              + eps cos(2phi) sigma_TT
                              + sqrt(2 eps (1+eps)) cos(phi) sigma_LT
                              + h sqrt(2 eps (1-eps)) sin(phi) sigma_LTp ]

`h` is the beam helicity; phi in Trento.  Same decomposition as CLAS6 eq. B4 and
COMPASS eq. 2.

---

## 4. Where it is allowed to be used

Fitted over Q^2 1.15 to 8.31, x_B 0.13 to 0.60, |t| up to about 1.8 GeV^2.
Decide deliberately what happens outside that box; do not let extrapolation
happen by accident, especially in Q^2 (see below).

---

## 5. Four weaknesses the generator will inherit

These are not caveats added for politeness.  Each is a measured failure.

**sigma_LT is the worst-described quantity in the model.**  The cos(phi)
modulation of every generated event comes from it.  Blind prediction gives 20.4
per point on the CLAS12 cross sections and 4.7 on Hall-A 2016.  The phi shape is
the least trustworthy thing the generator will produce.

**The Q^2 dependence is a single power Q^nQ over a range of 1.2 to 8.3, and it
does not fit.**  Hall-A 2016 and Hall-A 2021 both say so, and in opposite
directions at the two ends: at Q^2 = 1.5 the model undershoots sigma_T, at 8.3 it
overshoots sigma_U by 1.8.  With y21 in the fit the compromise costs the
asymmetries (eg1 1.36 -> 1.81 per point).

**Hall-A 2016 does not fit at all**: 4.5 per point with its normalisation pulled
to +2.7 sigma of the quoted 3.12 % and still not enough.  Seven points, the only
Rosenbluth separation on the proton.

**The neutron channel runs but rests on four points with 42 to 90 % errors.**
Dropping them moves n/p from 1.06 to 2.15 — nothing else in the database
constrains it.  `pi0n` output is indicative only.

**Positivity is worth checking before sampling.**  Verify the bracket in section
3 stays positive over the full phi range in the box you sample.  If the model can
drive it negative anywhere, sampling breaks; a guard is better than a surprise.

---

## 6. One thing to verify rather than assume

The fit uses CLAS6 with re-computed radiative corrections (`rc=True` in
`datasets.py` — the `s_u_rc`, `s_LT_rc`, `s_TT_rc` columns).  If the generator
applies RC through EXCLURAD with a *different* model inside it, the loop is not
closed and absolute rates are not self-consistent.  That closure is the whole
reason `rc_iter1` is called what it is; worth confirming it still holds.

---

## 7. What changed in the data on 2026-09-05

The fit above is not the August one refitted — the data underneath it changed.

* De Masi carried **no systematic at all** (the CLAS database export gives
  statistics only), so 703 phi points were the most heavily weighted set in the
  sample.  PRC 77 042201(R) quotes 0.016; it is in now.
* The Hall-A neutron systematics exist only as the bands of figure 5 of
  PRL 118 222002.  They were digitised, calibrated against the published error
  bars to 0.0 % in sigma_T.
* Hall-A 2016 and the neutron had been carrying **an invented 10 %** that was
  wrong in both directions at once: it hid a real discrepancy in y16 and
  inflated a false one in the neutron.
* Three overall normalisations (3.5 %, 3.12 %, 3.1 %) are now fitted as scales
  with Gaussian priors instead of being smeared over the points.
* Two of the three flavour phases had been sitting at 0.5 because slots 35 and
  36 kept the window `[0.5, 2.0]` of a magnitude those slots used to carry.
  Reopened to `[-pi, pi]`.

Full detail: `~/rc_iter1` commits `d2b8543`, `bc7eb13`, `8952a51`, and
`~/hepgen_fit/data/DATA_UPDATE_2026-09-05.md`.
