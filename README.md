# exclurad_py

QED radiative corrections and unweighted event generation for exclusive
meson electroproduction at CLAS12:

```
e(k1) + p(p) -> e'(k2) + h(pv) + B(pu) [+ gamma(k)]        h = rho0, phi, J/psi, pi0, eta
```

One RC engine — exact covariant Bardin–Shumeiko, the **full five structure
functions** `sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT'`, beam
helicity — serving every channel, with the cross-section model, the meson,
and the detection topology as **plug-in axes rather than compile-time
choices**.

## What the name means, and what it does not

**Same formalism as EXCLURAD, independent implementation.** This is
EXCLURAD's physics — the exact covariant Bardin–Shumeiko master formula,
the same `delta_VR`, `delta_inf`, `delta_vac`, the same five structure
functions — and it reproduces EXCLURAD to **0.15–0.20 %** over ~5000 bins
in both pi0 and eta (see `REPORT.md` §3.2).

It is **not** a line-by-line translation of the Fortran. EXCLURAD evaluates
the radiative leptonic tensor through long analytic contractions;
`exclurad_py` builds it numerically from helicity spinors and contracts it
with the hadronic tensor. Different method, same physics. That matters: a
transliteration agreeing with its parent proves nothing, whereas two
independent implementations agreeing to 0.2 % is evidence both are right.
It is also why the residual is 0.2 % rather than zero, and why this code is
smooth through phi = 0/180 where `exclurad.F` returns NaN.

**It also covers vector mesons, which EXCLURAD never did.** rho, phi and
J/psi run on the models of `~/MC_vector_mesons/diffrad_vm.f90` (ported and
verified to 8–9 digits) in two-structure-function mode. So the name
understates the scope: this replaces three codes with one — EXCLURAD
(pi0/eta), DIFFRAD (vector), and their generators. The Fortran is retained
as a frozen validation reference, never as a production path.

## Quick start

To produce a CLAS12 sample (card -> LUND -> GEMC -> checks), read `docs/Story_instruction.md`;
the production cards are `cards/{rad,born}_pi0_{rga,rgk6535,rgk7546}.txt`.

```bash
python3 -c "from exclurad_py import models; print(models.list_models())"
python3 -m exclurad_py.validation.selftest
```

```python
from exclurad_py import models, rc_factor

sf  = models.get("eta.tableB1", t_nucl=-0.2)     # published eta production model
res = rc_factor(Ebeam=10.6, Q2=1.5, W2=5.0, t=-0.2, phi=1.1,
                sfmodel=sf, h=+1, mx2_cut=0.5)
print(res["eta"], res["delta_VR"], res["sigma_F"])
```

Unweighted event generation with LUND output:

```python
from exclurad_py.generator import Generator, Box, write_lund
from exclurad_py import stamp

gen = Generator(sf, Ebeam=10.6,
                box=Box(Q2=(1.0, 2.5), W2=(4.5, 7.0), t=(-0.5, -0.15)),
                mx2_cut=0.5, cache="eta_table.npz")
gen.setup(verbose=True)
events = gen.generate(100_000)
write_lund("eta.lund", events, channel="eta", stamp=stamp(sf))
print(gen.report(), gen.health())
```

## Layout

| directory | contents |
|---|---|
| `exclurad_py/core/` | **frozen** RC machinery. Knows nothing about channels or models. Change only for physics reasons; every change invalidates the validation suite. |
| `exclurad_py/topology.py` | which hadron is detected, which is missing — plus the `sigma_LT` sign and `t_N` conventions that follow |
| `exclurad_py/models/` | the versioned axis: `pi0.vpk2013`, `eta.tableB1`, `jpsi.diffrad_vm`, … |
| `exclurad_py/generator/` | unweighted generation, photon sampling, lab frames, LUND |
| `exclurad_py/validation/` | internal consistency suite + legacy-code comparison |

## The three axes

Everything that varies across use cases is one of three things, and the
package is organized around exactly those:

1. **Channel** — rho, phi, J/psi, pi0, eta. Enters only as `(m_V, m_U)`.
2. **Model vintage** — `pi0.vpk2013` / `pi0.vpk2021`, `eta.tableB1` /
   `eta.refit2013` / `eta.vpk2021`, three J/psi models. This is where the
   physics content lives and where new work lands.
3. **Topology** — which hadron is detected. Easy to overlook, and wrong
   silently: EXCLURAD detects the **proton** and misses the meson, so
   `mV = m_p`, `mU = m_meson`, the SFs are functions of the nucleon
   transfer `t_N`, and `sigma_LT`, `sigma_LT'` flip sign.

## Spins: what the card does, and what lands in the file

`ihel` (card field 10) is the beam helicity, `ptarg` (field 20) the longitudinal
target polarisation. Both reach the physics: the Born cross section and the
radiative correction are evaluated at the event's own spins
(`events.py: _born(..., h)`, `rc_factor(..., h=h, lam=ptarg)`).

**`ihel = 0` does not average the double-spin part away.** The helicity is drawn
per event, 50/50, and the event is then accepted against `sigma(h)` — so the two
helicity subsets genuinely differ. Since 2026-09-23 the drawn value is written to
**LUND event-header column 5**, so one sample carries both spin states:

| column | content | card field |
|---:|---|---|
| 4 | target polarisation | `ptarg` |
| 5 | **beam helicity of that event** | `ihel`, or the per-event draw if `ihel = 0` |
| 7 | beam energy | `ebeam` |

Measured on a 400-event test (`pi0.amp2609`, Q2 1-6, W2 3.24-16, t -2..-0.1):

| card | A = (N+ - N-)/N |
|---|---|
| `ihel = 0`, `ptarg = +1` | **+0.210 +- 0.050** — this is A_LL const = sqrt(1-eps^2) sigma_T' / sigma_0 |
| `ihel = 0`, `ptarg = 0` | -0.035 +- 0.050, i.e. zero — A_LU goes as sin(phi) and integrates away |

So all four asymmetries of the polarised decomposition come out of **two** samples
(`ihel = 0` with `ptarg = +1` and `-1`) rather than four, by splitting on column 5.
Both helicities then come from identical running conditions, which cancels part of
the systematics.

**One condition, and it is not optional: `EXACT_ACCEPT=1`.** The eta table is built
at a single helicity (`events.py:347`), so without exact per-event re-evaluation the
Born stays right while the RC does not, and the helicity dependence of eta is biased.
The generator prints a warning when it detects this.

### Cross section and normalisation

Events are **unweighted** — LUND column 10 is always 1.0, because GEMC does not
propagate weights. The cross section lives beside the file, in
`inputs/config.json`:

```
sigma_nb   0.3203551992991448     nb, integrated over the card's box and cuts
trials     2608
accepted   400
```

`sigma_nb` = (accepted / trials) x majorant integral, i.e. sigma over the whole box,
not per event. Hence `sigma per event = sigma_nb / n_events` and the sample's
luminosity is `n_events / sigma_nb` [nb^-1]. With `ihel = 0` it is already the
helicity-averaged sigma, since h is drawn inside.

### Radiative correction with `ihel = 0`

eta = sigma_obs / sigma_Born, and both depend on the spins. Run born and rad from the
**same card** (`ihel = 0` in both), then:

- **unpolarised RC** — sum over both helicities, bin by bin:
  `eta(bin) = [N_rad(bin)/N_rad_tot x sigma_nb_rad] / [N_born(bin)/N_born_tot x sigma_nb_born]`
- **helicity-dependent RC** — apply the same formula separately to the column-5 = +1
  and column-5 = -1 subsets of **both** samples.

The second is only meaningful because the generator evaluated the RC at each event's
own helicity; the label is not attached afterwards.

## Models and provenance

```
>>> print(models.list_models("eta"))
eta.refit2013  2013refit  baryon_detected  2013 form, repo-default eta column (later refit; NOT published)
eta.tableB1    2016note   baryon_detected  CLAS-2016-101 App. B Table B.1 — the PUBLISHED production model
eta.vpk2021    2021       baryon_detected  March-2021 GT fit (dvmpx.F XSINIT, k=2)
```

The registry **refuses a model with no provenance**, and every output —
RC table, LUND file, plot — is stamped with the model name, version, and
git commit. Eight months from now you can tell which cross section made a
number. That is the whole point of the naming discipline.

Each model also declares a `validity` range. This is not bureaucracy: the
radiative tail evaluates the SFs at shifted kinematics, `W2~` down to
threshold and `Q2~` on both sides of the observed value, so a fit made on a
limited grid *will* be called outside it. Out-of-range calls are clamped by
default and **counted** — `sf.usage()` reports the fraction.

## Adding a model

```python
from exclurad_py.models import ProtonDetectedModel, register
from exclurad_py.topology import baryon_detected

@register
class Eta_MyFit2026(ProtonDetectedModel):
    name       = "eta.myfit2026"
    channel, version = "eta", "2026"
    topology   = baryon_detected(0.547862, meson="eta")
    validity   = dict(W2=(1.2, 20.0), Q2=(0.5, 12.0), mt=(0.0, 5.0))
    provenance = "fit of ... ; port verified against ... to N digits"
    reference  = "path or DOI"

    def _sfs_tn(self, t_nucl, xB, Q2):
        return sT, sL, sTT, sLT      # nb/GeV^2; do NOT apply the LT sign flip
```

## Validation

```bash
python3 -m exclurad_py.validation.selftest      # 8 internal checks
```

See `REPORT.md` for the current status, the numbers, and what is not yet
validated.
