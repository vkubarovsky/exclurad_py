# mesonrc

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

This replaces three codes with one: EXCLURAD (pi0/eta, 5 SF), DIFFRAD
(vector, 2 SF only), and their generators. The Fortran is retained as a
frozen validation reference, never as a production path.

## Quick start

```bash
python3 -c "from mesonrc import models; print(models.list_models())"
python3 -m mesonrc.validation.selftest
```

```python
from mesonrc import models, rc_factor

sf  = models.get("eta.tableB1", t_nucl=-0.2)     # published eta production model
res = rc_factor(Ebeam=10.6, Q2=1.5, W2=5.0, t=-0.2, phi=1.1,
                sfmodel=sf, h=+1, mx2_cut=0.5)
print(res["eta"], res["delta_VR"], res["sigma_F"])
```

Unweighted event generation with LUND output:

```python
from mesonrc.generator import Generator, Box, write_lund
from mesonrc import stamp

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
| `mesonrc/core/` | **frozen** RC machinery. Knows nothing about channels or models. Change only for physics reasons; every change invalidates the validation suite. |
| `mesonrc/topology.py` | which hadron is detected, which is missing — plus the `sigma_LT` sign and `t_N` conventions that follow |
| `mesonrc/models/` | the versioned axis: `pi0.vpk2014`, `eta.tableB1`, `jpsi.diffrad`, … |
| `mesonrc/generator/` | unweighted generation, photon sampling, lab frames, LUND |
| `mesonrc/validation/` | internal consistency suite + legacy-code comparison |

## The three axes

Everything that varies across use cases is one of three things, and the
package is organized around exactly those:

1. **Channel** — rho, phi, J/psi, pi0, eta. Enters only as `(m_V, m_U)`.
2. **Model vintage** — `pi0.vpk2014` / `pi0.vpk2021`, `eta.tableB1` /
   `eta.refit2013` / `eta.vpk2021`, three J/psi models. This is where the
   physics content lives and where new work lands.
3. **Topology** — which hadron is detected. Easy to overlook, and wrong
   silently: EXCLURAD detects the **proton** and misses the meson, so
   `mV = m_p`, `mU = m_meson`, the SFs are functions of the nucleon
   transfer `t_N`, and `sigma_LT`, `sigma_LT'` flip sign.

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
from mesonrc.models import ProtonDetectedModel, register
from mesonrc.topology import baryon_detected

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
python3 -m mesonrc.validation.selftest      # 8 internal checks
```

See `REPORT.md` for the current status, the numbers, and what is not yet
validated.
