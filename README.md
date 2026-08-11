# vmrad — radiative corrections for exclusive meson electroproduction (vector and pseudoscalar) with the full set of structure functions

QED radiative corrections (RC) for

```
e(k1) + p(p) -> e'(k2) + p'(p') + V(pv),      V = rho0, phi, J/psi
```

generalizing the DIFFRAD treatment (which uses only sigma_T and sigma_L) to
the **full set of five structure functions**

```
sigma_T, sigma_L, sigma_TT, sigma_LT, sigma_LT'
```

including beam polarization (helicity h), in the **exact covariant
Bardin–Shumeiko approach** (no soft-photon cutoff parameter), following the
EXCLURAD formalism [Afanasev, Akushevich, Burkert, Joo, PRD 66 (2002) 074004]
adapted to vector-meson production.

Radiation from the lepton line, vacuum polarization (lepton loops exactly,
hadronic from the standard fit) and the lepton vertex correction are
included; hadron-line radiation and two-photon exchange are neglected
(as in DIFFRAD/EXCLURAD).

## Master formula

```
sigma_obs = sigma_Born * exp(delta_inf) * (1 + delta_VR + delta_vac) + sigma_F
```

evaluated at fixed *observed* (Q2, W2, t, phi); `sigma_F` is the IR-finite
hard-photon integral in which the structure functions are evaluated at the
*shifted* (true) kinematics (W2~, Q2~, t~) of the hadronic vertex. See
`rc_formulas.pdf` for all formulas and conventions.

Instead of transcribing the (very long) analytic contractions of EXCLURAD,
the radiative leptonic tensor is computed **exactly** from helicity spinors
numerically and contracted with the hadronic tensor built from your five
structure functions. All conventions are locked by machine-precision
closure tests (see below).

## Install / run

Requires python3 + numpy + scipy (matplotlib for the example plot).

```bash
python3 tests/run_checks.py     # validation suite
python3 example.py              # rho0 demo: RC vs phi, A_LU before/after RC
```

## Usage

```python
from vmrad import rc_factor, demo_rho, TwoSF

sf = demo_rho()                                # demo model - replace with yours!
res = rc_factor(Ebeam=10.6, Q2=2.5, W2=12.0, t=-0.4, phi=1.1,
                sfmodel=sf, h=+1,              # beam helicity
                mx2_cut=(0.938 + 0.3)**2)      # missing-mass^2 cut (GeV^2)
print(res['eta'])          # RC factor sigma_obs/sigma_Born
print(res['delta_VR'], res['delta_inf'], res['delta_vac'], res['sigma_F'])
```

### Plugging in your own structure functions

Subclass `StructureFunctions`; return d sigma_i/dt in **nb/GeV^2**
(Hand convention, see the PDF) for possibly array-valued (W2, Q2, t) —
these get evaluated at shifted kinematics inside the photon integral,
so the model must be defined for W2 down to threshold, Q2 shifted both up
and down from the observed value, and t shifted accordingly:

```python
from vmrad import StructureFunctions

class MySF(StructureFunctions):
    mV = 0.77526
    def __call__(self, W2, Q2, t):
        ...
        return {'sT': sT, 'sL': sL, 'sTT': sTT, 'sLT': sLT, 'sLTp': sLTp}
```

`TwoSF(model)` zeroes sigma_TT/LT/LT' — the DIFFRAD-like mode, useful to
quantify how much the extra structure functions change the RC.

### Conventions (important when comparing to your fits!)

Born cross section (phi in the Trento-like frame defined in
`kinematics.py`, h = beam helicity):

```
dsigma/(dQ2 dW2 dt dphi) = Gamma_v/(2pi) * [ sigma_T + eps sigma_L
    + eps cos(2phi) sigma_TT + sqrt(2 eps (1+eps)) cos(phi) sigma_LT
    + h sqrt(2 eps (1-eps)) sin(phi) sigma_LT' ]
```

Some analyses define sigma_LT with sqrt(eps(1+eps)) (no factor 2) and/or
opposite signs of sigma_TT, sigma_LT — rescale accordingly.

## Validation (tests/run_checks.py)

1. **Born closure**: exact tensor contraction reproduces the canonical
   angular decomposition; all five coefficients = 1 to ~3e-8 (lepton-mass
   effects), for both helicities.
2. **Current conservation** of the radiative leptonic tensor (~1e-9) and
   **soft factorization** LR -> 4 F_IR L0 (verified at omega = 1e-4).
3. **IR consistency**: angular integral of the subtraction term matches the
   analytic (alpha/pi)(l_m - 1) d ln(v^2) slope of delta_inf to ~2e-6 —
   a non-circular check of the Bardin–Shumeiko bookkeeping.
4. **TwoSF reduction** and grid convergence (~1e-4 relative on sigma_F).

## Files

| file | content |
|---|---|
| `vmrad/kinematics.py` | invariants, CM four-vectors, v_max |
| `vmrad/spinors.py`    | exact Born + radiative leptonic tensors (helicity spinors) |
| `vmrad/hadronic.py`   | hadronic tensor from the 5 SFs; SF plug-in interface |
| `vmrad/models.py`     | *demo* rho/phi/J/psi parameterizations (replace!) |
| `vmrad/born.py`       | Born cross section |
| `vmrad/virtual.py`    | delta_VR, delta_inf, delta_vac |
| `vmrad/radiative.py`  | IR-subtracted hard-photon integral sigma_F |
| `vmrad/rc.py`         | driver: `rc_factor(...)` |
| `rc_formulas.pdf`     | full write-up of formulas and conventions |

## Pseudoscalar mesons (pi0, pi+, eta)

The five-SF decomposition is identical for pseudoscalars (it is EXCLURAD's
original case), so the same code applies to `ep -> e' p pi0`, `e' n pi+`,
`e' p eta`. Set two masses on your SF model:

* `mV` — the detected meson mass;
* `mU` — the **undetected recoil baryon** (proton by default; the *neutron*
  for pi+ n). It enters v = MX^2 - mU^2, the v limits and the Li2 term of
  delta_VR.

Demo models: `demo_pi0()`, `demo_pip()`, `demo_eta()`; example:
`python3 example_pseudoscalar.py` (RC vs phi for pi0 p and pi+ n).
Hadron-line radiation (including from the charged pi+) is neglected, as in
EXCLURAD.
