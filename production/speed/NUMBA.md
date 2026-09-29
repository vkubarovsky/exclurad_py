# numba: 2.494x more, on the production stack, with the physics intact

Branch `speed-numba`.  Opt-in with `RC_NUMBA=1`; without it the numpy path
runs and is byte-identical to production.

## Result

| | |
|---|---|
| **on farm COMPUTE NODES** | **2.494x**, median of 40 paired slurm tasks (10-90 %: 2.247-2.871) |
| on top of the merged `speed` | 1.657x -> **4.13x total** |
| campaign, end to end | **2.60 -> 0.63 s/event** |
| core-hours per 100 M | 72 000 -> **~17 400** |
| **meets the brief's <1 s/event target?** | **yes** |
| production stack change | **none** -- python 3.9.25 and numpy 1.23.5 unchanged |
| added | numba 0.60.0 + llvmlite 0.43.0 |

## What it is

`docs/cost_model.md` says the generator's cost is (integrand points) x
(cost per point), and that the integrand is one quantity:

    LRH = Re( L_R^{mu nu} H_{mu nu} )

evaluated ~232 000 times per written event.  `core/_kernel_numba.py`
computes it row by row with nothing larger than a 4x4 on the stack -- the
(N,4,4) complex temporaries never exist.  That is the thing numpy cannot do
however well it is vectorised, and it is why this is worth a dependency
when the pure-numpy work was not.

Both callers go through `core/integrand.py:lrh_batch`, so `sigma_F` and the
tail sampler -- the two halves of the point count -- are both covered.

The structure-function model stays outside the kernel: it is a Python
object, it cannot be jitted, and it is already evaluated once per block.

## Correctness: not byte-identical, and much closer than that sounds

The bar drops from byte-identity, so it was replaced with three stronger-
than-usual checks rather than a hand-wave.

**1. Same events, not merely the same distributions.**  4000 events, both
backends, same seed:

    26 291 lines, identical structure, same particles in the same order
    2 948 lines differ, all in trailing digits
    largest relative difference in ANY number in the file:  6.3e-08

No accept/reject decision flipped in 4000 events.  A KS test returns p = 1
because the samples are not merely compatible -- they are the same sample.

**2. On the production architecture it is far tighter than on the mac.**

| `sigma_F` over the production box | numba vs numpy |
|---|---:|
| mini (ARM) | 1.1e-9 median, 1.7e-7 worst |
| **ifarm x86 (where production runs)** | **1.9e-13 worst** |

1.9e-13 is essentially the brief's 1e-12 bar.  On the farm this is tight
enough that **39 of 40 compute-node LUND files came out BYTE-identical**,
and the 40th differed by one ulp in one number of 162 lines
(`4.3306413135e-04` against `...134e-04`, a radiated photon's p_y).

**3. Where the disagreement comes from, and why it is not the port.**
`rad_L` takes kappa1 = 2 k.k1 from `dot4`, which cancels catastrophically
when the photon is collinear with the lepton.  The error tracks
collinearity exactly:

| kappa1 / E1*om | median rel err |
|---|---:|
| < 1e-5 | 2.2e-10 |
| 1e-5 - 1e-4 | 1.5e-12 |
| > 1e-2 | 6.7e-16 |

and in that same collinear region numpy's own `kappa1` already differs from
the STABLE `2*om*kap_own` -- which `_channel_nodes` computes and `rad_L`
ignores -- by up to 2.5e-8.  Both backends are limited by the same
conditioning; neither is the more accurate one.  **Passing the stable kappa
into `rad_L` is an accuracy improvement available independently of numba,
and worth doing on its own.**

## Deployment: no stack change, which was the surprise

The expected blocker was "numba means a new python or a new numpy".  It
does not:

    ifarm, python 3.9.25 + numpy 1.23.5 (EXACTLY production) + numba 0.60.0 -> works

numba 0.60 declares `numpy<2.1,>=1.22`, so 1.23.5 is supported; pip merely
defaults to the newest allowed and has to be pinned.  What is added is
numba 0.60.0 and llvmlite 0.43.0 (which vendors LLVM).  Whether that is
acceptable inside the OSG container and `clas12-mcgen` is a packaging
decision, not a technical obstacle.

Install used (own venv, never a shared one):

    /usr/bin/python3 -m venv /work/clas12/vpk/venv-speed-nb39
    .../bin/python -m pip install --no-cache-dir numba "numpy==1.23.5" "scipy==1.9.3"

Two install notes that cost time: pip on lustre left a half-written
llvmlite that imported with `AttributeError: get_host_cpu_name` and needed
`--force-reinstall`; and numba pulls numpy 2.0.2 unless numpy is pinned.

## A control worth keeping

The numpy backend appeared to differ from the production baseline by 3.1e-8
on ifarm.  Running **unmodified** `production` code in the same venv gave
the identical 158 points and the identical 3.103e-8 -- so it is the numpy
*build* (pip wheel against the system RPM of the same 1.23.5), not this
branch.  Same class as the 2.3e-7 between numpy versions.  Byte-identity is
a within-build statement, not just a within-version one.
