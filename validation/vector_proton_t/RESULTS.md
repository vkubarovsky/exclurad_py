# Vector mesons in the proton-detected bin — what was measured

2026-10-08/09.  VPK: the same t definition in the vector channels as in pi0/eta.

## The bin

| | meson bin (was) | proton bin (now) |
|---|---|---|
| detected | the meson | the recoil proton |
| t held fixed | `(q - p_V)^2` | `(p - p')^2 = -2 M T_p` |
| missing mass | `MX^2(e'V)` | `MX^2(e'p)` |
| topology | `meson_detected`, `t_kind=photon` | `baryon_detected`, `t_kind=target` |

At Born the two are the SAME invariant — for 2 -> 2, `q - p_V = p' - p` — so a
card's t box carries over unchanged.  Verified to machine precision at nine
azimuths for all three channels (`born_identity.py`); rho is the one that
tests anything, because `SLT_OVER_SU = -0.30` makes sigma_LT non-zero and the
`lt_sign` flip and `phi_offset = pi` have to cancel exactly.  Its own phi
profile varies by min/max = 0.315, so the test is not vacuous.

## Are the SAMPLES the same?

40 000 radiative J/psi events per topology, same card and seed, `iacc none`,
`vcut 0`, both histogrammed in the same variables.

| variable | mean, meson bin | mean, proton bin | difference |
|---|---:|---:|---:|
| `t_N` | -1.84548 +- 0.00597 | -1.85173 +- 0.00593 | **-0.7 sigma** |
| `t_meson` | -1.87428 +- 0.00602 | -1.88726 +- 0.00598 | -1.5 sigma |
| `E_gamma` (lab) | +0.02101 +- 0.00046 | +0.02645 +- 0.00054 | **+7.7 sigma** |

Binned, the hadronic distributions agree (worst pull 2.2 sigma in `t_N`, 2.8
in `t_meson`, no trend in the means) and the photon spectrum does not:

| `E_gamma` | ratio proton/meson | pull |
|---|---:|---:|
| 0.25 - 0.60 GeV | 1.240 | 3.7 |
| 0.60 - 1.50 GeV | 1.466 | 4.1 |

**So the two bins give the same hadronic sample and a different radiative
tail.**  At `vcut = 0` the ceiling is the kinematic limit of `MX^2(e'p)` in one
and of `MX^2(e'V)` in the other -- they differ by factors of 4-5 -- and the
proton bin reaches harder photons.  Mean photon energy is 26 % higher.

That is the reason the switch is needed for the SAMPLES and not only for the
tables: the generated tail has to populate the missing mass the analysis cuts
on.

## RC in the new bin

`independent_tail.py` gained a `target` mode: the detected proton's direction
follows v at fixed `t_N`, where `cm_angle` freezes it.  Phase-space factor 1,
derived in the code -- at fixed E', `dt_N = -2|q| p' dcos`, so
`d3p'/(2E') = dE' dt_N dphi/(4|q|)` and p' cancels; `cm_angle` freezes `dcos`
instead and keeps `p'(v)/p'(0)`.

`check_target_tail.py`, exclurad_py non-exponentiated against the independent
first-order tail, three points:

| v interval | worst difference |
|---|---:|
| 0.03 - 0.10 | -0.53 % |
| 0.10 - 0.20 | -1.30 % |
| 0.20 - 0.50 | -2.87 % |

Soft limit `|M_R|^2/|M_B|^2` over the eikonal: 0.99995.  Accumulated on eta
this is about 0.2 %.

## A correction worth keeping

The first version of `compare_topologies.py` read the LUND columns one place
off -- `pz` where `E` belongs -- and produced a `t_N` table that looked
plausible (worst pull 1.0 sigma) and was not `t_N` at all.  The script now
asserts `E^2 - p^2 = m^2` on every particle line it reads.  The corrected
numbers are the ones above.
