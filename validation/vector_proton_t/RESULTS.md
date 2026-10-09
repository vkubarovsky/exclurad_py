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

## Why the meson bin has fewer hard photons — WITHDRAWN, rerunning

**The 40 000-event comparison above was run without the production
environment and its photon result must not be used.**  `EXACT_ACCEPT`,
`EXPO_TAIL`, `MAJ_SHAPE` and `MAJ_SAFETY` were all unset, so
`exact_per_event` stayed False and BOTH samples took the cell-based
`TailSampler` — the path `events.py:1339` records as biasing
`frac(v > 0.3)` by about 2x — with the cells sitting in different
coordinates in the two bins.  The log read `health: clean | stage-2 b_viol
0/40000`, which looks like production and is not: with `EXACT_ACCEPT` unset
stage 2 never runs and the count is 0 by construction.  That line is fixed
in fb2012d.

The hadronic result (`t_N` and `t_meson` means agreeing to under 1.5 sigma)
is unaffected in substance but is being regenerated too.

**And the argument that would have stopped it earlier**, from the J/psi
electroproduction session: each bin's joint density of (observed point,
photon) is the same `|M|^2` in different coordinates, so a different v
CEILING at a fixed observed point is a different SLICE, not missing phase
space.  Two exact samplers must give the same photon spectrum however
different the ceilings are.  A 26 % difference therefore accuses a sampler,
not the physics.  The ceiling measurement below is real and still worth
having — it is just not an explanation of a photon deficit.

### The ceiling, measured (stands as a fact about the coordinates)

3021 physical points over the J/psi box, `v_max_kin` at the SAME observed
point:

| | median | mean | max |
|---|---:|---:|---:|
| meson bin, `MX^2(e'V) - M_p^2` | 0.4473 | 0.4824 | 1.186 |
| proton bin, `MX^2(e'p) - m_psi^2` | 1.3003 | 1.3866 | 3.337 |

Higher in the proton bin at 100 % of points, median ratio 2.76, minimum
1.94; by t: 2.22, 2.92 and 4.33 in -7.0..-4.0, -4.0..-2.0 and -2.0..-0.45.

### The veto hypothesis, rejected (still rejected)

Events whose observed `t_meson` falls outside the Born-physical range at
their own observed (W2, Q2): 0 of 39999 in each sample.  The same pass
confirms the observed variables reconstruct from the LUND with the nominal
beam — Q2 0.0010-0.2456 against the card's 0.001-0.25, W2 16.57-20.57
against 16.4-20.8.  Not yet counted: events with `t_N` inside the card's
`[-7, -0.45]` whose `t_meson` is outside it, which is the one route by
which the BOX, rather than physics, could differ between the bins.

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
