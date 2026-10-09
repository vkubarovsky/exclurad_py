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

## Are the SAMPLES the same?  Rerun on the exact path — YES

The first answer to this was wrong and is withdrawn.  Those 40 000-event
samples ran with `EXACT_ACCEPT`, `EXPO_TAIL`, `MAJ_SHAPE` and `MAJ_SAFETY`
all unset, so both took the cell-based `TailSampler` — the path
`events.py:1339` records as biasing `frac(v > 0.3)` by about 2x — with the
cells in different coordinates in the two bins.  They gave a 26 % difference
in the mean photon energy and it was the sampler.

**Rerun with production's environment** — `MAJ_SHAPE=16,8,16,12
MAJ_SAFETY=2.5 EXACT_ACCEPT=1 EXPO_TAIL=1` — 40 000 radiative J/psi events
per bin, one card, `iacc none`, `vcut 0`, both histogrammed in the same
variables, production photons only (`parent == 0`):

| | meson bin | proton bin | difference |
|---|---:|---:|---:|
| mean `t_N` | -1.86461 | -1.85725 | +0.9 sigma |
| mean `t_meson` | -1.89522 | -1.88733 | +0.9 sigma |
| mean `E_gamma` | +0.02157 | +0.02106 | **-0.7 sigma** |

| `E_gamma` bin | meson | proton | ratio | pull |
|---|---:|---:|---:|---:|
| 0.00-0.02 | 36044 | 36172 | 1.0036 | 0.5 |
| 0.02-0.05 | 1159 | 1120 | 0.9664 | -0.8 |
| 0.05-0.10 | 855 | 800 | 0.9357 | -1.4 |
| 0.10-0.25 | 926 | 925 | 0.9989 | -0.0 |
| 0.25-0.60 | 638 | 631 | 0.9890 | -0.2 |
| 0.60-1.50 | 353 | 333 | 0.9433 | -0.8 |
| 1.50-12.0 | 25 | 19 | 0.76 | -1.0 |

**With both samplers exact the two bins give the same sample** — hadronic
and radiative, every pull under 1.5 sigma.  Which is what the J/psi
electroproduction session argued from the start and what I twice measured
my way out of: each bin's joint density of (observed point, photon) is the
same `|M|^2` in different coordinates, so a different v CEILING at a fixed
observed point is a different SLICE, not missing phase space.  Two exact
samplers must agree, and they do.

The box, their last loose end — events with `t_N` inside the card's
`[-7, -0.45]` whose `t_meson` is outside it: 2 / 40000 and 6 / 40000, i.e.
0.005 % and 0.015 %.

### So why switch at all

Not for the samples.  For the two things the bin IS:

1. **eta tables** computed from `rc_factor` are binned, and the difference
   between the two bins at a forced-equal numeric v window is 2.7-4.8 %
   (above).  That cannot be matched away, because `v` is a different
   physical quantity in the two.
2. **the exclusivity cut**.  `v = MX^2(e'p) - m_V^2` is what the analysis
   cuts on, and with `vcut = 0` and the cut applied offline the generator
   must be able to express it.

### The ceiling and the veto, both still facts about the coordinates

`v_max` is higher in the proton bin at 100 % of 3021 physical points,
median ratio 2.76 — and that is now known NOT to produce a different photon
spectrum, which is the useful part.  Events whose observed `t_meson` falls
outside the Born-physical range: 0 of 39999 in each sample.

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
