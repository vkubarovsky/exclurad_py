# Fermi motion for J/psi photoproduction off the deuteron — what was done and what it shows

Branch `jpsi-fermi`, worktree `~/exclurad_py_jpsi_fermi`, branched from `fermi` on 2026-09-26
(`fermi` was not yet merged into `production`). Written by the session that did the work;
the brief it followed is `JPSI_FERMI_BRIEF.md`. Numbers in section 5 come from
`validation/jpsi_fermi_study.py` on the samples listed there.

## 1. What was reused and what was added

The deuteron machinery of `fermi` (`generator/fermi.py`, the three hooks in `events.py`) has
no channel dependence and was used unchanged. Added:

| where | what |
|---|---|
| `topology.meson_detected` | gains `m_target` / `target`, defaulting to the recoil nucleon, exactly as `baryon_detected` does. Without it every vector-meson topology had a proton initial state whatever the recoil, and `Generator` reads `topology.target` to decide which nucleon of the deuteron is struck. |
| `models/vector.py` | `jpsin.roberts2026` = `Jpsi_Roberts2026` relabelled: same sigma(W), same threshold constant, neutron topology. Registered as the isoscalar baseline sigma_n = sigma_p (section 3). |
| `generator/lund.py`, `generator/decay.py` | channel aliases `jpsin` in `CHANNEL_PID`, `DECAY` and `DECAY_PID`. The last one is the trap the brief warned about in a new form: `run.py` decayed the J/psi, but the LUND writer consults its own copy of the decay table, and `channel in DECAY_PID` was silently False, so the neutron sample came out with an undecayed J/psi and no e+e- pair. Found by counting positron lines; the free-proton twin decayed correctly. |
| `runcard.py`, `run.py` | card field 22, `fermi_mstar_min` (float, default 0.30 = the pi0/eta value), passed to `Generator(fermi_m_star_min=...)`. Cards without the line are unchanged. |
| `validation/fermi_jpsi_threshold.py` | acceptance criterion 4 (section 4). |
| `validation/fermi_conservation.py` | learns a J/psi box when the model name contains `jpsi`. |
| `cards/born_jpsi_photo_d2.txt`, `cards/born_jpsi_photo_p.txt` | the quasi-real cards of section 2. |

Nothing in the forbidden list (`_propose`, `calibrate`, `_b_full`, `_b_cell`, `_cell_cdf`,
`sigma_total`) was touched. `generator/fermi.py` and the AV18 table were not copied.

## 2. Which photoproduction, and which flux

CLAS12 "photoproduction" with the electron in the Forward Tagger is quasi-real
electroproduction at Q2 of 0.01 to a few tenths. That is what the cards generate: Q2 box
0.01-0.10 GeV2, W2 16.29-30 GeV2 (from the free-nucleon threshold (M_N + M_J/psi)^2 up to
what Fermi motion allows at 10.6 GeV), t to -6 GeV2, Born, no radiator, no y cut. The flux
convention is the code's own and was not re-derived: |q| flux at the core (`core/hadronic.py`),
which is what the vector-meson channels were validated with, and which FERMI_NOTE.md section 5
shows is unaffected by the boost to the struck-nucleon frame. A real bremsstrahlung beam
needs a different normalisation and is not what these samples describe. Every comparison in
section 5 is between two samples generated in the same convention, so the convention drops
out of the ratios.

**Why no y cut.** With Fermi motion the generator's y and x_B cuts are evaluated in the
struck-nucleon frame (`_pass_yxb` uses `self._E`, which `_draw_fermi` has set to the beam
energy in that frame). A y window therefore does not select a laboratory photon energy on a
deuteron; the first version of the threshold test tried and the majorant came out empty. The
laboratory E_gamma = E - E' is read off the written electron afterwards, in the test and in
the study. This is a property of the `fermi` branch worth knowing for any deuteron card that
uses y or x_B cuts, not only for J/psi.

## 2b. The untagged mode: bremsstrahlung in the target (VPK, 2026-09-26)

VPK's instruction on the day: no scattered electron; the beam radiates in the 5 cm cell, the
photon energy comes from that process, and what is wanted is the total yield versus E_gamma
and the t distribution.  That is a different generator path from electroproduction and was
written as `generator/photo.py`:

- photon spectrum: thin-target bremsstrahlung,
  n_gamma(k) dk = (X/X0) (1/k) [4/3 (1 - y) + y^2] S(k) dk with y = k/E0, where S(k) is the Schiff
  intermediate-screening factor (screening radius = Bohr radius for H and D): 1 away from the
  endpoint, 0.91 at 100 MeV below it, 0.77 at 30 MeV, zero 0.4 MeV below E0 (`docs/jpsi_fermi/
  jpsi_photo_flux_endpoint.png`); it lowers the photon count in 6.5-10.6 GeV by 1.2 %; the vertex is drawn along the
  cell with density proportional to the upstream thickness and X = L/2 is the cell average
  (5 cm LD2 = 0.0065 X0, LH2 0.0056 X0: thin-target is exact at that level);
- nucleon: at rest, or the AV18 struck nucleon of `generator/fermi.py` with the same spectator
  prescription and the same m* floor;
- hadron side: the electroproduction model at Q2 = 0.  The Hand-convention sigma_T IS the
  real-photon dsigma/dt (the Hand flux is the real-photon energy), so nothing new is assumed:
  for `jpsi.roberts2026` this is sigma_tot(W) b exp(b t');
- sampling: k from the analytic bremsstrahlung cumulative; accept with sigma_tot(W)/sigma_max,
  sigma_tot tabulated once on a W2 grid; t from the exact inverse cumulative of dsigma/dt on a
  401-point grid; the meson and recoil built in the gamma N centre of mass and boosted with the
  actual photon and nucleon four-vectors;
- output: LUND with beam pid 22 and beam energy = k in the event header, target 2212 / 2112 /
  1000010020, J/psi as a status-0 parent with its e+e- daughters, recoil, spectator; the
  event vertex z in the LUND vertex column; `config.json` with photons per electron in the k
  range, the flux-weighted <sigma>, and the yield per incident electron per target nucleon.

Card: key = value lines (`cards/photo_jpsi_d2.txt`, `cards/photo_jpsi_p.txt`);
`python3 -m exclurad_py.generator.photo card.txt [--jobs N]`.

Checks (`validation/jpsi_photo_study.py`, 200 000 events per sample): four-momentum
photon + target = V + N' (+ N_s) to 5.5e-9 GeV, the LUND's ten significant digits; the
free-proton E_gamma spectrum against n_gamma(k) sigma_tot(W(k)): chi2/ndf = 22.1/23; the t'
slope at E_gamma 9.4-9.6 GeV: 1.64 against the model's 1.67 GeV^-2.  Speed: 0.08 ms per event
on a free nucleon, 0.21 ms on the deuteron, single core; no majorant violation.

## 3. The neutron model

All three J/psi models had `topology.target = p`. `gamma n -> J/psi n` needs a neutron
registration, and the choice made is the one the brief asked for: `jpsin.roberts2026` is the
proton model with the topology changed and nothing else. The exchange is gluonic and
isoscalar, so sigma_n = sigma_p at leading order; that equality is also the null hypothesis a
neutron measurement tests, so inventing a neutron parameterisation here would bury the
signal. Everything assumed in the parent, the t-slope b = 1.67, the dipole in Q2, R = Q2/m^2,
stays assumed; the provenance string of the class says so.

The registration is what makes the spectator a proton: `Generator._struck` reads
`topology.target`, so `jpsin` on a deuteron strikes the neutron and writes a spectator with
pid 2212, status 1. Which is also the detectable one.

## 4. The m* floor, and the acceptance tests

`fermi_m_star_min` rejects a Fermi draw whose off-shell struck mass m* falls below the floor.
For pi0 the default 0.30 GeV costs 0.14 % of the AV18 distribution. Near the J/psi threshold
the high-momentum tail is the signal, so the floor was revisited with the AV18 sampler
itself (`fermi_jpsi_threshold.py`, part A, 200 000 configurations, photon along +z):

| E_gamma [GeV] | allowed configurations | <|p_s|> of those | p_s > 300 MeV/c | removed by floor 0.30 | 0.20 | 0.10 |
|---|---|---|---|---|---|---|
| 7.0 | 0.88 % | 373 MeV/c | 60.8 % | 2.9 % | 0.7 % | 0.11 % |
| 7.5 | 4.44 % | 224 | 18.3 % | 0.7 % | 0.2 % | 0.04 % |
| 8.0 | 24.6 % | 117 | 4.4 % | 0.17 % | 0.06 % | 0.01 % |
| 8.2 | 44.3 % | 91 | 2.6 % | 0.10 % | 0.03 % | 0.01 % |
| 10.0 | 97.3 % | 86 | 1.9 % | 0.06 % | 0.02 % | 0.00 % |

This reproduces the table of the brief. **Chosen: 0.10 GeV** for the J/psi cards. It keeps
99.9 % of the allowed configurations at 7.0 GeV where 0.30 would drop 3 %; the boost to the
struck-nucleon frame is still bounded (gamma = E*/m* below 8 at the floor). The pi0/eta
default stays 0.30; the value is a card line, visible in every sample's `config.json`.

Acceptance (JPSI_FERMI_BRIEF.md section 7), all run on this branch on 2026-09-26:

1. `fermi_identity.py 300`: with the flag off the LUND bytes equal production's head,
   sha256, YES.
2. `fermi_conservation.py 400 jpsin.roberts2026 jpsi.roberts2026 --born`: max |P_in - P_out|
   5e-15 GeV with Fermi motion, spectator on shell to 3e-16, for both models.
3. `fermi_module.py`: all checks passed (unchanged module).
4. `fermi_jpsi_threshold.py 200000 5000`: on the deuteron, events appear below the free
   threshold, 117 of 5000 in the whole box, 10 of them between 6.5 and 7.5 GeV; on the free
   neutron none; the sub-threshold spectators have <|p_s|> = 195 MeV/c and 19 % above
   300 MeV/c (cross-section weighted, hence softer than the unweighted 373 MeV/c of the
   kinematics). Floor 0.30 against 0.10 changes the killed-draw fraction from 1.79 % to
   1.74 % and nothing visible in the yields at this statistics.

## 5. What Fermi motion does to the sample

Figures and tables in `docs/jpsi_fermi/`; samples on wd_14tb under `aao_norad_kin/jpsi_fermi/`.

### 5a. Untagged bremsstrahlung photoproduction, 10.6 GeV electrons on the 5 cm cell

Yield per incident electron and per target nucleon, nb x (photons per electron), 200 000 events
per sample (the free proton carries the LH2 radiation length, the two neutron samples LD2):

| E_gamma bin [GeV] | deuteron, gamma n | free proton | free neutron |
|---|---|---|---|
| 6.5-7.5 | 9.6e-7 (480 ev) | 0 | 0 |
| 7.5-8.0 | 3.5e-6 (1773 ev) | 0 | 0 |
| 8.0-8.2 | 4.0e-6 (1996 ev) | 0 | 0 |
| 8.2-9.0 | 5.43e-5 | 4.52e-5 | 5.33e-5 |
| 9.0-10.6 | 3.35e-4 | 2.99e-4 | 3.48e-4 |
| total | 3.98e-4 | 3.44e-4 | 4.01e-4 |

- **2.1 % of the deuteron events lie below the free-neutron threshold of 8.20 GeV**, none of
  the free-nucleon events; 0.24 % below 7.5 GeV.  The per-nucleon total is 0.6 % lower than on
  the free neutron: Fermi motion moves yield from the steeply rising region just above
  threshold to below it, and the two nearly cancel in the integral.
- The sub-threshold spectators are hard, exactly as the kinematics require: <|p_s|> = 209 MeV/c
  below 8.2 GeV (21 % above 300 MeV/c), 257 MeV/c below 8.0, **382 MeV/c below 7.5 GeV with 62 %
  above 300 MeV/c**, against 93 MeV/c and 3.3 % above threshold.  The struck neutron's m* falls
  from 0.92 GeV above threshold to 0.73 GeV below 7.5 GeV.
- The t distributions of the three samples are indistinguishable in shape (same b; the Fermi
  boost changes t only through W).

### 5b. Quasi-real electroproduction, Q2 0.01-0.1, 100 000 events per sample

Same picture with the electron in the event: 2.4 % of the deuteron events below 8.20 GeV in the
lab (258 events in 6.5-7.5 GeV, 1022 in 7.5-8.0, 1109 in 8.0-8.2); the observed W, computed
with the target at rest, differs from the true gamma N mass by -25 MeV on average with an rms
of 105 MeV, and 2.5 % of the events have an observed W below the free threshold; spectators
202 MeV/c below threshold and 382 MeV/c below 7.5 GeV, the same numbers as in 5a.


## 6. Known limits

- PWIA, no final-state interaction, no coherent gamma d -> J/psi d, unpolarised deuteron,
  spectator on shell: the scope of `fermi`. The sub-threshold events are exactly the
  large-spectator-momentum configurations where PWIA is weakest, and the brief's own
  comparison of AV18 against a Hulthen form found the sub-threshold yield wave-function
  dependent at the 40 % level at 7.0 GeV.
- The t-slope of the elementary amplitude is the assumed b = 1.67 GeV^-2 of
  `jpsi.roberts2026`; near threshold it is one of the quantities being measured.
- No background merging, no radiative corrections, no detector: generator level only.
- The majorant of the wide quasi-real box was exceeded in about 0.1 % of trials with the
  default settings (health line); the study samples were regenerated with
  `MAJ_SHAPE=16,8,16,12 MAJ_SAFETY=6`.
