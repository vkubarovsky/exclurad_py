# Fermi motion for J/psi photoproduction — working brief

Written 2026-09-26 by the session that built the deuteron target on branch
`fermi` (pi0/eta, RG-A and RG-B). Ask it, or VPK, before deviating from
anything marked **must**.

---

## 1. The task

Give J/psi photoproduction off a deuteron the same treatment pi0/eta now has:
Fermi momentum for the struck nucleon, spectator emitted as a real particle,
behind the existing card flag. The physics target is **γn → J/psi n near
threshold**, which is a neutron measurement and therefore necessarily a
deuteron measurement — there is no free neutron.

**Phase 1 scope:** PWIA, unpolarised deuteron, spectator on shell, struck
nucleon off shell by the spectator prescription, no final-state interactions.
Same scope as `fermi`, for the same reasons.

**Out of scope:** FSI, coherent production (γd → J/psi d, which is a real
background this machinery does not describe), tensor polarisation, and any
change to the proton-target physics.

---

## 2. Where to branch from, and where to work

**If `fermi` has been merged into `production`** (check: `git log --oneline
production | grep -i fermi`), branch from `production`. You inherit the whole
machinery and there is nothing to rebase later.

**If it has not**, branch from **`fermi`**, not from `production` — otherwise
you start without the machinery and have to invent it. Rebase onto
`production` once `fermi` lands.

Make your own worktree, never a second checkout of someone else's branch:

    git worktree add ~/exclurad_py_jpsi_fermi -b jpsi-fermi <base>

`/Users/vpk/exclurad_py` is `production` and is actively used; `/Users/vpk/exclurad_py_fermi`
is `fermi`. Git will refuse to let you check out a branch another worktree
holds — that refusal is the safety property, do not work around it.

**The stash stack is shared between worktrees.** Never `git stash` / `git stash
pop` bare; use a temporary WIP commit.

---

## 3. Read before writing any code

| what | where | why |
|---|---|---|
| `generator/fermi.py` | whole file, ~180 lines | the machinery you are reusing. **It has no channel dependence at all** — no mention of pi0, eta or jpsi. It works for J/psi as it stands. |
| `_draw_fermi`, `_boost_to_lab`, `_pass_fermi_frame` | `generator/events.py` | the three hooks. `_draw_fermi` returns before touching the RNG when the flag is off; that is what makes criterion 1 hold. |
| `FERMI_NOTE.md` §2 and §4 | | why `m*` had to enter the kinematics and where the k range stops. Both decisions are revisited below — §4 in particular does **not** carry over. |
| `Jpsi_Roberts2026._sfs` | `models/vector.py` | the model you will use, and why you do not need a photoproduction one |
| `FERMI_BRIEF.md` §5 | | the forbidden list, which is unchanged |

---

## 4. The good news: you already have a photoproduction model

`jpsi.roberts2026` **is** a photoproduction fit with a dipole bolted on:

    sigma_tot(W) = A (W-Wth)^alpha exp(-B (W-Wth))  nb   <- fitted to GlueX/CLAS12 points
    dsigma_T/dt  = sigma_tot * b * exp(b t')             <- b = 1.67, ASSUMED
    Q2 dependence = (m^2/(m^2+Q2))^2.4                   <- ASSUMED
    R = sigma_L/sigma_T = Q2/m^2                         <- ASSUMED

At **Q2 = 0** the dipole is exactly 1, `sigma_L = 0`, and `sigma_TT = sigma_LT
= sigma_LT' = 0` already: only the transverse piece survives, which is what a
real photon requires. Verified numerically (2026-09-26): `sigma_L` is
identically zero at Q2 = 0 and the Q2 = 0 column equals `sigma_tot * b`, i.e.
the photoproduction `dsigma/dt` at `t_min`. `validity` already covers it,
`Q2 = (1e-4, 30)`, `W2` from 16.4 against a threshold of 16.29.

**So do not write a new model.** The one assumed ingredient that still bites at
Q2 = 0 is the t-slope `b = 1.67`, and the provenance string says so itself:
*"Replace when a near-threshold t-slope and R are settled."* Near threshold the
t-slope is one of the observables being published, so treat it as a placeholder
and say so in whatever you produce.

**The trap is the flux, not the cross section.** The virtual-photon flux
diverges as Q2 → 0. CLAS12 "photoproduction" is quasi-real electroproduction
with the electron in the Forward Tagger at Q2 ~ 0.01-0.3, so running the
existing generator with a small-Q2 box gives you the measurement as CLAS12
makes it, with the flux convention that is already settled in this code
(`core/hadronic.py`: `|q|` flux at the core, Hand converted at the model
boundary). A **real** bremsstrahlung beam needs a different normalisation.
Decide which you are doing and write it down; do not re-derive the flux.

---

## 5. What is genuinely different from pi0, and must be decided explicitly

### The k ceiling from `FERMI_NOTE.md` §4 does not carry over

For pi0 the sampler caps at the spacelike limit `k = 3.5601 fm^-1` and the
generator additionally rejects `m* < 0.30 GeV` (`fermi_m_star_min`), because
as `m* -> 0` the struck nucleon goes lightlike and its rest frame needs an
unbounded boost. That floor costs 0.14 % of pi0 events and protects the frame.

**For near-threshold J/psi the high-k tail is the signal, not the tail.** The
free-nucleon threshold is `E_gamma = 8.201 GeV` on the neutron (8.208 on the
proton; the neutron is heavier, so its threshold is 7 MeV lower). Below it,
production happens *only* through Fermi motion, and only for nucleons moving
toward the photon. Fraction of deuteron configurations kinematically allowed,
full spectator vector, AV18:

| `E_gamma` [GeV] | allowed | `<\|p_s\|>` of those | spectator > 300 MeV/c |
|---|---|---|---|
| 7.0 | 0.84 % | **367 MeV/c** | **60 %** |
| 7.5 | 4.33 % | 222 | 18 % |
| 8.0 | 24.6 % | 116 | 4.3 % |
| 8.20 | 44.4 % | 90 | 2.6 % |
| 10.0 | 97.3 % | 85 | 1.9 % |

A 0.30 GeV floor would cut into the physics. **You must revisit it, and state
what you chose.** `fermi_m_star_min` is already a constructor parameter — the
change is a value, not a code change. Do not silently keep the pi0 default.

### The spectator becomes detectable, which inverts the pi0 conclusion

For pi0 only ~3 % of spectators clear the CVT threshold. Below the J/psi
threshold it is **60 %**, because the reaction self-selects large Fermi
momentum. Two pieces of luck compound: for a **struck neutron the spectator is
a proton**, which the CVT tracks, and with the Forward Tagger the photon energy
is known, so tagging the spectator closes the kinematics event by event.

The uncomfortable corollary, which belongs in whatever you write: the
detectable spectators are exactly the ones where PWIA is weakest.

### There is no neutron J/psi model

All three (`jpsi.diffrad`, `jpsi.diffrad_vm`, `jpsi.roberts2026`) have
`topology.target = p`. `γn → J/psi n` needs a registration. Physically the
exchange is gluonic and isoscalar, so `sigma_n = sigma_p` at leading order and
a relabelling is the correct *baseline* — it is also the null hypothesis the
measurement tests, so do not invent a neutron parameterisation and bury the
signal. **Make it a deliberate, documented choice.**

Note this also decides which nucleon `_draw_fermi` treats as struck: it reads
`sf.topology.target`, so the registration is what makes the spectator a proton.

### The sub-threshold yield is wave-function dependent

AV18 against an analytic Hulthen stand-in: identical above threshold, 6 % apart
at `E_gamma = 7.5`, **42 % apart at 7.0**. Below threshold the answer is a
measurement of the deuteron's high-k tail as much as of the elementary
amplitude. A real study needs AV18 / CD-Bonn / Paris, not this crude proxy.

---

## 6. What must not be touched

Unchanged from `FERMI_BRIEF.md` §5: **`_propose`, `calibrate`, `_b_full`,
`_b_cell`, `_cell_cdf`, `sigma_total`.** The two-stage rejection was composed
wrongly until `225d4d9` and the fix is subtle; if your task appears to require
touching them, stop and ask.

**Do not copy `generator/fermi.py` or `data/nk_d2_av18.dat`.** Use them. If you
need a change — and you will, for the ceiling — it goes into the shared module
as a parameter, not into a fork. Two copies of a wave function is how two
channels quietly stop agreeing.

---

## 7. Acceptance

1. **With the flag off, output is byte-identical to the base branch for the
   same seed** — sha256, not "agrees within errors". `validation/fermi_identity.py`
   already does this and resolves the base at run time; point it at your base.
   **This is the test that keeps your branch from disturbing anything else.**
2. Per-event four-momentum conservation: `beam + d = e' + J/psi + N' + N_s`, at
   machine precision. `validation/fermi_conservation.py` takes a model name.
3. The sampled `p_F` reproduces the table (`validation/fermi_module.py`, unchanged).
4. **New, and specific to this task:** the yield below the free-nucleon
   threshold must match the kinematic expectation above — 0.84 % at 7.0 GeV,
   24.6 % at 8.0 — within the cross-section weighting. If sub-threshold events
   do not appear at all, `fermi_m_star_min` is still at the pi0 default.

---

## 8. Traps that already cost this project real time

* **`/tmp` is `noexec` on the farm compute nodes.** sqlite-jdbc cannot `dlopen`
  the native library it unpacks, CCDB falls back to an unreachable MySQL, and
  `recon-util` dies in 2 s with an error three steps from its cause — after
  burning the full GEMC time. Fix in `docs/gemc_howto.tex`; use
  `JDK_JAVA_OPTIONS`, not `_JAVA_OPTIONS`, and unset it before `run-groovy`.
* **`hipo_to_ntuple.groovy` selects the hadron by `pid == 2212`.** On any
  neutron-target sample that picks the **spectator proton** instead of the
  detected neutron. It must be fixed before any `pi0n`, `etan` or neutron
  J/psi sample is reduced. Nobody has done this yet.
* **Measure the farm rate on a live task before sizing an array**
  (`grep "Begin of event" gemc.log`). Extrapolating from a 200-event test was
  wrong four times in one session; the rate varies by a factor two with load.
* **Background merging is standard here** (`production/farm/sim_array.sbatch`,
  50 nA over eleven detectors). The deuteron pi0 samples of 2026-09-26 do not
  have it and say so; do not repeat that by accident.

---

## 9. What to hand back

A branch that passes §7, plus a short note recording: the ceiling you chose for
`k` and why, how the neutron model was registered, which flux convention the
photoproduction run uses, and the sub-threshold yield against the kinematic
expectation. Do not merge to `production` yourself.
