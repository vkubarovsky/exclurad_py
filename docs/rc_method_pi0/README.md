# pi0 RC two ways, one bin — first pass (2026-10-21)

Code: `production/analysis/rc_method_pi0.py` (numbers), `rc_method_pi0_plots.py` (figures).
Data: `prod_pi0_rga_t25_20260928` merged ntuples, 1M rad + 1M Born per polarity, `pi0.amp2609`,
copied to `/Volumes/wd_14tb/rc_method_pi0`. Generator commits in the sidecars (2470395, 408d833,
ac3e572, b1897dc, ee4cc1b) all contain 2a1c904 and 225d4d9; the ntuple groovy is from after 2a1c904.

**Bin:** 1.5 < Q² < 2.5, 0.15 < x_B < 0.30, 0.2 < −t < 0.6, inbending + outbending, 12 φ bins.
The bin first proposed (Q² 2–3, x_B 0.2–0.3, −t 0.2–0.4) held ~800 Born events per polarity — too few.

**A** = `rc_factor` (same engine) averaged on a 3×3×3×3 grid per φ bin, selection = `MM²(e'p) < mx2_cut`.
**B** = (N_rad σ_rad/N_gen) / (N_born σ_born/N_gen), reconstructed, same cut chain on both arms.

## Result

| selection (B, reconstructed) | δ_B (bin) |
|---|---|
| topology + DIS | 1.173 |
| + M(γγ), MM²(e'p) 3σ | 0.942 |
| + M(e'γγ) 3σ, coplanarity θ(X, π⁰) < 4°, MP_T < 0.75 | 0.925 |
| + MM²(e'pγγ), ME 3σ — full chain | **0.899 ± 0.018** |

A equals B's integral at **mx2_cut = 0.19 GeV²** (0.15–0.23 at 1σ). φ by φ, χ² of A against B is
minimal at 0.17: **9.7 for 12 bins**. 0.08 gives 22, 0.30 gives 17, no cut gives 231.
Within ±5–8 % per φ bin, **one cut makes them agree everywhere in this bin** (outcome three).
That is a statement at this precision, about this bin, and not yet about the whole grid.

## What was found on the way

1. **φ is defined as in EXCLURAD, in both methods:** φ = Trento azimuth of the recoil proton + 180°.
   The analysis must use the same definition as the RC engine. Taking φ from the γγ pair gives a
   different variable for radiative events, and the generator-level closure then fails without a cut.
2. **Generator-level closure (tight cuts) holds to ~3 %, and the 3 % is `itarg`.** With the same MM²(e'p) cut,
   B_gen/A ≈ 1.03. In the bin, 2.8 % of Born-arm events have MM²(e'p) > 0.05, from the pre-vertex
   radiator that `itarg = 1` switches on in the Born arm too. `rc_factor` has no `itarg`. This
   inflates B relative to A by about +3 % and moves the equivalent cut up by about 0.05.
3. **The note's literal windows do not fit this MC.** Tables 7–10 are data widths after momentum
   corrections. Our GEMC sample is wider (σ M_γγ 0.011 against 0.0073), and the ME peak sits at
   +0.07 GeV. The literal MM²(e'pγγ) and ME windows keep about 3 % of Born events. Each window here is
   therefore ±3σ around the Born-MC peak, per polarity and per proton detector, which is the note's own
   rule applied to MC. Proton smearing (note Sec. 4.7) is off, because this MC is already wider than
   the data. Neither the vertex cuts nor the same-sector veto are applied: the ntuple has no vertex and
   no sector.

## Not done yet

- Other bins. The equivalent cut must be checked for stability across the grid. The brief's point is
  that it is not expected to be stable.
- Correcting B for `itarg`, or regenerating the Born arm with `itarg = 0` (the vector channels are
  already being switched).

## What B is, and acceptance (p6)

B is the note's Eq. 20: the ratio of reconstructed yields, rad over Born, each normalised to its own
σ/N_gen. No acceptance is divided out, and none should be: the note applies the Born acceptance
separately, so δ_B carries Acc_rad/Acc_Born inside it. Exactly:
δ_B = η_gen(no cut) × Acc_rad/Acc_Born. The cuts act only through that acceptance ratio. In this bin
the ratio is 0.95 after topology + DIS and 0.73 after the full chain. A has no acceptance at all, so
A against B measures how well one `mx2_cut` stands in for Acc_rad/Acc_Born.

## The note's grid (`rc_method_pi0_grid.py`, `rc_method_pi0_grid.pdf`)

The full note grid (10 Q² × 8 x_B × 8 −t, 24 φ) cannot be filled from 1M + 1M per polarity: after the
full chain the sample holds 56k Born and 40k rad events. So the comparison is φ-integrated, on two levels:

- 26 note (Q², x_B) cells, −t integrated, with ≥ 300 events per arm (errors 2–7 %).
- 44 note (Q², x_B, −t) cells with ≥ 300 per arm (errors 5–9 %). Indicative only.

In each cell, A is evaluated at the centroid of the reconstructed rad-MC kinematics, with 12 φ points
weighted by its own σ_Born; no Born MC enters A.

- **The best single cut for all cells is mx2_cut = 0.14 GeV²**, with χ² = 42 for 26 cells (2D). At 0.17
  it is 54, and at 0.08 it is 101. One cut does not describe every cell within its errors.
- **Q² < 2:** the 4 cells at Q² 1.0–1.5 agree with A(0.14) to 0–2 % (errors ≈ 2 %); at Q² 1.5–2.0 the
  two lowest-x_B cells sit 6–8 % below. The cell-by-cell equivalent cut there is 0.07–0.16.
- **Q² > 3 and x_B > 0.38:** B sits 10–18 % above A(0.14), at 2–3σ per cell, and the equivalent cut is
  0.25 to above 0.5. These cells have centroids at −t ≈ 0.8–1.0. Whether this comes from the physics
  of the cut chain or from A evaluated at a single centroid in a wide, t_min-truncated cell is not yet
  separated.
- Against y and W there is no clear trend at this precision (g3).

`itarg` still biases B by about +3 % against A in every cell. Correcting for it lowers the equivalent
cut by about 0.05.

## The note's cuts as printed, every combination of Q², x_B, −t (`rc_method_pi0_combos.py`, `rc_method_pi0_combos.pdf`)

**What A can cut on:** only `v_cut` (MM²(e'p) < v_cut + m²_π0). `rc_factor` has no missing energy,
no MM²(e'pγγ), no MP_T and no coplanarity. B applies all of them, event by event, to both MC arms.

**Cuts in B, Sec. 3.5 literally:**
- DIS: Q² > 1, W > 2, p_e' > 2.
- π⁰ identification: M_γγ within 3σ (σ from the Born MC), γγ opening angle > 1°.
- Exclusivity: |ME| < 0.2, |MM²(e'pγγ)| < 0.004, MP_T < 0.75, coplanarity θ(X, π⁰) < 4°.

On this uncorrected MC these windows keep 39–43 % of Born and 28–32 % of rad; they act on both arms alike.

**A per bin, any binning:** A_bin = N / Σ 1/η(x_i) over the accepted rad events in the bin, with
η = σ_obs/σ_Born at each event's own reconstructed kinematics. Rad events follow σ_Born·η·Acc, so this
is the Born- and acceptance-weighted mean of η, and no Born MC enters A. η is finite for 99.9 % of events.

**Result.** At v_cut = 0.2, A is 5–10 % above B almost everywhere: the literal exclusivity windows remove
more radiation than v_cut = 0.2 does. **One v_cut ≈ 0.065 GeV² fits every combination within its errors:**

| binning (bins integrated over the rest) | cells | best v_cut | χ² at 0.06 |
|---|---|---|---|
| Q² | 9 | 0.068 | 11 |
| x_B | 8 | 0.069 | 6 |
| −t | 8 | 0.069 | 14 |
| Q² × x_B | 22 | 0.066 | 26 |
| Q² × −t | 28 | 0.065 | 31 |
| x_B × −t | 35 | 0.065 | 35 |
| Q² × x_B × −t | 35 | 0.055 | 30 |

The per-bin errors are 2–10 %, so this holds at that precision: outcome three of the brief. The
earlier, wider ±3σ-from-MC windows gave an equivalent cut of 0.14 and 10–18 % misses at high Q²/x_B.
**The equivalent v_cut follows the exclusivity windows, not the kinematics.** It must therefore be
re-derived for every analysis's cut chain, which needs the rad + Born MC at least once.

Still in B: `itarg` (+3 %), which lowers the equivalent v_cut by about 0.03 at this slope.

## v after the other cuts (`rc_method_pi0_vdist.py`, `rc_method_pi0_vdist.pdf`)

The note has no cut on MM²(e'p), so v = MM²(e'p) − m²_π0 was plotted after the whole Sec. 3.5 chain:

- **Reconstructed v:** rad and Born look almost the same. The resolution (σ ≈ 0.1 GeV²) is wider than
  the radiative tail that survives, so the analysis cannot see a v_cut on the reconstructed MM²(e'p).
- **Generated v of the accepted rad events:** the chain's efficiency against the true radiated v is
  flat up to v ≈ 0.01. It falls to **50 % at v = 0.057** and is near zero above 0.2. Only 2.0 % of the
  accepted rad yield has v > 0.066, and 0.09 % has v > 0.2.
- That 50 % point, read straight off the cut chain, agrees with the v_cut ≈ 0.065 found by matching A
  to B. The chain acts as a soft cut on v of about that size, and A's sharp step at v_cut stands in for
  it. They agree because the soft edge is roughly symmetric about the step.

## Which cut removes the radiation, and how v relates to ME and MM²(e'p'π⁰) (`rc_method_pi0_vcuts_each.py`)

For an unseen radiated photon k: v = MM²(e'p') − m²_π = 2 p_π·k = 2 E_k E_π (1 − β_π cos θ_πk);
ME = E_k; MM²(e'p'π⁰) = k² = 0; MP_T = k_T.

Each cut alone (after DIS + π⁰ ID), as an efficiency against the generated v:

| cut | 50 % point in v (GeV²) |
|---|---|
| \|ME\| < 0.2 | 0.085 |
| \|MM²(e'pγγ)\| < 0.004 | 0.080 |
| coplanarity θ(X, π⁰) < 4° | 0.24 |
| MP_T < 0.75 | 4.5 (inactive) |
| all together | **0.055** |

- **ME** cuts the photon energy directly. Through v = 2E_kE_π(1 − β cos θ_πk), each photon angle gives a
  different v limit. The radiated photon sits a median 21° from the π⁰.
- **MM²(e'p'π⁰)** is 0 for any real photon, so it removes radiation only through its resolution, which
  grows with E_k. Here it acts almost exactly like the ME cut.
- An equivalent v_cut is therefore a property of the ME and MM²(e'pγγ) windows and of the detector
  resolution, and it has to be derived for that pair of cuts.

## Every cut variable, N−1 (`rc_method_pi0_cutvars.py`, `rc_method_pi0_cutvars.pdf`)

Each variable is shown with all the OTHER cuts applied, rad and Born, on linear and log scales. The
note's cuts are drawn as solid lines (Sec. 3.5; M_γγ from Sec. 3.4). The coplanarity angle is
θ(X, π⁰): X is the missing particle of ep → e'p'X and π⁰ is the reconstructed γγ pair. For a radiative
event X = π⁰ + k, so the angle is the photon's pull on X.

Given all the others, each cut keeps (Born / rad):

| cut | Born | rad |
|---|---|---|
| Q² > 1 | 81 % | 80 % |
| W > 2 | 74 % | 75 % |
| \|ME\| < 0.2 | **71 %** | **69 %** |
| \|MM²(e'pγγ)\| < 0.004 | **80 %** | **77 %** |
| p_e' > 2, M_γγ, opening > 1°, MP_T < 0.75, coplanarity < 4° | 100 % | 100 % |

Only ME and MM²(e'pγγ) separate rad from Born. With those two applied, MP_T and the coplanarity cut
remove nothing more. Q² and W only trim the generation box (Q² ≥ 0.8, W ≥ 1.8). After all cuts,
MM²(e'p) and M(e'γγ) lie inside the Tables 7–10 windows, so those windows would add nothing.

## φ fit and structure functions (`rc_method_pi0_phifit.py`), and the report

The pseudo-data are the even half of the reconstructed rad MC, after the note's cuts. They are
corrected with δ_B (built from the odd half and the Born MC) and with δ_A (v_cut = 0.06 and 0.2), then
fitted for σ_U, σ_TT and σ_LT against the generated Born distribution.

In wide2 (1.0<Q²<2.5, 0.10<x_B<0.30, 0.09<−t<1.0; 6013 events):
- v_cut = 0.2 gives σ_U 10 % low.
- σ_TT agrees for all three corrections within ±2–3 % of σ_U.
- σ_LT (−0.02 σ_U) is not measurable at this statistics.

The full write-up is `rc_method_pi0_report.tex/.pdf`. `publish.sh` builds it and copies it to
`~/OneDrive/Work/2026_pi0_RC_method/`.

## Interaction point (`rc_method_pi0_vertex.py`, `v5_vertex.png`)

Read from 8 DSTs per arm and polarity (40k events each), via `vz.groovy` on ifarm:
- **Generated:** both cards draw vz uniform in −5.5 … −0.5 cm (the 5 cm LH₂ cell centred at −3 cm),
  with itarg = 1 in both arms. Rad and Born draw independently (correlation 0.004).
- **GEMC keeps the LUND vertex:** MC::Particle vz = MC::Lund vz to 0.000 cm. The gcard's
  BEAM_V/SPREAD_V are not applied to LUND input.
- **Reconstructed electron vz:** resolution σ₆₈ = 0.89 cm inbending, 1.23 outbending, with offsets
  +0.17 and −0.09 cm. Identical in rad and Born.
- **The note's vertex cut is ambiguous.** Sec. 2 places the target at −5.5 … −0.5 with an electron
  window of (−13, 12)/(−18, 10). Tables 7–10 say |vz| < 2.2/2.5. Read literally, |vz| < 2.2 keeps
  36 % of electrons; |vz + 3| < 2.2 keeps 74 %. It acts the same on both arms.
- **For the next campaign:** in `events.py` `_draw_vertex`, vz is drawn from the card range only if
  itarg = 1. With itarg = 0 (production `cd0e621`, every Born card) every Born event gets vz = 0, a
  point 3 cm downstream of the cell centre, while rad stays spread over the cell. The September π⁰ and
  15.10 φ campaigns are not affected (both arms itarg = 1). Reported to Launch generator 2026-10-22.
