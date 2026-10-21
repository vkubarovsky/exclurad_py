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
| + M(e'γγ) 3σ, cone < 4°, MP_T < 0.75 | 0.925 |
| + MM²(e'pγγ), ME 3σ — full chain | **0.899 ± 0.018** |

A equals B's integral at **mx2_cut = 0.19 GeV²** (0.15–0.23 at 1σ). φ by φ, χ² of A against B is
minimal at 0.17: **12.3 for 12 bins**. 0.08 gives 24, 0.30 gives 21, no cut gives 239.
Within ±5–8 % per φ bin, **one cut makes them agree everywhere in this bin** (outcome three).
That is a statement at this precision, about this bin, and not yet about the whole grid.

## What was found on the way

1. **φ is not the same variable in A and B.** For the proton-detected topology, EXCLURAD's φ is the
   proton's azimuth + 180°. The note bins in φ of the γγ pair. For Born events the two are the same; for
   radiative ones they are not. With no cut, the radiative tail sits at φ_γγ ≈ 180° and at φ_p ≈ 0°
   (p3, last panel). After the full exclusivity chain the difference is within the errors (p4, open vs
   filled points). With loose cuts it is not.
2. **Generator-level closure holds to ~3 %, and the 3 % is `itarg`.** With the same MM²(e'p) cut,
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
- φ_γγ against φ_p at intermediate cut levels, quantified.
