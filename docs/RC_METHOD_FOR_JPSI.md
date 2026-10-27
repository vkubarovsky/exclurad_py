# The RC-method study for J/ψ: how the π⁰ study was done, and what to change

Written 2026-10-27 by the "Pi0 radiation correction study" session for the "J/psi radiation correction
study" session, at VPK's request. The π⁰ study is on branch `rc-method-comparison` of
`github.com/vkubarovsky/exclurad_py`, worktree `~/exclurad_py_rcmethod`. Results are in
`docs/rc_method_pi0/README.md`, and the report is at OneDrive `Work/2026_pi0_RC_method/`.

**Scope for J/ψ, set by VPK: everything below except the φ dependence.** No φ fits and no structure
functions. All comparisons are integrated over φ.

## 1. The question

Two ways to get from a measured yield to a Born cross section:

- **B: rad/Born MC** (what the analyses do). Rad and Born twins, the same GEMC, reconstruction and cuts.
  Per bin:
  `δ_B = (N_rad^rec · σ_rad / N_rad^gen) / (N_Born^rec · σ_Born / N_Born^gen)`.
  σ is `sigma_nb` from each arm's `prov/*.config.json` (`campaign_lib.sigma_nb`), averaged over the chunks.
  Do not divide out acceptance: δ_B = η_gen(no cut) × Acc_rad/Acc_Born exactly, and the cuts act only
  through the acceptance ratio.
- **A: the formula.** `rc_factor` in `exclurad_py/core/rc.py`, the same engine that generated both arms,
  gives η = σ_obs/σ_Born. The ONLY exclusivity cut it knows is `mx2_cut` (a missing-mass-squared upper
  limit). It knows nothing of missing energy, total missing mass, MP_T or coplanarity.

Measure A against B, find which single cut value makes them agree, and check whether that value holds
everywhere. Say in the report that this is not a validation of the RC physics: both methods use the
same engine. What differs is that A knows one number about the selection while B knows the whole selection.

## 2. First decision: which J/ψ model and topology, and therefore what `mx2_cut` means

The engine has two J/ψ models with opposite topologies (checked 2026-10-27):

| model | topology | `mx2_cut` cuts | t | `t_nucl` |
|---|---|---|---|---|
| `jpsi.diffrad_vm` | meson_detected | MM²(e'J/ψ), missing **proton** | kinematic | none |
| `jpsi.diffrad_vm_pdet` | baryon_detected | MM²(e'p), missing **J/ψ** | fixed nucleon | yes |

Use the model the campaign was generated with: `sf.model` in any `.info` / `config.json`. Define the
analysis's v the same way, `v = MX² − m_U²`, with m_U the missing particle of that topology.

- For `_pdet`, set `m.t_nucl = t` per event before every `rc_factor` call. A model built once with a
  fixed `t_nucl` has no t dependence and fails silently: memory note "frozen t_nucl", 9 % on C_rad.
- For the three-track topology without the proton, check which model and v make sense before building
  anything.

## 3. Data: check before trusting

- **Campaign:** `prod_jpsi_f18in_20261009` (rad + Born). Ask the "Launch generator" session for its
  current state. 800 J/ψ Born DSTs were voided on 2026-10-22 by the point-target bug and are being
  regenerated. Use only a Born arm made from `bbe6628` or later.
- **Vertex**, the trap we found:
  - Before `bbe6628`, `itarg = 0` made the generator write vz = 0 for every event.
  - GEMC keeps the LUND vertex exactly. `MC::Particle vz == MC::Lund vz`, and the gcard's
    `BEAM_V`/`SPREAD_V` are not applied to LUND input.
  - So a point in the LUND is a point in the detector, and the two arms see different acceptance.
  - Check one LUND chunk per arm: column 14 must spread over −5.5 … −0.5 cm.
  - Script: `production/analysis/rc_method_pi0_vertex.py` plus `vz.groovy` (in the vertex section of
    `docs/rc_method_pi0/README.md`). Plot the vertex of both arms before anything else; it is VPK's rule.
- **`itarg`:** today's Born cards have `itarg = 0`, rad cards `itarg = 1`.
  - B therefore contains the pre-vertex external radiation; `rc_factor` does not.
  - Expect A and B to differ by that (+1.25 ± 0.11 % in σ for J/ψ, per Launch generator). State it,
    and do not chase it as a method difference.
  - In the September π⁰ sample both arms had `itarg = 1`, and the residual was +3 % in B.
- **Commits:** read `git_commit` in the sidecars of both arms. If they differ, say so in the report.
- **Copies:** copy ntuples with plain `rsync` (no `--info=stats2`) and check the exit status. Verify
  against a sha256 manifest computed on ifarm. A swallowed rsync error once left four directories
  silently empty.

## 4. Reading the ntuples

- Use `production/analysis/campaign_lib.load`, never a reader of your own.
  - `MC::GenMatch`'s `found` flag is set for zero-momentum rows and for wrong particles; `load` handles both.
  - Vector record: 73 floats, rec block at 19, slots 0 e', 1 p, 2 h+, 3 h−, 4 γ(decay), 5 γ(prod).
    `CHANNELS["jpsi"]` carries it.
  - Reading the π⁰ layout (50/55 floats, rec block at 13) on a vector file does not fail; it returns numbers.
- `load(..., rec=True, require=TOPOLOGY[...])` selects the reconstructed topology. Use the J/ψ analysis's
  own topology (FT electron; FD leptons and proton; the three-track set without the proton, if used).
- `load` filters the rows. If you need the generated kinematics of the same reconstructed events
  (generated v, see §7), keep the mask. In `rc_method_pi0.load` I return `ok` for exactly that.

## 5. The scripts, and what to change for J/ψ

All in `production/analysis/` on `rc-method-comparison`:

| script | what it does | for J/ψ |
|---|---|---|
| `rc_method_pi0.py` | reader, kinematics (Q², x_B, t, W, exclusivity variables), MC-calibrated windows, cut levels | rewrite `kinematics()` for e+e− and the J/ψ topology; keep the structure |
| `rc_method_pi0_combos.py` | **the main tool**: note cuts → per-event η → A and B in every binning combination | reuse almost as is: swap the cut function, the binning variables and edges, and the model in `eta_chunk` |
| `rc_method_pi0_combos_plots.py` | one variable at a time, two-variable maps, all-cells list, χ² against `v_cut`, equivalent-cut histogram, B/A − 1 in % under every plot | reuse; change the variable names and labels |
| `rc_method_pi0_cutvars.py` | every cut variable N−1 (all other cuts applied), rad vs Born, with the analysis cut lines | reuse; change the variable list |
| `rc_method_pi0_vdist.py` | v after the other cuts: reconstructed (lin and log), generated v of accepted rad events, the chain's efficiency against generated v | reuse; v as in §2 |
| `rc_method_pi0_vcuts_each.py` | each cut alone as an efficiency against generated v; v against E_k and the photon angle | reuse; recompute the relations of §7 for J/ψ |
| `rc_method_pi0_vertex.py` | vertex of both arms (generated, true, reconstructed) | reuse with `vz.groovy` |
| `rc_method_pi0_plots.py`, `_grid*.py`, `_phifit.py` | first single-bin pass, centroid grid, φ fit | **skip**: superseded by `_combos`, or φ |

Report template: `docs/rc_method_pi0/rc_method_pi0_report.tex`. Publishing: `docs/rc_method_pi0/publish.sh`.
Change its topic folder, e.g. `~/OneDrive/Work/2026_Jpsi_RC_method`, and ask Launch generator which
folder it prefers.

## 6. The method, step by step

1. **Cuts:** take the J/ψ analysis's own cuts, literally (DIS/kinematic, PID, exclusivity windows).
   Apply them identically to both arms.
   - Our MC is wider than corrected data, so literal windows may keep only 30–40 % of events. That is
     fine for B, because both arms are cut alike. Report the survival fraction per cut.
   - Alternatively, centre windows on the Born-MC peak with ±3σ of the MC itself (`calibrate()`).
   - **The equivalent cut depends on which you choose** (π⁰: 0.065 literal against 0.14 with wider
     windows). Pick one and say which.
2. **Generator-level closure first:**
   - With the same cut on the *generated* MX², B_gen must follow A, because it is the same engine.
   - Any residual is `itarg` or a bug. Find it before going on.
3. **A per bin, for any binning**, without Born MC:
   - Compute η per accepted rad event at its own reconstructed kinematics, for a list of `mx2_cut` values.
   - `A_bin = N / Σ 1/η(x_i)`. This is the Born- and acceptance-weighted mean of η, because rad events
     follow σ_Born·η·Acc.
   - Cache η per event in an `.npz`, and check the cache's shape including the number of cuts. My first
     cache check compared only rows and silently reused a table with fewer cuts.
4. **Binning:** every combination of the analysis's variables (for J/ψ probably Q², t, W or E_γ): each
   alone, each pair, all three.
   - Keep cells with ≥ 100–200 events per arm.
   - Quote B ± its Poisson error, A at the analysis's nominal `mx2_cut`, and B/A − 1 in %.
5. **Scan `mx2_cut`:**
   - For each binning, compute χ² of A against B against one global cut, and the cell-by-cell equivalent
     cut (where A = B).
   - Three outcomes: no cut fits; one fits in the integral but not cell by cell; one fits everywhere.
     State which one holds, and at what precision.
6. **Report per cell**, not only integrated: +15 % in one corner and −15 % in another is not agreement.

## 7. Understanding which cuts remove the radiation

This is what made the π⁰ result interpretable.

- Plot every cut variable N−1 for rad and Born, and the fraction each cut keeps given all the others.
  For π⁰ only missing energy and MM²(e'p'π⁰) cut rad harder than Born; MP_T and coplanarity added nothing.
- With an unseen photon k: ME = E_k, the total missing mass squared is k² = 0 for any real photon (it
  cuts only through resolution, which grows with E_k), and v = 2 p_U·k for the missing particle U.
  Write the J/ψ version for your topology.
- Measure the chain's efficiency against the **generated** v of the accepted rad events.
  - For π⁰ it was a soft edge with its 50 % point at v = 0.057, the same as the equivalent `v_cut` from
    matching A to B.
  - That is the physical explanation of the equivalent cut, and the strongest single plot in the π⁰ report.
- **J/ψ-specific question:** do the generated rad events, and `rc_factor`, include radiation from the
  decay leptons (final-state radiation in J/ψ → e+e−)? If the MC has it and `rc_factor` does not, it is a
  real A–B difference that a missing-mass cut on the e'p system cannot fix. Find out before interpreting.
  The "J/psi electroproduction RC generator" session knows the state of FSR.

## 8. Traps I hit (all cost time)

- **macOS multiprocessing:** `Pool` uses `spawn`, which re-imports the script. Code outside
  `if __name__ == "__main__":` hangs at 0 % CPU.
- **pdflatex stamps the build time into the PDF.** Track only the `.tex` in git, or every publish looks dirty.
- **Never switch branches in `~/exclurad_py`** (shared checkout). Make your own worktree:
  `git worktree add -b <branch> ~/exclurad_py_<name> production`.
  Commit to your branch, push, and never merge into `production`. Production changes go through Launch generator.
- LUND, ntuples and caches go on `/Volumes/wd_14tb`, never `$HOME` or the internal SSD.
- **Plots in English only.**
- **Statistics:** with ~1M per arm, per-cell errors were 2–10 %. Say how far the conclusion goes.

## 9. Deliverables

- A README section per study with the numbers.
- Figures with B/A − 1 in % under every comparison.
- A LaTeX report: methods, sample and caveats (model, commits, `itarg`, vertex, statistics), every cut
  variable N−1, how the cuts act on v, A against B in all combinations, χ² against `v_cut`, and conclusions.
- Published to OneDrive with a `publish.sh` and a README carrying the commit.
- Tell VPK when it is done.
