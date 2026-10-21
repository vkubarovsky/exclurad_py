# Two ways to correct for radiation, and what the difference costs

Task for a new agent, written 2026-10-21. Branch `rc-method-comparison`,
worktree `~/exclurad_py_rcmethod`, forked from `production` at `17712bd`.

## The question

Two ways to get from a measured yield to a Born cross section.

**A — the EXCLURAD way.** Generate the radiative sample only. Correct each
bin by a factor computed from the Bardin–Shumeiko formula at that bin's
kinematics, with one parameter: a cut `v_cut` (equivalently `mx2_cut`). No
Born simulation at all.

**B — the two-arm way, what we do now.** Generate radiative *and* Born,
pass both through the same GEMC and the same analysis cuts, and form
`N_rad / N_born` bin by bin.

**Measure the difference between A and B.** That is the whole task.

## What this is NOT

It is not a validation of the radiative correction. Both ways use the same
engine — `exclurad_py/core/rc.py` — and the same physics. If you find a
disagreement, the RC calculation is not thereby wrong.

What differs is only this: **A knows one number about the selection, and B
knows the selection.** The real cut chain is a region in many variables —
missing mass, missing momentum, coplanarity, vertex, timing, fiducial
boundaries — and no single `v_cut` is equivalent to it. So A and B must
differ. The question is by how much, where, and whether a `v_cut` can be
chosen to make them agree.

Say this in the report, or the result will be misread as "EXCLURAD is wrong".

## Why it matters

If the difference is small, nobody needs a Born arm: every CLAS12 analysis
can use a formula, and we save half of every campaign. If it is large, this
is the measurement that says why the Born arm is not optional, and it is
worth writing up on its own.

What we already believe, from `~/clas12_pi0_rc`, and why we expect the
difference to be real: sensitivity to `v_cut` is **0.41 % per 0.01 GeV²**
(median, up to 3.2 %), against **0.28 % median** for an entire change of
physics model. A 0.02 GeV² shift in the cut outweighs changing the model
threefold. And a hard `v_cut` keeps only **65–74 %** of the sample — it is
not a correction to the answer, it is most of it.

Expect a real effect. Do not expect it to be uniform.

## Do not generate anything. The data exists.

| campaign | arms | state |
|---|---|---|
| `/volatile/clas12/vpk/prod_phi_f18in_20261015` | rad + Born | **complete, 1600 DSTs** — use this |
| `/volatile/clas12/vpk/prod_jpsi_f18in_20261009` | rad + Born | Born being regenerated with `itarg=0`; use after |
| `/volatile/clas12/storyf/exclurad_amp2614_validation_50a5074/fullchain_1M_20261007` | rad only | Story's π⁰, no Born arm |

**Start with φ.** It is the only campaign that is finished in both arms, in
four GEMC configurations: inbending 45 nA, outbending 50 nA, and both again
with no background. 1M generated events per arm.

Its two arms both carry `itarg = 1`. That is wrong for an absolute Born
cross section and is being fixed elsewhere, but it is *consistent between
the arms*, so it does not bias a comparison of methods. Use it, and say so.

The ntuples are small (2.2 GB for all 1600) and can be copied to a laptop:
`/Volumes/wd_14tb/phi_plots` already holds a copy, along with `prov/`, which
carries `sigma_nb` per chunk.

## How I would do it

The elegant part: **both methods can be evaluated on the same radiative
events.** You are not comparing two samples, you are comparing two ways of
weighting one.

1. **Reproduce B first, and check it against something known.** Read the rad
   and Born ntuples with `production/analysis/campaign_lib.py`, apply the
   matching rule it implements, bin in (Q², x_B, t, φ), and form
   `N_rad/N_born` per bin. Check the integral against
   `sigma_rad/sigma_born` from the `prov/` sidecars — they must agree.

2. **Compute A on the same events.** `rc_factor(Ebeam, Q2, W2, t, phi,
   sfmodel, mx2_cut=...)` returns `sigma_born`, `sigma_obs` and `eta` at a
   point. Evaluate it at each bin's kinematics and form the same ratio.
   `eta` is `sigma_obs/sigma_born`, so A's prediction for the bin ratio is
   the cross-section-weighted `eta` over that bin.

3. **Scan `mx2_cut`.** This is the heart of it. There is no right value —
   that is the point. Scan it, and for each value plot A/B bin by bin. Three
   outcomes, all publishable:
   - no `v_cut` makes them agree → the one-parameter approximation fails
   - one `v_cut` agrees globally but not bin by bin → it fails *locally*,
     which is worse, because an integral can hide it
   - one `v_cut` agrees everywhere → A is enough and we have been wasting
     half of every campaign

4. **Then make the cuts realistic.** Steps 1–3 can be done with the four-track
   requirement alone. The interesting number comes when a real exclusivity
   chain is applied — a missing-mass window above all. Apply it identically
   to both arms, and watch how A/B moves as the chain is tightened. That
   motion *is* the answer to the question.

5. **Report per bin, not only integrated.** An integrated agreement built
   from +15 % in one corner and −15 % in another is not agreement.

## Traps, every one of them measured here

- **`MC::GenMatch`'s `found` flag is not a reconstruction.** It is set for
  `REC::Particle` rows with zero momentum (58 % of π⁰n neutrons, 8–14 %
  elsewhere) and for rows that are a *different particle* — a 7.7 GeV
  generated electron matched to a 0.2 GeV reconstructed photon, in 100 % of
  the events with a soft "electron". `campaign_lib.load` already handles
  this; do not write your own reader without it.
- **Two record layouts.** Vector mesons 73 floats with the rec block at 19;
  π⁰/η 50 floats at 13. Reading one with the other's offsets does not fail,
  it returns numbers.
- **`t_min` varies by a factor 800** across a wide box. A bin near the
  forward edge is partly outside the physical region; integrate over the
  intersection, not the rectangle.
- **Our samples are generated with `vcut = 0` on purpose** — no missing-mass
  cut at generation — so the cut chain can be applied offline, identically to
  data. That is what makes this study possible at all. Do not regenerate with
  a cut.
- **The generator draws *observed* kinematics.** An event's `t` and `W` are
  what a detector would measure, not the vertex values. `rc_factor` expects
  observed kinematics too, so this matches — but do not "undo" radiation by
  hand anywhere.
- **`f_rad/f_born × C_rad` is the cross-section-weighted `eta` in any
  binning.** That identity is the bridge between the two methods; use it as a
  closure test before trusting anything.
- EXCLURAD in Fortran cannot run `amp2614` and returns NaN at `vcut = 0`.
  "The EXCLURAD way" here means our engine used as a formula, not the Fortran.

## Deliverable

A note with: the method, the `v_cut` scan, A/B bin by bin for at least the φ
campaign, and a plain statement of which of the three outcomes above holds.
Plots in English. Put it in `docs/` on this branch and publish the PDF the
way the project publishes notes.

## How this worktree was made, and the rule behind it

```bash
cd ~/exclurad_py                       # always on `production`, never switched
git worktree add -b rc-method-comparison ~/exclurad_py_rcmethod production
```

**Never `git checkout` a branch inside `~/exclurad_py`.** It is the shared
checkout, several agents read it expecting `production`, and switching it
takes their work with it. Every branch gets its own worktree; there are nine
already. This rule was broken on 2026-10-15 and the mess was found by another
agent six days later.

Work in `~/exclurad_py_rcmethod`. Commit and push to `rc-method-comparison`.
**Do not merge into `production`** — VPK reviews a branch before it is merged.
