# Merging `fermi` into `production`

Rewritten 2026-09-26 by the session that did the work on this branch, after the
rebase. The earlier version of this file described a merge with a conflict to
resolve; that situation no longer exists.

## State

```
production   fa24418
fermi        rebased onto fa24418, 21 commits, working tree clean
```

**The integration is now a fast-forward.** There is nothing to resolve.

## What was done on this branch before handing it over

* **Rebased onto `production`**, per `FERMI_BRIEF.md` §2, which rules out
  merging back and forth so the diff stays readable as one change.
* **`docs/gemc_howto.tex` resolved here**, by the two sessions that wrote the
  halves rather than by whoever merges. `production`'s item on the
  `/tmp`-is-`noexec` sqlite failure is kept in full — it is the better of the
  two, because `JDK_JAVA_OPTIONS` is *prepended* so an explicit `-Xmx` still
  wins, where `_JAVA_OPTIONS` would override it silently, and because it unsets
  the variable before `run-groovy`. From this branch only the measured timing
  budget for RG-A at 10.6 GeV was added, which `production` did not have.
* **The two run registries folded into one.** `production/RUNS.md` is deleted;
  its content is an entry in `docs/PRODUCTIONS.md`. Git would not have flagged
  this — the paths differ, so a merge would have left both.
* **The byte-identity anchor now resolves at run time** (`git rev-parse
  origin/production`) instead of the literal `16412f8`. Pinned to a stale
  commit the test passed while proving nothing, and failed silently.

## Acceptance, re-run on the rebased tree

| criterion | result |
|---|---|
| 1 byte-identity at `fermi = none` | **PASS** against `fa24418`, not the fork point |
| 2 sampled p_F vs \|psi(p)\|² | PASS, chi²/ndf 1.044 |
| 3 four-momentum conservation | PASS, 4 channels, worst 8.9e-15 |
| 4 B and ms/event | PASS, 1.02 -> 1.78 ms/event, B unchanged |
| 5 W smearing | PASS, 131 MeV, 4.4 % below the box edge |
| 6 cell edges vs controls | PASS, Q² 1.6 sigma at edges against 2.1 on controls |

Verified from the production side and recorded here so it need not be repeated:
the branch does not touch `_propose`, `calibrate`, `_b_full`, `_b_cell`,
`_cell_cdf` or `sigma_total` — the hard constraint of `FERMI_BRIEF.md` §5.

## What a reviewer should still weigh

**The two 1M samples carry no merged background**, which is now standard here.
As a Fermi-versus-no-Fermi comparison they stand; as production samples they do
not. `docs/PRODUCTIONS.md` states which conclusions survive the background being
restored and which do not, rather than a blanket disclaimer.

## To integrate

```bash
git checkout production && git merge --ff-only fermi
```

Pushing is a separate decision and belongs to Valery.
