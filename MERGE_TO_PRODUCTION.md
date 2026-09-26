# Merging `fermi` into `production` — what the merging session needs

Written 2026-09-26 by the session that runs the pi0 production on `production`,
after dry-running the merge and aborting it. Two readers: the agent working on
this branch (§1 is yours, and only you can do it), and whoever performs the
merge (§2 onwards).

## 0. State at the time of writing

```
origin/production   fa24418
origin/fermi        e0dd4a9
local  fermi        bd56502     <- 17 commits, NOT pushed
```

The two branches diverged at `16412f8`. Since then `production` gained five
commits (the gemc fixes the first real farm run forced, the measured wall
clock, `run_report.py`, and `docs/PRODUCTIONS.md`), and `fermi` gained
seventeen.

**What I checked so the reviewer does not have to repeat it:** the branch does
not touch `_propose`, `calibrate`, `_b_full`, `_b_cell`, `_cell_cdf` or
`sigma_total` — not one line. That was the one hard constraint in
`FERMI_BRIEF.md` §5, and it is honoured. Everything new lives in
`generator/fermi.py`, the card field defaults to `none`, the draw returns
before doing anything when it is off, and `ptarg != 0` is refused.

## 1. For the agent on this branch, before anyone merges

**Push.** Seventeen commits exist only on this machine. Until
`git push origin fermi` runs, a merge on another host gets `e0dd4a9` — the
brief, and none of the work.

**Two untracked files.** `production/gemc_check/fermi_compare.py` and
`production/runs/fermi_influence_generator.png` are in the working tree and in
no commit. Commit them or delete them; do not leave the decision to someone
who does not know what they are.

**Decide about the two registries — this is the one that will be missed.**
You wrote `production/RUNS.md` (201 lines, the runs launched from this branch).
`production` has `docs/PRODUCTIONS.md` (153 lines, the pi0 production
campaigns). Git will not report a conflict, because the paths differ: the merge
will silently leave the repository with two run registries, which is worse than
either. One of them has to become the file, and the other has to fold into it.
You know what is in yours; fold it yourself rather than leaving it to the
merge.

## 2. The merge itself

```bash
git -C /Users/vpk/exclurad_py_fermi push origin fermi
cd /Users/vpk/exclurad_py          # production is checked out here
git fetch origin && git merge origin/fermi
```

Work in `/Users/vpk/exclurad_py`. Do not check out `fermi` there — git will
refuse anyway, because this worktree holds it, and that refusal is a feature.

**One real conflict, dry-run verified:** `docs/gemc_howto.tex`. Both sides
appended to it — this branch added its own items, `production` added the
`/tmp`-is-`noexec` finding and the measured wall clock. Both sets are worth
keeping; resolve by keeping both, not by taking a side.

## 3. Acceptance, on the merged tree and not before

Criterion 1 of `FERMI_BRIEF.md` — with `fermi = none`, byte-identical output
for the same seed — has to be re-run **after** the merge. Passing it on the
branch is not enough: `production` moved five commits since the branch point,
and the test has never seen those together with this branch. Everything else
in §8 of the brief was measured on the branch and does not need repeating
unless the merge changes a file it touched.

## 4. The farm, at the time of writing

```
fp_none    667 PENDING     (this branch)
fp_av18      1 PENDING     (this branch)
post_rad    29 PENDING     (pi0 production, finishing the 200/200)
post_born3   3 PENDING     (same)
```

Nothing here blocks the merge. It is recorded because the 668 jobs from this
branch and the 32 from the production run share one fairshare, so each slows
the other; if the pi0 production needs to close first, the `fp_none` array is
the thing to hold.
