#!/usr/bin/env python3
"""Rewrite the LUND event-header line into the format the generator now writes.

    fix_lund_header.py <dir> [--dry]

Until 2026-09-23 run.py never passed the card's target polarisation or beam
energy to write_lund, so column 4 was 0.000 and column 7 the 10.6 default in
every file (commits 2ce1b03, 5e6101b fixed the generator).  This repairs files
already on disk: column 4 is taken from the file's own `# run.ptarg:` comment,
column 7 from `# run.Ebeam:`, and the whole line is re-emitted with

    f"{npart:4d} {A:4d} {Z:4d} {pol_target:6.3f} {pol_beam:6.3f} "
    f"{beam_pid:5d} {beam_energy:8.4f} {target_pid:6d} {process:4d} {weight:12.6e}"

which is lund.py's own format string.  Re-emitting rather than substituting
matters: an earlier patch replaced the five characters "0.000" with "-1.000",
which is one wider, so negative-polarisation files ended up a column out of
step with freshly generated ones.  Everything except column 4 and 7 is carried
across unchanged, and particle lines are never touched.

Idempotent: a file already in canonical form is rewritten to itself.  Each file
is written to a temporary name in the same directory and moved into place, so
an interrupted run leaves no half-written LUND.
"""
import glob
import os
import re
import sys
import tempfile

HDR = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+"
                 r"(\d+)\s+(\d+\.?\d*)\s+(-?\d+)\s+(\d+)\s+(\S+)\s*$")


def canon(npart, A, Z, ptarg, pbeam, bpid, ebeam, tpid, proc, w):
    return (f"{npart:4d} {A:4d} {Z:4d} {ptarg:6.3f} {pbeam:6.3f} "
            f"{bpid:5d} {ebeam:8.4f} {tpid:6d} {proc:4d} {w:12.6e}\n")


def comments(path):
    """(ptarg, Ebeam) from the comment header, or (None, None)."""
    pt = eb = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("# run.ptarg:"):
                pt = float(line.split(":", 1)[1])
            elif line.startswith("# run.Ebeam:"):
                eb = float(line.split(":", 1)[1])
    return pt, eb


def fix(path, dry=False):
    pt, eb = comments(path)
    if pt is None:
        return 0, 0, "no run.ptarg"
    n_hdr = n_chg = 0
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        with open(path) as src, open(tmp, "w") as dst:
            for line in src:
                m = HDR.match(line) if not line.startswith("#") else None
                if m:
                    n_hdr += 1
                    g = m.groups()
                    new = canon(int(g[0]), int(g[1]), int(g[2]), pt, float(g[4]),
                                int(g[5]), eb if eb is not None else float(g[6]),
                                int(g[7]), int(g[8]), float(g[9]))
                    if new != line:
                        n_chg += 1
                    line = new
                dst.write(line)
        if dry or n_chg == 0:
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return n_hdr, n_chg, None


def main(root, dry):
    files = sorted(glob.glob(os.path.join(root, "**", "*.lund"), recursive=True))
    print(f"{'DRY RUN: ' if dry else ''}{len(files)} files under {root}")
    nf = nh = nc = 0
    skipped = []
    for i, f in enumerate(files, 1):
        h, c, why = fix(f, dry)
        if why:
            skipped.append((f, why)); continue
        nh += h; nc += c
        if c:
            nf += 1
        if i % 200 == 0:
            print(f"  {i}/{len(files)}  headers {nh}  changed {nc}", flush=True)
    print(f"DONE  files {len(files)}  files changed {nf}  header lines {nh}  "
          f"lines changed {nc}  skipped {len(skipped)}")
    for f, why in skipped[:5]:
        print("  skipped:", os.path.basename(f), why)


if __name__ == "__main__":
    main(sys.argv[1], "--dry" in sys.argv)
