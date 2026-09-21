#!/usr/bin/env python3
"""Count events in LUND files.

    lundcount FILE_OR_DIR ...          per-file counts and a total
    lundcount -m FILE ...              also the breakdown by particle multiplicity
    lundcount -q DIR                   total only
    lundcount -c FILE ...              cross-check against the header's run.n_events

An event is one 10-column header line followed by as many particle lines as
its first column says.  Counting the header lines is exact and does not care
how many particles an event has -- unlike grepping for "6 1 1", which counts
only the events with six particles.  Directories are searched recursively
for *.lund.  Works on any LUND file, not only exclurad_py's.
"""
import os
import sys


def count(path, want_mult=False):
    n = 0
    mult = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) != 10:
                continue          # particle line (14 columns) or junk
            k = int(t[0])
            n += 1
            if want_mult:
                mult[k] = mult.get(k, 0) + 1
            for _ in range(k):
                next(fh, None)
    return n, mult


def header_n(path):
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                return None
            if line.startswith("# run.n_events:"):
                return int(line.split()[-1])
    return None


def main(argv):
    flags = {a for a in argv if a.startswith("-")}
    paths = [a for a in argv if not a.startswith("-")]
    if not paths or "-h" in flags or "--help" in flags:
        print(__doc__.strip()); return 0
    quiet, want_mult, check = "-q" in flags, "-m" in flags, "-c" in flags
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, f) for f in sorted(names) if f.endswith(".lund")]
        else:
            files.append(p)
    total, tot_mult, bad = 0, {}, 0
    for f in files:
        n, mult = count(f, want_mult)
        total += n
        for k, v in mult.items():
            tot_mult[k] = tot_mult.get(k, 0) + v
        if not quiet:
            extra = ""
            if want_mult:
                extra += "   " + " ".join(f"{k}p:{v}" for k, v in sorted(mult.items()))
            if check:
                h = header_n(f)
                if h is not None and h != n:
                    extra += f"   *** header says {h} ***"; bad += 1
                elif h is not None:
                    extra += "   header ok"
            print(f"{n:9d}  {f}{extra}")
    print(f"{total:9d}  TOTAL  ({len(files)} files)"
          + ("   " + " ".join(f"{k}p:{v}" for k, v in sorted(tot_mult.items())) if want_mult else "")
          + (f"   *** {bad} files disagree with their header ***" if check and bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
