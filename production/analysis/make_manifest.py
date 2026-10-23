#!/usr/bin/env python3
"""Write a MANIFEST.txt for a campaign directory, derived from the data itself.

    python3 make_manifest.py <campaign dir> [--write]

Without --write it prints to stdout.  With --write it puts MANIFEST.txt in the
campaign directory.

WHY THIS EXISTS.  On 2026-10-23 a second session found
/Volumes/wd_14tb/jpsi_plots, whose 1600 .f32 files are named EXACTLY like the
current ones, loaded them with a library pinned to the old 73-float record,
and ran a whole analysis on the superseded pre-22-October production without a
single error.  Nothing in the tree said which generation a file belonged to:
the directory was named after a plot book, the filenames carry no vintage, and
a library that expects the old record reads old files happily.  VPK's words
for the result were that nobody can tell who is who or where anything came
from.

Everything below is READ FROM THE FILES, never typed in: the provenance
sidecars already carry the git commit, the branch, the model and the cross
section, and the .f32 size gives the record length.  A manifest that is typed
can drift from its directory; one that is derived cannot.  The only
hand-written part is an optional NOTES.txt in the campaign directory, copied
in verbatim, for the history that is not in any file -- what this supersedes
and what is known to be wrong with it.
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime

# (short, long) record lengths per channel family, longest first
LAYOUT = {"jpsi": (80, 73), "phi": (80, 73), "rho": (80, 73),
          "pi0": (55, 50), "pi0n": (55, 50), "eta": (55, 50), "etan": (55, 50)}


def card_fields(path):
    """The positional values of a run card, comments stripped."""
    out = []
    try:
        for line in open(path):
            body = line.split("!")[0].strip()
            if body:
                out.append(body)
    except OSError:
        return []
    return out


def nfields(sz, channel):
    """Record length of a .f32 from its size, trying only this channel's two."""
    for n in LAYOUT.get(channel, (80, 73, 55, 50)):
        if sz % (n * 4) == 0:
            return n
    return None


def arm_report(root, arm, channel):
    """Everything derivable about one arm directory."""
    d = os.path.join(root, arm)
    r = {"arm": arm}
    provs = sorted(glob.glob(os.path.join(d, "prov", "*.config.json")))
    r["n_chunks"] = len(provs)
    if not provs:
        return r

    by_date = defaultdict(list)
    commits, branches, models, tags = Counter(), Counter(), Counter(), Counter()
    sigmas, accepted = [], []

    # THREADED, because this is the whole cost.  Each sidecar is a separate
    # small read on lustre and there are two per chunk; done serially, 1400 of
    # them took longer than the rest of the script by orders of magnitude and
    # the first run never finished.  The reads are independent, so a pool of
    # 32 turns minutes into seconds.  Nothing here writes.
    def read_one(f):
        try:
            idx = int(os.path.basename(f).split("_")[-1].split(".")[0])
        except ValueError:
            idx = -1
        mt = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d")
        try:
            return idx, mt, json.load(open(f))
        except Exception:
            return idx, mt, None

    with ThreadPoolExecutor(max_workers=32) as pool:
        loaded = list(pool.map(read_one, provs))

    for idx, mt, j in loaded:
        by_date[mt].append(idx)
        if j is None:
            continue
        st = j.get("stamp") or {}
        commits[st.get("git_commit", "?")] += 1
        branches[st.get("git_branch", "?")] += 1
        tags[st.get("version_tag", "?")] += 1
        sf = st.get("sf") or {}
        models[sf.get("model", "?")] += 1
        rep = j.get("report") or {}
        if rep.get("sigma_nb") is not None:
            sigmas.append((idx, float(rep["sigma_nb"])))
        if rep.get("accepted") is not None:
            accepted.append(int(rep["accepted"]))

    r["dates"] = {k: (min(v), max(v), len(v)) for k, v in sorted(by_date.items())}
    r["commits"] = commits.most_common()
    r["branches"] = branches.most_common()
    r["tags"] = tags.most_common()
    r["models"] = models.most_common()
    r["sigmas"] = sigmas
    r["accepted_total"] = sum(accepted)

    # the ntuples: record length and count, per output directory
    r["outs"] = {}
    for od in sorted(glob.glob(os.path.join(d, "out_*"))):
        fs = glob.glob(os.path.join(od, "*.f32"))
        if not fs:
            r["outs"][os.path.basename(od)] = (0, None, None)
            continue
        sizes = Counter(os.path.getsize(f) for f in fs)
        sz, _ = sizes.most_common(1)[0]
        nf = nfields(sz, channel)
        recs = sz // (nf * 4) if nf else None
        r["outs"][os.path.basename(od)] = (len(fs), nf, recs)
        if len(sizes) > 1:
            r["outs"][os.path.basename(od)] = (len(fs), nf, recs, dict(sizes))

    # scandir, not glob: it does one directory read instead of building and
    # filtering a full path list for every one of several hundred entries.
    n_dst = 0
    for dd in glob.glob(os.path.join(d, "dst_*")):
        try:
            n_dst += sum(1 for e in os.scandir(dd) if e.name.endswith(".hipo"))
        except OSError:
            pass
    r["n_dst"] = n_dst
    r["n_lund"] = len(glob.glob(os.path.join(d, "lund_osg", "*.lund")))
    return r


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = os.path.abspath(sys.argv[1].rstrip("/"))
    write = "--write" in sys.argv

    cards = sorted(glob.glob(os.path.join(root, "cards", "rga_*.txt")))
    model = ""
    for c in cards:
        f = card_fields(c)
        if f:
            model = f[0].split()[0]
            break
    channel = model.split(".")[0] if model else "?"

    arms = sorted(b for b in os.listdir(root)
                  if os.path.isdir(os.path.join(root, b))
                  and os.path.isdir(os.path.join(root, b, "prov")))

    L = []
    L.append("MANIFEST -- %s" % os.path.basename(root))
    L.append("=" * 78)
    L.append("")
    L.append("Derived from the directory's own files by")
    L.append("  production/analysis/make_manifest.py, %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    L.append("Nothing here is typed in except the NOTES section at the end.")
    L.append("")
    L.append("path      %s" % root)
    L.append("channel   %s" % channel)
    L.append("model     %s" % model)
    L.append("")

    if cards:
        L.append("CARDS -- the positional fields that differ between the arms")
        sets = {os.path.basename(c): card_fields(c) for c in cards}
        names = list(sets)
        if len(names) == 2:
            a, b = sets[names[0]], sets[names[1]]
            L.append("  %-10s %-28s %-28s" % ("field", names[0], names[1]))
            for i in range(max(len(a), len(b))):
                va = a[i] if i < len(a) else "(absent)"
                vb = b[i] if i < len(b) else "(absent)"
                if va != vb:
                    L.append("  %-10d %-28s %-28s" % (i + 1, va.split()[0] if va.split() else va,
                                                      vb.split()[0] if vb.split() else vb))
            L.append("")
            L.append("  Every field above is a difference between the two arms, so it is")
            L.append("  a physics setting the two-arm correction REMOVES.  Anything not")
            L.append("  listed is identical in both arms and therefore cancels.")
        L.append("")

    for arm in arms:
        r = arm_report(root, arm, channel)
        L.append("-" * 78)
        L.append("ARM  %s" % arm)
        if not r["n_chunks"]:
            L.append("  no provenance -- nothing generated here")
            L.append("")
            continue
        L.append("  chunks with provenance   %d" % r["n_chunks"])
        L.append("  LUND files               %d" % r["n_lund"])
        L.append("  DST files                %d" % r["n_dst"])
        L.append("  generated (by file date, chunk range, count):")
        for d, (lo, hi, n) in r["dates"].items():
            L.append("      %s   chunks %d-%d   %d" % (d, lo, hi, n))
        if len(r["dates"]) > 1:
            L.append("      NOTE: more than one date -- this arm MIXES vintages.")
        for label, key in (("code commit", "commits"), ("branch", "branches"),
                           ("version tag", "tags"), ("sf model", "models")):
            vals = r[key]
            if len(vals) == 1:
                L.append("  %-24s %s" % (label, vals[0][0]))
            else:
                L.append("  %-24s MIXED: %s" % (label, ", ".join("%s (%d)" % v for v in vals)))
        if any("dirty" in str(c[0]) for c in r["commits"]):
            L.append("      NOTE: a commit marked -dirty means the working tree had")
            L.append("            uncommitted changes when those chunks were generated.")
        if r["sigmas"]:
            s = [v for _, v in r["sigmas"]]
            L.append("  sigma_nb over all %d chunks   mean %.6g nb   min %.6g   max %.6g"
                     % (len(s), sum(s) / len(s), min(s), max(s)))
            for d, (lo, hi, _n) in r["dates"].items():
                sub = [v for i, v in r["sigmas"] if lo <= i <= hi]
                if sub and len(r["dates"]) > 1:
                    L.append("      chunks %d-%d (%s)   mean %.6g nb" % (lo, hi, d, sum(sub) / len(sub)))
            L.append("      Normalise with the range you actually analysed, not with all of it.")
        L.append("  ntuples:")
        for od, v in r["outs"].items():
            if v[0] == 0:
                L.append("      %-10s none" % od)
            elif len(v) == 4:
                L.append("      %-10s %4d files  MIXED RECORD LENGTHS %s -- refuse to read as one set"
                         % (od, v[0], v[3]))
            else:
                L.append("      %-10s %4d files  %s floats x %s records each" % (od, v[0], v[1], v[2]))
        L.append("")

    notes = os.path.join(root, "NOTES.txt")
    L.append("=" * 78)
    L.append("NOTES -- hand written, the history that is in no file")
    L.append("")
    if os.path.exists(notes):
        L.extend("  " + ln.rstrip() for ln in open(notes))
    else:
        L.append("  (no NOTES.txt in this directory.  Write one: what this supersedes,")
        L.append("   what supersedes it, and what is known to be wrong with it.)")
    L.append("")

    text = "\n".join(L) + "\n"
    if write:
        open(os.path.join(root, "MANIFEST.txt"), "w").write(text)
        print("wrote %s/MANIFEST.txt (%d lines)" % (root, len(L)))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
