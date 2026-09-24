#!/usr/bin/env python3
"""Append a finished run_prod_v2 run to an existing wd_14tb sample.

    append_run.py <sample dir> <channel> [--dry]

The v2 loop already writes exactly 5000 events per file, so this is not a
re-chunking: each incoming file becomes one output file, numbered on from the
end of the existing sequence, with the merge.* provenance header the rest of
the sample carries.  Sources are the `incoming_<host>/` directories the puller
filled; they are left in place until the append is verified.

Also rewrites manifest.csv in full -- md5 of every file, old ones included,
because the 2026-09-23/24 header repairs changed files the manifest had
already recorded.
"""
import csv
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_lund import merge, _header  # noqa: E402

SPINS = [("pos", "pos"), ("pos", "neg"), ("neg", "pos"), ("neg", "neg")]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(sample, channel, dry=False, only=None):
    incoming = sorted(glob.glob(f"{sample}/incoming_*"))
    if only:
        incoming = [d for d in incoming if os.path.basename(d) in only]
    print(f"sources: {', '.join(os.path.basename(d) for d in incoming)}")

    # Route by the file's OWN header, not by its path: run_prod_v2 writes the
    # spin into the directory tree, slurm_submit does not (it keeps the old
    # <round>/<combo>/<channel>/<date>_rad_v1/lund layout), and a path-based
    # glob silently skipped all 574 slurm files of the 21-23.09 run.
    bucket = {k: [] for k in SPINS}
    n_seen = 0
    for d in incoming:
        host = os.path.basename(d).replace("incoming_", "")
        for f in sorted(glob.glob(f"{d}/**/*.lund", recursive=True)):
            kv, _ = _header(f)
            ih, pt = kv.get("run.ihel"), kv.get("run.ptarg")
            if ih is None or pt is None:
                print(f"  ! no spin in header, skipped: {f}"); continue
            key = ("pos" if float(ih) > 0 else "neg", "pos" if float(pt) > 0 else "neg")
            bucket[key].append((f, host)); n_seen += 1
    print(f"  {n_seen} incoming files routed by header")

    added = []
    for bh, tp in SPINS:
        rel = f"beam_helicity_{bh}/target_pol_{tp}/rad"
        out = f"{sample}/{rel}"
        have = sorted(glob.glob(f"{out}/*.lund"))
        srcs = bucket[(bh, tp)]
        print(f"  {bh}/{tp}: have {len(have)}, appending {len(srcs)}")
        if dry or not srcs:
            continue
        rows, info = merge(srcs, out, f"{channel}_rad", chunk=5000,
                           start=len(have), verbose=True)
        for r in rows:
            r.update(beam_helicity=bh, target_pol=tp, mode="rad")
        added += rows
    if dry:
        return

    # --- manifest: keep what the old rows say about sources/seeds, refresh md5
    man = f"{sample}/manifest.csv"
    old = list(csv.DictReader(open(man))) if os.path.exists(man) else []
    fields = ["file", "beam_helicity", "target_pol", "mode", "events",
              "hosts", "sources", "seeds", "md5"]
    by_key = {(r["file"], r["beam_helicity"], r["target_pol"], r["mode"]): r for r in old}
    for r in added:
        by_key[(r["file"], r["beam_helicity"], r["target_pol"], r["mode"])] = r
    n_re = 0
    for k, r in by_key.items():
        f = f"{sample}/beam_helicity_{r['beam_helicity']}/target_pol_{r['target_pol']}/{r['mode']}/{r['file']}"
        if os.path.exists(f):
            m = md5(f)
            if m != r.get("md5"):
                n_re += 1
            r["md5"] = m
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in sorted(by_key):
            w.writerow({x: by_key[k].get(x, "") for x in fields})
    print(f"manifest: {len(by_key)} rows, {n_re} md5 refreshed (header repairs), "
          f"{len(added)} rows added")


if __name__ == "__main__":
    only = [a for a in sys.argv[3:] if a.startswith("incoming_")] or None
    main(sys.argv[1], sys.argv[2], "--dry" in sys.argv, only)
