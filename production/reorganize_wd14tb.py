"""Rebuild the September-2026 polarised campaign into the wd_14tb sample layout.

    python3 production/reorganize_wd14tb.py <channel> [--dry]

Layout written (see README at /Volumes/wd_14tb/exclurad_py/):

    exclurad_py/<model>/<channel>/<sample>/
        README.md  manifest.csv  cards/  tables/
        beam_helicity_{pos,neg}/target_pol_{pos,neg}/{born,rad}/<channel>_<mode>_NNNNNN.lund

Sources stay untouched.  Every source file is identified by the md5 of its
event text before it is used, so a file that reached the disk twice by two
sync paths (eta_n did) is consumed once.  Within a spin/mode the order is
fixed -- source directory, cycle, date directory, part -- so the merge is
reproducible.  merge_lund does the copying and the byte-level check.
"""
import csv
import glob
import hashlib
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_lund import merge, _header  # noqa: E402

D = "/Volumes/wd_14tb"
OUT = f"{D}/exclurad_py/amp2609"
SAMPLE = "E10.6_Q1-6_t0.1-2_vcut0_pol"
CHUNK = 5000

# channel -> (sub-directory name used by the generator, [(source root, host)])
SOURCES = {
    "pi0_p": ("pi0", [(f"{D}/pol_pi0_p_clonfarm11_20260918", "clonfarm11"),
                      (f"{D}/pol_pi0_p_slurm_20260918", "slurm")]),
    "eta_p": ("eta", [(f"{D}/pol_eta_p_ifarm_20260918", "ifarm2402")]),
    "pi0_n": ("pi0n", [(f"{D}/pol_pi0_n_20260918", "vpkmacmini")]),
    "eta_n": ("etan", [(f"{D}/pol_eta_n_20260918", "phallbvpk-mac"),
                       (f"{D}/pol_eta_n_20260919", "phallbvpk-mac")]),
}
SPIN = {"hp_pp": ("beam_helicity_pos", "target_pol_pos"),
        "hp_pm": ("beam_helicity_pos", "target_pol_neg"),
        "hm_pp": ("beam_helicity_neg", "target_pol_pos"),
        "hm_pm": ("beam_helicity_neg", "target_pol_neg")}


def sort_key(path):
    """(cycle, date-dir, part) -- cycle 0 for slurm-style trees without cycles."""
    m = re.search(r"/cyc(\d+)/", path)
    cyc = int(m.group(1)) if m else 0
    m = re.search(r"/(\d{8})_[\d.]+_(?:born|rad)_v1/", path)
    date = m.group(1) if m else ""
    m = re.search(r"_(\d+)\.lund$", path)
    part = int(m.group(1)) if m else 0
    return (cyc, date, part)


def body_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for line in fh:
            if not line.startswith(b"#"):
                h.update(line)
    return h.hexdigest()


UNPOL = {   # the 15.09 campaign: ihel=0 (helicity-averaged), no target pol, vcut 0.2/0.3
    "pi0_p": ("pi0", f"{D}/amp2609_prod_20260915/pi0", "E10.6_Q1.2-8_t0.001-1.8_vcut0.2_unpol"),
    "eta_p": ("eta", f"{D}/amp2609_prod_20260915/eta", "E10.6_Q1.2-8_t0.001-1.8_vcut0.3_unpol"),
}


def main(channel, dry=False, unpol=False):
    if unpol:
        gen_ch, root, sample = UNPOL[channel]
        roots = [(root, "clonfarm11")]
        spins = {"avg": ("beam_helicity_zero", "target_pol_zero")}
        extra = (("run.ihel", "0"), ("run.ptarg", "0.0"))
        combo_dir = ""          # no per-combo subdirectory in that layout
    else:
        gen_ch, roots = SOURCES[channel]
        sample, spins, extra, combo_dir = SAMPLE, SPIN, (), None
    out = f"{OUT}/{channel}/{sample}"
    os.makedirs(out, exist_ok=True)
    manifest, summary = [], []
    seen_md5 = {}
    dups = []
    t0 = time.time()
    for combo, (hdir, tdir) in spins.items():
        for mode in ("born", "rad"):
            srcs = []
            for root, host in roots:
                sub = combo if combo_dir is None else combo_dir
                pat = f"{root}/{sub}/**/{gen_ch}/*_{mode}_v1/lund/*.lund" if sub else f"{root}/**/{gen_ch}/*_{mode}_v1/lund/*.lund"
                for f in sorted(glob.glob(pat, recursive=True), key=sort_key):
                    md5 = body_md5(f)
                    if md5 in seen_md5:
                        dups.append((f, seen_md5[md5]))
                        continue
                    seen_md5[md5] = f
                    srcs.append((f, host))
            if not srcs:
                print(f"  {combo}/{mode}: no sources")
                continue
            odir = f"{out}/{hdir}/{tdir}/{mode}"
            prefix = f"{channel}_{mode}"
            print(f"{combo}/{mode}: {len(srcs)} source files -> {odir}")
            if dry:
                continue
            rows, info = merge(srcs, odir, prefix, chunk=CHUNK, extra=extra)
            for r in rows:
                r.update(beam_helicity=hdir.split("_")[-1], target_pol=tdir.split("_")[-1], mode=mode)
                manifest.append(r)
            summary.append(dict(combo=combo, beam=hdir, target=tdir, mode=mode, **info))
    if dry:
        print(f"dry run; {len(dups)} duplicate source files would be skipped")
        return
    # --- cards and tables: from the first source root, one per combo
    os.makedirs(f"{out}/cards", exist_ok=True); os.makedirs(f"{out}/tables", exist_ok=True)
    for combo, (hdir, tdir) in spins.items():
        root = roots[0][0]
        if unpol:
            cards = sorted(glob.glob(f"{root}/**/inputs/gen_input.txt", recursive=True), key=sort_key)
            for c in cards[:2]:
                shutil.copy2(c, f"{out}/cards/{hdir}_{tdir}_{os.path.basename(os.path.dirname(os.path.dirname(c)))}_gen_input.txt")
            tabs = sorted(glob.glob(f"{root}/**/etatable.npz", recursive=True), key=sort_key)
            if tabs: shutil.copy2(tabs[0], f"{out}/tables/{hdir}_{tdir}_etatable.npz")
            continue
        for c in (f"{root}/{combo}.txt", f"{root}/{combo}/card_born.txt", f"{root}/{combo}/card_rad.txt",
                  f"{root}/{combo}/cyc01/card_born.txt", f"{root}/{combo}/cyc01/card_rad.txt"):
            if os.path.exists(c):
                shutil.copy2(c, f"{out}/cards/{hdir}_{tdir}_{os.path.basename(c)}")
        tabs = sorted(glob.glob(f"{root}/{combo}/**/etatable.npz", recursive=True), key=sort_key)
        if tabs:
            shutil.copy2(tabs[0], f"{out}/tables/{hdir}_{tdir}_etatable.npz")
    # --- manifest
    with open(f"{out}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "beam_helicity", "target_pol", "mode", "events",
                                           "hosts", "sources", "seeds", "md5"])
        w.writeheader()
        for r in manifest:
            w.writerow({k: r[k] for k in w.fieldnames})
    # --- README
    kv, _ = _header(srcs[0][0])
    tot = {"born": 0, "rad": 0}
    lines = [f"# {channel} — amp2609 — {sample}", "",
             f"Generated {'2026-09-15..18' if unpol else '2026-09-18..21'} with exclurad_py (exact Bardin–Shumeiko RC), model `{kv.get('sf.model')}`.",
             f"Beam {kv.get('run.Ebeam')} GeV.  Box: Q2 {kv.get('run.box.Q2')}, W2 {kv.get('run.box.W2')}, "
             f"t {kv.get('run.box.t')} GeV2; vcut {kv.get('run.vcut')}"
             + (" (no cut on the radiative tail); " if float(kv.get('run.vcut', 0)) == 0 else " (v = MX2 - m_miss2 cut on the radiative tail); ")
             f"y 0.25–0.90, xB 0.13–0.60; idecay 1; pre-vertex radiator on; "
             + ("beam helicity averaged (ihel=0), unpolarised target." if unpol else "|target polarisation| = 1."),
             f"Model validity: W2 {kv.get('sf.validity.W2')}, Q2 {kv.get('sf.validity.Q2')}, |t| {kv.get('sf.validity.mt')} "
             f"(out of range: {kv.get('sf.out_of_range')}).", "",
             "Every file holds exactly 5000 events; the final partial chunk of each stream was discarded.",
             "Events were copied byte for byte from the source runs; only the comment header was rewritten,",
             "and it lists the source files, event ranges, hosts and code versions of that chunk.",
             "`manifest.csv` has the same per file, with seeds and md5.", "",
             "## Sources", ""]
    for root, host in roots:
        lines.append(f"- `{root}` — {host}")
    if dups:
        lines.append(f"- {len(dups)} source files were present twice (two sync paths) and consumed once; "
                     f"see manifest for the copy used.")
    lines += ["", "## Contents", "", "| beam helicity | target pol | mode | files | events | discarded tail | hosts | code |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for s in summary:
        tot[s["mode"]] += s["n_out"]
        lines.append(f"| {s['beam'].split('_')[-1]} | {s['target'].split('_')[-1]} | {s['mode']} | {s['files']} | "
                     f"{s['n_out']} | {s['tail']} | {' ; '.join(s['hosts'])} | {' ; '.join(s['tags'])} |")
    lines += ["", f"**Total: born {tot['born']:,} · rad {tot['rad']:,}**", "",
              "## Verification", "",
              "For every stream the sha256 of the source event text (minus the discarded tail) equals the sha256 of",
              "the event text re-read from the written files:", ""]
    for s in summary:
        lines.append(f"- {s['beam']}/{s['target']}/{s['mode']}: `{s['sha256'][:16]}…` ({s['n_in']} in, {s['n_out']} out)")
    lines += ["", f"Built {time.strftime('%Y-%m-%d %H:%M')} by production/reorganize_wd14tb.py in {time.time()-t0:.0f} s.",
              "Header note: `sf.t_nucl` in the source headers was runtime state (last event's t), not the card value; dropped.", ""]
    with open(f"{out}/README.md", "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {out}/README.md and manifest.csv; total born {tot['born']}, rad {tot['rad']}; "
          f"{len(dups)} duplicate source files skipped; {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], dry="--dry" in sys.argv, unpol="--unpol" in sys.argv)
