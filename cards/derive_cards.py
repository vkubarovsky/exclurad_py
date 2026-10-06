#!/usr/bin/env python3
"""Derive a set of run cards for one model vintage from the reference cards.

    python3 cards/derive_cards.py amp2614

Written because a version tag that is typed twice gets out of step.  On
2026-10-05 night_2614.sh was made from night_2613.sh with
sed s/amp2613/amp2614/g, which left --job-name=g13_* untouched -- that string
contains no "amp2613" -- so an amp2614 campaign ran under names claiming the
previous model.  Here the vintage is an argument and appears once.

Substitution is by FIELD INDEX, the way exclurad_py/runcard.py reads the file
(field 1 = model, 4 = iborn, 13 = outroot).  Matching on the value is what made
's|^5000$|...|' silently fail against "5000    ! nev".

The output root is made RELATIVE on purpose: the reference neutron cards carry
one person's /volatile path, and anyone else running them straight gets a
PermissionError from os.makedirs before anything happens.
"""
import os
import sys

VINT = sys.argv[1] if len(sys.argv) > 1 else "amp2614"
HERE = os.path.dirname(os.path.abspath(__file__))

# (parent card, channel prefix, iborn, output name)
JOBS = [("rad_pi0_rga.txt",       "pi0",  0, f"rad_pi0_rga_{VINT}.txt"),
        ("born_pi0_rga.txt",      "pi0",  1, f"born_pi0_rga_{VINT}.txt"),
        ("rad_eta_rga.txt",       "eta",  0, f"rad_eta_rga_{VINT}.txt"),
        ("born_eta_rga.txt",      "eta",  1, f"born_eta_rga_{VINT}.txt"),
        ("rad_pi0n_rgb_none.txt", "pi0n", 0, f"rad_pi0n_rgb_{VINT}.txt"),
        ("rad_pi0n_rgb_none.txt", "pi0n", 1, f"born_pi0n_rgb_{VINT}.txt"),
        ("rad_etan_rgb_none.txt", "etan", 0, f"rad_etan_rgb_{VINT}.txt"),
        ("rad_etan_rgb_none.txt", "etan", 1, f"born_etan_rgb_{VINT}.txt")]

HDR = (f"! DERIVED by cards/derive_cards.py {VINT} from {{parent}} -- do not edit\n"
       f"! the model line by hand; re-derive instead.  Only the model and iborn\n"
       f"! differ from the parent, plus a relative output root.\n")

for parent, ch, ib, out in JOBS:
    src = open(os.path.join(HERE, parent)).read().split("\n")
    res, k = [], 0
    for line in src:
        bare = line.split("!")[0]
        if bare.strip():
            k += 1
            if k == 1:
                res.append(f"{ch}.{VINT}".ljust(35) + f"! model  <channel>.<vintage>")
                continue
            if k == 4:
                res.append(f"{ib}".ljust(35) +
                           "! iborn   1 = Born only, 0 = full RC")
                continue
            if k == 13:
                res.append("./exclurad_out".ljust(35) +
                           "! output root -- RELATIVE; the wrapper substitutes it")
                continue
        res.append(line)
    with open(os.path.join(HERE, out), "w") as f:
        f.write(HDR.format(parent=parent) + "\n".join(res))
    print(f"  {out:34} <- {parent:24} {ch}.{VINT}  iborn={ib}")
print(f"\n{len(JOBS)} cards derived for {VINT}")
