#!/usr/bin/env python3
"""LaTeX numbers and tables for the report, from summary.json and nothing else.

    python3 production/analysis/rc_method_jpsi_tables.py docs/rc_method_jpsi

Writes numbers.tex (one \\newcommand per quoted number) and tab_*.tex, which
the report \\input's.  A number in the text that is not here is a number that
can drift from the figures.
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "docs/rc_method_jpsi"
s = json.load(open(os.path.join(D, "summary.json")))
POLS = ("inb", "outb")
TOP = {"3trk": "e'e^+e^-", "4trk": "e'e^+e^-p'"}

N = {}
def num(name, val, fmt="%.4f"):
    N[name] = fmt % val

num("Ffsr", s["F_fsr"])
num("hard", s["effective"]["hard"])
num("hardpct", 100 * (1 - s["effective"]["hard"]), "%.2f")
for t, tag in (("3trk", "Three"), ("4trk", "Four")):
    e = s["effective"][t]
    num(f"Kfifty{tag}", e["K50"], "%.3f"); num(f"vfifty{tag}", e["v50"], "%.2f"); num(f"Ffifty{tag}", e["F50"])
    p = s["topo"][f"{t}_polarity"]
    num(f"polchi{tag}", p["chi2"], "%.0f"); num(f"poln{tag}", p["n"], "%d")
    num(f"polmean{tag}", p["mean_diff_pct"], "%+.1f")
for pol, tag in (("inb", "In"), ("outb", "Out")):
    num(f"vher{tag}", s["sample"][f"3trk_{pol}"]["v_her"], "%.3f")
    for t, tt in (("3trk", "Three"), ("4trk", "Four")):
        sm = s["sample"][f"{t}_{pol}"]
        num(f"ngenRad{tag}", sm["rad"]["n_gen"], "%d"); num(f"nchRad{tag}", sm["rad"]["n_chunks"], "%d")
        num(f"ngenBorn{tag}", sm["born"]["n_gen"], "%d"); num(f"nchBorn{tag}", sm["born"]["n_chunks"], "%d")
        num(f"sigratio{tag}", sm["rad"]["sigma_nb"] / sm["born"]["sigma_nb"], "%.5f")
        num(f"nsel{tt}{tag}", sm["rad"]["n_sel"], "%d"); num(f"nselB{tt}{tag}", sm["born"]["n_sel"], "%d")
        num(f"Bher{tt}{tag}", s["cutvars"][f"{t}_{pol}"]["mariana"]["B"])
        num(f"Bherr{tt}{tag}", s["cutvars"][f"{t}_{pol}"]["mariana"]["B_err"])
sr, sb = s["sample"]["3trk_inb"]["rad"], s["sample"]["3trk_inb"]["born"]
kept = (sr["sigma_nb"] / sr["sigma_pre"]) / (sb["sigma_nb"] / sb["sigma_pre"])
num("keptRad", sr["sigma_nb"] / sr["sigma_pre"]); num("keptBorn", sb["sigma_nb"] / sb["sigma_pre"])
num("keptratio", kept); num("keptpct", 100 * (1 - kept), "%.2f")
num("sigpreIn", sr["sigma_pre"] / sb["sigma_pre"], "%.5f")
num("expectclos", 100 * (s["effective"]["hard"] * kept - 1), "%+.2f")
mu, sg = s["sample"]["3trk_inb"]["born"]["cal"]
num("mxmu", mu, "%.3f"); num("mxsig", sg, "%.3f")
cl = s["closure"]["t_proton"]["integrated"]
num("closB", cl["B"], "%.4f"); num("closA", cl["A"], "%.4f"); num("closdev", 100 * (cl["B"] / cl["A"] - 1), "%+.2f")
g = s["gencuts"]
num("Fgen", g["fsr_truth"]["F_gen"]); num("mxsoft", 100 * (1 - g["fsr_truth"]["mx_keeps_soft"]), "%.1f")
for t, tt in (("3trk", "Three"), ("4trk", "Four")):
    ch = g["chains"][t]["her window"]
    keys = list(ch)
    num(f"gnocut{tt}", ch["no cut"]["B"]); num(f"gmee{tt}", ch["M(ee)"]["B"])
    last = [k for k in keys if k.startswith("+")][-1]
    num(f"gfull{tt}", ch[last]["B"])
# the production part alone (electron line + pre-vertex radiator), her bins,
# 3 tracks inbending, at her MX cut: the number her 0.85 stands for
for i, (q, tag) in enumerate(zip([q for q in s["mariana"]["3trk"] if q["pol"] == "inb"], ("One", "Two", "Three"))):
    num(f"prod{tag}", q["A_int"] * q["ext"] * s["effective"]["hard"], "%.3f")
for i, (v, tag) in enumerate(zip(s["gen10M_corrected"]["values"], ("One", "Two", "Three"))):
    num(f"genc{tag}", v[0], "%.3f")
with open(os.path.join(D, "numbers.tex"), "w") as f:
    for k, v in N.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))

# ---- her three W bins -------------------------------------------------------
gen = s["gen10M_corrected"]["values"]
with open(os.path.join(D, "tab_mariana.tex"), "w") as f:
    f.write("\\begin{tabular}{llcccccc}\\toprule\n")
    f.write("topology & W (GeV) & method 1 inb & method 1 outb & method 2 nominal & method 2 effective & generated 10M$^*$ & M.~Tenorio \\\\\\midrule\n")
    for t in ("3trk", "4trk"):
        rows = {(q["pol"], tuple(q["W"])): q for q in s["mariana"][t]}
        for i, W in enumerate(sorted({tuple(q["W"]) for q in s["mariana"][t]})):
            a, b = rows[("inb", W)], rows[("outb", W)]
            f.write(f"${TOP[t]}$ & {W[0]:.2f}--{W[1]:.2f} & {a['B']:.3f}$\\pm${a['B_err']:.3f} & "
                    f"{b['B']:.3f}$\\pm${b['B_err']:.3f} & {a['A']:.3f} & {a['A_eff']:.3f} & "
                    f"{gen[i][0]:.3f} & 0.75 \\\\\n")
        f.write("\\midrule\n" if t == "3trk" else "")
    f.write("\\bottomrule\\end{tabular}\n")

# ---- chi2 of method 1 against effective method 2 -----------------------------
with open(os.path.join(D, "tab_chi2.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrr}\\toprule\n")
    f.write("topology & binning & $\\chi^2$/cells inb & mean dev.\\ inb & $\\chi^2$/cells outb & mean dev.\\ outb \\\\\\midrule\n")
    lab = {"W": "W", "Q2": "$Q^2$", "mt": "$|t|$", "maps": "$(W,|t|)$ at 3 $Q^2$", "mariana": "her 3 W bins"}
    for t in ("3trk", "4trk"):
        for b in ("W", "Q2", "mt", "maps", "mariana"):
            a = s["chi2_eff"][f"{t}_inb_{b}"]; o = s["chi2_eff"][f"{t}_outb_{b}"]
            f.write(f"${TOP[t]}$ & {lab[b]} & {a['chi2']:.1f}/{a['n']} & {a['wmean_dev_pct']:+.1f}\\% & "
                    f"{o['chi2']:.1f}/{o['n']} & {o['wmean_dev_pct']:+.1f}\\% \\\\\n")
        f.write("\\midrule\n" if t == "3trk" else "")
    f.write("\\bottomrule\\end{tabular}\n")

# ---- N-1 per cut ---------------------------------------------------------------
with open(os.path.join(D, "tab_cutvars.tex"), "w") as f:
    for t in ("3trk", "4trk"):
        names = list(s["cutvars"][f"{t}_inb"]["mariana"]["cuts"])
        f.write("\\begin{tabular}{lcccc}\\toprule\n")
        f.write(f"${TOP[t]}$: cut & rad keeps (inb) & Born keeps (inb) & $r$ inb & $r$ outb \\\\\\midrule\n")
        for n in names:
            a = s["cutvars"][f"{t}_inb"]["mariana"]["cuts"][n]; o = s["cutvars"][f"{t}_outb"]["mariana"]["cuts"][n]
            f.write(f"{n.replace('_', ' ').replace('|', '$|$').replace('>', '$>$')} & {a['f_rad']:.4f} & {a['f_born']:.4f} & "
                    f"{a['r']:.4f}$\\pm${a['r_err']:.4f} & {o['r']:.4f}$\\pm${o['r_err']:.4f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n\n\\medskip\n\n")

# ---- generator-level chain ----------------------------------------------------
with open(os.path.join(D, "tab_gencuts.tex"), "w") as f:
    f.write("\\begin{tabular}{llcccc}\\toprule\n")
    f.write("topology & step & her window & W 4.15--4.29 & W 4.29--4.40 & W 4.40--4.48 \\\\\\midrule\n")
    for t in ("3trk", "4trk"):
        ch = s["gencuts"]["chains"][t]
        for step in ch["her window"]:
            f.write(f"${TOP[t]}$ & {step} & " + " & ".join(f"{ch[w][step]['B']:.4f}" for w in ch) + " \\\\\n")
        f.write("\\midrule\n" if t == "3trk" else "")
    f.write("\\bottomrule\\end{tabular}\n")
print("written", D)
