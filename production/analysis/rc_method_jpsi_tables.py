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


# =============================================================================
# C_rad TABLES FOR THE CROSS SECTION, method 1, from the combos run itself
# =============================================================================
# F18 combined as her summed yields combine the two polarities: each polarity
# enters with weight  Q_pol x Acc_Born,pol(bin),  Acc_Born = n_born / N_gen,born,
# charges from her note, Table 1 (F18 inbending 35.0234, outbending 32.7332).
CHARGE = {"inb": 35.0234, "outb": 32.7332}
RUN = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/wd_14tb/rc_method_jpsi/out"
cmb = json.load(open(os.path.join(RUN, "rc_method_jpsi_combos.json")))


def cellmap(topo, pol, binning):
    return {json.dumps(c["range"], sort_keys=True): c for c in cmb["cells"][f"{topo}_{pol}"] if c["binning"] == binning}


def combine(ci, co, topo):
    # one polarity below threshold: the other alone, flagged -- the two agree
    # within errors everywhere they both exist (polarity chi2 51/67, 32/50)
    if "B" not in ci and "B" not in co:
        return None
    if "B" not in ci or "B" not in co:
        c = ci if "B" in ci else co
        return c["B"], c["B_err"], True
    w = {}
    for pol, c in (("inb", ci), ("outb", co)):
        ngen = cmb["sample"][f"{topo}_{pol}"]["born"]["n_gen"]
        w[pol] = CHARGE[pol] * c["n_born"] / ngen
    W = w["inb"] + w["outb"]
    C = (w["inb"] * ci["B"] + w["outb"] * co["B"]) / W
    E = ((w["inb"] * ci["B_err"]) ** 2 + (w["outb"] * co["B_err"]) ** 2) ** 0.5 / W
    return C, E, False


def fmt(c):
    """Value +- error; italic when the error exceeds 10 % (not a measurement), dagger when one polarity only."""
    if c is None:
        return "---"
    t = f"{c[0]:.3f}$\\pm${c[1]:.3f}"
    if c[1] > 0.10 * c[0]:
        t = f"\\textit{{{t}}}"
    if len(c) > 2 and c[2]:
        t += "$^\\dagger$"
    return t


CSV = ["topology,binning,Q2_lo,Q2_hi,W_lo,W_hi,t_lo,t_hi,C_inb,err_inb,C_outb,err_outb,C_F18,err_F18,F18_one_polarity,n_rad_inb,n_rad_outb"]


def rows(topo, binning):
    ci, co = cellmap(topo, "inb", binning), cellmap(topo, "outb", binning)
    out = []
    for k in ci:
        a, b = ci[k], co.get(k, {})
        f = combine(a, b, topo)
        rg = a["range"]
        out.append((rg, (a["B"], a["B_err"]) if "B" in a else None,
                    (b["B"], b["B_err"]) if "B" in b else None, f, a["n_rad"], b.get("n_rad", 0)))
        g = lambda v, i: "" if v not in rg else rg[v][i]
        val = lambda x, i: "" if x is None else f"{x[i]:.5f}"
        aa = (a["B"], a["B_err"]) if "B" in a else None
        bb = (b["B"], b["B_err"]) if "B" in b else None
        CSV.append(",".join(map(str, [topo, binning, g("Q2", 0), g("Q2", 1), g("W", 0), g("W", 1), g("mt", 0), g("mt", 1),
                                       val(aa, 0), val(aa, 1), val(bb, 0), val(bb, 1), val(f, 0), val(f, 1),
                                       "" if f is None else int(f[2]),
                                       a["n_rad"], b.get("n_rad", 0)])))
    return out


def rng(r, v, d=3):
    return f"{r[v][0]:.{d}f}--{r[v][1]:.{d}f}"


HEAD = "inbending & outbending & \\textbf{F18 combined} & $n_{\\rm rad}$ \\\\\\midrule\n"


def table(fname, spec):
    """spec: list of (topology, binning, column labels, column formatter)."""
    with open(os.path.join(D, fname), "w") as f:
        for topo, binning, cols, cf in spec:
            f.write("\\begin{tabular}{" + "l" * len(cols) + "cccr}\\toprule\n")
            f.write(f"\\multicolumn{{{len(cols) + 4}}}{{l}}{{${TOP[topo]}$}}\\\\\n")
            f.write(" & ".join(cols) + " & " + HEAD)
            for r_, a, b, c, na, nb in rows(topo, binning):
                f.write(" & ".join(cf(r_)) + f" & {fmt(a)} & {fmt(b)} & \\textbf{{{fmt(c)}}} & {na + nb} \\\\\n")
            f.write("\\bottomrule\\end{tabular}\n\n\\medskip\n\n")


both = ("3trk", "4trk")
table("tab_crad_hardW.tex", [(t, "mariana", ["W (GeV)"], lambda r: [rng(r, "W", 2)]) for t in both])
table("tab_crad_eff.tex", [("3trk", "her_eff3", ["$Q^2$ (GeV$^2$)", "W (GeV)"], lambda r: [rng(r, "Q2", 4), rng(r, "W", 4)]),
                           ("4trk", "her_eff4", ["$Q^2$ (GeV$^2$)", "W (GeV)"], lambda r: [rng(r, "Q2", 4), rng(r, "W", 4)])])
table("tab_crad_Q2.tex", [(t, "her_Q2", ["$Q^2$ (GeV$^2$)"], lambda r: [rng(r, "Q2", 2)]) for t in both])
table("tab_crad_WQ2.tex", [(t, "her_WQ2", ["$Q^2$ (GeV$^2$)", "W (GeV)"], lambda r: [rng(r, "Q2", 2), rng(r, "W", 2)]) for t in both])
for t in both:
    table(f"tab_crad_maps_{t}.tex",
          [(t, f"map_Q2_{a}_{b}", ["$Q^2$ (GeV$^2$)", "W (GeV)", "$|t|$ (GeV$^2$)"],
            lambda r: [rng(r, "Q2", 3), rng(r, "W", 2), rng(r, "mt", 2)]) for a, b in cmb["qslices"]])
for t in both:
    table(f"tab_crad_1d_{t}.tex", [(t, v, [lab], (lambda r, v=v, d=d: [rng(r, v, d)]))
                                   for v, lab, d in (("W", "W (GeV)", 2), ("Q2", "$Q^2$ (GeV$^2$)", 3), ("mt", "$|t|$ (GeV$^2$)", 1))])
with open(os.path.join(D, "crad_tables.csv"), "w") as f:
    f.write("\n".join(CSV) + "\n")
print("C_rad tables written:", len(CSV) - 1, "rows in crad_tables.csv")
