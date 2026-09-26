#!/usr/bin/env python3
"""numbers.json (from jpsi_photo_report_figs.py) -> LaTeX macros for the report."""
import json
import sys

n = json.load(open(sys.argv[1]))
out = []


def m(name, val, fmt="%.4f"):
    out.append("\\newcommand{\\%s}{%s}" % (name, fmt % val if not isinstance(val, str) else val))


r = n["ratio"]
m("nD", n["d2"]["n"], "%d"); m("nN", n["n"]["n"], "%d"); m("nP", n["p"]["n"], "%d")
m("nDM", n["d2"]["n"] / 1e6, "%.0f")
m("fitAt", r["fit_at_9p5"]); m("fitErr", r["fit_err"]); m("fitSlope", r["slope_per_GeV"]); m("fitSlopeErr", r["slope_err"])
m("fitChi", r["chi2"], "%.1f"); m("fitNdf", r["ndf"], "%d")
m("totalRatio", r["total_ratio"]); m("noFluxTotal", r["noflux_total"]); m("meanFlux", r["mean_flux_factor"])
for key, tag in (("8.2-8.4", "A"), ("8.4-8.6", "B"), ("8.6-9.0", "C"), ("9.0-9.5", "D"), ("9.5-10.0", "E"), ("10.0-10.6", "F")):
    b = r[key]
    m("rat" + tag, b["ratio"]); m("ratErr" + tag, b["err"]); m("ratNF" + tag, b["noflux_ratio"])
    m("yOn" + tag, b["on"] * 1e6, "%.2f"); m("yOff" + tag, b["off"] * 1e6, "%.2f")
f = n["flux"]
m("photonsInRange", f["photons_per_e_LD2_6p5_10p6"] * 1e3, "%.3f"); m("photonsAboveTh", f["photons_per_e_LD2_above_th"] * 1e3, "%.3f")
m("tipLoss", 100 * f["endpoint_reduction_6p5_10p6"], "%.2f")
for d, v in f["S_at"].items():
    m("Sat" + {"300": "iii", "100": "i", "30": "xxx", "10": "x", "3": "c", "1": "a"}[d], v, "%.3f")
b = n["below"]
m("fracBelow", 100 * b["frac"], "%.2f"); m("yieldBelow", b["yield_nb"] * 1e6, "%.2f")
m("fracBelowEight", 100 * n["d2"]["frac_below_80"], "%.2f"); m("fracBelowSeven", 100 * n["d2"]["frac_below_75"], "%.2f")
for key, tag in (("6.5-7.0", "A"), ("7.0-7.5", "B"), ("7.5-8.0", "C"), ("8.0-8.2", "D")):
    m("cnt" + tag, b["counts"][key], "%d")
sp = n["spectator"]
for key, tag in (("above", "Ab"), ("below", "Be"), ("below_8.0", "Ei"), ("below_7.5", "Se")):
    m("ps" + tag, sp[key]["ps_mean_MeV"], "%.0f"); m("fps" + tag, 100 * sp[key]["frac_gt300"], "%.1f"); m("ms" + tag, sp[key]["mstar_mean"], "%.3f")
    m("nsp" + tag, sp[key]["n"], "%d")
for x, tag in (("d2", "D"), ("n", "N"), ("p", "P")):
    m("yield" + tag, n[x]["yield_nb"] * 1e6, "%.2f"); m("sig" + tag, n[x]["sigma_flux_weighted_nb"], "%.4f")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[2], len(out), "macros")
