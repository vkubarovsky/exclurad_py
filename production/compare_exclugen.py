"""exclurad_py vs exclugen (OSG v2, 10.6 GeV, 100k each) for pi0 and eta.

exclugen = exact first-order RC built on EXCLURAD -- the same formalism
exclurad_py reproduces to 0.15-0.20% at the RC-factor level.  So this compares
the GENERATOR layer, with the RC engine already validated underneath.

Handled:
 * exclugen output is UNWEIGHTED (VPK: none of these generators weight).
   LUND header col 10 is `sigev`, the event cross section, written for
   information -- gen_lib.f90:386 -- NOT a weight. Cols 11-15 are
   xB, y, W2, Q2, nu, used directly instead of recomputing invariants.
 * E_gamma is the INTERNAL radiated photon (LUND particle 7, parent 0); the
   decay photons (parent 4) are excluded by the reader.
 * Comparison is restricted to the overlap box, since exclurad_py was generated
   on a narrower box than exclugen's full Q2 0.2-10 / W2 3.2-20.

NOT handled (a real difference, stated not corrected):
 * exclugen has PRE-VERTEX EXTERNAL RADIATION on (Tsai straggling in the LH2
   cell); exclurad_py is internal-only.  External loss shifts the effective beam
   energy per event but does not appear as particle 7, so the internal
   photon spectra remain comparable.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})
P = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for row, ch in enumerate(["pi0", "eta"]):
    m = np.load(f"{P}/{ch}_vs_exclugen_kin.npz")
    r = np.load(f"{P}/ref_{ch}_exclugen.npz")
    w = r["weight"]

    lo = dict(Q2=max(m["Q2"].min(), r["Q2"].min()), W2=max(m["W2"].min(), r["W2"].min()),
              mt=max((-m["t"]).min(), (-r["t"]).min()))
    hi = dict(Q2=min(m["Q2"].max(), r["Q2"].max()), W2=min(m["W2"].max(), r["W2"].max()),
              mt=min((-m["t"]).max(), (-r["t"]).max()))
    km = ((m["Q2"] >= lo["Q2"]) & (m["Q2"] <= hi["Q2"]) & (m["W2"] >= lo["W2"]) &
          (m["W2"] <= hi["W2"]) & (-m["t"] >= lo["mt"]) & (-m["t"] <= hi["mt"]))
    kr = ((r["Q2"] >= lo["Q2"]) & (r["Q2"] <= hi["Q2"]) & (r["W2"] >= lo["W2"]) &
          (r["W2"] <= hi["W2"]) & (-r["t"] >= lo["mt"]) & (-r["t"] <= hi["mt"]))

    print(f"=== {ch}: overlap Q2 {lo['Q2']:.2f}-{hi['Q2']:.2f}, "
          f"W2 {lo['W2']:.2f}-{hi['W2']:.2f}, |t| {lo['mt']:.3f}-{hi['mt']:.3f}")
    print(f"    exclurad_py {km.sum()} events, exclugen {kr.sum()} events "
          f"(sum w = {w[kr].sum():.4g})")
    for cut in (0.05, 0.15, 0.30, 0.60, 1.0):
        fm = (m["E_gamma"][km] > cut).mean()
        fr = (r["E_gamma"][kr] > cut).mean()
        print(f"      Eg>{cut:4.2f} : exclurad_py {fm:.4f}   exclugen {fr:.4f}   "
              f"ratio {fm/max(fr,1e-9):.2f}")

    for j, (vm, vr, lab, xlog) in enumerate([
            (m["Q2"][km], r["Q2"][kr], r"$Q^2$  [GeV$^2$]", False),
            (-m["t"][km], -r["t"][kr], r"$|t|$  [GeV$^2$]", False),
            (m["E_gamma"][km], r["E_gamma"][kr], r"$E_\gamma$  [GeV]", False)]):
        a = ax[row, j]
        if j == 2:
            sm, sr = vm > 0.05, vr > 0.05
            b = np.linspace(0.05, min(np.percentile(vr[sr], 99), 3.0), 40)
            a.hist(vm[sm], bins=b, density=True, histtype="step", lw=2, color=BLUE,
                   label="exclurad_py")
            a.hist(vr[sr], bins=b, density=True, histtype="step",
                   lw=2, color=VERM, label="exclugen")
            a.set_yscale("log")
        else:
            b = np.linspace(min(vm.min(), vr.min()),
                            max(np.percentile(vm, 99), np.percentile(vr, 99)), 40)
            a.hist(vm, bins=b, density=True, histtype="step", lw=2, color=BLUE,
                   label="exclurad_py")
            a.hist(vr, bins=b, density=True, histtype="step", lw=2,
                   color=VERM, label="exclugen")
            a.set_yscale("log")
        a.set_xlabel(lab); a.set_ylabel("normalized")
        a.set_title(f"({chr(97+row*3+j)})  {ch}", loc="left", fontsize=9)
        if j == 0:
            a.legend(frameon=False, fontsize=8)
    print()

fig.tight_layout()
fig.savefig(f"{P}/compare_vs_exclugen.png", bbox_inches="tight")
print("wrote compare_vs_exclugen.png")
