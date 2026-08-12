"""Compare mesonrc 100k samples against the diffrad_vm reference samples.

TWO THINGS THAT ARE NOT DIRECTLY COMPARABLE, and why:

1. The reference samples are WEIGHTED (weights span 0.006-78, mean 1.24).
   mesonrc events are unweighted.  Every reference distribution below is
   therefore filled with its weight; mesonrc's with unit weight.

2. The RADIATIVE FRACTION is a bookkeeping choice, not an observable.  It
   is fixed by where the soft/hard split v_min sits: mesonrc places it so
   the soft branch keeps ~35% of the cross section, diffrad_vm uses its own
   threshold.  Comparing "fraction of events carrying a photon" compares
   the two conventions, not the physics.  What IS comparable is the photon
   energy spectrum ABOVE A COMMON THRESHOLD, normalized to the same total.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_lund import read, invariants

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})
P = os.path.dirname(os.path.abspath(__file__))
REF = "/Users/vpk/MC_vector_mesons/samples_v2"
EGCUT = 0.15                     # common photon threshold, GeV

PAIRS = [("phi", 333, f"{REF}/phi_rc_100k.lund"),
         ("jpsi", 443, f"{REF}/jpsi_rc_100k.lund")]

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
row = 0
for ch, pid, refpath in PAIRS:
    kin = np.load(f"{P}/{ch}_kin.npz")
    ref = read(refpath, pid)
    rQ2, rW2, rt = invariants(ref)
    rw = ref["weight"]

    # restrict both to the overlap box so we compare the same region
    lo = dict(Q2=max(kin["Q2"].min(), rQ2.min()), W2=max(kin["W2"].min(), rW2.min()),
              mt=max((-kin["t"]).min(), (-rt).min()))
    hi = dict(Q2=min(kin["Q2"].max(), rQ2.max()), W2=min(kin["W2"].max(), rW2.max()),
              mt=min((-kin["t"]).max(), (-rt).max()))
    mk = ((kin["Q2"] >= lo["Q2"]) & (kin["Q2"] <= hi["Q2"]) &
          (kin["W2"] >= lo["W2"]) & (kin["W2"] <= hi["W2"]) &
          (-kin["t"] >= lo["mt"]) & (-kin["t"] <= hi["mt"]))
    mr = ((rQ2 >= lo["Q2"]) & (rQ2 <= hi["Q2"]) & (rW2 >= lo["W2"]) &
          (rW2 <= hi["W2"]) & (-rt >= lo["mt"]) & (-rt <= hi["mt"]))

    for j, (var_m, var_r, lab, logy) in enumerate([
            (kin["Q2"][mk], rQ2[mr], r"$Q^2$  [GeV$^2$]", True),
            (-kin["t"][mk], -rt[mr], r"$|t|$  [GeV$^2$]", True),
            (kin["E_gamma"][mk], ref["E_gamma"][mr], r"$E_\gamma$  [GeV]", True)]):
        a = ax[row, j]
        if j == 2:
            sm = var_m > EGCUT; sr = var_r > EGCUT
            b = np.linspace(EGCUT, min(np.percentile(var_r[sr], 99.5), 4.0), 40)
            a.hist(var_m[sm], bins=b, density=True, histtype="step", lw=2,
                   color=BLUE, label="mesonrc (unweighted)")
            a.hist(var_r[sr], bins=b, weights=rw[mr][sr], density=True,
                   histtype="step", lw=2, color=VERM, label="diffrad_vm (weighted)")
            a.set_yscale("log")
            a.set_title(f"({chr(97+row*3+j)})  {ch}: photon spectrum, "
                        f"$E_\\gamma>{EGCUT}$ GeV", loc="left", fontsize=9)
        else:
            b = np.linspace(min(var_m.min(), var_r.min()),
                            max(np.percentile(var_m, 99.5), np.percentile(var_r, 99.5)), 40)
            a.hist(var_m, bins=b, density=True, histtype="step", lw=2, color=BLUE,
                   label="mesonrc")
            a.hist(var_r, bins=b, weights=rw[mr], density=True, histtype="step",
                   lw=2, color=VERM, label="diffrad_vm")
            if logy:
                a.set_yscale("log")
            a.set_title(f"({chr(97+row*3+j)})  {ch}: {lab.split('[')[0].strip()}",
                        loc="left", fontsize=9)
        a.set_xlabel(lab); a.set_ylabel("normalized")
        if j == 0:
            a.legend(frameon=False, fontsize=8)

    fr_m = (kin["E_gamma"][mk] > EGCUT).mean()
    fr_r = np.average(ref["E_gamma"][mr] > EGCUT, weights=rw[mr])
    print(f"{ch}: overlap box Q2 {lo['Q2']:.3f}-{hi['Q2']:.3f}, "
          f"W2 {lo['W2']:.2f}-{hi['W2']:.2f}, |t| {lo['mt']:.3f}-{hi['mt']:.3f}")
    print(f"   fraction with E_gamma > {EGCUT} GeV :  mesonrc {fr_m:.4f}   "
          f"diffrad_vm {fr_r:.4f}   ratio {fr_m/fr_r:.3f}")
    sm = kin["E_gamma"][mk] > EGCUT; sr = ref["E_gamma"][mr] > EGCUT
    print(f"   median E_gamma above cut     :  mesonrc {np.median(kin['E_gamma'][mk][sm]):.4f}   "
          f"diffrad_vm {np.median(ref['E_gamma'][mr][sr]):.4f}")
    row += 1

fig.tight_layout()
fig.savefig(f"{P}/compare_vs_diffrad_vm.png", bbox_inches="tight")
print("wrote compare_vs_diffrad_vm.png")
