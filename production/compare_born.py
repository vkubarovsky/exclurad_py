"""Born-only comparison: exclurad_py vs exclugen (pi0, eta) and diffrad_vm (rho, phi, jpsi).

No radiative corrections anywhere -- so any disagreement here is the Born
cross section, the virtual-photon flux convention, or the phase-space
sampling.  It cannot be the radiative tail.  That is the whole point.

Cuts are matched on both sides:
 * y, xB and (jpsi) |p|/theta of e', proton and both decay daughters are
   applied to exclurad_py at generation, from the reference run cards;
 * for exclugen the reference itself runs itarg=1 (external radiator), so
   events are required to have 10.6 - e_eff < 1 MeV, which keeps 96.9% of
   them and removes the external-radiation confound;
 * both samples are then restricted to the (Q2, W2, |t|) overlap.
All samples are UNWEIGHTED on both sides.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})
P = os.path.dirname(os.path.abspath(__file__))
M = 0.93827208816
E0 = 10.6

MESON_PID = {"pi0": 111, "eta": 221, "rho": 113, "phi": 333, "jpsi": 443}


def read_lund(path, meson_pid, need_eeff=False, max_ev=None):
    Q2 = []; W2 = []; t = []; keep = []
    with open(path) as fh:
        while True:
            l = fh.readline()
            if not l:
                break
            if l.startswith("#"):
                continue
            f = l.split()
            if len(f) < 10:
                continue
            try:
                npart = int(f[0])
            except ValueError:
                continue
            hdr_kin = len(f) >= 15
            scat = None; rec = None; eeff = E0
            for i in range(npart):
                p = fh.readline().split()
                if len(p) < 11:
                    continue
                pid = int(p[3]); st = int(p[2])
                px, py, pz, E = map(float, p[6:10])
                if i == 0 and st == 0 and pid == 11:
                    eeff = E        # exclugen writes a beam DOC line (status 0)
                elif pid == 11 and st == 1 and scat is None:
                    scat = (E, px, py, pz)
                elif pid == 2212:
                    rec = (E, px, py, pz)
            if scat is None or rec is None:
                continue
            if hdr_kin:
                W2.append(float(f[12])); Q2.append(float(f[13]))
            else:
                E, px, py, pz = scat
                q2 = 2 * E0 * E - 2 * E0 * pz
                nu = E0 - E
                Q2.append(q2); W2.append(M * M + 2 * M * nu - q2)
            t.append(2 * M * M - 2 * M * rec[0])
            keep.append((E0 - eeff) < 1e-3 if need_eeff else True)
            if max_ev and len(Q2) >= max_ev:
                break
    m = np.array(keep, dtype=bool)
    return (np.array(Q2)[m], np.array(W2)[m], -np.array(t)[m])


REF = {
 "pi0":  ("exclugen", sorted(glob.glob("/Volumes/wd_14tb/mc/exclugen/pi0/20260810_10.6_born_osg_v2/osg/*.lund")), True),
 "eta":  ("exclugen", sorted(glob.glob("/Volumes/wd_14tb/mc/exclugen/eta/20260810_10.6_born_osg_v2/osg/*.lund")), True),
 "rho":  ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/rho_born_100k.lund"], False),
 "phi":  ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/phi_born_100k.lund"], False),
 "jpsi": ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/jpsi_born_100k.lund"], False),
}

def mine_files(ch):
    d = f"/Volumes/wd_14tb/mc/exclurad_py/{ch}/20260813_10.6_born_v1/lund"
    return sorted(glob.glob(f"{d}/{ch}_exclurad_py_born_10.6GeV*.lund"))

chans = [c for c in sys.argv[1:] or list(REF) if mine_files(c)]
fig, ax = plt.subplots(len(chans), 3, figsize=(13, 3.4 * len(chans)), squeeze=False)

for row, ch in enumerate(chans):
    refname, files, need_eeff = REF[ch]
    mp = [read_lund(f, MESON_PID[ch]) for f in mine_files(ch)]
    mine = tuple(np.concatenate([q[i] for q in mp]) for i in range(3))
    parts = [read_lund(f, MESON_PID[ch], need_eeff) for f in files]
    ref = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    lo = [max(mine[i].min(), ref[i].min()) for i in range(3)]
    hi = [min(mine[i].max(), ref[i].max()) for i in range(3)]
    mm = np.ones(len(mine[0]), bool); rm = np.ones(len(ref[0]), bool)
    for i in range(3):
        mm &= (mine[i] >= lo[i]) & (mine[i] <= hi[i])
        rm &= (ref[i] >= lo[i]) & (ref[i] <= hi[i])

    print(f"=== {ch} BORN vs {refname}:  exclurad_py {mm.sum()}  ref {rm.sum()}")
    for i, lab in enumerate(("Q2", "W2", "|t|")):
        a, b = mine[i][mm], ref[i][rm]
        print(f"     {lab:>4}  mean {a.mean():9.4f} vs {b.mean():9.4f}  "
              f"ratio {a.mean()/b.mean():6.3f}   median {np.median(a):9.4f} vs {np.median(b):9.4f}")

    for j, (lab, unit) in enumerate((("Q^2", "GeV$^2$"), ("W^2", "GeV$^2$"), ("|t|", "GeV$^2$"))):
        A = ax[row, j]
        a, b = mine[j][mm], ref[j][rm]
        bins = np.linspace(min(a.min(), b.min()),
                           max(np.percentile(a, 99.5), np.percentile(b, 99.5)), 45)
        A.hist(b, bins=bins, density=True, histtype="step", lw=2.4, color=VERM,
               label=f"{refname}  (reference)")
        A.hist(a, bins=bins, density=True, histtype="step", lw=1.6, color=BLUE,
               ls="--", label="exclurad_py  (new)")
        A.set_yscale("log")
        A.set_xlabel(rf"${lab}$  [{unit}]"); A.set_ylabel("normalized")
        A.set_title(f"({chr(97 + row * 3 + j)})  {ch.upper()} BORN", loc="left",
                    fontsize=10, fontweight="bold")
        A.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
out = f"{P}/compare_born.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
