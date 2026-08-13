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

# Split by reference: pseudoscalars are compared to exclugen, vector
# mesons to diffrad_vm.  Two pages rather than one, since the groups share
# neither the reference nor the physics.
GROUPS = {"ps": ["pi0", "eta"], "vm": ["rho", "phi", "jpsi"]}
args = [a for a in sys.argv[1:] if not a.startswith("-")]
if args and args[0] in GROUPS:
    chans = [c for c in GROUPS[args[0]] if mine_files(c)]
    OUTNAME = f"compare_born_{args[0]}.png"
    TITLE = ("pseudoscalar mesons vs exclugen" if args[0] == "ps"
             else "vector mesons vs diffrad_vm")
else:
    chans = [c for c in args or list(REF) if mine_files(c)]
    OUTNAME, TITLE = "compare_born.png", "Born"
from matplotlib.gridspec import GridSpec

nrow = len(chans)
# 3 rows per channel: histogram, ratio, and a blank spacer so the next
# channel's title does not land on the previous ratio panel.
fig = plt.figure(figsize=(13, 4.4 * nrow))
gs = GridSpec(3 * nrow, 3, figure=fig,
              height_ratios=[3, 1, 0.55] * nrow, hspace=0.08, wspace=0.28)

for row, ch in enumerate(chans):
    refname, files, need_eeff = REF[ch]
    mp = [read_lund(f, MESON_PID[ch]) for f in mine_files(ch)]
    mine = tuple(np.concatenate([q[i] for q in mp]) for i in range(3))
    parts = [read_lund(f, MESON_PID[ch], need_eeff) for f in files]
    ref = tuple(np.concatenate([p_[i] for p_ in parts]) for i in range(3))

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
              f"ratio {a.mean()/b.mean():6.3f}")

    for j, (lab, unit) in enumerate((("Q^2", "GeV$^2$"), ("W^2", "GeV$^2$"),
                                     ("|t|", "GeV$^2$"))):
        A = fig.add_subplot(gs[3 * row, j])
        R = fig.add_subplot(gs[3 * row + 1, j], sharex=A)
        a, b = mine[j][mm], ref[j][rm]
        bins = np.linspace(lo[j], max(np.percentile(a, 99.5),
                                      np.percentile(b, 99.5)), 41)
        ha, _ = np.histogram(a, bins=bins)
        hb, _ = np.histogram(b, bins=bins)
        wa, wb = ha / ha.sum(), hb / hb.sum()
        ctr = 0.5 * (bins[:-1] + bins[1:])

        A.step(ctr, wb, where="mid", lw=2.4, color=VERM,
               label=f"{refname}  (reference)")
        A.step(ctr, wa, where="mid", lw=1.6, color=BLUE, ls="--",
               label="exclurad_py  (new)")
        A.set_yscale("log")
        A.set_ylabel("normalized")
        A.set_title(f"({chr(97 + row * 3 + j)})  {ch.upper()} BORN",
                    loc="left", fontsize=10, fontweight="bold")
        A.legend(frameon=False, fontsize=8, loc="upper right")
        A.tick_params(labelbottom=False)

        # ratio with Poisson errors on both samples
        ok = (ha > 0) & (hb > 0)
        r = np.full(len(ctr), np.nan); e = np.full(len(ctr), np.nan)
        r[ok] = wa[ok] / wb[ok]
        e[ok] = r[ok] * np.sqrt(1.0 / ha[ok] + 1.0 / hb[ok])
        R.axhline(1.0, color=GREY, lw=1, ls="--")
        R.errorbar(ctr[ok], r[ok], yerr=e[ok], fmt="o", ms=2.5, lw=0.8,
                   color=BLUE, capsize=0)
        R.set_ylim(0.4, 1.6)
        R.set_ylabel("new/ref", fontsize=8)
        R.set_xlabel(rf"${lab}$  [{unit}]")
        R.grid(alpha=0.25, lw=0.5)
    print()

# tight_layout fights the GridSpec hspace that pairs each histogram with
# its ratio panel; the spacing is set explicitly in the GridSpec instead.
fig.suptitle(f"BORN only (no radiative corrections) -- {TITLE}",
             fontsize=12, fontweight="bold", y=0.995)
out = f"{P}/{OUTNAME}"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
