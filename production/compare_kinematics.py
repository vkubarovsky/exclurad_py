"""Particle-level comparison: exclurad_py vs the reference generator.

Layout follows ~/MC_vector_mesons/<ch>/plot_compare.py `make_kinematics_plot`
-- one row per particle, columns |p| / theta vs |p| / theta -- but each 1D
panel overlays the two generators and carries a ratio strip, and the 2D
column is split into the two generators side by side.  The originals
validate ONE sample; this one compares two.

    python3 production/compare_kinematics.py <ps|vm|channel> [born|rad]
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BLUE, VERM, GREY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
P = os.path.dirname(os.path.abspath(__file__))
E0 = 10.6

# final state actually written to LUND, in the order the rows should appear
PARTS = {
 "pi0":  [(11, r"$e'$", 20), (2212, r"$p$", 90), (111, r"$\pi^0$", 40)],
 "eta":  [(11, r"$e'$", 20), (2212, r"$p$", 90), (221, r"$\eta$", 40)],
 "rho":  [(11, r"$e'$", 20), (2212, r"$p$", 90), (113, r"$\rho^0$", 40),
          (211, r"$\pi^+$", 40), (-211, r"$\pi^-$", 40)],
 "phi":  [(11, r"$e'$", 20), (2212, r"$p$", 90), (333, r"$\phi$", 40),
          (321, r"$K^+$", 40), (-321, r"$K^-$", 40)],
 "jpsi": [(11, r"$e'$", 20), (2212, r"$p$", 90), (443, r"$J/\psi$", 40),
          (-11, r"$e^+$", 40), (11.5, r"$e^-$ (decay)", 40)],
}
REF = {
 "pi0":  ("exclugen", sorted(glob.glob("/Volumes/wd_14tb/mc/exclugen/pi0/20260810_10.6_born_osg_v2/osg/*.lund")), True),
 "eta":  ("exclugen", sorted(glob.glob("/Volumes/wd_14tb/mc/exclugen/eta/20260810_10.6_born_osg_v2/osg/*.lund")), True),
 "rho":  ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/rho_born_100k.lund"], False),
 "phi":  ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/phi_born_100k.lund"], False),
 "jpsi": ("diffrad_vm", ["/Users/vpk/MC_vector_mesons/samples_v2/jpsi_born_100k.lund"], False),
}
GROUPS = {"ps": ["pi0", "eta"], "vm": ["rho", "phi", "jpsi"]}
MESON_PID = {111, 221, 113, 333, 443}


def mine_files(ch, mode="born"):
    d = f"/Volumes/wd_14tb/mc/exclurad_py/{ch}/*_10.6_{mode}_v1/lund"
    return sorted(glob.glob(f"{d}/{ch}_exclurad_py_{mode}_10.6GeV*.lund"))


def read(path, slots, need_eeff=False):
    """slots: list of pid keys. Returns {slot: [(|p|,theta_deg), ...]}.

    The scattered electron is the FIRST status-1 pid==11 after the optional
    beam DOC line; a second one (J/psi decay) goes to the 11.5 slot, so the
    two electrons never collide in the same histogram.
    """
    out = {s: [] for s in slots}
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
            rows, eeff, n_el = [], E0, 0
            for i in range(npart):
                q = fh.readline().split()
                if len(q) < 11:
                    continue
                pid, st = int(q[3]), int(q[2])
                px, py, pz = map(float, q[6:9])
                if i == 0 and st == 0 and pid == 11:
                    eeff = float(q[9])
                    continue
                # Keep status-1 tracks, plus the parent meson wherever it is
                # written as documentation (status 0) because the reference
                # decayed it: exclugen sends pi0/eta -> gamma gamma and
                # diffrad_vm sends phi -> K+K-, so the meson line is status 0
                # there and status 1 here.  Comparing it needs both.
                if st != 1 and pid not in MESON_PID:
                    continue
                key = pid
                if pid == 11:
                    n_el += 1
                    key = 11 if n_el == 1 else 11.5
                rows.append((key, px, py, pz))
            if need_eeff and (E0 - eeff) > 1e-3:
                continue              # exclugen ran itarg=1: drop radiated beam
            for key, px, py, pz in rows:
                if key in out:
                    p = np.hypot(np.hypot(px, py), pz)
                    th = np.degrees(np.arctan2(np.hypot(px, py), pz))
                    out[key].append((p, th))
    return out


def gather(files, slots, need_eeff=False):
    acc = {s: [] for s in slots}
    for fn in files:
        for k, v in read(fn, slots, need_eeff).items():
            acc[k].extend(v)
    return {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v]))
            for k, v in acc.items() if v}


def panel(fig, gs, r, c, a, b, label, unit, refname, thmax=None):
    """1D overlay + ratio strip. Returns the mean ratio."""
    hi = thmax if thmax else np.percentile(np.concatenate([a, b]), 99.5)
    bins = np.linspace(0, hi, 50)
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    wa, wb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
    ctr = 0.5 * (bins[:-1] + bins[1:])

    AX = fig.add_subplot(gs[r, c])
    RX = fig.add_subplot(gs[r + 1, c], sharex=AX)
    AX.step(ctr, wb, where="mid", lw=2.2, color=VERM, label=refname)
    AX.step(ctr, wa, where="mid", lw=1.5, ls="--", color=BLUE, label="exclurad_py")
    AX.set_yscale("log")
    AX.set_ylabel("norm.", fontsize=8)
    AX.set_title(label, loc="left", fontsize=9, fontweight="bold")
    AX.legend(frameon=False, fontsize=7)
    AX.tick_params(labelbottom=False)

    ok = (ha > 3) & (hb > 3)
    ratio = np.full(len(ctr), np.nan); err = np.full(len(ctr), np.nan)
    ratio[ok] = wa[ok] / wb[ok]
    err[ok] = ratio[ok] * np.sqrt(1 / ha[ok] + 1 / hb[ok])
    RX.axhline(1, color=GREY, lw=1, ls="--")
    RX.errorbar(ctr[ok], ratio[ok], yerr=err[ok], fmt="o", ms=2.2, lw=0.7,
                color=BLUE, capsize=0)
    RX.set_ylim(0.3, 1.7)
    RX.set_ylabel("new/ref", fontsize=7)
    RX.set_xlabel(unit, fontsize=8)
    RX.grid(alpha=0.25, lw=0.5)
    return a.mean() / b.mean()


def two_d(fig, gs, r, c, pm, th, pmax, thmax, title):
    AX = fig.add_subplot(gs[r:r + 2, c])
    h = AX.hist2d(pm, th, bins=50, cmap="plasma", cmin=1,
                  range=[[0, pmax], [0, thmax]])
    fig.colorbar(h[3], ax=AX, pad=0.02, fraction=0.045)
    AX.set_xlabel(r"$|p|$ (GeV/c)", fontsize=8)
    AX.set_ylabel(r"$\theta$ (deg)", fontsize=8)
    AX.set_title(title, loc="left", fontsize=8.5)
    AX.grid(False)


def page(chans, mode, outname, title):
    rows = sum(len(PARTS[c]) for c in chans)
    fig = plt.figure(figsize=(16, 3.5 * rows))
    gs = GridSpec(3 * rows, 4, figure=fig,
                  height_ratios=[3, 1, 0.55] * rows,
                  width_ratios=[1, 1, 1.15, 1.15],
                  hspace=0.12, wspace=0.30)
    r = 0
    for ch in chans:
        refname, rfiles, need_eeff = REF[ch]
        slots = [p for p, _, _ in PARTS[ch]]
        A = gather(mine_files(ch, mode), slots)
        B = gather(rfiles, slots, need_eeff)
        print(f"\n=== {ch.upper()} {mode.upper()} vs {refname} "
              f"({len(A.get(11,[[]])[0])} vs {len(B.get(11,[[]])[0])} events)")
        for pid, lab, thmax in PARTS[ch]:
            if pid not in A or pid not in B:
                print(f"   {lab}: missing in "
                      f"{'new' if pid not in A else 'ref'} -- skipped")
                r += 3
                continue
            pa, ta = A[pid]; pb, tb = B[pid]
            pmax = np.percentile(np.concatenate([pa, pb]), 99)
            tag = f"{ch} {lab}"
            rp = panel(fig, gs, r, 0, pa, pb, f"{tag}   $|p|$",
                       r"$|p|$ (GeV/c)", refname)
            rt = panel(fig, gs, r, 1, ta, tb, f"{tag}   $\\theta$",
                       r"$\theta$ (deg)", refname, thmax=thmax)
            two_d(fig, gs, r, 2, pa, ta, pmax, thmax, f"{tag}  exclurad_py")
            two_d(fig, gs, r, 3, pb, tb, pmax, thmax, f"{tag}  {refname}")
            print(f"   {lab:>14}  <|p|> {pa.mean():7.3f} vs {pb.mean():7.3f}"
                  f"  ({rp:5.3f})   <theta> {ta.mean():6.2f} vs {tb.mean():6.2f}"
                  f"  ({rt:5.3f})")
            r += 3
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.998)
    out = os.path.join(P, outname)
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    what = a[0] if a else "vm"
    mode = a[1] if len(a) > 1 else "born"
    chans = GROUPS.get(what, [what])
    chans = [c for c in chans if mine_files(c, mode)]
    if not chans:
        sys.exit(f"no exclurad_py {mode} samples for {what}")
    nm = ("pseudoscalar mesons vs exclugen" if what == "ps" else
          "vector mesons vs diffrad_vm" if what == "vm" else f"{what}")
    page(chans, mode, f"kin_{what}_{mode}.png",
         f"exclurad_py {mode.upper()} -- particle kinematics, {nm}")
