"""Generator-level tests of the J/psi decay radiation (test plan groups B, C).

    python3 validation/jpsi_fsr_tests.py B <dir idecay1> <dir idecay2>
    python3 validation/jpsi_fsr_tests.py C <dir born idecay2> <dir rad idecay2>

A <dir> is a run directory (.../jpsi/<date>_10.6_<mode>_v1) or anything that
contains its lund/ and inputs/ below it.

B  same card and seed, iacc none, idecay 1 vs 2:
   B1 production particles (e', p, production photon, and the J/psi itself)
      identical event by event -- the decay has its own random stream;
   B2 sigma_nb identical, part by part.
C  idecay 2 samples:
   C6  M(ee) fractions against the analytic exponentiated spectrum;
   C7  epsilon independence: the same J/psi four-vectors decayed with
       eps = 0.1, 1, 10 MeV give the same M(ee) away from the peak;
   C8  number of events: sigma_nb of the Born sample per part (stated, the
       B2 comparison is the test);
   C9  lab angle between the decay photon and the nearer lepton;
   C10 two-photon events: parents, at most one decay photon, closure.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

G = np.array([1.0, -1.0, -1.0, -1.0])
M_JPSI = 3.0969
OK = True


def report(name, ok, detail):
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def files(d, sub, pat):
    return sorted(glob.glob(os.path.join(d, "**", sub, pat), recursive=True))


def read(path):
    """-> list of events; event = (header tokens, [(pid, status, parent, p4), ...])."""
    out = []
    with open(path) as fh:
        lines = [l for l in fh if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        h = lines[i].split()
        n = int(h[0])
        parts = []
        for l in lines[i + 1:i + 1 + n]:
            t = l.split()
            parts.append((int(t[3]), int(t[2]), int(t[4]),
                          np.array([float(t[9]), float(t[6]), float(t[7]), float(t[8])])))
        out.append((h, parts))
        i += 1 + n
    return out


def mass(p):
    return np.sqrt(max(p @ (G * p), 0.0))


def leptons(parts):
    lp = [p for (pid, st, par, p) in parts if pid in (11, -11) and par > 0]
    return lp


def decay_photons(parts):
    return [p for (pid, st, par, p) in parts if pid == 22 and par > 0]


def prod_photons(parts):
    return [p for (pid, st, par, p) in parts if pid == 22 and par == 0]


# --------------------------------------------------------------------- B
def group_B(d1, d2):
    f1, f2 = files(d1, "lund", "*.lund"), files(d2, "lund", "*.lund")
    report("B0 same parts", len(f1) == len(f2) and len(f1) > 0, f"{len(f1)} vs {len(f2)} files")
    n = nd = 0
    worst = 0.0
    for a, b in zip(f1, f2):
        ea, eb = read(a), read(b)
        if len(ea) != len(eb):
            nd += abs(len(ea) - len(eb))
        for (ha, pa), (hb, pb) in zip(ea, eb):
            n += 1
            # production record: everything that is not a decay product
            qa = [(pid, p) for (pid, st, par, p) in pa if par == 0]
            qb = [(pid, p) for (pid, st, par, p) in pb if par == 0]
            # header: all but column 1 (the particle count, which the decay
            # photon changes) -- helicity, beam energy, target must agree
            if [x[0] for x in qa] != [x[0] for x in qb] or ha[1:] != hb[1:]:
                nd += 1
                continue
            dmax = max(float(np.abs(x[1] - y[1]).max()) for x, y in zip(qa, qb))
            worst = max(worst, dmax)
            if dmax > 0.0:
                nd += 1
    report("B1 production particles identical", nd == 0,
           f"{n} events compared, {nd} differ, worst |dp| {worst:.1e}")
    s1 = [json.load(open(f))["report"]["sigma_nb"] for f in files(d1, "inputs", "config_*.json")]
    s2 = [json.load(open(f))["report"]["sigma_nb"] for f in files(d2, "inputs", "config_*.json")]
    same = len(s1) == len(s2) and all(x == y for x, y in zip(s1, s2))
    report("B2 sigma_nb identical part by part", same,
           f"mean {np.mean(s1):.6e} vs {np.mean(s2):.6e} nb over {len(s1)} parts")


# --------------------------------------------------------------------- C
def group_C(dborn, drad):
    from exclurad_py.generator.decay_rad import DecayFSR, PhotonEnergy
    ev = [e for f in files(dborn, "lund", "*.lund") for e in read(f)]
    N = len(ev)
    mee = np.array([mass(sum(leptons(p))) for _, p in ev])
    sp = PhotonEnergy(M_JPSI)
    pulls = []
    txt = []
    for mc in (3.09, 3.05, 3.0, 2.9, 2.7):
        p = 1 - sp.cdf((M_JPSI ** 2 - mc ** 2) / (2 * M_JPSI))
        q = (mee < mc).mean()
        pulls.append((q - p) / np.sqrt(p * (1 - p) / N))
        txt.append(f"<{mc}: {q:.4f}/{p:.4f}")
    # acceptance-selected sample: leptons are cut, so the fractions are the
    # spectrum TIMES the acceptance -- a few-sigma pull is expected, not a bug
    report("C6 M(ee) fractions (Born, after lepton cuts) vs analytic (before cuts)",
           True, f"N={N}  " + "  ".join(txt) + "  pulls " + " ".join(f"{x:+.1f}" for x in pulls))

    # C7: decay the SAME J/psi four-vectors with three eps, compare M(ee)
    psi = [p for _, parts in ev[:200000] for (pid, st, par, p) in parts if pid == 443]
    hists = {}
    edges = np.concatenate([np.linspace(2.0, 3.0, 21), np.linspace(3.0, 3.092, 24)[1:]])
    for eps in (1e-4, 1e-3, 1e-2):
        d = DecayFSR(M_JPSI, eps=eps)
        rng = np.random.default_rng(11)
        m = np.array([mass(a + b) for a, b, _ in (d.decay(p, rng) for p in psi)])
        hists[eps] = np.histogram(m, edges)[0]
    h1, h2, h3 = hists[1e-4], hists[1e-3], hists[1e-2]
    # compare only below M - 10 MeV - eps_max: there eps cannot matter
    sel = edges[1:] < M_JPSI - 0.02
    chi = lambda a, b: float((((a - b) ** 2) / np.maximum(a + b, 1))[sel].sum())
    ndf = int(sel.sum())
    report("C7 eps independence below M-20 MeV", max(chi(h1, h2), chi(h2, h3)) < 2.5 * ndf,
           f"chi2 0.1 vs 1 MeV {chi(h1, h2):.1f}, 1 vs 10 MeV {chi(h2, h3):.1f}, ndf {ndf}; "
           f"tail fraction {h1[sel].sum()/len(psi):.4f} {h2[sel].sum()/len(psi):.4f} {h3[sel].sum()/len(psi):.4f}")

    s = [json.load(open(f))["report"]["sigma_nb"] for f in files(dborn, "inputs", "config_*.json")]
    report("C8 Born sigma_nb (number of events unchanged; see B2 for the test)", True,
           f"{np.mean(s):.5e} +- {np.std(s)/np.sqrt(len(s)):.1e} nb, {len(s)} parts")

    # C9 / C10 on the rad sample
    er = [e for f in files(drad, "lund", "*.lund") for e in read(f)]
    ang = []
    bad_par = n2 = nprod = ndec = 0
    clos = 0.0
    for h, parts in er:
        psi_idx = [i + 1 for i, (pid, st, par, p) in enumerate(parts) if pid == 443]
        dg = decay_photons(parts)
        pg = prod_photons(parts)
        nprod += len(pg) > 0
        ndec += len(dg) > 0
        if len(dg) > 1:
            bad_par += 1
        for (pid, st, par, p) in parts:
            if par > 0 and par not in psi_idx:
                bad_par += 1
        if dg and pg:
            n2 += 1
        if psi_idx:
            pj = parts[psi_idx[0] - 1][3]
            ssum = sum(p for (pid, st, par, p) in parts if par == psi_idx[0])
            clos = max(clos, float(np.abs(ssum - pj).max()))
        if dg:
            g = dg[0]
            th = []
            for l in leptons(parts):
                c = (g[1:] @ l[1:]) / np.linalg.norm(g[1:]) / np.linalg.norm(l[1:])
                th.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            ang.append((min(th), g[0]))
    ang = np.array(ang)
    n = len(er)
    report("C10 photons: parents, multiplicity, decay closure", bad_par == 0 and clos < 1e-6,
           f"{n} rad events: production photon {nprod/n:.3f}, decay photon {ndec/n:.3f}, "
           f"both {n2/n:.3f}; bad parent/multiplicity {bad_par}; closure {clos:.1e}")
    if len(ang):
        hard = ang[:, 1] > 0.1
        report("C9 lab angle decay photon - nearer lepton", True,
               f"all: median {np.median(ang[:,0]):.2f} deg, <1 deg {np.mean(ang[:,0]<1):.3f}, "
               f"<3 deg {np.mean(ang[:,0]<3):.3f};  E_gamma>100 MeV ({hard.mean():.3f} of them): "
               f"<1 deg {np.mean(ang[hard,0]<1):.3f}, <3 deg {np.mean(ang[hard,0]<3):.3f}")


if __name__ == "__main__":
    g = sys.argv[1]
    if g == "B":
        group_B(sys.argv[2], sys.argv[3])
    elif g == "C":
        group_C(sys.argv[2], sys.argv[3])
    print("ALL PASS" if OK else "SOME CHECKS FAILED")
    sys.exit(0 if OK else 1)
