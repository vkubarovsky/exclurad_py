"""Why amp2614 gives sigma_rad/sigma_Born = 1.285: the rad arm has vcut = 0, i.e. no cut at all.

1. True hadronic W' = M(p' g g) against observed W, rad and Born arms (generated block).
2. eta(v_cut) = rc_factor at Born-generated points, no MC; the last column (no cut) is what
   sigma_rad/sigma_Born measures, up to the itarg asymmetry of the two arms.
amp2614 ntuples are 55-float records (with vz); usage: python rc_method_pi0_eta_vscan.py 4000
"""
import sys, glob, numpy as np
from multiprocessing import Pool
import rc_method_pi0 as R
C = "/Volumes/wd_14tb/prod_pi0_amp2614_20261021"
EB = 10.6
VC = [0.06, 0.2, 0.5, 1.0, 2.0, 4.0, None]

def chunk(rows):
    from exclurad_py import models
    from exclurad_py.core.rc import rc_factor
    m = models.get("pi0.amp2614", t_nucl=-0.3)
    out = np.full((len(rows), len(VC)), np.nan)
    for i, (Q2, xB, mt, ph) in enumerate(rows):
        m.t_nucl = -mt; W2 = R.MP**2 + Q2*(1/xB - 1)
        try:
            for j, v in enumerate(VC):
                out[i, j] = rc_factor(EB, Q2, W2, -mt, np.radians(ph), m, h=+1,
                                      mx2_cut=None if v is None else v + R.MPI**2)["eta"]
        except Exception:
            pass
    return out

def gen(arm, nfiles=40):
    a = np.concatenate([np.fromfile(f, "<f4").reshape(-1, 55) for f in
                        sorted(glob.glob(f"{C}/{arm}/out_inb/*bgnone*.f32"))[:nfiles]]).astype(float)
    p = [a[:, 1+3*k:4+3*k] for k in range(4)]
    g = R.kinematics(*p)
    E = lambda v, m: np.sqrt((v**2).sum(1) + m**2)
    P = p[1] + p[2] + p[3]; Eh = E(p[1], R.MP) + E(p[2], 0) + E(p[3], 0)
    Wt = np.sqrt(np.maximum(Eh**2 - (P**2).sum(1), 0))
    Wo = np.sqrt(R.MP**2 + g["Q2"]*(1/g["xB"] - 1))
    return g, Wo, Wt

if __name__ == "__main__":
    # 1. true vs observed W in the rad arm (Born as the check: Wt == Wo)
    for arm in ("born", "rad"):
        g, Wo, Wt = gen(arm)
        d = Wo - Wt
        print(f"{arm}: n={len(Wo)}  Wobs<1.8: {(Wo<1.8).mean():.4f}  Wtrue<1.8: {(Wt<1.8).mean():.3f}  "
              f"Wtrue<1.6: {(Wt<1.6).mean():.3f}  Wtrue<1.4: {(Wt<1.4).mean():.3f}  |Wo-Wt|>0.01: {(np.abs(d)>0.01).mean():.3f}")
        if arm == "rad":
            for lo, hi in ((1.8, 2.0), (2.0, 2.2), (2.2, 2.5), (2.5, 3.0), (3.0, 4.0)):
                s = (Wo >= lo) & (Wo < hi)
                print(f"   Wobs {lo}-{hi}: n={s.sum():6d}  <Wtrue>={Wt[s].mean():.3f}  frac Wtrue<1.8={np.mean(Wt[s]<1.8):.3f}  frac Wtrue<1.4={np.mean(Wt[s]<1.4):.3f}")
            h, e = np.histogram(Wt, bins=[1.07, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.5, 3, 4.2])
            print("   Wtrue hist:", dict(zip([f"{x:.2f}" for x in e[:-1]], (h/len(Wt)).round(3))))
            rg = g
    # 2. eta(v_cut) at Born points
    g, Wo, _ = gen("born", 20)
    kin = np.column_stack([g["Q2"], g["xB"], g["mt"], g["phi"]])
    idx = np.random.default_rng(2).choice(len(kin), int(sys.argv[1]), replace=False)
    with Pool(8) as p:
        eta = np.concatenate(p.map(chunk, np.array_split(kin[idx], 128)))
    ok = np.isfinite(eta).all(1); W = Wo[idx]
    print(f"\neta vs v_cut (GeV^2), Born-weighted, n={ok.sum()}")
    print("        " + "".join(f"{str(v):>9}" for v in VC))
    for lo, hi in ((1.8, 4.2), (1.8, 2.0), (2.0, 2.2), (2.2, 2.5), (2.5, 3.0), (3.0, 4.2)):
        s = ok & (W >= lo) & (W < hi)
        print(f"W {lo}-{hi} ({s.sum()/ok.sum():.2f}) " + "".join(f"{eta[s, j].mean():9.4f}" for j in range(len(VC))))
    e = eta[ok, -1]; print(f"<eta_nocut> = {e.mean():.4f} +- {e.std()/np.sqrt(len(e)):.4f}")
