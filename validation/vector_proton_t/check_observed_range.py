# Does the MESON bin lose hard-photon events because the observed t_meson
# falls outside the Born-physical range at the observed (W2, Q2)?
import glob, math, sys
import numpy as np
sys.path.insert(0, "/data/vpk/exclurad_py_vt")
from exclurad_py.core.kinematics import EventKin
MP, MJ, EB = 0.93827208816, 3.0969, 10.6
def read(d):
    f = glob.glob(d + "/**/lund/*.lund", recursive=True)[0]
    out = []
    cur = None
    for line in open(f):
        p = line.split()
        if len(p) == 10:
            if cur and all(k in cur for k in "epV"):
                out.append(cur)
            cur = {}
            continue
        if len(p) < 11 or cur is None:
            continue
        v4 = np.array([float(p[9]), float(p[6]), float(p[7]), float(p[8])])
        assert abs(v4[0]**2 - v4[1:]@v4[1:] - float(p[10])**2) < 1e-4
        pid = p[3]
        if pid == "11" and p[4] == "0" and "e" not in cur: cur["e"] = v4
        elif pid == "2212": cur["p"] = v4
        elif pid == "443": cur["V"] = v4
        elif pid == "22" and p[4] == "0": cur["g"] = cur.get("g", 0.0) + v4[0]
    return out
k1 = np.array([EB, 0, 0, math.sqrt(EB*EB - 0.51099895e-3**2)])
tgt = np.array([MP, 0, 0, 0.0])
for tag in ("mdet", "pdet"):
    ev = read("/data/vpk/vt_test_20261008/" + tag)
    nout = n = 0; Eg_out = []; Eg_in = []
    Q2s, W2s = [], []
    for c in ev:
        q = k1 - c["e"]
        Q2 = -(q[0]**2 - q[1:]@q[1:])
        s = tgt + q
        W2 = s[0]**2 - s[1:]@s[1:]
        Q2s.append(Q2); W2s.append(W2)
        d_ = q - c["V"]
        tM = d_[0]**2 - d_[1:]@d_[1:]
        try:
            lo, hi = EventKin.t_limits(Q2, W2, MJ, MP, M=MP, t_kind="photon")
        except Exception:
            continue
        n += 1
        if tM < lo - 1e-9 or tM > hi + 1e-9:
            nout += 1; Eg_out.append(c.get("g", 0.0))
        else:
            Eg_in.append(c.get("g", 0.0))
    Q2s = np.array(Q2s); W2s = np.array(W2s)
    print("%s: observed Q2 %.4f..%.4f (card 0.001-0.25), W2 %.2f..%.2f (card 16.4-20.8)"
          % (tag, Q2s.min(), Q2s.max(), W2s.min(), W2s.max()))
    print("   t_meson outside the Born range at its own (W2,Q2): %d / %d = %.2f %%"
          % (nout, n, 100.0*nout/max(n,1)))
    if Eg_out:
        eo, ei = np.array(Eg_out), np.array(Eg_in)
        print("   mean E_gamma  outside %.4f   inside %.4f   (n_out %d)"
              % (eo.mean(), ei.mean(), len(eo)))
        print("   E_gamma > 0.25 GeV: %.2f %% of the outside set, %.2f %% of the inside"
              % (100*(eo>0.25).mean(), 100*(ei>0.25).mean()))
