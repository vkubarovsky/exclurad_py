#!/usr/bin/env python3
"""Are the two t bins the same SAMPLE?  Both LUNDs in the SAME variables.

Generate once per topology from validation/vector_proton_t/card_{mdet,pdet}.txt
-- identical but for the model line -- then run this.

The assert on E^2 - p^2 = m^2 is there because the first version of this
script read the LUND columns one place off (pz where E belongs) and produced
a table that looked plausible and was not.  The columns are
    0 idx  1 lifetime  2 status  3 pid  4 parent  5 daughter
    6 px   7 py        8 pz      9 E    10 mass   11-13 vx,vy,vz
"""
# LUND columns, verified against E^2 - p^2 = m^2:
#   0 idx  1 lifetime  2 status  3 pid  4 parent  5 daughter
#   6 px   7 py        8 pz      9 E    10 mass   11-13 vx,vy,vz
import glob, math, sys
import numpy as np
MP, EB = 0.93827208816, 10.6
def read(d, mesonpid):
    f = glob.glob(d + "/**/lund/*.lund", recursive=True)[0]
    tN, tM, Eg = [], [], []
    cur = None
    for line in open(f):
        p = line.split()
        if len(p) == 10:                                   # header
            if cur and all(k in cur for k in "epV"):
                k2, pr, pv = cur["e"], cur["p"], cur["V"]
                k1 = np.array([EB, 0, 0, math.sqrt(EB*EB - 0.51099895e-3**2)])
                q = k1 - k2
                tN.append(2*MP*MP - 2*MP*pr[0])
                d_ = q - pv
                tM.append(d_[0]**2 - d_[1:]@d_[1:])
                Eg.append(cur.get("g", 0.0))
            cur = {}
            continue
        if len(p) < 11 or cur is None:
            continue
        v4 = np.array([float(p[9]), float(p[6]), float(p[7]), float(p[8])])
        chk = abs(v4[0]**2 - v4[1:]@v4[1:] - float(p[10])**2)
        assert chk < 1e-4, ("column layout wrong", chk, p[:4])
        pid = p[3]
        if pid == "11" and p[4] == "0" and "e" not in cur: cur["e"] = v4
        elif pid == "2212": cur["p"] = v4
        elif pid == mesonpid: cur["V"] = v4
        elif pid == "22" and p[4] == "0": cur["g"] = cur.get("g", 0.0) + v4[0]
    return np.array(tN), np.array(tM), np.array(Eg)
D = "/data/vpk/vt_test_20261008"
a = read(D+"/mdet", "443"); b = read(D+"/pdet", "443")
print("events: mdet %d   pdet %d" % (len(a[0]), len(b[0])))
print("t_N range: mdet %+.4f..%+.4f   pdet %+.4f..%+.4f  (card box -7.00..-0.45)"
      % (a[0].min(), a[0].max(), b[0].min(), b[0].max()))
for k, nm, ed in ((0, "t_N = 2M^2 - 2M E_p  (lab)", np.linspace(-7.0, -0.45, 9)),
                  (1, "t_meson = (q - p_V)^2", np.linspace(-9.0, -0.4, 9)),
                  (2, "E_gamma (lab)", np.array([0,0.02,0.05,0.1,0.25,0.6,1.5,12.0]))):
    ha,_ = np.histogram(a[k], bins=ed); hb,_ = np.histogram(b[k], bins=ed)
    na, nb = ha.sum(), hb.sum()
    print("\n  %s   (in range: mdet %d, pdet %d)" % (nm, na, nb))
    print("  %-20s %8s %8s %9s %8s" % ("bin", "mdet", "pdet", "ratio", "pull"))
    for i in range(len(ed)-1):
        if ha[i] < 20 and hb[i] < 20: continue
        fa, fb = ha[i]/na, hb[i]/nb
        r = fb/fa if fa else float("nan")
        e = r*math.sqrt(1/max(ha[i],1)+1/max(hb[i],1))
        print("  %8.3f-%-10.3f %8d %8d %9.4f %8.2f" % (ed[i],ed[i+1],ha[i],hb[i],r,(r-1)/e))
