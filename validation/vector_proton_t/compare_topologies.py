# THE DECISIVE TEST: both LUNDs histogrammed in the SAME variables.
import glob, math, sys
import numpy as np
MP, EB = 0.93827208816, 10.6
def read(d):
    f = glob.glob(d + "/**/lund/*.lund", recursive=True)[0]
    tN, tM, Q2, W2, Eg = [], [], [], [], []
    cur = None
    for line in open(f):
        p = line.split()
        if len(p) < 10:
            continue
        if len(p) == 10 and "e" not in p[0]:          # header: 10 fields
            if cur and "e" in cur and "p" in cur and "V" in cur:
                k2 = cur["e"]; pr = cur["p"]; pv = cur["V"]
                k1 = np.array([EB, 0, 0, math.sqrt(EB*EB - 0.51099895e-3**2)])
                q = k1 - k2
                Q2.append(-(q[0]**2 - q[1:]@q[1:]))
                tgt = np.array([MP,0,0,0.0])
                s = tgt + q
                W2.append(s[0]**2 - s[1:]@s[1:])
                tN.append(2*MP*MP - 2*MP*pr[0])
                d_ = q - pv
                tM.append(d_[0]**2 - d_[1:]@d_[1:])
                Eg.append(cur.get("g", 0.0))
            cur = {}
            continue
        pid = p[3]; v4 = np.array([float(p[8]), float(p[5]), float(p[6]), float(p[7])])
        if cur is None: cur = {}
        if pid == "11" and p[4] == "0" and "e" not in cur: cur["e"] = v4
        elif pid == "2212": cur["p"] = v4
        elif pid == "443": cur["V"] = v4
        elif pid == "22" and p[4] == "0": cur["g"] = cur.get("g",0.0) + v4[0]
    return (np.array(tN), np.array(tM), np.array(Q2), np.array(W2), np.array(Eg))
D="/data/vpk/vt_test_20261008"
a = read(D+"/mdet"); b = read(D+"/pdet")
print("events: mdet %d   pdet %d" % (len(a[0]), len(b[0])))
names = ["t_N = (p-p')^2", "t_meson = (q-p_V)^2", "Q2", "W2", "E_gamma(lab)"]
EDG = [np.linspace(-7.0,-0.45,9), np.linspace(-7.0,-0.45,9),
       np.linspace(0.001,0.25,7), np.linspace(16.4,20.8,7),
       np.array([0,0.02,0.05,0.1,0.25,0.6,1.5,12.0])]
for k,(nm,ed) in enumerate(zip(names,EDG)):
    ha,_ = np.histogram(a[k], bins=ed); hb,_ = np.histogram(b[k], bins=ed)
    na, nb = ha.sum(), hb.sum()
    print("\n  %s   (mdet %d, pdet %d in range)" % (nm, na, nb))
    print("  %-18s %8s %8s %9s %8s" % ("bin","mdet","pdet","ratio","sigma"))
    for i in range(len(ed)-1):
        if ha[i]<20 and hb[i]<20: continue
        fa, fb = ha[i]/na, hb[i]/nb
        r = fb/fa if fa else float("nan")
        e = r*math.sqrt(1/max(ha[i],1)+1/max(hb[i],1))
        print("  %7.3f-%-9.3f %8d %8d %9.4f %8.2f" % (ed[i],ed[i+1],ha[i],hb[i],r,(r-1)/e if e else 0))
