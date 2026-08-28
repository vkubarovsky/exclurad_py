"""How far down in W~^2 does the radiative tail reach at the published v_cut,
and how much of the RC comes from below the amp2021 validity floor?"""
import math, sys
import numpy as np
sys.path.insert(0,"/Users/vpk/exclurad_py")
from exclurad_py import models
from exclurad_py.core.rc import rc_factor
from exclurad_py.core.constants import M_P

M_PI0, M_ETA = 0.1349768, 0.547862
VCUT = {"pi0": 0.17, "eta": 0.20}

class Rec:
    """Delegate to the model, record every W2 it is asked at."""
    def __init__(self, m, floor=None):
        self._m = m; self.W2 = []; self.floor = floor
    def __getattr__(self, k): return getattr(self._m, k)
    def __setattr__(self, k, v):
        if k in ("_m","W2","floor"): object.__setattr__(self,k,v)
        else: setattr(self._m, k, v)
    def __call__(self, W2, Q2, t):
        self.W2.append(np.atleast_1d(np.asarray(W2,dtype=float)).copy())
        out = self._m(W2, Q2, t)
        if self.floor is not None:
            w = np.atleast_1d(np.asarray(W2,dtype=float))
            mask = w < self.floor
            if np.any(mask):
                out = {k: np.where(mask, 0.0, np.asarray(v,dtype=float))
                       for k,v in out.items()}
        return out

def load(fn):
    rows=[]
    for line in open(fn):
        v=line.split()
        if len(v)<12: continue
        v=[float(x) for x in v[:12]]
        Q2,xB,mt=v[0],v[1],v[2]
        W2=M_P*M_P+Q2*(1-xB)/xB
        rows.append((Q2,xB,mt,W2))
    return rows

ch = sys.argv[1] if len(sys.argv)>1 else "pi0"
mM = M_PI0 if ch=="pi0" else M_ETA
fn = f"/Users/vpk/pi0_eta_amplitude_model/data/strfun_{ch}.data"
rows = sorted(load(fn), key=lambda r: r[3])       # lowest W2 first
print(f"{ch}: {len(rows)} bins, W2 from {rows[0][3]:.2f} to {rows[-1][3]:.2f}")
mx2 = mM*mM + VCUT[ch]
print(f"v_cut = {VCUT[ch]}  ->  MX2 cut = {mx2:.4f} GeV^2\n")
print(f"{'Q2':>5} {'xB':>6} {'-t':>5} {'W2':>6} | {'min W~2':>8} {'dW2':>6} | "
      f"{'eta':>7} | {'d(eta) if W2<3.3 zeroed':>22} {'<3.6':>8} {'<4.0':>8}")
print("-"*104)
for (Q2,xB,mt,W2) in rows[:5] + rows[len(rows)//2:len(rows)//2+2] + rows[-2:]:
    base = models.get(f"{ch}.amp2021", t_nucl=-mt, Ebeam=5.75)
    rec  = Rec(base)
    r0 = rc_factor(5.75, Q2, W2, -mt, math.pi/2, rec, h=+1, mx2_cut=mx2)
    allW = np.concatenate(rec.W2)
    wmin = allW.min()
    outs=[]
    for floor in (3.3, 3.6, 4.0):
        m2 = Rec(models.get(f"{ch}.amp2021", t_nucl=-mt, Ebeam=5.75), floor=floor)
        r  = rc_factor(5.75, Q2, W2, -mt, math.pi/2, m2, h=+1, mx2_cut=mx2)
        outs.append((r["eta"]-r0["eta"])/r0["eta"]*100)
    print(f"{Q2:5.2f} {xB:6.3f} {mt:5.2f} {W2:6.2f} | {wmin:8.3f} {W2-wmin:6.3f} | "
          f"{r0['eta']:7.4f} | {outs[0]:22.3f} {outs[1]:8.3f} {outs[2]:8.3f}")
