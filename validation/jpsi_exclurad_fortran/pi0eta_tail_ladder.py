"""THE COMPARISON: independent first-order tail vs exclurad_py, pi0/eta,
production bin, v ladder.  Both non-exponentiated."""
import sys, numpy as np
sys.path.insert(0, "/Users/vpk/exclurad_py-wt-tail")
sys.path.insert(0, "/Users/vpk/exclurad_py-wt-tail/validation/jpsi_exclurad_fortran")
import independent_tail as IT
from exclurad_py.models import get
from exclurad_py.core.rc import rc_factor

PTS = [("pi0.amp2614", 2.0, 9.0, -0.30), ("pi0.amp2614", 1.2, 6.0, -0.30),
       ("pi0.amp2614", 4.0, 9.0, -0.50), ("eta.amp2614", 2.0, 9.0, -0.30)]
EDGES = [0.03, 0.1, 0.3, 1.0, 3.0]
PHI = 20.0

def py_tail(nm, Q2, W2, t, va, vb, mU):
    """exclurad_py first-order tail between two v cuts, non-exponentiated.

    DIFFERENCE OF ETA, not of sigma_F.  sigma_F is the Bardin-Shumeiko
    SUBTRACTED hard term, so sigma_F(vb) - sigma_F(va) is the tail MINUS the
    soft approximation over the same interval; the missing piece is
    delta_inf(vb) - delta_inf(va), which eta carries.  Differencing sigma_F
    alone made the independent calculation look 5.8x high at v = 0.03-0.1 and
    1.2x at v = 1-2.3 -- the ratio falling with v is the signature of exactly
    that term.  py_values() in this directory had it right all along.

    Helicity-AVERAGED, to match the spin-summed leptonic tensor of the
    independent side.
    """
    out = []
    for v in (va, vb):
        sf = get(nm, t_nucl=t)
        r = [rc_factor(Ebeam=10.6, Q2=Q2, W2=W2, t=t, phi=np.radians(PHI),
                       sfmodel=sf, h=h, mx2_cut=v + mU ** 2,
                       exponentiate=False) for h in (+1, -1)]
        out.append((r[0]["sigma_obs"] + r[1]["sigma_obs"])
                   / (r[0]["sigma_born"] + r[1]["sigma_born"]))
    return out[1] - out[0]

def main():
    for nm, Q2, W2, t in PTS:
        p = IT.Point(Q2, W2, t, PHI, mode="target", model_name=nm)
        mU = float(p.model.mU)
        print(f"\n=== {nm}  Q2={Q2} W2={W2} t={t}  phi={PHI}deg   "
              f"vmax={p.vmax:.4f}")
        print(f"{'v interval':>14} {'independent':>13} {'exclurad_py':>13} "
              f"{'ratio':>9} {'diff':>10}")
        tot_i = tot_p = 0.0
        for va, vb in zip(EDGES[:-1], EDGES[1:]):
            vb_ = min(vb, p.vmax * (1 - 1e-9))
            if vb_ <= va:
                continue
            xg, wg = np.polynomial.legendre.leggauss(12)
            la, lb = np.log(va), np.log(vb_)
            vv = np.exp(0.5 * (lb - la) * xg + 0.5 * (lb + la))
            ww = 0.5 * (lb - la) * wg * vv
            ind = float(np.sum(ww * np.array(
                [p.dtail_dv(v, nt=240, nps=72) for v in vv])))
            pyv = py_tail(nm, Q2, W2, t, va, vb_, mU)
            tot_i += ind; tot_p += pyv
            r = ind / pyv if pyv else float("nan")
            print(f"{va:6.2f}-{vb_:6.2f} {ind:13.6f} {pyv:13.6f} "
                  f"{r:9.4f} {100*(ind/pyv-1) if pyv else float('nan'):+9.2f}%")
        print(f"{'TOTAL >0.03':>14} {tot_i:13.6f} {tot_p:13.6f} "
              f"{tot_i/tot_p:9.4f} {100*(tot_i/tot_p-1):+9.2f}%")

if __name__ == "__main__":
    main()
