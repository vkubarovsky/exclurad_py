"""Радиационная поправка к пучковой спиновой асимметрии rho.

    python3 production/rc_bsa_rho.py [-o rc_bsa_rho.png]

Чего здесь НЕ считается: delta для сечения.  Для асимметрии это не тот
ответ.  Асимметрия -- отношение

    A(phi) = [sigma+(phi) - sigma-(phi)] / [sigma+(phi) + sigma-(phi)] ,

и если записать sigma_obs^h = sigma_born^h * eta^h, то при eta+ = eta-
поправка сокращается ТОЧНО.  Значит измеряемый эффект -- это отклонение
eta+ от eta-, а вовсе не величина самой eta.

Считается моментом sin(phi), потому что именно он извлекается в анализе:

    M = Int A(phi) sin(phi) dphi / Int sin^2(phi) dphi ,

и отчёт идёт как M_obs / M_born - 1, то есть во сколько процентов
радиационные поправки сдвигают ИЗВЛЕКАЕМУЮ величину sigma_LT'/sigma_0.

У модели rho.diffrad_vm sigma_LT' тождественно нуль -- она двухфункциональная.
Асимметрия из неё вышла бы нулём, делить было бы не на что.  Поэтому
sigma_LT'/sigma_U подставляется из измерения (env SLTP_OVER_SU, механизм
_frac в models/base.py): наблюдаемая статьи sigma_LT'/sigma_0 при
sigma_0 = sigma_T + eps*sigma_L -- это ровно та же доля.

Чего здесь НЕТ и что квадратурой не берётся: миграция.  Радиационный хвост
уводит событие по phi и по t, и в окно эксклюзивности попадают события,
рождённые в другом бине; модуляция от этого размывается.  Это считается
только генератором, на событиях.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exclurad_py import models                                   # noqa: E402
from exclurad_py.core.rc import rc_factor                        # noqa: E402
from exclurad_py.core.born import born_xsec                      # noqa: E402
from exclurad_py.core.kinematics import EventKin                 # noqa: E402

M_P = 0.93827208816
E0 = 10.6

# Окно эксклюзивности статьи: 0.85 < M_X < 1.10 ГэВ по недостающей массе
# e'pi+pi-X, то есть по протону отдачи.  v = M_X^2 - m_p^2.
VCUT = 1.10 ** 2 - M_P ** 2

# Пять бинов {Q2, x_B} и точки по -t, снятые с Рис. 6, вместе с измеренным
# sigma_LT'/sigma_0.  Значения с рисунка: для ОТНОШЕНИЯ A_obs/A_born они
# почти безразличны, но пусть будут настоящие.
BINS = [
    ("1", 2.73, 0.230, [(0.22, -0.20), (0.36, -0.34), (0.52, -0.36), (0.82, -0.29)]),
    ("2", 2.74, 0.265, [(0.21, -0.20), (0.36, -0.28), (0.54, -0.27), (0.85, -0.23)]),
    ("3", 2.86, 0.397, [(0.27, -0.14), (0.45, -0.17), (0.95, -0.06)]),
    ("4", 5.68, 0.426, [(0.42, -0.24), (1.20, -0.11)]),
    ("5", 5.18, 0.529, [(0.80, -0.10), (1.50, -0.03)]),
]


def w2_of(Q2, xB):
    return M_P ** 2 + Q2 * (1.0 - xB) / xB


def moments(Q2, W2, t, sltp, nphi=16):
    """Момент sin(phi) асимметрии, борновской и наблюдаемой."""
    os.environ["SLTP_OVER_SU"] = "%.6f" % sltp
    sf = models.get("rho.diffrad_vm")
    sf.Ebeam = E0
    mx2 = VCUT + sf.mU ** 2
    phis = (np.arange(nphi) + 0.5) * 2.0 * np.pi / nphi
    ab, ao = [], []
    for ph in phis:
        try:
            kin = EventKin(E0, Q2, W2, t, ph, sf.mV, v=0.0, mU=sf.mU,
                           t_kind=getattr(sf.topology, "t_kind", "photon"),
                           phi_offset=float(getattr(sf.topology, "phi_offset", 0.0)))
            bp = born_xsec(kin, sf, +1)
            bm = born_xsec(kin, sf, -1)
            op = rc_factor(E0, Q2, W2, t, ph, sf, h=+1, mx2_cut=mx2)["sigma_obs"]
            om = rc_factor(E0, Q2, W2, t, ph, sf, h=-1, mx2_cut=mx2)["sigma_obs"]
        except Exception:
            return None
        if not all(np.isfinite(x) and x > 0 for x in (bp, bm, op, om)):
            return None
        ab.append((bp - bm) / (bp + bm))
        ao.append((op - om) / (op + om))
    s = np.sin(phis)
    norm = np.sum(s * s)
    return float(np.sum(np.array(ab) * s) / norm), float(np.sum(np.array(ao) * s) / norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="rc_bsa_rho.png")
    ap.add_argument("-n", "--nphi", type=int, default=16)
    a = ap.parse_args()

    print("rho BSA, E = %.1f GeV, окно 0.85 < M_X < 1.10 -> v_cut = %.4f ГэВ^2\n"
          % (E0, VCUT), flush=True)
    print("%-4s %6s %6s %7s %7s %10s %10s %9s"
          % ("бин", "Q2", "xB", "W2", "-t", "A_born", "A_obs", "сдвиг"), flush=True)
    rows = []
    for name, Q2, xB, pts in BINS:
        W2 = w2_of(Q2, xB)
        for mt, sltp in pts:
            r = moments(Q2, W2, -mt, sltp, a.nphi)
            if r is None:
                print("%-4s %6.2f %6.3f %7.2f %7.2f      недостижимо" % (name, Q2, xB, W2, mt), flush=True)
                continue
            mb, mo = r
            sh = 100.0 * (mo / mb - 1.0) if mb else float("nan")
            print("%-4s %6.2f %6.3f %7.2f %7.2f %10.5f %10.5f %8.2f%%"
                  % (name, Q2, xB, W2, mt, mb, mo, sh), flush=True)
            rows.append((name, Q2, xB, mt, mb, mo, sh))

    if not rows:
        sys.exit("ни одной точки не посчиталось")
    sh = np.array([r[6] for r in rows])
    print("\nсдвиг извлекаемой sigma_LT'/sigma_0: медиана %.2f %%, диапазон %.2f..%.2f %%"
          % (np.median(sh), sh.min(), sh.max()))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=140)
    colors = plt.cm.viridis(np.linspace(0, .85, len(BINS)))
    ci = {b[0]: colors[i] for i, b in enumerate(BINS)}
    for name, Q2, xB, mt, mb, mo, s in rows:
        ax.plot(mt, s, "o", ms=8, color=ci[name])
    for i, b in enumerate(BINS):
        ax.plot([], [], "o", color=colors[i],
                label=r"$Q^2$=%.2f, $x_B$=%.3f" % (b[1], b[2]))
    ax.axhline(0, color="0.4", lw=.8)
    ax.set_xlabel(r"$-t$ (GeV$^2$)", fontsize=11)
    ax.set_ylabel(r"shift of extracted $\sigma_{LT'}/\sigma_0$ (%)", fontsize=11)
    ax.set_title(r"$\rho^0$ BSA: radiative shift of the $\sin\phi$ moment, "
                 r"$M_X$ window 0.85-1.10 GeV", fontsize=11)
    ax.grid(alpha=.3, lw=.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("записано:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
