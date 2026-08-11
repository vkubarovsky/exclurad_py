"""
Example: RC factors for exclusive pseudoscalar electroproduction
(pi0 p and pi+ n) with the full set of structure functions, vs. the
two-SF (sigma_T/sigma_L only) treatment.

Produces rc_example_ps.png and prints tables.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vmrad import rc_factor, demo_pi0, demo_pip, TwoSF
from vmrad.constants import M_P, M_N

EB, Q2, W2, T = 10.6, 2.2, 7.0, -0.3        # CLAS12-like kinematics

cases = [
    ("pi0 p", demo_pi0(), M_P + 0.20),      # MX(=p) cut: M_p + 200 MeV
    ("pi+ n", demo_pip(), M_N + 0.20),      # MX(=n) cut: M_n + 200 MeV
]

phis = np.linspace(0.0, 2 * np.pi, 13)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for icase, (name, SF, mxcut) in enumerate(cases):
    SF2 = TwoSF(SF)
    print(f"\n{name}:  E={EB} Q2={Q2} W2={W2} t={T}  MX<{mxcut:.3f} GeV")
    print(f"{'phi/pi':>7} | {'eta full h=+':>12} {'eta full h=-':>12} "
          f"{'eta T/L':>10} | {'ALU born':>9} {'ALU obs':>9}")
    rows = []
    for phi in phis:
        rp = rc_factor(EB, Q2, W2, T, phi, SF, h=+1, mx2_cut=mxcut ** 2)
        rm = rc_factor(EB, Q2, W2, T, phi, SF, h=-1, mx2_cut=mxcut ** 2)
        r2 = rc_factor(EB, Q2, W2, T, phi, SF2, h=+1, mx2_cut=mxcut ** 2)
        alu_b = (rp['sigma_born'] - rm['sigma_born']) / \
                (rp['sigma_born'] + rm['sigma_born'])
        alu_o = (rp['sigma_obs'] - rm['sigma_obs']) / \
                (rp['sigma_obs'] + rm['sigma_obs'])
        rows.append((phi, rp['eta'], rm['eta'], r2['eta'], alu_b, alu_o))
        print(f"{phi/np.pi:7.3f} | {rp['eta']:12.5f} {rm['eta']:12.5f} "
              f"{r2['eta']:10.5f} | {alu_b:9.5f} {alu_o:9.5f}")
    rows = np.array(rows)
    a = ax[icase]
    a.plot(rows[:, 0] / np.pi, rows[:, 1], 'o-', label='full SFs, h=+1')
    a.plot(rows[:, 0] / np.pi, rows[:, 2], 's-', label='full SFs, h=-1')
    a.plot(rows[:, 0] / np.pi, rows[:, 3], 'k--', label=r'$\sigma_T,\sigma_L$ only')
    a.set_xlabel(r'$\phi/\pi$'); a.set_ylabel(r'RC factor $\eta$')
    a.set_title(f'{name},  $Q^2$={Q2}, $W^2$={W2}, $t$={T} GeV$^2$')
    a.legend(); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig('rc_example_ps.png', dpi=150)
print("\nsaved rc_example_ps.png")
