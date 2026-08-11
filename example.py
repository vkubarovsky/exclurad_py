"""
Example: RC factors for exclusive rho0 electroproduction with the full set of
structure functions vs. the two-SF (DIFFRAD-like sigma_T/sigma_L) treatment.

Produces rc_example.png and prints a table.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vmrad import rc_factor, demo_rho, TwoSF
from vmrad.constants import M_P

EB, Q2, W2, T = 10.6, 2.5, 12.0, -0.4       # GeV, GeV^2
MX2CUT = (M_P + 0.30) ** 2                  # missing-mass cut MX < M + 300 MeV
SF = demo_rho()
SF2 = TwoSF(SF)

phis = np.linspace(0.0, 2 * np.pi, 13)
rows = []
print(f"rho0:  E={EB} GeV  Q2={Q2}  W2={W2}  t={T}  MX2cut={MX2CUT:.3f} GeV^2")
print(f"{'phi/pi':>7} | {'eta full h=+':>12} {'eta full h=-':>12} "
      f"{'eta T/L h=+':>12} | {'ALU born':>9} {'ALU obs':>9}")
for phi in phis:
    rp = rc_factor(EB, Q2, W2, T, phi, SF, h=+1, mx2_cut=MX2CUT)
    rm = rc_factor(EB, Q2, W2, T, phi, SF, h=-1, mx2_cut=MX2CUT)
    r2 = rc_factor(EB, Q2, W2, T, phi, SF2, h=+1, mx2_cut=MX2CUT)
    alu_b = (rp['sigma_born'] - rm['sigma_born']) / \
            (rp['sigma_born'] + rm['sigma_born'])
    alu_o = (rp['sigma_obs'] - rm['sigma_obs']) / \
            (rp['sigma_obs'] + rm['sigma_obs'])
    rows.append((phi, rp['eta'], rm['eta'], r2['eta'], alu_b, alu_o))
    print(f"{phi/np.pi:7.3f} | {rp['eta']:12.5f} {rm['eta']:12.5f} "
          f"{r2['eta']:12.5f} | {alu_b:9.5f} {alu_o:9.5f}")

rows = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(rows[:, 0] / np.pi, rows[:, 1], 'o-', label='full SFs, h=+1')
ax[0].plot(rows[:, 0] / np.pi, rows[:, 2], 's-', label='full SFs, h=-1')
ax[0].plot(rows[:, 0] / np.pi, rows[:, 3], 'k--', label=r'$\sigma_T,\sigma_L$ only')
ax[0].set_xlabel(r'$\phi/\pi$'); ax[0].set_ylabel(r'RC factor $\eta$')
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title(r'$\rho^0$, $Q^2$=2.5, $W^2$=12, $t$=-0.4 GeV$^2$')

ax[1].plot(rows[:, 0] / np.pi, rows[:, 4], 'o-', label=r'$A_{LU}$ Born')
ax[1].plot(rows[:, 0] / np.pi, rows[:, 5], 's-', label=r'$A_{LU}$ observed')
ax[1].set_xlabel(r'$\phi/\pi$'); ax[1].set_ylabel(r'$A_{LU}$')
ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title('beam-spin asymmetry before/after RC')
fig.tight_layout()
fig.savefig('rc_example.png', dpi=150)
print("\nsaved rc_example.png")
