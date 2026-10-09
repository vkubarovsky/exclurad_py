# If the meson bin is not vetoing events, what limits its photons?
# v_max_kin at the SAME observed point in the two parametrisations.
import sys, math
import numpy as np
sys.path.insert(0, "/data/vpk/exclurad_py_vt")
from exclurad_py.core.kinematics import EventKin
MP, MJ, EB = 0.93827208816, 3.0969, 10.6
rng = np.random.default_rng(3)
rows = []
for _ in range(4000):
    Q2 = rng.uniform(0.001, 0.25); W2 = rng.uniform(16.57, 20.57)
    t = rng.uniform(-7.0, -0.45)
    try:
        vm = EventKin.vmax_kin(EB, Q2, W2, t, MJ, M=MP, m=0.51099895e-3,
                               mU=MP, t_kind="photon")
        vp = EventKin.vmax_kin(EB, Q2, W2, t, MP, M=MP, m=0.51099895e-3,
                               mU=MJ, t_kind="target")
    except Exception:
        continue
    if not (np.isfinite(vm) and np.isfinite(vp)) or vm <= 0 or vp <= 0:
        continue
    rows.append((Q2, W2, t, float(vm), float(vp)))
a = np.array(rows)
print("v_max_kin at the same observed point, %d physical points" % len(a))
print("  meson bin  (v = MX^2(e'V) - M_p^2):  median %.4f   mean %.4f   max %.3f"
      % (np.median(a[:,3]), a[:,3].mean(), a[:,3].max()))
print("  proton bin (v = MX^2(e'p) - m_psi^2): median %.4f   mean %.4f   max %.3f"
      % (np.median(a[:,4]), a[:,4].mean(), a[:,4].max()))
r = a[:,4]/a[:,3]
print("  ratio proton/meson: median %.2f   10%% %.2f   90%% %.2f   min %.2f"
      % (np.median(r), np.percentile(r,10), np.percentile(r,90), r.min()))
print("  fraction of points where the proton ceiling is higher: %.1f %%" % (100*(r>1).mean()))
for lo,hi in ((-7.0,-4.0),(-4.0,-2.0),(-2.0,-0.45)):
    s=(a[:,2]>=lo)&(a[:,2]<hi)
    if s.sum()<50: continue
    print("    t %5.1f..%-5.2f  meson %.4f  proton %.4f  ratio %.2f"
          % (lo,hi,np.median(a[s,3]),np.median(a[s,4]),np.median(a[s,4]/a[s,3])))
