"""Regenerate pi0/eta to match the exclugen OSG v2 reference samples.

Reference: ~/exclugen/out/{pi0,eta}/20260810_10.6_rad_osg_v2/osg/ (100k each)
  E=10.6, W 1.8-4.5, Q2 0.2-10, full cos(theta*),
  generated-tail cap v_gen = 0.5 (pi0) / 0.6 (eta),
  model = CLAS12 pi0/eta March-2021 parameterization.

So: model pi0.vpk2021 / eta.vpk2021 (NOT the 2013/TableB1 sets used for the
published-RC validation), and mx2_cut = v_gen + m_meson^2.

The box here is narrower than the reference's, deliberately: the phi run
showed a coarse eta-table over a very wide box gives 14% interpolation
error, which corrupts the soft/hard split (soft = sigma_obs - sigma_hard).
A fine table is built and its interpolation error CHECKED before any events
are generated; the comparison is then done on the overlap region.
"""
import os, sys, time, json
import numpy as np
from exclurad_py import models, stamp
from exclurad_py.core.constants import M_PI0, M_ETA
from exclurad_py.generator import Generator, Box, write_lund

N = int(os.environ.get("NEV", "100000"))
OUT = os.path.dirname(os.path.abspath(__file__))

CFG = {
  # mmiss MUST be the same mU the topology uses, since mx2 = v_gen + mmiss^2
  # is converted straight back to v = MX^2 - mU^2 inside the generator.  A
  # mismatch silently delivers a different v_gen than the one written here.
  "pi0": dict(model="pi0.vpk2021", mmiss=M_PI0, v_gen=0.5,
              box=Box(Q2=(1.0, 5.0), W2=(4.0, 12.0), t=(-2.0, -0.10))),
  "eta": dict(model="eta.vpk2021", mmiss=M_ETA, v_gen=0.6,
              box=Box(Q2=(1.0, 5.0), W2=(4.5, 12.0), t=(-2.0, -0.10))),
}
channels = sys.argv[1:] or list(CFG)
summary = {}
for ch in channels:
    c = CFG[ch]
    print(f"=== {ch}: {c['model']}  v<{c['v_gen']} ===", flush=True)
    sf = models.get(c["model"], t_nucl=-0.30)
    mx2 = c["v_gen"] + c["mmiss"] ** 2
    g = Generator(sf, Ebeam=10.6, box=c["box"], mx2_cut=mx2, helicity=0,
                  seed=20260813, table_shape=(8, 8, 8, 8),
                  majorant_shape=(6, 6, 6, 6), tail_shape=(3, 3, 2, 2),
                  safety=2.5, cache=os.path.join(OUT, f"tab2_{ch}.npz"))
    t0 = time.time(); g.setup(verbose=True); t_setup = time.time() - t0
    chk = g.table.check_interpolation(n=25)
    print(f"  setup {t_setup:.0f}s   eta table interp: {chk}", flush=True)
    if chk.get("rms", 9) > 0.02:
        print(f"  *** WARNING: eta interpolation rms {chk['rms']:.3f} > 0.02 -- "
              f"soft/hard split will be unreliable ***", flush=True)

    t0 = time.time(); ev = g.generate(N, progress=max(N // 5, 1))
    t_gen = time.time() - t0
    print(f"  {len(ev)} events in {t_gen:.0f}s ({1e3*t_gen/len(ev):.2f} ms/ev)", flush=True)

    st = stamp(sf, extra=dict(box=c["box"].as_dict(), mx2_cut=mx2, v_gen=c["v_gen"],
                              Ebeam=10.6, n_events=len(ev),
                              reference="exclugen out/%s/20260810_10.6_rad_osg_v2" % ch,
                              eta_interp=chk))
    lund = os.path.join(OUT, f"{ch}_exclurad_py_vs_exclugen_{N//1000}k.lund")
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st)
    np.savez_compressed(os.path.join(OUT, f"{ch}_vs_exclugen_kin.npz"),
                        Q2=np.array([e["Q2"] for e in ev]),
                        W2=np.array([e["W2"] for e in ev]),
                        t=np.array([e["t"] for e in ev]),
                        E_gamma=np.array([e["photon"][0] if e["photon"] is not None else 0.0 for e in ev]),
                        eta=np.array([e["eta"] for e in ev]))
    r = g.report()
    summary[ch] = dict(model=c["model"], n=n, eta_interp=chk, box=c["box"].as_dict(),
                       v_gen=c["v_gen"], ms_per_event=1e3*t_gen/len(ev),
                       **{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()},
                       health=g.health())
    print(f"  wrote {n} -> {lund}", flush=True)
    print("  health:", g.health() or "clean", flush=True)

with open(os.path.join(OUT, "summary_vs_exclugen.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("ALL DONE", flush=True)
