"""Generate 100k unweighted events with hard photons: pi0, eta, phi, J/psi.

Boxes for phi and J/psi match the diffrad_vm runs that produced
~/MC_vector_mesons/samples_v2/*_100k.lund so the comparison is like-for-like
(input_phi_rc_100k.txt, input_jpsi_rc_100k.txt): Ebeam 10.6, cutv 0 (no
missing-mass cut, so v runs to the kinematic limit).

pi0/eta use their published-analysis cuts (v < 0.17 / 0.20) since no
diffrad_vm reference exists for them.

NOTE on the radiative fraction: it is NOT a physical observable to compare
directly.  It is set by where the soft/hard split v_min is placed, which is
a generator bookkeeping choice.  The comparable quantity is the photon
energy spectrum above a common threshold.
"""
import os, sys, time, json
import numpy as np
from exclurad_py import models, stamp
from exclurad_py.generator import Generator, Box, write_lund

N = int(os.environ.get("NEV", "100000"))
OUT = os.path.dirname(os.path.abspath(__file__))

CFG = {
  "pi0":  dict(model="pi0.vpk2013", kw=dict(t_nucl=-0.30),
               box=Box(Q2=(1.0, 4.5), W2=(4.0, 10.0), t=(-0.8, -0.15)),
               mx2=0.17 + 0.13957**2, safety=2.0),
  "eta":  dict(model="eta.tableB1", kw=dict(t_nucl=-0.30),
               box=Box(Q2=(1.0, 4.5), W2=(4.5, 10.0), t=(-0.8, -0.15)),
               mx2=0.20 + 0.5473**2, safety=2.0),
  "phi":  dict(model="phi.diffrad_vm", kw={},
               box=Box(Q2=(0.4, 6.0), W2=(5.0, 20.5), t=(-4.0, -0.2)),
               mx2=None, safety=2.5),
  "jpsi": dict(model="jpsi.diffrad_vm", kw={},
               box=Box(Q2=(0.02, 0.25), W2=(16.6, 20.6), t=(-5.0, -0.2)),
               mx2=None, safety=4.0),
}

channels = sys.argv[1:] or list(CFG)
summary = {}
for ch in channels:
    c = CFG[ch]
    print(f"=== {ch}: {c['model']} ===", flush=True)
    sf = models.get(c["model"], **c["kw"])
    g = Generator(sf, Ebeam=10.6, box=c["box"], mx2_cut=c["mx2"], helicity=0,
                  seed=20260812, table_shape=(4, 4, 4, 6),
                  majorant_shape=(6, 6, 6, 6), tail_shape=(2, 2, 2, 2),
                  safety=c["safety"],
                  cache=os.path.join(OUT, f"etatable_{ch}.npz"))
    t0 = time.time(); g.setup(verbose=True); t_setup = time.time() - t0
    print(f"  setup {t_setup:.1f}s", flush=True)

    t0 = time.time(); ev = g.generate(N, progress=max(N // 10, 1))
    t_gen = time.time() - t0
    print(f"  generated {len(ev)} in {t_gen:.1f}s ({1e3*t_gen/len(ev):.2f} ms/ev)", flush=True)

    lund = os.path.join(OUT, f"{ch}_exclurad_py_{N//1000}k.lund")
    st = stamp(sf, extra=dict(box=c["box"].as_dict(), mx2_cut=c["mx2"],
                              Ebeam=10.6, n_events=len(ev)))
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st)

    # kinematics dump for the comparison
    rad = np.array([e["radiative"] for e in ev])
    egam = np.array([e["photon"][0] if e["photon"] is not None else 0.0 for e in ev])
    np.savez_compressed(os.path.join(OUT, f"{ch}_kin.npz"),
                        Q2=np.array([e["Q2"] for e in ev]),
                        W2=np.array([e["W2"] for e in ev]),
                        t=np.array([e["t"] for e in ev]),
                        phi=np.array([e["phi"] for e in ev]),
                        radiative=rad, E_gamma=egam,
                        scattered=np.array([e["scattered"] for e in ev]),
                        meson=np.array([e["meson"] for e in ev]),
                        recoil=np.array([e["recoil"] for e in ev]))
    r = g.report()
    summary[ch] = dict(model=c["model"], n=n, setup_s=t_setup, gen_s=t_gen,
                       ms_per_event=1e3*t_gen/len(ev), **{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()},
                       health=g.health(), lund=lund)
    print(f"  wrote {n} -> {lund}", flush=True)
    print("  health:", g.health() or "clean", flush=True)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("ALL DONE", flush=True)
