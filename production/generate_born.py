"""Born-only generation for all five channels, written to /Volumes/wd_14tb.

WHY BORN FIRST.  The radiative phi/J/psi samples disagree with diffrad_vm by
2.8x / 5.3x in hard-photon rate.  The cross-section models are now VERIFIED
identical (validation/vm_models: 8-9 digits), so the disagreement is either
the RC/tail treatment or the sampling.  A Born-only comparison separates
them: with no radiative tail at all, any remaining difference in dN/dQ2,
dN/dt, dN/dW2 is the Born cross section, the virtual-photon flux convention
or the phase-space sampling -- nothing to do with RC.

Boxes match the reference Born run cards:
  pi0/eta  exclugen  20260810_10.6_born_osg_v2 (W 1.8-4.5, Q2 0.2-10, full cth)
  phi      diffrad_vm input_phi_born_100k.txt  (Q2 0.4-10, y .25-.9, |t| .2-8)
  jpsi     diffrad_vm input_jpsi_born_100k.txt (Q2 .001-.25, y .79-.99, |t| 0-7)
  rho      diffrad_vm defaults

CAVEAT: exclugen's Born run has itarg=1, i.e. PRE-VERTEX EXTERNAL RADIATION
IS ON even in its Born sample.  exclurad_py Born is pure Born.  Expect a
difference in the scattered-electron spectrum from that alone.
"""
import os, sys, json, time
import numpy as np
from exclurad_py import models, stamp
from exclurad_py.generator import Generator, Box, write_lund

N = int(os.environ.get("NEV", "100000"))
ROOT = "/Volumes/wd_14tb/mc/exclurad_py"
DATE, EB = "20260813", "10.6"

CFG = {
 "pi0":  dict(model="pi0.vpk2021", kw=dict(t_nucl=-0.30),
              box=Box(Q2=(1.0, 5.0), W2=(4.0, 12.0), t=(-2.0, -0.10)), safety=2.5),
 "eta":  dict(model="eta.vpk2021", kw=dict(t_nucl=-0.30),
              box=Box(Q2=(1.0, 5.0), W2=(4.5, 12.0), t=(-2.0, -0.10)), safety=2.5),
 "rho":  dict(model="rho.diffrad_vm", kw={},
              box=Box(Q2=(0.5, 6.0), W2=(5.0, 20.5), t=(-3.0, -0.15)), safety=3.0),
 "phi":  dict(model="phi.diffrad_vm", kw={},
              box=Box(Q2=(0.4, 6.0), W2=(5.0, 20.5), t=(-4.0, -0.2)), safety=3.0),
 "jpsi": dict(model="jpsi.diffrad_vm", kw={},
              box=Box(Q2=(0.02, 0.25), W2=(16.6, 20.6), t=(-5.0, -0.2)), safety=4.0),
}

channels = sys.argv[1:] or list(CFG)
summary = {}
for ch in channels:
    c = CFG[ch]
    tag = f"{DATE}_{EB}_born_v1"
    d = os.path.join(ROOT, ch, tag)
    for sub in ("lund", "inputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    print(f"=== {ch} BORN: {c['model']} -> {d}", flush=True)

    sf = models.get(c["model"], **c["kw"])
    g = Generator(sf, Ebeam=10.6, box=c["box"], mx2_cut=None, helicity=0,
                  seed=20260813, majorant_shape=(7, 7, 7, 7),
                  safety=c["safety"], born_only=True)
    t0 = time.time(); g.setup(); t_setup = time.time() - t0
    t0 = time.time(); ev = g.generate(N, progress=max(N // 4, 1))
    t_gen = time.time() - t0

    st = stamp(sf, extra=dict(mode="BORN ONLY (no radiative corrections)",
                              box=c["box"].as_dict(), Ebeam=10.6, n_events=len(ev)))
    lund = os.path.join(d, "lund", f"{ch}_exclurad_py_born_{EB}GeV.lund")
    n = write_lund(lund, ev, channel=ch, topology=sf.topology, stamp=st)
    np.savez_compressed(os.path.join(d, f"{ch}_born_kin.npz"),
                        Q2=np.array([e["Q2"] for e in ev]),
                        W2=np.array([e["W2"] for e in ev]),
                        t=np.array([e["t"] for e in ev]),
                        phi=np.array([e["phi"] for e in ev]),
                        scattered=np.array([e["scattered"] for e in ev]),
                        meson=np.array([e["meson"] for e in ev]),
                        recoil=np.array([e["recoil"] for e in ev]))
    r = g.report()
    cfg = dict(channel=ch, model=c["model"], box=c["box"].as_dict(), Ebeam=10.6,
               mode="born", n=n, seed=20260813, safety=c["safety"])
    with open(os.path.join(d, "inputs", "config.json"), "w") as fh:
        json.dump(dict(config=cfg, stamp=st, report={k: (float(v) if isinstance(v, float) else v)
                                                     for k, v in r.items()}), fh, indent=2, default=str)
    with open(os.path.join(d, "README.md"), "w") as fh:
        fh.write(f"# exclurad_py {ch} BORN, {EB} GeV, {n} events\n\n"
                 f"Model `{c['model']}`, box {c['box'].as_dict()}.\n"
                 f"Born only: no radiative corrections, no photon, eta=1.\n"
                 f"Unweighted; LUND weight column is 1.0.\n\n"
                 f"Layout mirrors `/Volumes/wd_14tb/mc/exclugen/`.\n")
    summary[ch] = dict(n=n, ms_per_event=1e3*t_gen/len(ev), setup_s=t_setup,
                       **{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()},
                       health=g.health(), path=lund)
    print(f"  {n} events, {1e3*t_gen/len(ev):.2f} ms/ev, health: {g.health() or 'clean'}", flush=True)

with open(os.path.join(ROOT, f"summary_born_{DATE}.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("ALL DONE", flush=True)
