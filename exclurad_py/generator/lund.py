"""
LUND output for the CLAS12 simulation chain (GEMC).

Format (one header line, then one line per particle):

header:   nParticles  A  Z  polTarget  polBeam  beamPID  beamE  nucleonPID
          processID  weight
particle: index  lifetime  status  PID  parentIndex  daughterIndex
          px py pz  E  mass  vx vy vz

Events written here are UNWEIGHTED -- every event is equally probable and
the weight column is 1.0.  That is deliberate: GEMC does not carry weights
through the simulation, and downstream CLAS12 analysis counts events.  A
weighted mode exists in the generator for diagnostics, but it must not be
written to LUND, and `write_lund` refuses events whose weight differs from
one unless `allow_weighted=True` is passed explicitly.

Provenance (code version, model name and version, cuts) is written as
leading comment lines.  GEMC ignores lines starting with '#'; if a
downstream tool does not, pass `comments=False`.
"""

import numpy as np

# PDG codes
PID = {"e-": 11, "gamma": 22, "pi0": 111, "pi+": 211, "eta": 221,
       "rho0": 113, "phi": 333, "jpsi": 443, "p": 2212, "n": 2112}

CHANNEL_PID = {"pi0": PID["pi0"], "eta": PID["eta"], "rho": PID["rho0"],
               "phi": PID["phi"], "jpsi": PID["jpsi"]}


def _fmt_particle(index, pid, mass, p4, parent=0, daughter=0,
                  status=1, lifetime=0.0, vertex=(0.0, 0.0, 0.0)):
    E, px, py, pz = p4
    return (f"{index:4d} {lifetime:6.2f} {status:3d} {pid:8d} "
            f"{parent:3d} {daughter:3d} "
            f"{px:18.10e} {py:18.10e} {pz:18.10e} {E:18.10e} {mass:14.8f} "
            f"{vertex[0]:8.4f} {vertex[1]:8.4f} {vertex[2]:8.4f}")


NAME_TO_PID = {"pi0": 111, "pi+": 211, "eta": 221, "rho0": 113, "phi": 333,
               "J/psi": 443, "p": 2212, "n": 2112, "gamma": 22}

# daughters written for a decayed meson; mirrors generator.decay.DECAY
DECAY_PID = {"rho":  (211, -211, 0.13957039),
             "phi":  (321, -321, 0.493677),
             "jpsi": (-11, 11, 0.51099895e-3),
             "pi0":  (22, 22, 0.0),
             "eta":  (22, 22, 0.0)}


def event_lines(ev, channel, topology=None, target_pid=PID["p"],
                beam_energy=10.6, pol_beam=None, pol_target=0.0, weight=1.0,
                process_id=0, vertex=(0.0, 0.0, 0.0)):
    """LUND lines for one assembled event (see frames.assemble_event).

    PID assignment is TOPOLOGY-DRIVEN, not slot-driven.  `ev["meson"]` is
    the *detected* hadron and `ev["recoil"]` the *missing* one -- and in the
    EXCLURAD pi0/eta configuration the detected hadron is the PROTON and the
    missing one is the meson.  Labelling by slot name gives an eta with the
    proton's momentum, which is why the topology is required here whenever
    it is not the plain meson-detected case.
    """
    if topology is not None:
        det_pid = NAME_TO_PID.get(topology.detected)
        mis_pid = NAME_TO_PID.get(topology.missing)
        if det_pid is None or mis_pid is None:
            raise KeyError(f"no PDG code for topology particles "
                           f"{topology.detected!r} / {topology.missing!r}")
    else:
        det_pid, mis_pid = CHANNEL_PID[channel], PID["p"]

    def mass_of(p4):
        return float(np.sqrt(max(p4[0] ** 2 - np.dot(p4[1:], p4[1:]), 0.0)))

    # (pid, mass, p4, status, parent, daughter)
    parts = [[PID["e-"], 0.51099895e-3, ev["scattered"], 1, 0, 0],
             [det_pid, mass_of(ev["meson"]), ev["meson"], 1, 0, 0],
             [mis_pid, mass_of(ev["recoil"]), ev["recoil"], 1, 0, 0]]
    if ev.get("photon") is not None:
        parts.append([PID["gamma"], 0.0, ev["photon"], 1, 0, 0])

    # A decayed vector meson becomes documentation (status 0) pointing at the
    # first daughter, and the two daughters are appended as the tracked
    # particles.  Without this the daughters computed in run.py are silently
    # dropped and GEMC is handed an undecayed rho/phi/J/psi.
    d = ev.get("daughters")
    if d is not None and d[0] is not None and channel in DECAY_PID:
        pid_a, pid_b, m_d = DECAY_PID[channel]
        # Locate the meson by PID, not by slot position: it is the DETECTED
        # hadron for the vector mesons but the MISSING one for pi0/eta, so
        # parts[1] is the proton in the EXCLURAD topology.  Matching on
        # CHANNEL_PID is unambiguous even when a radiated photon is already
        # in the list -- that is pid 22, the pi0 is 111.
        meson_pid = CHANNEL_PID[channel]
        mi = next((i for i, p in enumerate(parts) if p[0] == meson_pid), None)
        if mi is not None:
            parts[mi][3] = 0                     # meson -> documentation
            parts[mi][5] = len(parts) + 1        # first daughter (1-based)
            for pid_d, p4_d in ((pid_a, d[0]), (pid_b, d[1])):
                parts.append([pid_d, m_d, p4_d, 1, mi + 1, 0])

    # Beam polarisation is LUND header column 5, and it is where the event's
    # own helicity belongs.  The generator already draws h per event when the
    # card says ihel = 0 (events.py:463), but until now that number died
    # inside the generator: the file said 0.000 for every event, so a
    # beam-spin asymmetry could not be measured from one sample at all --
    # you had to run twice with ihel = +1 and -1 and hope the two samples
    # matched.  Pass pol_beam explicitly to override.
    if pol_beam is None:
        pol_beam = float(ev.get("helicity", 0.0) or 0.0)

    head = (f"{len(parts):4d} {1:4d} {1:4d} {pol_target:6.3f} {pol_beam:6.3f} "
            f"{PID['e-']:5d} {beam_energy:8.4f} {target_pid:6d} "
            f"{process_id:4d} {weight:12.6e}")
    lines = [head]
    for i, (pid, mass, p4, status, parent, daughter) in enumerate(parts, start=1):
        lines.append(_fmt_particle(i, pid, mass, p4, parent=parent,
                                   daughter=daughter, status=status,
                                   vertex=vertex))
    return lines


def write_lund(path, events, channel, topology=None, stamp=None,
               comments=True, allow_weighted=False, **kw):
    """Write a list of assembled events to a LUND file.

    events : iterables of dicts from frames.assemble_event, each optionally
             carrying an 'weight' key (must be 1.0 unless allow_weighted).
    stamp  : provenance dict (exclurad_py.provenance.stamp) written as comments.
    """
    from ..provenance import header_lines

    n_written = 0
    with open(path, "w") as fh:
        if comments and stamp is not None:
            for line in header_lines(stamp):
                fh.write(line + "\n")
            fh.write("# NOTE: events are UNWEIGHTED; weight column is 1.0\n")
        for ev in events:
            w = float(ev.get("weight", 1.0))
            if not allow_weighted and abs(w - 1.0) > 1e-12:
                raise ValueError(
                    "refusing to write weighted events to LUND (weight="
                    f"{w}); GEMC does not propagate weights. Generate "
                    "unweighted, or pass allow_weighted=True knowingly.")
            if not ev.get("ok", True):
                continue
            # Вершина -- СВОЯ у каждого события.  До появления
            # до-вершинного радиатора здесь стоял общий (0,0,0) на весь
            # файл: мишень в выводе не была представлена никак.
            kw_ev = dict(kw)
            if "vz" in ev and "vertex" not in kw_ev:
                kw_ev["vertex"] = (0.0, 0.0, float(ev["vz"]))
            for line in event_lines(ev, channel, topology=topology,
                                    weight=w, **kw_ev):
                fh.write(line + "\n")
            n_written += 1
    return n_written
