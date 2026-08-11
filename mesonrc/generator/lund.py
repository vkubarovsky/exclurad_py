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


def event_lines(ev, channel, topology=None, target_pid=PID["p"],
                beam_energy=10.6, pol_beam=0.0, pol_target=0.0, weight=1.0,
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

    parts = [(PID["e-"], 0.51099895e-3, ev["scattered"]),
             (det_pid, mass_of(ev["meson"]), ev["meson"]),
             (mis_pid, mass_of(ev["recoil"]), ev["recoil"])]
    if ev.get("photon") is not None:
        parts.append((PID["gamma"], 0.0, ev["photon"]))

    head = (f"{len(parts):4d} {1:4d} {1:4d} {pol_target:6.3f} {pol_beam:6.3f} "
            f"{PID['e-']:5d} {beam_energy:8.4f} {target_pid:6d} "
            f"{process_id:4d} {weight:12.6e}")
    lines = [head]
    for i, (pid, mass, p4) in enumerate(parts, start=1):
        lines.append(_fmt_particle(i, pid, mass, p4, vertex=vertex))
    return lines


def write_lund(path, events, channel, topology=None, stamp=None,
               comments=True, allow_weighted=False, **kw):
    """Write a list of assembled events to a LUND file.

    events : iterables of dicts from frames.assemble_event, each optionally
             carrying an 'weight' key (must be 1.0 unless allow_weighted).
    stamp  : provenance dict (mesonrc.provenance.stamp) written as comments.
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
            for line in event_lines(ev, channel, topology=topology,
                                    weight=w, **kw):
                fh.write(line + "\n")
            n_written += 1
    return n_written
