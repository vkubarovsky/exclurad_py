"""
Two-body meson decay and the acceptance cuts the reference generators apply.

Why this exists
---------------
`diffrad_vm` decays the vector meson (`decay_rho`, line 1117: "vector meson
-> h+ h- isotropic in rest frame") and then applies |p| / theta cuts to the
scattered electron, the recoil proton AND both decay daughters
(`check_acceptance`, line 1196).  Those cuts are applied AT GENERATION, so
100% of the reference events satisfy them.

That matters for any comparison of radiative rates: a radiative event has
lost beam energy, so its daughters are softer and more likely to fail the
acceptance.  Comparing a reference sample that lives inside the acceptance
against a exclurad_py sample with no acceptance at all compares two different
populations, and biases the radiative fraction downward in the reference.

So: decay isotropically the same way, apply the same cuts to both, then
compare.  The decay is isotropic in the meson rest frame -- matching
diffrad_vm exactly.  This is consistent with the two-SF mode: with
sigma_TT = sigma_LT = sigma_LT' = 0 and no SDMEs there is no angular
information to impose, so isotropic is the right choice, not an
approximation.
"""

import numpy as np

# meson -> (h+ pid, h- pid, daughter mass)
DECAY = {
    "rho":  (211, -211, 0.13957039),      # rho0 -> pi+ pi-
    "phi":  (321, -321, 0.493677),        # phi -> K+ K-
    "jpsi": (-11, 11, 0.51099895e-3),     # J/psi -> e+ e-
    # pi0/eta -> gamma gamma.  exclugen writes 2 photons per meson and GEMC
    # needs them: an undecayed pi0 deposits nothing in the calorimeter.
    # Isotropic in the meson rest frame is exact here -- a pseudoscalar
    # decaying to two photons has no preferred direction.
    "pi0":  (22, 22, 0.0),
    "eta":  (22, 22, 0.0),
}
# Free-neutron channels decay exactly like their proton counterparts -- the
# meson is the same particle.  They are listed EXPLICITLY rather than matched
# by prefix: `ch in DECAY` is a silent False for a missing key, so when the
# neutron channels were renamed from "pi0"/"eta" the pi0 -> gamma gamma decay
# simply stopped happening and the LUND came out with 3 particles instead of
# 5, with no error anywhere.
DECAY["pi0n"] = DECAY["pi0"]
DECAY["etan"] = DECAY["eta"]


# Which event slot holds the MESON.  The core kinematics names its slots by
# role, not by particle: ev["meson"] is the DETECTED hadron and ev["recoil"]
# the missing one.  Under the EXCLURAD pi0/eta topology the detected hadron is
# the PROTON, so decaying ev["meson"] there would blow apart the proton.
_NUCLEONS = ("p", "n")


def meson_slot(topology, channel):
    """'meson' or 'recoil' -- whichever slot actually holds the meson.

    Decided by WHAT the missing particle is, never by comparing its name to
    the channel name.  That comparison held only while every channel was
    named after its meson: the free-neutron channels are "pi0n"/"etan" while
    their missing particle is still "pi0"/"eta", so the test came out False,
    the slot fell back to "meson" -- which under baryon_detected holds the
    DETECTED NUCLEON -- and the neutron was decayed into two photons while
    the pi0 was written as an undecayed status-0 parent.  Four-momentum then
    failed by several GeV per event.
    """
    if topology is None:
        return "meson"
    return "meson" if getattr(topology, "missing", "") in _NUCLEONS else "recoil"


def boost_from_rest(p4_parent, p4_rest):
    """Boost a rest-frame 4-vector into the frame where the parent is p4_parent."""
    M = np.sqrt(max(p4_parent[0] ** 2 - np.dot(p4_parent[1:], p4_parent[1:]), 1e-30))
    beta = p4_parent[1:] / p4_parent[0]
    b2 = float(np.dot(beta, beta))
    if b2 < 1e-30:
        return p4_rest.copy()
    gamma = p4_parent[0] / M
    bp = float(np.dot(beta, p4_rest[1:]))
    E = gamma * (p4_rest[0] + bp)
    vec = (p4_rest[1:] + beta * ((gamma - 1.0) * bp / b2 + gamma * p4_rest[0]))
    return np.concatenate([[E], vec])


def decay_two_body(p4_meson, m_daughter, rng):
    """Isotropic two-body decay in the meson rest frame -> (p4_plus, p4_minus)."""
    M2 = p4_meson[0] ** 2 - np.dot(p4_meson[1:], p4_meson[1:])
    M = np.sqrt(max(M2, 1e-30))
    if M <= 2.0 * m_daughter:
        return None, None
    pstar = np.sqrt(M * M / 4.0 - m_daughter ** 2)
    ct = rng.uniform(-1.0, 1.0)
    st = np.sqrt(max(1.0 - ct * ct, 0.0))
    ph = rng.uniform(0.0, 2.0 * np.pi)
    n = np.array([st * np.cos(ph), st * np.sin(ph), ct])
    E = np.sqrt(pstar ** 2 + m_daughter ** 2)
    a = np.concatenate([[E], pstar * n])
    b = np.concatenate([[E], -pstar * n])
    return boost_from_rest(p4_meson, a), boost_from_rest(p4_meson, b)


# --------------------------------------------------------------------------- #
def p_theta(p4):
    """|p| [GeV] and polar angle [deg] of a lab 4-vector."""
    p = np.sqrt(np.sum(np.asarray(p4)[..., 1:] ** 2, axis=-1))
    pz = np.asarray(p4)[..., 3]
    th = np.degrees(np.arccos(np.clip(pz / np.maximum(p, 1e-12), -1.0, 1.0)))
    return p, th


# Acceptance windows transcribed from the reference run cards
# (input_jpsi_{born,rc}_100k.txt).  phi/rho cards impose no |p|/theta cuts.
ACCEPTANCE = {
    "jpsi": dict(electron=(0.1, 2.5, 2.0, 20.0),
                 proton=(0.4, 4.5, 5.0, 30.0),
                 daughter=(1.0, 9.0, 5.0, 40.0)),
}


def in_acceptance(channel, scattered, recoil, daughters=None):
    """Boolean mask: does the event pass the reference's |p|/theta cuts?

    daughters: sequence of two lab 4-vectors, or None to skip that cut.
    Channels with no acceptance card return all-True.
    """
    acc = ACCEPTANCE.get(channel)
    n = len(np.atleast_2d(scattered))
    if acc is None:
        return np.ones(n, dtype=bool)

    def win(p4, w):
        pmin, pmax, tmin, tmax = w
        p, th = p_theta(np.atleast_2d(p4))
        return (p >= pmin) & (p <= pmax) & (th >= tmin) & (th <= tmax)

    m = win(scattered, acc["electron"]) & win(recoil, acc["proton"])
    if daughters is not None and "daughter" in acc:
        for d in daughters:
            m = m & win(d, acc["daughter"])
    return m
