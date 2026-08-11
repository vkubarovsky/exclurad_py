"""
Hadronic-CM -> laboratory transformation and event assembly.

`core.kinematics.EventKin` builds every four-vector in the CM of the
observed virtual photon and the target, with q along +z and the leptons in
the x-z plane.  A generator has to deliver laboratory vectors: beam along
+z, target at rest, and a random azimuth about the beam (the physics has no
preferred orientation, so that angle is generated uniformly).

The transformation is
    1. boost by -beta_target so the target is at rest;
    2. rotate so the beam k1 points along +z;
    3. rotate by a uniform random angle about z.

Four-momentum conservation is CLOSED ON THE RECOIL, not imposed by fiat:
    p_recoil = k1 + p_target - k2 - p_meson - k_photon
so whatever the sampler produced, the written event balances exactly.  The
recoil's invariant mass is then checked against the expected value; a
failure means the sampled point was not on the constraint surface and the
event is rejected rather than written.
"""

import numpy as np

from ..core.spinors import dot4


def boost_matrix(beta):
    """Pure boost with velocity beta (3,) acting on contravariant 4-vectors."""
    b2 = float(np.dot(beta, beta))
    if b2 < 1e-30:
        return np.eye(4)
    g = 1.0 / np.sqrt(1.0 - b2)
    L = np.zeros((4, 4))
    L[0, 0] = g
    L[0, 1:] = -g * beta
    L[1:, 0] = -g * beta
    L[1:, 1:] = np.eye(3) + (g - 1.0) / b2 * np.outer(beta, beta)
    return L


def rotation_to_z(vec):
    """Rotation matrix (3,3) taking `vec` onto +z."""
    n = np.asarray(vec, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(np.dot(n, z))
    s = float(np.linalg.norm(v))
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LabTransform:
    """CM -> lab transformation for one observed kinematic point."""

    def __init__(self, kin, azimuth=0.0):
        self.kin = kin
        beta = kin.p[1:] / kin.p[0]           # target velocity in the CM
        self.B = boost_matrix(beta)           # boost to the target rest frame
        k1_rest = self.B @ kin.k1
        R = rot_z(azimuth) @ rotation_to_z(k1_rest[1:])
        self.R4 = np.eye(4)
        self.R4[1:, 1:] = R

    def __call__(self, v4):
        """Transform one or many 4-vectors (…,4) from the CM to the lab."""
        v4 = np.atleast_2d(np.asarray(v4, dtype=float))
        out = v4 @ self.B.T @ self.R4.T
        return out


# --------------------------------------------------------------------------- #
def assemble_event(kin, meson4=None, photon4=None, azimuth=0.0, tol=1e-6):
    """Build the laboratory four-vectors of one event.

    Parameters
    ----------
    kin     : EventKin at the OBSERVED point (fixes k1, k2 and the frame)
    meson4  : (4,) detected-hadron momentum in the CM.  For a radiative
              event this is the hadron at inelasticity v, NOT kin.pv --
              pass what the sampler produced.  None uses kin.pv (soft event).
    photon4 : (4,) real-photon momentum in the CM, or None for a soft event
    azimuth : rotation about the beam axis

    Returns a dict of lab four-vectors: beam, target, scattered, meson,
    photon (or None), recoil -- plus the closure residual.
    """
    T = LabTransform(kin, azimuth)
    pv_cm = kin.pv if meson4 is None else np.asarray(meson4, dtype=float)
    stack = [kin.k1, kin.k2, pv_cm, kin.p]
    if photon4 is not None:
        stack.append(np.asarray(photon4, dtype=float))
    lab = T(np.array(stack))

    beam, scat, meson, target = lab[0], lab[1], lab[2], lab[3]
    photon = lab[4] if photon4 is not None else None

    # target must be at rest after the transformation
    if np.linalg.norm(target[1:]) > 1e-8 * max(1.0, target[0]):
        raise RuntimeError("target not at rest after transformation")

    recoil = beam + target - scat - meson
    if photon is not None:
        recoil = recoil - photon

    m2 = dot4(recoil, recoil)
    expect = kin.mU ** 2
    resid = abs(m2 - expect) / max(1.0, abs(expect))

    return dict(beam=beam, target=target, scattered=scat, meson=meson,
                photon=photon, recoil=recoil, recoil_m2=float(m2),
                recoil_m2_expected=float(expect), closure=float(resid),
                ok=bool(resid < tol))
