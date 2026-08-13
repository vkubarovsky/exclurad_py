"""
Structure-function model interface.

Every model in this package is one class carrying, in addition to the
numbers, the metadata needed to answer "which cross section produced this
table?" eight months later:

    name        "pi0.vpk2021"        registry key, stamped into every output
    channel     "pi0"
    version     "2021"
    provenance  where it came from and how the port was verified
    reference   paper / note / repo path
    validity    dict of (lo, hi) ranges for W2, Q2, -t
    topology    a exclurad_py.topology.Topology instance

`validity` is not bureaucracy: the radiative tail evaluates the structure
functions at shifted kinematics -- W2~ down to threshold, Q2~ on both sides
of the observed value -- so a fit made on a limited grid *will* be called
outside it.  Each model declares its range and its out-of-range policy;
`OutOfRange.FREEZE` (clamp to the boundary) is the default because a
hard zero puts a discontinuity inside the photon integral, but the number
of out-of-range calls is counted and reported.
"""

import numpy as np

from ..topology import Topology, FIXED_NUCLEON

SF_KEYS = ("sT", "sL", "sTT", "sLT", "sLTp")


class OutOfRange:
    FREEZE = "freeze"    # clamp arguments to the validity boundary
    ZERO = "zero"        # return 0 outside (discontinuous -- use with care)
    RAISE = "raise"      # abort


class StructureFunctions:
    """Base class for all SF models.

    Subclasses implement `_sfs(W2, Q2, t)` returning a dict with the five
    keys of SF_KEYS, each d sigma_i/dt in **nb/GeV^2** (Hand convention),
    broadcasting over numpy arrays.  Callers use `__call__`, which applies
    the validity policy, the topology sign convention, and the call
    counters.

    Do not override `__call__` in a subclass; override `_sfs`.
    """

    name = None
    channel = None
    version = None
    provenance = ""
    reference = ""
    validity = {}                 # e.g. dict(W2=(1.2, 12.0), Q2=(0.1, 8.0))
    out_of_range = OutOfRange.FREEZE
    topology: Topology = None

    def __init__(self):
        self.n_calls = 0
        self.n_points = 0
        self.n_out_of_range = 0

    # -- the two masses the core kinematics needs -------------------------- #
    @property
    def mV(self):
        """Mass of the hadron defining the kinematics (EventKin mV)."""
        return self.topology.m_detected

    @property
    def mU(self):
        """Mass of the undetected system (EventKin mU)."""
        return self.topology.m_missing

    # -- implemented by subclasses ----------------------------------------- #
    def _sfs(self, W2, Q2, t):
        raise NotImplementedError

    # -- public entry point ------------------------------------------------ #
    def __call__(self, W2, Q2, t):
        W2 = np.atleast_1d(np.asarray(W2, dtype=float))
        Q2 = np.atleast_1d(np.asarray(Q2, dtype=float))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        W2, Q2, t = np.broadcast_arrays(W2, Q2, t)

        W2c, Q2c, tc, inrange = self._apply_validity(W2, Q2, t)

        sf = self._sfs(W2c, Q2c, tc)
        missing = set(SF_KEYS) - set(sf)
        if missing:
            raise KeyError(f"{type(self).__name__}._sfs omitted {sorted(missing)}")

        out = {}
        for k in SF_KEYS:
            v = np.broadcast_to(np.asarray(sf[k], dtype=float), W2.shape).copy()
            if k in ("sLT", "sLTp"):
                v = v * self.topology.lt_sign
            if self.out_of_range == OutOfRange.ZERO:
                v = np.where(inrange, v, 0.0)
            out[k] = v

        self.n_calls += 1
        self.n_points += W2.size
        self.n_out_of_range += int(np.count_nonzero(~inrange))
        return out

    # -- validity handling -------------------------------------------------- #
    def _apply_validity(self, W2, Q2, t):
        inrange = np.ones(W2.shape, dtype=bool)
        vals = {"W2": W2, "Q2": Q2, "mt": -t}
        clamped = {"W2": W2, "Q2": Q2, "mt": -t}
        for key, (lo, hi) in (self.validity or {}).items():
            if key not in vals:
                raise KeyError(f"validity key {key!r} not one of W2, Q2, mt")
            v = vals[key]
            inrange &= (v >= lo) & (v <= hi)
            clamped[key] = np.clip(v, lo, hi)

        if self.out_of_range == OutOfRange.RAISE and not inrange.all():
            bad = np.argwhere(~inrange)[0]
            raise ValueError(
                f"{self.name}: kinematics outside validity range at index "
                f"{tuple(bad)}: W2={W2[tuple(bad)]:.4g} Q2={Q2[tuple(bad)]:.4g} "
                f"t={t[tuple(bad)]:.4g}; validity={self.validity}")

        if self.out_of_range == OutOfRange.FREEZE:
            return clamped["W2"], clamped["Q2"], -clamped["mt"], inrange
        return W2, Q2, t, inrange

    # -- provenance --------------------------------------------------------- #
    def stamp(self) -> dict:
        d = dict(model=self.name, channel=self.channel, version=self.version,
                 provenance=self.provenance, reference=self.reference,
                 validity={k: list(v) for k, v in (self.validity or {}).items()},
                 out_of_range=self.out_of_range)
        d.update(self.topology.stamp())
        return d

    def usage(self) -> dict:
        frac = self.n_out_of_range / self.n_points if self.n_points else 0.0
        return dict(calls=self.n_calls, points=self.n_points,
                    out_of_range=self.n_out_of_range, out_of_range_frac=frac)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r} v{self.version}>"


# --------------------------------------------------------------------------- #
class ProtonDetectedModel(StructureFunctions):
    """Base for EXCLURAD-convention pi0/eta models.

    In this topology the core kinematics is built from the DETECTED PROTON,
    so the `t` it passes here is *not* the nucleon transfer the fit is a
    function of.  The nucleon transfer t_N is fixed by the observed bin and
    supplied at construction; the shifted kinematics of the radiative tail
    enter only through (W2~, Q2~), exactly as EXCLURAD does it.

    Subclasses implement `_sfs_tn(t_N, xB, Q2) -> (sT, sL, sTT, sLT)`.
    The sigma_LT sign flip of the detected-proton frame is applied by the
    base class via topology.lt_sign -- do NOT apply it in `_sfs_tn`.
    """

    def __init__(self, t_nucl, Ebeam=10.6):
        super().__init__()
        if t_nucl > 0:
            raise ValueError("t_nucl must be negative (spacelike)")
        self.t_nucl = float(t_nucl)
        self.Ebeam = float(Ebeam)
        if self.topology.sf_t != FIXED_NUCLEON:
            raise TypeError("ProtonDetectedModel requires a FIXED_NUCLEON topology")

    def _sfs_tn(self, t_nucl, xB, Q2):
        raise NotImplementedError

    def _sfs(self, W2, Q2, t):
        from ..core.constants import M_P
        xB = Q2 / np.maximum(W2 + Q2 - M_P * M_P, 1e-12)
        ok = (xB > 0) & (xB < 1) & (Q2 > 0)
        xBs = np.where(ok, xB, 0.5)
        Q2s = np.where(ok, Q2, 1.0)

        # t is the OBSERVED nucleon transfer and is FIXED: radiation comes
        # off the lepton line, so the measured proton -- and hence
        # t = (p-p')^2 -- is untouched by the photon.  The `t` argument
        # handed to us inside the radiative integral is the SHIFTED one and
        # must NOT be used here.  Only W2~ and Q2~ shift.
        #
        # For GENERATION, where every event has a different observed t, the
        # caller updates `self.t_nucl` per event (Generator._set_event_t).
        # Evaluating at one frozen t_nucl for a whole run removes the
        # model's t-dependence entirely and the sampled |t| spectrum
        # collapses to phase space -- mean |t| 2.27 vs exclugen's 0.386.
        sT, sL, sTT, sLT = self._sfs_tn(self.t_nucl, xBs, Q2s)
        z = np.zeros_like(np.asarray(sT, dtype=float))
        g = lambda a: np.where(ok, a, 0.0)   # noqa: E731
        return {"sT": g(sT), "sL": g(sL), "sTT": g(sTT),
                "sLT": g(sLT), "sLTp": z}

    def stamp(self):
        d = super().stamp()
        d["t_nucl"] = self.t_nucl
        d["Ebeam"] = self.Ebeam
        return d


# --------------------------------------------------------------------------- #
class TwoSF(StructureFunctions):
    """DIFFRAD-like mode: keep sigma_T, sigma_L; zero the interference SFs.

    Used to quantify how much the full SF set changes the RC.  Wraps any
    model and inherits its metadata (with ".twoSF" appended to the name).
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.name = f"{base.name}.twoSF"
        self.channel = base.channel
        self.version = base.version
        self.provenance = f"sigma_TT/LT/LT' forced to zero; base: {base.provenance}"
        self.reference = base.reference
        self.validity = base.validity
        self.out_of_range = base.out_of_range
        self.topology = base.topology

    def _sfs(self, W2, Q2, t):
        sf = self.base._sfs(W2, Q2, t)
        z = np.zeros_like(np.asarray(sf["sT"], dtype=float))
        return {"sT": sf["sT"], "sL": sf["sL"], "sTT": z, "sLT": z, "sLTp": z}
