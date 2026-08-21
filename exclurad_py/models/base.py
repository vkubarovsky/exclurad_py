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

import os

import numpy as np

from ..topology import Topology, FIXED_NUCLEON
from ..core.constants import M_P

SF_KEYS = ("sT", "sL", "sTT", "sLT", "sLTp")


class OutOfRange:
    """What to do when kinematics leave the declared validity range.

    DEFAULT IS `REPORT`: evaluate the model as written and only COUNT the
    excursions.  Clamping (`FREEZE`) substitutes a cross section the model
    never produced, and there is no evidence that the boundary value is
    closer to the truth than the model's own extrapolation -- an analytic
    fit is at least self-consistent, while a clamp invents a plateau and a
    kink in the derivative.  It also hides itself: the event is generated
    at its true kinematics but carries a foreign weight.

    Measured cost of the old FREEZE default: with validity Q2 > 0.5 and a
    generation box starting at 0.3, HALF the sample was clamped, and pi0's
    Born chi2/dof against exclugen was 24.25 (Q2) and 10.51 (|t|). With the
    clamp removed: 1.96 and 0.93 -- statistical agreement.
    """

    REPORT = "report"    # evaluate as-is, only count excursions  <- default
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
    out_of_range = OutOfRange.REPORT
    topology: Topology = None
    # Beam energy the model may need for epsilon.  Generator overwrites it
    # from the card; the default only matters for a standalone call.
    Ebeam = 10.6

    # Azimuthal moments declared as a FRACTION OF sigma_U = sigma_T +
    # eps sigma_L.  A class that sets one of these has the base class
    # OVERWRITE whatever its own _sfs returned for that structure function.
    #
    # This is how measured ratios get into a model that does not fit them:
    # sigma_LT'/sigma_0 = +0.10 for pi0 (CLAS, forward bins +0.05..+0.18,
    # flat in -t) and sigma_LT/sigma_0 = -0.30 for rho (-0.2..-0.4).  eta
    # borrows pi0's until it has its own measurement.
    #
    # As a fraction of sigma_U, never of sigma_T: they coincide only while
    # sigma_L is zero, and that is a property of today's fits, not of the
    # statement being made.  The values are in the MODEL's convention;
    # topology.lt_sign then carries them into the core's frame.
    SLT_OVER_SU = None
    SLTP_OVER_SU = None

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
    def _epsilon(self, xB, Q2):
        """Virtual-photon polarisation at this (xB, Q2) and self.Ebeam.

        Needed by any model stating a structure function as a fraction of
        sigma_U.  Same expression EventKin uses, written in the variables a
        structure-function model is handed; checked against
        EventKin.epsilon() to machine precision from eps 0.96 down to 0.25.
        """
        xB = np.asarray(xB, dtype=float)
        Q2 = np.asarray(Q2, dtype=float)
        nu = Q2 / np.maximum(2.0 * M_P * xB, 1e-12)
        y = nu / self.Ebeam
        g2 = (2.0 * M_P * xB) ** 2 / np.maximum(Q2, 1e-12)
        num = 1.0 - y - 0.25 * g2 * y * y
        den = 1.0 - y + 0.5 * y * y + 0.25 * g2 * y * y
        return np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)

    def _frac(self, name):
        """The declared sigma_LT / sigma_LT' fraction, or an env override.

        The override exists for ONE purpose: the physical sigma_LT'/sigma_U
        of 0.10 makes A_LU about 4%, and 100k radiative events measure the
        radiative correction to it only to +-12% -- resolving a 10% shift
        would take 1.2M events per channel, some seventy hours.  Raising the
        fraction to 1.0 multiplies A_LU by ten, drops the error on the ratio
        to about 0.014, and turns the same sample into a 7-sigma test of the
        whole helicity path.  validation/asymmetry/run_check.py does exactly
        this in quadrature; the override extends it to generated events.

        It shouts, because a model silently different from the card's is the
        failure mode that has cost the most this week.
        """
        v = os.environ.get(name)
        if v is None:
            return getattr(self, name)
        v = float(v)
        if not StructureFunctions._warned.get(name):
            StructureFunctions._warned[name] = True
            print(f" *** {name} ПЕРЕОПРЕДЕЛЕНО ИЗ ОКРУЖЕНИЯ: "
                  f"{getattr(self, name)} -> {v}.  Это испытательный режим, "
                  f"не физика модели. ***", flush=True)
        return v

    _warned = {}

    def _apply_su_fractions(self, sf, W2, Q2):
        """Overwrite sigma_LT / sigma_LT' if the class declared a fraction."""
        slt, sltp = self._frac("SLT_OVER_SU"), self._frac("SLTP_OVER_SU")
        if slt is None and sltp is None:
            return sf
        xB = Q2 / np.maximum(W2 - M_P * M_P + Q2, 1e-12)
        sU = sf["sT"] + self._epsilon(xB, Q2) * sf["sL"]
        sf = dict(sf)
        if slt is not None:
            sf["sLT"] = slt * sU
        if sltp is not None:
            sf["sLTp"] = sltp * sU
        return sf

    def _sfs(self, W2, Q2, t):
        raise NotImplementedError

    # -- public entry point ------------------------------------------------ #
    def __call__(self, W2, Q2, t):
        W2 = np.atleast_1d(np.asarray(W2, dtype=float))
        Q2 = np.atleast_1d(np.asarray(Q2, dtype=float))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        W2, Q2, t = np.broadcast_arrays(W2, Q2, t)

        W2c, Q2c, tc, inrange = self._apply_validity(W2, Q2, t)

        sf = self._apply_su_fractions(self._sfs(W2c, Q2c, tc), W2c, Q2c)
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
        # REPORT / ZERO / RAISE: hand the model its true arguments
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

    Subclasses implement `_sfs_tn(t_N, xB, Q2) -> (sT, sL, sTT, sLT)`, or
    `-> (sT, sL, sTT, sLT, sLTp)` when the fit carries a fifth structure
    function.  Both arities are accepted; a 4-tuple means sigma_LT' = 0.
    The sigma_LT sign flip of the detected-proton frame is applied by the
    base class via topology.lt_sign -- do NOT apply it in `_sfs_tn`.
    """

    def __init__(self, t_nucl, Ebeam=10.6):
        super().__init__()
        if t_nucl > 0:
            raise ValueError("t_nucl must be negative (spacelike)")
        self.t_nucl = float(t_nucl)
        self.Ebeam = float(Ebeam)
        # VPK 2026-08-15: the generator must reproduce ORIGINAL EXCLURAD's
        # corrections.  The Gilman/Hand conversion kf is therefore FROZEN at
        # the OBSERVED (W2, Q2) of the current point -- set by rc_factor,
        # born_xsec and TailSampler alongside t_nucl -- not evaluated at the
        # shifted tail kinematics.  Frozen kf cancels in eta(v_cut), which is
        # exactly why the pre-kf code matched the published 2013 tables to
        # 0.88% max; the shifted variant drifted to 1.77%.
        self.kf_W2Q2 = None
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
        _sfs = self._sfs_tn(self.t_nucl, xBs, Q2s)
        if len(_sfs) == 5:
            sT, sL, sTT, sLT, sLTp = _sfs
        elif len(_sfs) == 4:
            sT, sL, sTT, sLT = _sfs
            sLTp = None                      # fit has no fifth SF
        else:
            raise ValueError(
                f"{type(self).__name__}._sfs_tn returned {len(_sfs)} values; "
                f"expected 4 (sT, sL, sTT, sLT) or 5 (+ sLTp)")

        # FLUX-CONVENTION CONVERSION (found 2026-08-15; attribution corrected
        # same day).  The pi0/eta fits are HAND-convention -- the PRC paper
        # states it (Eq. 4, 'The Hand convention was adopted') and DVMPX's
        # FLUXW numerically equals Gamma(K_H=(W2-M2)/2M) to 5 digits.  OUR
        # core, however, normalises the hadronic tensor covariantly:
        # kap*pref ~ (W2-M2)^2/sqrt(lam_q), i.e. an effective K = |q| flux
        # (hadronic.py) -- correct for the diffrad_vm VM family, which is
        # defined in that convention and validated as-is.  Hand-defined SFs
        # must therefore be handed to the core multiplied by K_G/K_H
        # = sqrt(lam_q)/(W2-M2)*... = 2M|q|/(W2-M2).  Verified: born/DVMPX
        # was K_H/K_G to 4 digits before (0.9225@Q2=0.35, 0.4029@Q2=7) and
        # 1.0000 +/- 0.05% after.  Effect: high-Q2 population was
        # underweighted -- invisible for pi0 (0.2% of events above Q2=4),
        # a factor ~3 for eta (4% there).  eta(v_cut) ratios: untouched at
        # the observed point (flux cancels in the ratio).
        if self.kf_W2Q2 is not None:
            W2k, Q2k = self.kf_W2Q2          # frozen at the observed point
        else:
            W2k, Q2k = W2, Q2                # standalone call: args ARE observed
        nu = (W2k + Q2k - M_P * M_P) / (2.0 * M_P)
        kf = 2.0 * M_P * np.sqrt(np.maximum(nu * nu + Q2k, 0.0)) \
            / np.maximum(W2k - M_P * M_P, 1e-12)

        z = np.zeros_like(np.asarray(sT, dtype=float))
        g = lambda a: np.where(ok, a * kf, 0.0)   # noqa: E731
        # sigma_LT' rides the SAME Hand -> covariant flux conversion as the
        # other four: it comes out of the same fit, in the same convention.
        #
        # It also takes topology.lt_sign, applied for both sLT and sLTp in
        # StructureFunctions.__call__ -- correct under the Trento convention
        # because sin(phi + pi) = -sin(phi) flips exactly like cos does.
        #
        # And it must NOT get the factor 2 that dvmpw.F:57 puts on it, for
        # the same reason sigma_LT must not (see b1d64d0): dvmpw doubles
        # both OFF-DIAGONAL structure functions -- sigma_LT on the symmetric
        # (LX + XL) combination, sigma_LT' on the antisymmetric i(LX - XL) --
        # because EXCLURAD's tensor absorbs a compensating 1/2 on each.  Ours
        # is canonical and takes them bare.
        #
        # NOTE for whoever supplies the first real sigma_LT': exclugen's own
        # assembly is WRONG for this term.  dvmpw.F:62 multiplies S_LTP by
        # SIN(2*PHI_G) where the canonical fifth term is sin(phi).  Dormant
        # only because dvmpw.F:158 sets SIGMA_LTP = 0.
        return {"sT": g(sT), "sL": g(sL), "sTT": g(sTT),
                "sLT": g(sLT), "sLTp": z if sLTp is None else g(sLTp)}

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
