"""
Unweighted event generation with exact radiative corrections.

Structure ("measured-first", as in exclugen and diffrad_vm v2)
--------------------------------------------------------------
1. Sample the OBSERVED kinematics (Q2, W2, t, phi) from sigma_obs -- the
   radiatively corrected cross section, not the Born one.  Unweighting is
   by accept/reject against a piecewise-constant majorant built in a
   pre-scan, so every surviving event is equally probable.
2. For each accepted event, split sigma_obs into the unresolved (soft)
   and resolved (hard) branches at v_min and throw for the branch.
3. If hard, sample the photon (v, Omega_k) from the positive-definite tail
   density and rebuild the detected hadron at that v.
4. Close four-momentum conservation on the recoil baryon and write LUND.

Sampling the observed variables from sigma_obs -- rather than sampling Born
and correcting afterwards -- is what makes the generator inherit the exact
RC instead of approximating it.

Cost
----
The observed-kinematics majorant needs one rc_factor evaluation per
pre-scan cell (~0.1 s each), which dominates setup.  Generation itself is
block-vectorized and costs tens of microseconds per event.  Setup is
therefore amortized: generate large samples per configuration, not many
small ones.
"""

import numpy as np

from ..core.kinematics import EventKin
from ..core.rc import rc_factor
from .tail import TailSampler, choose_vmin
from .frames import assemble_event


class Box:
    """Sampling box in the observed variables."""

    def __init__(self, Q2, W2, t, phi=(0.0, 2.0 * np.pi)):
        self.Q2, self.W2, self.t, self.phi = Q2, W2, t, phi

    def volume(self):
        return ((self.Q2[1] - self.Q2[0]) * (self.W2[1] - self.W2[0])
                * (self.t[1] - self.t[0]) * (self.phi[1] - self.phi[0]))

    def as_dict(self):
        return dict(Q2=list(self.Q2), W2=list(self.W2),
                    t=list(self.t), phi=list(self.phi))


class Generator:
    """Unweighted generator for one channel / model / beam energy.

    Parameters
    ----------
    sfmodel : structure-function model from mesonrc.models
    Ebeam   : beam energy (GeV)
    box     : Box of observed (Q2, W2, t, phi)
    mx2_cut : missing-mass-squared cut defining the exclusive window
    helicity: +1, -1, or 0 (0 = average, generated 50/50)
    """

    def __init__(self, sfmodel, Ebeam, box, mx2_cut=None, helicity=0,
                 seed=1234, n_scan=(4, 4, 4, 8), safety=2.0):
        self.sf = sfmodel
        self.Ebeam = float(Ebeam)
        self.box = box
        self.mx2_cut = mx2_cut
        self.helicity = helicity
        self.rng = np.random.default_rng(seed)
        self.n_scan = n_scan
        self.safety = float(safety)

        self.scan = None
        self.n_overweight_kin = 0
        self.max_overweight_kin = 1.0
        self.n_trials = 0
        self.n_accept = 0
        self.n_hard = 0
        self.n_soft = 0
        self.n_closure_fail = 0

    # ------------------------------------------------------------------ #
    def _point(self, Q2, W2, t, phi, h):
        """sigma_obs at one observed point, or None if unphysical."""
        try:
            kin = EventKin(self.Ebeam, Q2, W2, t, phi, self.sf.mV,
                           v=0.0, mU=self.sf.mU)
            vk = EventKin.vmax_kin(self.Ebeam, Q2, W2, t, self.sf.mV,
                                   mU=self.sf.mU)
            vm = vk if self.mx2_cut is None else min(
                vk, self.mx2_cut - self.sf.mU ** 2)
            if vm <= 0:
                return None
            r = rc_factor(Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t, phi=phi,
                          sfmodel=self.sf, h=h, mx2_cut=None if
                          self.mx2_cut is None else self.mx2_cut)
            return kin, vm, r
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return None

    def prescan(self, verbose=False):
        """Build the piecewise-constant majorant over observed kinematics.

        A single global maximum would be useless: the cross section spans
        orders of magnitude across the box.  Cells are uniform in
        (Q2, W2, t, phi); each cell's majorant is the largest sigma_obs
        found at its corners and centre, times a safety factor.
        """
        nQ, nW, nt, nph = self.n_scan
        edges = dict(
            Q2=np.linspace(*self.box.Q2, nQ + 1),
            W2=np.linspace(*self.box.W2, nW + 1),
            t=np.linspace(*self.box.t, nt + 1),
            phi=np.linspace(*self.box.phi, nph + 1))
        M = np.zeros((nQ, nW, nt, nph))
        h = self.helicity if self.helicity != 0 else +1

        for i in range(nQ):
            for j in range(nW):
                for k in range(nt):
                    for l in range(nph):
                        best = 0.0
                        for fq in (0.02, 0.5, 0.98):
                            for fw in (0.02, 0.5, 0.98):
                                for ft in (0.02, 0.5, 0.98):
                                    for fp in (0.02, 0.5, 0.98):
                                        Q2 = edges["Q2"][i] + fq * np.diff(edges["Q2"])[i]
                                        W2 = edges["W2"][j] + fw * np.diff(edges["W2"])[j]
                                        t = edges["t"][k] + ft * np.diff(edges["t"])[k]
                                        ph = edges["phi"][l] + fp * np.diff(edges["phi"])[l]
                                        got = self._point(Q2, W2, t, ph, h)
                                        if got is None:
                                            continue
                                        s = got[2]["sigma_obs"]
                                        if np.isfinite(s):
                                            best = max(best, abs(s))
                        M[i, j, k, l] = best * self.safety
            if verbose:
                print(f"  prescan Q2 slice {i+1}/{nQ}")

        self.scan = dict(edges=edges, M=M, shape=(nQ, nW, nt, nph))
        tot = M.sum()
        self.scan["cdf"] = np.cumsum(M.ravel()) / tot if tot > 0 else None
        self.scan["total"] = tot
        return self.scan

    # ------------------------------------------------------------------ #
    def _propose(self):
        s = self.scan
        flat = int(np.searchsorted(s["cdf"], self.rng.random()))
        flat = min(flat, int(np.prod(s["shape"])) - 1)
        i, j, k, l = np.unravel_index(flat, s["shape"])
        e = s["edges"]
        u = self.rng.random(4)
        Q2 = e["Q2"][i] + u[0] * np.diff(e["Q2"])[i]
        W2 = e["W2"][j] + u[1] * np.diff(e["W2"])[j]
        t = e["t"][k] + u[2] * np.diff(e["t"])[k]
        ph = e["phi"][l] + u[3] * np.diff(e["phi"])[l]
        return Q2, W2, t, ph, s["M"][i, j, k, l]

    def generate(self, n_events, progress=None):
        """Generate n_events unweighted events. Returns a list of dicts."""
        if self.scan is None:
            self.prescan()

        out = []
        while len(out) < n_events:
            Q2, W2, t, phi, M = self._propose()
            h = self.helicity if self.helicity != 0 else \
                (+1 if self.rng.random() < 0.5 else -1)
            self.n_trials += 1

            got = self._point(Q2, W2, t, phi, h)
            if got is None:
                continue
            kin, v_max, r = got
            s_obs = r["sigma_obs"]
            if not np.isfinite(s_obs) or s_obs <= 0:
                continue
            if s_obs > M:
                self.n_overweight_kin += 1
                self.max_overweight_kin = max(self.max_overweight_kin, s_obs / M)
            if self.rng.random() * M > s_obs:
                continue
            self.n_accept += 1

            # --- soft / hard branch ---------------------------------- #
            v_min, soft_frac = choose_vmin(kin, self.sf, h, v_max, s_obs)
            photon4 = meson4 = None
            if v_min is not None and self.rng.random() > soft_frac:
                ts = TailSampler(kin, self.sf, h, v_max, v_min=v_min)
                ph = ts.sample(self.rng, 1)
                photon4 = ph["k4"][0]
                meson4 = ts._meson_at_v(ph["v"][0])[0][0]
                self.n_hard += 1
            else:
                self.n_soft += 1

            ev = assemble_event(kin, meson4=meson4, photon4=photon4,
                                azimuth=self.rng.uniform(0, 2 * np.pi))
            if not ev["ok"]:
                self.n_closure_fail += 1
                continue
            ev.update(weight=1.0, Q2=Q2, W2=W2, t=t, phi=phi, helicity=h,
                      radiative=photon4 is not None)
            out.append(ev)
            if progress and len(out) % progress == 0:
                print(f"  {len(out)}/{n_events}")
        return out

    # ------------------------------------------------------------------ #
    def report(self):
        eff = self.n_accept / self.n_trials if self.n_trials else float("nan")
        n = self.n_hard + self.n_soft
        return dict(
            trials=self.n_trials, accepted=self.n_accept, efficiency=eff,
            hard=self.n_hard, soft=self.n_soft,
            radiative_fraction=self.n_hard / n if n else float("nan"),
            closure_failures=self.n_closure_fail,
            overweight_kinematics=self.n_overweight_kin,
            max_overweight_kinematics=self.max_overweight_kin)

    def health(self):
        """Human-readable warnings, empty if the run looks sound."""
        r = self.report()
        msgs = []
        if r["overweight_kinematics"] > 0.001 * max(r["trials"], 1):
            msgs.append(f"kinematics majorant exceeded in "
                        f"{r['overweight_kinematics']} of {r['trials']} trials "
                        f"(max x{r['max_overweight_kinematics']:.2f}) -- "
                        f"increase n_scan or safety")
        if r["closure_failures"]:
            msgs.append(f"{r['closure_failures']} events failed four-momentum "
                        f"closure and were discarded")
        if r["efficiency"] < 0.01:
            msgs.append(f"low unweighting efficiency {r['efficiency']:.4f} -- "
                        f"a finer prescan would speed generation up")
        return msgs
