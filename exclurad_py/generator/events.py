"""
Unweighted event generation with exact radiative corrections.

Structure ("measured-first", as in exclugen and diffrad_vm v2)
--------------------------------------------------------------
1. Sample the OBSERVED kinematics (Q2, W2, t, phi) from sigma_obs -- the
   radiatively corrected cross section, not the Born one -- by accept/reject
   against a piecewise-constant majorant.  Every surviving event is equally
   probable, so the output is UNWEIGHTED.
2. Split sigma_obs into the unresolved (soft) and resolved (hard) branches
   at v_min and throw for the branch.
3. If hard, sample the photon from the positive-definite tail density and
   rebuild the detected hadron at that inelasticity.
4. Close four-momentum conservation on the recoil and write LUND.

Sampling the observed variables from sigma_obs -- rather than sampling Born
and applying a correction afterwards -- is what makes the generator inherit
the exact RC instead of approximating it.

Cost, and how it is managed
---------------------------
`rc_factor` costs ~0.1 s per point, so it cannot be called per event, and a
prescan that calls it thousands of times cannot be afforded either.  Two
devices fix this:

  * sigma_obs = sigma_Born * eta, with sigma_Born exact per event
    (microseconds) and eta interpolated from a precomputed RCTable.  The
    table's interpolation error is measured, not assumed.
  * the majorant is built from sigma_Born alone (cheap, fine grid) times a
    scalar bound on eta taken from the table.

The photon branch reuses one TailSampler per prescan cell rather than
building one per event (~0.4 s each).  Within a cell the tail density
varies smoothly; APPROXIMATION FLAGGED: the photon spectrum is therefore
that of the cell centre, not of the exact event kinematics.  The soft/hard
branching itself uses the event's own sigma_obs and is exact.  Making the
photon exact needs a final accept/reject against the event's true density,
batched over a block of events -- see REPORT.md.
"""

import json
import os

import numpy as np

from ..core.born import born_xsec
from ..core.kinematics import EventKin
from ..core.rc import rc_factor
from .tail import TailSampler, choose_vmin, photon_at
from .frames import assemble_event
from .tables import RCTable


class _SoftOnly(Exception):
    """Internal: event's radiative tail is below resolution -- emit soft."""


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
    """Unweighted generator for one channel / model / beam energy."""

    def __init__(self, sfmodel, Ebeam, box, mx2_cut=None, helicity=0,
                 seed=1234, majorant_shape=(8, 8, 8, 12),
                 table_shape=(5, 5, 5, 8), tail_shape=(2, 2, 2, 2),
                 safety=1.6, cache=None, born_only=False, cuts=None,
                 v_min=None, maj_cache=None):
        """born_only=True: no radiative corrections at all.

        sigma_obs == sigma_Born, eta == 1, no photon is ever emitted, and
        neither the RC table nor the tail samplers are built.  This is the
        discriminating test when a radiative sample disagrees with another
        generator: if the BORN distributions already differ, the problem is
        the cross section, the flux convention or the sampling -- not the
        radiative tail.
        """
        self.sf = sfmodel
        self.Ebeam = float(Ebeam)
        self.box = box
        self.mx2_cut = mx2_cut
        self.helicity = helicity
        self.rng = np.random.default_rng(seed)
        self.majorant_shape = tuple(majorant_shape)
        self.table_shape = tuple(table_shape)
        self.tail_shape = tuple(tail_shape)
        self.safety = float(safety)
        self.cache = cache
        # The majorant prescan (~19 s) is identical for every worker of a
        # parallel run.  Rebuilding it per worker is why --jobs made things
        # SLOWER than one core: 12 x 19 s of duplicated setup against 40 s
        # of actual generation each.  Cached to disk and shared instead.
        self.maj_cache = maj_cache
        self.born_only = bool(born_only)
        # Cuts the REFERENCE generators apply at generation time.  Without
        # these the two samples are different populations and any rate
        # comparison is meaningless -- see decay.py.
        self.cuts = dict(cuts or {})
        # Explicit soft/hard split, as exclugen takes it (vmin=1e-4).  None
        # -> choose_vmin picks one.  Exposing it matters: the radiative
        # FRACTION is set by this number, so two generators cannot be
        # compared on it unless both are told the same value.
        self.v_min_fixed = v_min
        self.n_cut_yxb = 0
        self.n_cut_acc = 0
        self.n_vmin_fallback = 0

        self.table = None
        self.maj = None
        self._tail_cache = {}

        self.n_trials = 0
        self.n_accept = 0
        self.n_hard = 0
        self.n_soft = 0
        # INSTRUMENTATION: where hard events are lost between "branch chosen"
        # and "photon written".  The tail sampler is built at the tail-CELL
        # CENTRE but its draw is tested against THIS event's v_max, and every
        # failure silently falls through to the soft branch.
        # ETA_PROPOSAL_CAP caps the TABLE eta used in the accept weight.  The
        # unclipped table is honest but its 1e3-1e5 corners crater the
        # unweighting efficiency (~100x).  With a cap the PROPOSAL is biased,
        # so an unweighted sample is only valid after reweighting each event
        # by eta_exact/min(eta_table, cap) -- production/weighted_val.py.
        # Interim measure until the two-stage exact-eta accept is built in.
        _cap = float(os.environ.get("ETA_PROPOSAL_CAP", "0"))
        self.eta_proposal_cap = _cap if _cap > 0 else None
        # exact per-event soft/hard split and photon density (see generate())
        self.exact_per_event = os.environ.get("EXACT_PER_EVENT", "") == "1"
        # stage-2 exact accept: unweighted output follows sigma_obs_exact even
        # with a capped proposal.  Implies exact_per_event (needs rc_e anyway).
        self.exact_accept = os.environ.get("EXACT_ACCEPT", "") == "1"
        if self.exact_accept:
            self.exact_per_event = True
        # VPK 2026-08-15: production scheme is EXPONENTIATED (variant A)
        self.expo_tail = os.environ.get("EXPO_TAIL", "") == "1"
        if self.expo_tail:
            self.exact_per_event = True
        self._B = None            # per-majorant-cell bound for stage 2
        self._B_global = 3.0
        self.n_accept2 = 0
        self.n_b_viol = 0
        self.max_b_viol = 1.0
        self.n_hard_chosen = 0     # rng said "hard"
        self.n_drop_no_tail = 0    # no tail sampler for the cell at all
        self.n_drop_v_ge_vmax = 0  # drawn v >= this event's v_max
        self.n_drop_photon_at = 0  # photon_at() could not build the photon
        self.n_drop_except = 0     # ValueError/ZeroDivisionError in the draw
        self.n_closure_fail = 0
        self.n_overweight = 0
        self.max_overweight = 1.0

    # ------------------------------------------------------------------ #
    @property
    def _t_kind(self):
        """Which invariant the box's t axis means -- from the topology.

        rc_factor takes this from the model's topology; the generator did
        NOT, so it built every event in the default "photon" convention
        (t = (q-p')^2) while pi0/eta topologies mean "target"
        (t = (p-p')^2). The sampled t and the t you recompute from the
        written proton then disagree -- 0.42 vs 3.33 -- and the |t| spectrum
        came out 8x too hard against exclugen.
        """
        return getattr(self.sf.topology, "t_kind", "photon")

    def _kin(self, Q2, W2, t, phi):
        return EventKin(self.Ebeam, Q2, W2, t, phi, self.sf.mV,
                        phi_offset=float(getattr(getattr(self.sf, 'topology', None),
                                                 'phi_offset', 0.0)),
                        v=0.0, mU=self.sf.mU, t_kind=self._t_kind)

    def _vmax(self, Q2, W2, t):
        vk = EventKin.vmax_kin(self.Ebeam, Q2, W2, t, self.sf.mV,
                               mU=self.sf.mU, t_kind=self._t_kind)
        if self.mx2_cut is None:
            return vk
        return min(vk, self.mx2_cut - self.sf.mU ** 2)

    def _set_event_t(self, t):
        """Point a fixed-t_N model at THIS event's observed transfer.

        pi0/eta models are functions of the nucleon transfer supplied at
        construction (see models.base.ProtonDetectedModel). In generation t
        changes event to event, so it has to be updated -- while the model
        still ignores the shifted t inside the radiative integral, because
        t does not shift.
        """
        if hasattr(self.sf, "t_nucl"):
            self.sf.t_nucl = float(t)

    def _born(self, Q2, W2, t, phi, h):
        self._set_event_t(t)
        try:
            return born_xsec(self._kin(Q2, W2, t, phi), self.sf, h)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None

    # ------------------------------------------------------------------ #
    def setup(self, verbose=False):
        """Build the eta table and the Born-based majorant."""
        h = self.helicity if self.helicity != 0 else +1

        if self.born_only:
            self.table = None
            eta_hi = 1.0
        else:
            eta_hi = None
        self.table = None
        # A polarised run needs stage 2.  The eta table is built at ONE
        # helicity, so the proposal it feeds is helicity-blind; with
        # sigma_LT' nonzero the true eta differs between the two signs and
        # an unweighted sample drawn against that proposal is biased in the
        # beam-spin asymmetry -- silently, since every other distribution
        # looks right.  EXACT_ACCEPT=1 removes the bias, because stage 2
        # re-accepts against sigma_obs evaluated at the EVENT's helicity.
        # Probe the model rather than trusting a flag: fits gain a fifth
        # structure function without anyone editing this file.
        if not self.born_only and not self.exact_accept:
            try:
                _p = self.sf(np.array([sum(self.box.W2) / 2.0]),
                             np.array([sum(self.box.Q2) / 2.0]),
                             np.array([sum(self.box.t) / 2.0]))
                if np.any(np.abs(_p["sLTp"]) > 0.0):
                    print(" *** WARNING: sigma_LT' is nonzero and "
                          "EXACT_ACCEPT is off.  The eta table is built at a "
                          "single helicity, so the beam-spin asymmetry of "
                          "this sample will be biased.  Set EXACT_ACCEPT=1. "
                          "***", flush=True)
            except Exception:
                pass

        if self.born_only:
            pass
        elif self.cache:
            self.table = RCTable.load(self.cache, self.sf, self.Ebeam,
                                      self.box, self.mx2_cut, h,
                                      self.table_shape)
        if self.table is None and not self.born_only:
            if verbose:
                print("building eta table "
                      f"({int(np.prod(self.table_shape))} RC points)...")
            self.table = RCTable(self.sf, self.Ebeam, self.box, self.mx2_cut,
                                 h, self.table_shape).build(verbose=verbose)
            if self.cache:
                self.table.save(self.cache)

        eta_hi = 1.0 if self.born_only else float(np.nanmax(self.table.eta))
        # the majorant bound must match the (possibly capped) proposal weight,
        # or the acceptance ratio min(eta,cap)/eta_max craters to ~1e-5
        if not self.born_only and self.eta_proposal_cap:
            eta_hi = min(eta_hi, self.eta_proposal_cap)

        if self.maj_cache and os.path.exists(self.maj_cache):
            d = np.load(self.maj_cache, allow_pickle=True)
            key = json.loads(str(d["key"]))
            if key == self._maj_key(eta_hi):
                self.maj = dict(edges={k: d["e_" + k] for k in ("Q2", "W2", "t", "phi")},
                                M=d["M"], shape=tuple(d["shape"]),
                                cdf=d["cdf"], eta_hi=float(d["eta_hi"]))
                if verbose:
                    print(f" majorant loaded from {os.path.basename(self.maj_cache)}")
                return self

        nQ, nW, nt, nph = self.majorant_shape
        edges = dict(Q2=np.linspace(*self.box.Q2, nQ + 1),
                     W2=np.linspace(*self.box.W2, nW + 1),
                     t=np.linspace(*self.box.t, nt + 1),
                     phi=np.linspace(*self.box.phi, nph + 1))
        M = np.zeros(self.majorant_shape)
        # Stratified random sub-sampling inside each cell. A full 3^4 corner
        # scan costs 81 Born evaluations per cell (~140k calls for a 6x6x6x8
        # grid, minutes); n_probe random points plus the cell corners in the
        # steepest variable is enough, and the safety factor plus the
        # overweight counter catch any cell that was underestimated.
        rng = np.random.default_rng(20260811)
        n_probe = 24
        for i in range(nQ):
            for j in range(nW):
                for k in range(nt):
                    for l in range(nph):
                        u = rng.random((n_probe, 4))
                        u[0] = [0.02, 0.02, 0.02, 0.02]
                        u[1] = [0.98, 0.98, 0.98, 0.98]
                        u[2] = [0.5, 0.5, 0.5, 0.5]
                        Q2s = edges["Q2"][i] + u[:, 0] * np.diff(edges["Q2"])[i]
                        W2s = edges["W2"][j] + u[:, 1] * np.diff(edges["W2"])[j]
                        ts_ = edges["t"][k] + u[:, 2] * np.diff(edges["t"])[k]
                        phs = edges["phi"][l] + u[:, 3] * np.diff(edges["phi"])[l]
                        best = 0.0
                        for Q2, W2, t, ph in zip(Q2s, W2s, ts_, phs):
                            b = self._born(Q2, W2, t, ph, h)
                            if b is not None and np.isfinite(b):
                                best = max(best, abs(b))
                        M[i, j, k, l] = best
        M *= eta_hi * self.safety
        tot = M.sum()
        self.maj = dict(edges=edges, M=M, shape=self.majorant_shape,
                        cdf=np.cumsum(M.ravel()) / tot if tot > 0 else None,
                        eta_hi=eta_hi)
        return self

    def _maj_key(self, eta_hi):
        return dict(model=self.sf.name, Ebeam=self.Ebeam, box=self.box.as_dict(),
                    mx2_cut=self.mx2_cut, helicity=self.helicity,
                    shape=list(self.majorant_shape), safety=self.safety,
                    born_only=self.born_only, eta_hi=round(float(eta_hi), 9),
                    cuts={k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in sorted(self.cuts.items())})
        # NOTE: t_nucl is deliberately NOT part of the key. Since the
        # per-event-t fix it is mutable state -- the generator rewrites it
        # for every event -- so including it recorded whatever t the last
        # calibration event happened to have (-0.432 instead of the card's
        # -0.30). The key never matched, every worker silently rebuilt the
        # majorant, and --jobs looked far slower than it is.

    def save_majorant(self, path):
        """Persist the CALIBRATED majorant so parallel workers can share it."""
        s = self.maj
        np.savez_compressed(path, M=s["M"], cdf=s["cdf"],
                            shape=np.array(s["shape"]), eta_hi=s["eta_hi"],
                            key=json.dumps(self._maj_key(s["eta_hi"])),
                            **{"e_" + k: v for k, v in s["edges"].items()})
        return path

    def _pass_yxb(self, Q2, W2):
        ylo, yhi = self.cuts.get("y", (0.0, 1.0))
        xlo, xhi = self.cuts.get("xB", (0.0, 1.0))
        if (ylo, yhi, xlo, xhi) == (0.0, 1.0, 0.0, 1.0):
            return True
        M = self.sf.topology.m_detected if False else 0.93827208816
        nu = (W2 + Q2 - M * M) / (2.0 * M)
        if nu <= 0:
            return False
        y = nu / self.Ebeam
        xB = Q2 / (W2 + Q2 - M * M)
        return (ylo <= y <= yhi) and (xlo <= xB <= xhi)

    # ------------------------------------------------------------------ #
    def _propose(self):
        s = self.maj
        flat = int(np.searchsorted(s["cdf"], self.rng.random()))
        flat = min(flat, int(np.prod(s["shape"])) - 1)
        i, j, k, l = np.unravel_index(flat, s["shape"])
        e, u = s["edges"], self.rng.random(4)
        return (e["Q2"][i] + u[0] * np.diff(e["Q2"])[i],
                e["W2"][j] + u[1] * np.diff(e["W2"])[j],
                e["t"][k] + u[2] * np.diff(e["t"])[k],
                e["phi"][l] + u[3] * np.diff(e["phi"])[l],
                s["M"][i, j, k, l], (i, j, k, l))

    def _tail_cell(self, cell):
        """Map a fine majorant cell onto the coarser tail-sampler grid.

        One TailSampler costs ~0.4 s to build, so one per majorant cell
        (hundreds) is unaffordable. The tail grid is deliberately coarse:
        within a tail cell the photon density is treated as constant in the
        observed kinematics. APPROXIMATION -- see REPORT.md gap 3.
        """
        return tuple(int(c * ts // ms) for c, ms, ts
                     in zip(cell, self.majorant_shape, self.tail_shape))

    def _tail_for_cell(self, cell, h, sigma_obs):
        key = (self._tail_cell(cell), h)
        if key in self._tail_cache:
            return self._tail_cache[key]
        e = self.maj["edges"]
        tc = key[0]
        # centre of the TAIL cell, in each observed variable
        def centre(name, idx, n_tail):
            lo, hi = e[name][0], e[name][-1]
            w = (hi - lo) / n_tail
            return lo + (idx + 0.5) * w
        cQ2 = centre("Q2", tc[0], self.tail_shape[0])
        cW2 = centre("W2", tc[1], self.tail_shape[1])
        ct = centre("t", tc[2], self.tail_shape[2])
        cph = centre("phi", tc[3], self.tail_shape[3])
        try:
            kin = self._kin(cQ2, cW2, ct, cph)
            vm = self._vmax(cQ2, cW2, ct)
            if vm <= 0:
                self._tail_cache[key] = None
                return None
            # sigma_obs MUST be evaluated at the same point as sigma_hard.
            #
            # sigma_hard below is built at the tail-CELL CENTRE, but the caller
            # passes this EVENT's sigma_obs (Born x interpolated eta).  With
            # tail_shape=(2,2,2,2) a tail cell spans e.g. Q2 0.2-5.1, across
            # which sigma_born moves by orders of magnitude -- so
            # frac = sigma_hard(centre)/sigma_obs(event) is not a probability
            # at all.  It collapsed to a median of 6e-4 where the correct value
            # at fixed kinematics is 0.59-0.62, which is what emptied the hard
            # branch and left every RAD sample effectively Born.
            #
            # One exact rc_factor per tail cell (there are only
            # prod(tail_shape) of them) puts numerator and denominator back at
            # the same kinematics and takes the eta table out of the split.
            self._set_event_t(ct)
            rc_c = rc_factor(Ebeam=self.Ebeam, Q2=cQ2, W2=cW2, t=ct, phi=cph,
                             sfmodel=self.sf, h=h, mx2_cut=self.mx2_cut,
                             t_kind=self._t_kind)
            sigma_obs_c = float(rc_c["sigma_obs"])
            if not np.isfinite(sigma_obs_c) or sigma_obs_c <= 0:
                self._tail_cache[key] = None
                return None

            if self.v_min_fixed:
                v_min = min(self.v_min_fixed, 0.5 * vm)
                from .tail import TailSampler as _TS
                ts_probe = _TS(kin, self.sf, h, vm, v_min=v_min)
                soft, hard, hf = ts_probe.split(sigma_obs_c)
                soft_frac = 1.0 - hf
                if soft <= 0:
                    v_min, soft_frac = choose_vmin(kin, self.sf, h, vm,
                                                   sigma_obs_c)
                    self.n_vmin_fallback += 1
            else:
                v_min, soft_frac = choose_vmin(kin, self.sf, h, vm, sigma_obs_c)
            if v_min is None:
                self._tail_cache[key] = None
                return None
            self._tail_cache[key] = dict(
                ts=TailSampler(kin, self.sf, h, vm, v_min=v_min),
                soft_frac=soft_frac, buf=None, ibuf=0)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            self._tail_cache[key] = None
        return self._tail_cache[key]

    def _b_cell(self, cell):
        """Stage-2 bound for this majorant cell (calibrated; global fallback)."""
        if self._B is None:
            return self._B_global
        v = float(self._B[cell])
        return v if v > 0 else self._B_global

    def _draw_photon(self, entry, block=4096):
        """One (v, nhat) from a cell's sampler, refilling in blocks.

        Drawing photons one at a time wastes the vectorization: sample(1)
        still proposes a whole candidate block. Buffering cuts the
        per-event cost by an order of magnitude.
        """
        buf, i = entry["buf"], entry["ibuf"]
        if buf is None or i >= len(buf["v"]):
            buf = entry["ts"].sample(self.rng, block)
            entry["buf"], entry["ibuf"] = buf, 0
            i = 0
        entry["ibuf"] = i + 1
        return dict(v=buf["v"][i], nhat=buf["nhat"][i])

    # ------------------------------------------------------------------ #
    def calibrate(self, n_probe=20000, margin=1.35, verbose=False):
        """Raise the majorant where a trial run exceeds it, then reset.

        A prescan samples each cell at a finite number of points and can
        miss its true maximum; the wider the box, the more often.  Any
        exceedance biases an unweighted sample -- those events are
        under-represented -- so it must be fixed BEFORE generating, not
        reported afterwards.

        This makes a throwaway pass, records the worst sigma_obs/M per
        cell, inflates those cells by that ratio times `margin`, and
        discards the events.  The generated sample is then produced against
        the corrected majorant.
        """
        s = self.maj
        shape = s["shape"]
        worst = np.ones(shape)
        h0 = self.helicity if self.helicity != 0 else +1
        for _ in range(n_probe):
            Q2, W2, t, phi, M, cell = self._propose()
            if M <= 0 or not self._pass_yxb(Q2, W2):
                continue
            h = h0 if self.helicity != 0 else (+1 if self.rng.random() < 0.5 else -1)
            b = self._born(Q2, W2, t, phi, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = 1.0 if self.born_only else float(self.table(Q2, W2, t, phi)[0])
            if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                eta = self.eta_proposal_cap
            r = (b * eta) / M
            if r > worst[cell]:
                worst[cell] = r
        # stage-2 bound calibration: exact/proposal ratio per cell, from up to
        # 300 probe points (rc_factor is ~0.1-1 s, the full prescan would be
        # unaffordable).  Cells never probed fall back to the global max.
        # born_only has no eta table (there is no radiative proposal to
        # correct), so stage 2 is both meaningless and a crash: self.table
        # is None.  The two other table reads, lines ~467 and ~544, already
        # guard with `1.0 if self.born_only else ...`; this one did not.
        # Never hit before because run_production_100k.sh only ever exports
        # EXACT_ACCEPT=1 for the rad cards -- iborn=1 with the production
        # environment first happened on 2026-08-17.
        if self.exact_accept and not self.born_only:
            bmax = np.zeros(shape)
            n_b = 0
            for _ in range(n_probe):
                if n_b >= 300:
                    break
                Q2, W2, t, phi, Mv, cell = self._propose()
                if Mv <= 0 or not self._pass_yxb(Q2, W2):
                    continue
                # Draw the helicity the way generate() does, not h0.  The
                # bound this loop calibrates is r = sigma_obs(h) / (born(h) *
                # eta_table), and eta_table is built at h0, so r = eta(h) /
                # eta(h0) -- identically 1 at h0 and NOT 1 at the other sign
                # once sigma_LT' is nonzero.  Probing only h0 would calibrate
                # the bound on the one helicity where it is trivially tight.
                # Today this is a no-op: sigma_LT' = 0 in every fit we carry,
                # so nothing depends on h at all.  With a 5% asymmetry the
                # miss would be ~0.06% (eta 0.88255 vs 0.88305 at v_cut 0.1)
                # against a 1.3 safety factor -- small, but there is no
                # reason to leave it wrong.
                h_p = h0 if self.helicity != 0 else \
                    (+1 if self.rng.random() < 0.5 else -1)
                b = self._born(Q2, W2, t, phi, h_p)
                if b is None or not np.isfinite(b) or b <= 0:
                    continue
                eta = float(self.table(Q2, W2, t, phi)[0])
                if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                    eta = self.eta_proposal_cap
                try:
                    self._set_event_t(t)
                    so = float(rc_factor(
                        Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t, phi=phi,
                        sfmodel=self.sf, h=h_p, mx2_cut=self.mx2_cut,
                        t_kind=self._t_kind)["sigma_obs"])
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    continue
                if not np.isfinite(so) or so <= 0:
                    continue
                n_b += 1
                r = so / (b * eta)
                if r > bmax[cell]:
                    bmax[cell] = r
            self._B_global = float(max(bmax.max() * 1.3, 1.05))
            self._B = np.where(bmax > 0, bmax * 1.3, 0.0)
            if verbose:
                print(f" stage-2 bounds: {n_b} probes, global B = "
                      f"{self._B_global:.2f}")

        n_bad = int((worst > 1.0).sum())
        s["M"] = s["M"] * np.where(worst > 1.0, worst * margin, 1.0)
        tot = s["M"].sum()
        s["cdf"] = np.cumsum(s["M"].ravel()) / tot if tot > 0 else None
        self.n_trials = self.n_accept = 0
        self.n_overweight = 0
        self.max_overweight = 1.0
        self.n_cut_yxb = self.n_cut_acc = 0
        if verbose:
            print(f" calibrated majorant: {n_bad} of {int(np.prod(shape))} cells "
                  f"raised (worst factor {worst.max():.2f})")
        return dict(cells_raised=n_bad, worst=float(worst.max()))

    # ------------------------------------------------------------------ #
    def generate(self, n_events, progress=None):
        if self.maj is None:
            self.setup()
        out = []
        while len(out) < n_events:
            Q2, W2, t, phi, M, cell = self._propose()
            h = self.helicity if self.helicity != 0 else \
                (+1 if self.rng.random() < 0.5 else -1)
            self.n_trials += 1

            # y / xB cuts first: they are arithmetic and reject before the
            # (much more expensive) Born evaluation.
            if not self._pass_yxb(Q2, W2):
                self.n_cut_yxb += 1
                continue

            b = self._born(Q2, W2, t, phi, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = 1.0 if self.born_only else float(self.table(Q2, W2, t, phi)[0])
            if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                eta = self.eta_proposal_cap
            s_obs = b * eta
            if s_obs <= 0:
                continue
            if s_obs > M:
                self.n_overweight += 1
                self.max_overweight = max(self.max_overweight, s_obs / M)
            if self.rng.random() * M > s_obs:
                continue
            self.n_accept += 1

            v_max = self._vmax(Q2, W2, t)
            if v_max <= 0:
                continue
            self._set_event_t(t)
            kin = self._kin(Q2, W2, t, phi)

            photon4 = meson4 = None
            if self.exact_per_event and not self.born_only:
                # EXACT PER-EVENT RADIATION (env EXACT_PER_EVENT=1).
                # The cell-based path holds the soft/hard split and the photon
                # density constant across a tail cell (16 cells over the whole
                # box), which biased frac(v > 0.3) by ~2x against the exact
                # quadrature even after the population reweighting.  Here both
                # come from THIS event's kinematics: one rc_factor (~0.1 s)
                # and one TailSampler (~0.4 s) per ACCEPTED event -- affordable
                # because accepted events are ~1e4, not the 1e7 trials.
                try:
                    rc_e = rc_factor(Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t,
                                     phi=phi, sfmodel=self.sf, h=h,
                                     mx2_cut=self.mx2_cut,
                                     t_kind=self._t_kind)
                    so_e = float(rc_e["sigma_obs"])

                    # STAGE 2 (optional, EXACT_ACCEPT=1): re-accept on
                    # sigma_obs_exact / (proposal * B_cell).  Removes the
                    # capped-proposal bias from the UNWEIGHTED sample itself;
                    # B_cell is calibrated in calibrate(), violations counted.
                    if self.exact_accept:
                        prop = float(s_obs)
                        B = self._b_cell(cell)
                        r = so_e / (prop * B) if prop > 0 else 0.0
                        if r > 1.0:
                            self.n_b_viol += 1
                            self.max_b_viol = max(self.max_b_viol, r)
                        if self.rng.random() > min(r, 1.0):
                            continue          # rejected by the exact stage
                        self.n_accept2 += 1

                    vmn = min(self.v_min_fixed or 1.0e-4, 0.5 * v_max)
                    # degenerate tail: photons with v < ~10*vmn are below any
                    # missing-mass resolution; treating the event as soft is
                    # physically free and avoids the near-threshold sampler
                    # stall that cost 9 h on the J/psi box (vmax ~ 1e-3).
                    if v_max < 10.0 * vmn:
                        raise _SoftOnly

                    if self.expo_tail:
                        # EXPONENTIATED scheme (VPK decision 2026-08-15):
                        # the observable frac(v > x) must equal
                        # [sigma_obs(vmax)-sigma_obs(x)]/sigma_obs(vmax) with
                        # the YFS-exponentiated soft factor -- what the
                        # published eta(v_cut) tables mean.  Hard fraction and
                        # the v-draw both come from the sigma_obs ladder; the
                        # bare tail supplies only the photon ANGLES.
                        mU2 = self.sf.mU ** 2

                        def sobs(x):
                            return float(rc_factor(
                                Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t, phi=phi,
                                sfmodel=self.sf, h=h, mx2_cut=x + mU2,
                                t_kind=self._t_kind)["sigma_obs"])

                        N0 = max(so_e - sobs(vmn), 0.0)
                        if self.rng.random() < N0 / so_e:
                            self.n_hard_chosen += 1
                            nodes = np.geomspace(vmn, v_max, 8)
                            N = [N0] + [max(so_e - sobs(x), 0.0)
                                        for x in nodes[1:-1]] + [0.0]
                            u = self.rng.random() * N0
                            j = next(k for k in range(7) if N[k + 1] <= u)
                            lo, hi = nodes[j], nodes[j + 1]
                            nl, nh = N[j], N[j + 1]
                            if nh > 0 and nl > nh and u > 0:
                                # log-log interpolation: N(v) ~ power law
                                fr = np.log(u / nl) / np.log(nh / nl)
                                v_star = float(
                                    lo * (hi / lo) ** np.clip(fr, 0.0, 1.0))
                            else:
                                # LAST segment: N(v_max) == 0 makes log-log
                                # degenerate and piled every deep photon onto
                                # the segment's left edge (pi0 scheme-A came
                                # out with EXACTLY zero events above v=0.3).
                                # Invert N(v)=u by geometric bisection with
                                # exact sigma_obs -- ~5 rc calls on the ~10%
                                # deepest hard events only.
                                lo_v, hi_v = lo, hi
                                for _ in range(5):
                                    mid = float(np.sqrt(lo_v * hi_v))
                                    Nm = max(so_e - sobs(mid), 0.0)
                                    if Nm > u:
                                        lo_v = mid
                                    else:
                                        hi_v = mid
                                v_star = float(np.sqrt(lo_v * hi_v))
                            ts_e = TailSampler(kin, self.sf, h, v_max,
                                               v_min=vmn)
                            s = ts_e.sample(self.rng, 1)
                            got2 = photon_at(kin, v_star, s["nhat"][0])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                    else:
                        ts_e = TailSampler(kin, self.sf, h, v_max, v_min=vmn)
                        _, _, hf = ts_e.split(so_e)
                        if np.isfinite(hf) and \
                                self.rng.random() < max(hf, 0.0):
                            self.n_hard_chosen += 1
                            s = ts_e.sample(self.rng, 1)
                            got2 = photon_at(kin, float(s["v"][0]),
                                             s["nhat"][0])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                except _SoftOnly:
                    photon4 = meson4 = None
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    photon4 = meson4 = None
                    self.n_drop_except += 1
                got = None
            elif self.born_only:
                got = None
            else:
                got = self._tail_for_cell(cell, h, s_obs)
            if got is None:
                if not self.born_only and not self.exact_per_event:
                    self.n_drop_no_tail += 1
            else:
                if self.rng.random() > got["soft_frac"]:
                    self.n_hard_chosen += 1
                    try:
                        s = self._draw_photon(got)
                        v = float(s["v"])
                        if v < v_max:
                            # rebuild at THIS event's kinematics: omega
                            # depends on z, which the cell centre does not
                            # share (see tail.photon_at)
                            got2 = photon_at(kin, v, s["nhat"])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                        else:
                            self.n_drop_v_ge_vmax += 1
                    except (ValueError, ZeroDivisionError):
                        photon4 = meson4 = None
                        self.n_drop_except += 1

            if photon4 is not None:
                self.n_hard += 1
            else:
                self.n_soft += 1

            ev = assemble_event(kin, meson4=meson4, photon4=photon4,
                                azimuth=self.rng.uniform(0, 2 * np.pi))
            if not ev["ok"]:
                self.n_closure_fail += 1
                continue
            ev.update(weight=1.0, Q2=Q2, W2=W2, t=t, phi=phi, helicity=h,
                      eta=eta, radiative=photon4 is not None)

            acc = self.cuts.get("iacc", "none")
            if acc != "none":
                from .decay import in_acceptance, DECAY, decay_two_body
                dau = None
                ch = self.sf.channel
                if ch in DECAY:
                    a, bq = decay_two_body(ev["meson"], DECAY[ch][2], self.rng)
                    if a is None:
                        self.n_cut_acc += 1
                        continue
                    ev["daughters"] = (a, bq)
                    dau = (a, bq)
                if not in_acceptance(acc, ev["scattered"], ev["recoil"], dau)[0]:
                    self.n_cut_acc += 1
                    continue
            out.append(ev)
            if progress and len(out) % progress == 0:
                print(f"  {len(out)}/{n_events}")
        return out

    # ------------------------------------------------------------------ #
    def sigma_total(self):
        """Absolute cross section of THIS run, nb (accept-reject estimator).

        sigma = (n_accept / n_trials) * integral of the majorant over the box.
        The generator was printing none of this, so a delta = sigma_obs/sigma_born
        test had no normalisation and could only check the SHAPE (2026-08-17).
        """
        if self.maj is None or self.n_trials == 0:
            return float("nan")
        e = self.maj["edges"]
        dQ2 = np.diff(e["Q2"])[:, None, None, None]
        dW2 = np.diff(e["W2"])[None, :, None, None]
        dt = np.diff(e["t"])[None, None, :, None]
        dph = np.diff(e["phi"])[None, None, None, :]
        vol = dQ2 * dW2 * dt * dph
        return float((self.n_accept / self.n_trials)
                     * float((self.maj["M"] * vol).sum()))

    def report(self):
        eff = self.n_accept / self.n_trials if self.n_trials else float("nan")
        n = self.n_hard + self.n_soft
        return dict(trials=self.n_trials, accepted=self.n_accept,
                    sigma_nb=self.sigma_total(),
                    efficiency=eff, hard=self.n_hard, soft=self.n_soft,
                    radiative_fraction=self.n_hard / n if n else float("nan"),
                    closure_failures=self.n_closure_fail,
                    overweight=self.n_overweight,
                    max_overweight=self.max_overweight,
                    cut_yxb=self.n_cut_yxb, cut_acceptance=self.n_cut_acc,
                    vmin_fallback=self.n_vmin_fallback,
                    tail_cells_built=len(self._tail_cache),
                    hard_chosen=self.n_hard_chosen,
                    accept2=self.n_accept2, b_viol=self.n_b_viol,
                    max_b_viol=self.max_b_viol,
                    drop_no_tail=self.n_drop_no_tail,
                    drop_v_ge_vmax=self.n_drop_v_ge_vmax,
                    drop_photon_at=self.n_drop_photon_at,
                    drop_except=self.n_drop_except)

    def health(self):
        r = self.report()
        msgs = []
        if r["overweight"] > 1e-3 * max(r["trials"], 1):
            msgs.append(f"majorant exceeded in {r['overweight']}/{r['trials']} "
                        f"trials (max x{r['max_overweight']:.2f}) -- raise "
                        f"safety or refine majorant_shape")
        if r["closure_failures"]:
            msgs.append(f"{r['closure_failures']} events failed four-momentum "
                        f"closure and were discarded")
        if r["efficiency"] < 0.01:
            msgs.append(f"unweighting efficiency {r['efficiency']:.4f} is low "
                        f"-- refine majorant_shape")
        return msgs
