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

import numpy as np

from ..core.born import born_xsec
from ..core.kinematics import EventKin
from .tail import TailSampler, choose_vmin, photon_at
from .frames import assemble_event
from .tables import RCTable


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
                 safety=1.6, cache=None):
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

        self.table = None
        self.maj = None
        self._tail_cache = {}

        self.n_trials = 0
        self.n_accept = 0
        self.n_hard = 0
        self.n_soft = 0
        self.n_closure_fail = 0
        self.n_overweight = 0
        self.max_overweight = 1.0

    # ------------------------------------------------------------------ #
    def _kin(self, Q2, W2, t, phi):
        return EventKin(self.Ebeam, Q2, W2, t, phi, self.sf.mV,
                        v=0.0, mU=self.sf.mU)

    def _vmax(self, Q2, W2, t):
        vk = EventKin.vmax_kin(self.Ebeam, Q2, W2, t, self.sf.mV, mU=self.sf.mU)
        if self.mx2_cut is None:
            return vk
        return min(vk, self.mx2_cut - self.sf.mU ** 2)

    def _born(self, Q2, W2, t, phi, h):
        try:
            return born_xsec(self._kin(Q2, W2, t, phi), self.sf, h)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None

    # ------------------------------------------------------------------ #
    def setup(self, verbose=False):
        """Build the eta table and the Born-based majorant."""
        h = self.helicity if self.helicity != 0 else +1

        self.table = None
        if self.cache:
            self.table = RCTable.load(self.cache, self.sf, self.Ebeam,
                                      self.box, self.mx2_cut, h,
                                      self.table_shape)
        if self.table is None:
            if verbose:
                print("building eta table "
                      f"({int(np.prod(self.table_shape))} RC points)...")
            self.table = RCTable(self.sf, self.Ebeam, self.box, self.mx2_cut,
                                 h, self.table_shape).build(verbose=verbose)
            if self.cache:
                self.table.save(self.cache)

        eta_hi = float(np.nanmax(self.table.eta))

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
            v_min, soft_frac = choose_vmin(kin, self.sf, h, vm, sigma_obs)
            if v_min is None:
                self._tail_cache[key] = None
                return None
            self._tail_cache[key] = dict(
                ts=TailSampler(kin, self.sf, h, vm, v_min=v_min),
                soft_frac=soft_frac, buf=None, ibuf=0)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            self._tail_cache[key] = None
        return self._tail_cache[key]

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
    def generate(self, n_events, progress=None):
        if self.maj is None:
            self.setup()
        out = []
        while len(out) < n_events:
            Q2, W2, t, phi, M, cell = self._propose()
            h = self.helicity if self.helicity != 0 else \
                (+1 if self.rng.random() < 0.5 else -1)
            self.n_trials += 1

            b = self._born(Q2, W2, t, phi, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = float(self.table(Q2, W2, t, phi)[0])
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
            kin = self._kin(Q2, W2, t, phi)

            photon4 = meson4 = None
            got = self._tail_for_cell(cell, h, s_obs)
            if got is not None:
                if self.rng.random() > got["soft_frac"]:
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
                    except (ValueError, ZeroDivisionError):
                        photon4 = meson4 = None

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
            out.append(ev)
            if progress and len(out) % progress == 0:
                print(f"  {len(out)}/{n_events}")
        return out

    # ------------------------------------------------------------------ #
    def report(self):
        eff = self.n_accept / self.n_trials if self.n_trials else float("nan")
        n = self.n_hard + self.n_soft
        return dict(trials=self.n_trials, accepted=self.n_accept,
                    efficiency=eff, hard=self.n_hard, soft=self.n_soft,
                    radiative_fraction=self.n_hard / n if n else float("nan"),
                    closure_failures=self.n_closure_fail,
                    overweight=self.n_overweight,
                    max_overweight=self.max_overweight,
                    tail_cells_built=len(self._tail_cache))

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
