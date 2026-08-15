"""
Precomputed RC tables.

The generator cannot call `rc_factor` per event: one RC point costs ~0.1 s
(a 4x10^4-node photon quadrature), which is six orders of magnitude too
slow for 10^6 events.  It does not need to.  The RC factor

    eta(Q2, W2, t, phi) = sigma_obs / sigma_Born

is a smooth, slowly varying function of order unity, while the violent
variation of the cross section lives entirely in sigma_Born, which is a
direct model evaluation and costs microseconds.  So:

    sigma_obs(x) = sigma_Born(x) * eta_interp(x)

with eta tabulated once on a coarse grid and linearly interpolated, and
sigma_Born evaluated exactly per event.  The interpolation error on eta
propagates directly to the event weight, so the table records its own
convergence: `RCTable.check_interpolation()` re-evaluates eta exactly at
random points and reports the residual.

Tables are cached to .npz with a full provenance stamp -- a table built
with pi0.vpk2014 must never be silently reused for pi0.vpk2021, so the
loader refuses a cache whose model name, version, beam energy or cut does
not match what was asked for.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..core.kinematics import EventKin
from ..core.rc import rc_factor
from ..provenance import stamp


class RCTable:
    """eta on a (Q2, W2, t, phi) grid, with cache and interpolation."""

    def __init__(self, sfmodel, Ebeam, box, mx2_cut=None, helicity=+1,
                 shape=(5, 5, 5, 8)):
        self.sf = sfmodel
        self.Ebeam = float(Ebeam)
        self.box = box
        self.mx2_cut = mx2_cut
        self.helicity = helicity
        self.shape = tuple(shape)

        self.axes = [
            np.linspace(*box.Q2, shape[0]),
            np.linspace(*box.W2, shape[1]),
            np.linspace(*box.t, shape[2]),
            np.linspace(*box.phi, shape[3]),
        ]
        self.eta = None
        self.valid = None

    # ------------------------------------------------------------------ #
    def _key(self):
        return dict(model=self.sf.name, version=self.sf.version,
                    Ebeam=self.Ebeam, mx2_cut=self.mx2_cut,
                    helicity=self.helicity, shape=list(self.shape),
                    box=self.box.as_dict(),
                    t_nucl=getattr(self.sf, "t_nucl", None),
                    # bump when the build filter or interpolation changes,
                    # so stale cached tables are not silently reused
                    interp="log-v2",
                    flux="gilman-v3")

    def build(self, verbose=False, jobs=None):
        """Evaluate eta at every grid node. This is the expensive step.

        Embarrassingly parallel -- every node is independent -- so it is
        split across `jobs` processes (default: all cores but one). A
        4096-node table is ~7 minutes serial and well under a minute on 14
        cores, which matters because a rad run cannot start without it.

        Workers rebuild the model from its registry name rather than
        receiving a pickled object, so this works under spawn as well as
        fork.
        """
        if jobs is None:
            jobs = max(1, (os.cpu_count() or 2) - 1)
        if jobs > 1:
            try:
                return self._build_parallel(jobs, verbose)
            except Exception as exc:            # fall back rather than fail
                if verbose:
                    print(f"  parallel table build unavailable ({exc}); serial")
        return self._build_serial(verbose)

    def _build_parallel(self, jobs, verbose):
        Q2a, W2a, ta, pa = self.axes
        nodes = [(i, j, k, l, float(Q2), float(W2), float(t), float(ph))
                 for i, Q2 in enumerate(Q2a) for j, W2 in enumerate(W2a)
                 for k, t in enumerate(ta) for l, ph in enumerate(pa)]
        kw = dict(t_nucl=getattr(self.sf, "t_nucl", None),
                  Ebeam=getattr(self.sf, "Ebeam", None))
        args = dict(model=self.sf.name, ctor=kw, Ebeam=self.Ebeam,
                    mx2_cut=self.mx2_cut, helicity=self.helicity)
        chunks = [nodes[i::jobs] for i in range(jobs)]
        if verbose:
            print(f"  eta table: {len(nodes)} nodes on {jobs} processes...")
        eta = np.ones(self.shape)
        valid = np.zeros(self.shape, dtype=bool)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for res in ex.map(_eta_chunk, [(c, args) for c in chunks]):
                for (i, j, k, l), e in res:
                    eta[i, j, k, l] = e
                    valid[i, j, k, l] = True
        if valid.any():
            eta[~valid] = float(np.exp(np.log(eta[valid]).mean()))
        self.eta, self.valid = eta, valid
        return self

    def _build_serial(self, verbose=False):
        Q2a, W2a, ta, pa = self.axes
        eta = np.ones(self.shape)
        valid = np.zeros(self.shape, dtype=bool)
        n_done = 0
        for i, Q2 in enumerate(Q2a):
            for j, W2 in enumerate(W2a):
                for k, t in enumerate(ta):
                    for l, ph in enumerate(pa):
                        try:
                            # fixed-t_N models (pi0/eta) must be pointed at
                            # THIS node's transfer, exactly as the generator
                            # does per event -- otherwise the whole table is
                            # built at one stale t_nucl.
                            if hasattr(self.sf, "t_nucl"):
                                self.sf.t_nucl = float(t)
                            r = rc_factor(Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t,
                                          phi=ph, sfmodel=self.sf,
                                          h=self.helicity,
                                          mx2_cut=self.mx2_cut)
                            e = r["eta"]
                            # keep every positive finite value -- see the
                            # matching comment in _eta_chunk
                            if np.isfinite(e) and e > 0.0:
                                eta[i, j, k, l] = e
                                valid[i, j, k, l] = True
                        except (ValueError, ZeroDivisionError,
                                FloatingPointError):
                            pass
                        n_done += 1
            if verbose:
                print(f"  eta table: Q2 slice {i+1}/{len(Q2a)} "
                      f"({n_done}/{int(np.prod(self.shape))} nodes)")
        # fill unusable nodes with the GEOMETRIC mean of the valid ones so the
        # interpolator stays finite at the edges of phase space; with eta
        # spanning decades the arithmetic mean is dominated by the few
        # largest nodes and would poison every neighbouring cell
        if valid.any():
            eta[~valid] = float(np.exp(np.log(eta[valid]).mean()))
        self.eta, self.valid = eta, valid
        return self

    # ------------------------------------------------------------------ #
    def __call__(self, Q2, W2, t, phi):
        """Multilinear interpolation of eta. Arrays welcome."""
        if self.eta is None:
            raise RuntimeError("table not built; call build() or load()")
        pts = [np.atleast_1d(np.asarray(x, dtype=float))
               for x in (Q2, W2, t, phi)]
        pts = np.broadcast_arrays(*pts)
        out = np.ones(pts[0].shape)

        idx, frac = [], []
        for ax, x in zip(self.axes, pts):
            if len(ax) == 1:
                idx.append(np.zeros(x.shape, dtype=int))
                frac.append(np.zeros(x.shape))
                continue
            i = np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2)
            idx.append(i)
            frac.append(np.clip((x - ax[i]) / (ax[i + 1] - ax[i]), 0.0, 1.0))

        # Interpolate log(eta), not eta: the value spans several decades and
        # linear interpolation between e.g. 1 and 1e3 gives ~500 halfway,
        # where the geometric (log-linear) 30 is the right order.  This is
        # also what makes the un-clipped high nodes usable at all.
        out[...] = 0.0
        for corner in range(16):
            w = np.ones(pts[0].shape)
            sel = []
            for d in range(4):
                bit = (corner >> d) & 1
                if len(self.axes[d]) == 1:
                    sel.append(idx[d])
                    continue
                sel.append(idx[d] + bit)
                w = w * (frac[d] if bit else (1.0 - frac[d]))
            out += w * np.log(self.eta[tuple(sel)])
        return np.exp(out)

    # ------------------------------------------------------------------ #
    def check_interpolation(self, n=25, rng=None):
        """Exact vs interpolated eta at random points inside the box."""
        rng = rng or np.random.default_rng(0)
        res = []
        for _ in range(n):
            Q2 = rng.uniform(*self.box.Q2)
            W2 = rng.uniform(*self.box.W2)
            t = rng.uniform(*self.box.t)
            ph = rng.uniform(*self.box.phi)
            try:
                if hasattr(self.sf, "t_nucl"):
                    self.sf.t_nucl = float(t)
                ex = rc_factor(Ebeam=self.Ebeam, Q2=Q2, W2=W2, t=t, phi=ph,
                               sfmodel=self.sf, h=self.helicity,
                               mx2_cut=self.mx2_cut)["eta"]
            except (ValueError, ZeroDivisionError):
                continue
            ip = float(self(Q2, W2, t, ph)[0])
            res.append((ex, ip, ip - ex))
        if not res:
            return dict(n=0)
        d = np.array([r[2] for r in res])
        return dict(n=len(res), mean=float(d.mean()), rms=float(d.std()),
                    max_abs=float(np.abs(d).max()))

    # ------------------------------------------------------------------ #
    def save(self, path):
        np.savez_compressed(
            path, eta=self.eta, valid=self.valid,
            axes=np.array(self.axes, dtype=object), allow_pickle=True,
            key=json.dumps(self._key()),
            stamp=json.dumps(stamp(self.sf, extra=self._key())))
        return path

    @classmethod
    def load(cls, path, sfmodel, Ebeam, box, mx2_cut=None, helicity=+1,
             shape=(5, 5, 5, 8)):
        """Load a cached table, refusing a mismatched one."""
        obj = cls(sfmodel, Ebeam, box, mx2_cut, helicity, shape)
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        key = json.loads(str(d["key"]))
        want = obj._key()
        if key != want:
            diff = {k: (key.get(k), want.get(k))
                    for k in set(key) | set(want) if key.get(k) != want.get(k)}
            raise ValueError(
                f"cached RC table {path} does not match this configuration; "
                f"differences (cached, requested): {diff}. Delete the cache "
                f"or use a different path -- reusing an eta table built with "
                f"another model or cut silently corrupts the sample.")
        obj.eta = d["eta"]
        obj.valid = d["valid"]
        obj.axes = list(d["axes"])
        return obj


# --------------------------------------------------------------------------- #
def _eta_chunk(payload):
    """Worker: evaluate eta at a list of grid nodes. Top level, so picklable."""
    nodes, args = payload
    from .. import models
    from ..core.rc import rc_factor
    import numpy as _np

    ctor = {k: v for k, v in args["ctor"].items() if v is not None}
    sf = models.get(args["model"], **ctor)
    out = []
    for (i, j, k, l, Q2, W2, t, ph) in nodes:
        try:
            if hasattr(sf, "t_nucl"):
                sf.t_nucl = float(t)
            r = rc_factor(Ebeam=args["Ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                          sfmodel=sf, h=args["helicity"],
                          mx2_cut=args["mx2_cut"])
            e = r["eta"]
            # eta legitimately spans several decades on wide boxes: it blows
            # up where sigma_born -> 0 (the pi0 box holds nodes with eta ~ 1e5,
            # median 129 among those the old 0.2 < e < 5 filter DISCARDED).
            # Clipping them biased the radiative tail soft by exactly the
            # missing enhancement; keep every positive finite value.
            if _np.isfinite(e) and e > 0.0:
                out.append(((i, j, k, l), float(e)))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass
    return out
