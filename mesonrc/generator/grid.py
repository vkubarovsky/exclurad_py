"""
Cell grid and majorant for hard-photon sampling.

The photon phase space is (v, cos theta_k, phi_k) in each of two importance
channels centred on the incident and the scattered lepton.  The density has
two collinear peaks spanning many orders of magnitude, so a single global
majorant is useless: the grid below is piecewise-constant over cells, with
the peaks flattened by the substitution

    x = ln(a - cos theta_c),     a = E_lepton / |k_lepton|

which is the same substitution the RC quadrature uses.  dcos theta = e^x dx,
so the sampled density carries an explicit e^x Jacobian.

Cells are uniform in (ln v, x, phi_k).  For each cell we store
  * I  -- the cell's contribution to the integral (mean over sub-samples
          times the cell volume), used to pick a cell;
  * M  -- max over the same sub-samples times a safety factor, used as the
          rejection majorant.
Sub-sampling is done for ALL cells in one vectorized pass.

A majorant estimated from a finite number of sub-samples can be exceeded by
a real point.  Those overweight events are counted and reported, never
clipped: an overweight rate above a per-mille or a max ratio far above 1
means the grid is too coarse for the model, and the generator says so
instead of quietly biasing the sample.
"""

import numpy as np


class ChannelGeometry:
    """Angular frame and x-substitution for one importance channel."""

    def __init__(self, kvec, m):
        ak = float(np.linalg.norm(kvec))
        E = float(np.sqrt(ak * ak + m * m))
        self.ak, self.E = ak, E
        self.zc = kvec / ak
        ref = np.zeros(3)
        ref[int(np.argmin(np.abs(self.zc)))] = 1.0
        x = np.cross(ref, self.zc)
        self.xc = x / np.linalg.norm(x)
        self.yc = np.cross(self.zc, self.xc)

        # a - 1 computed stably;  a = E/|k|
        self.am1 = m * m / (ak * (E + ak))
        self.x_min = np.log(self.am1)
        self.x_max = np.log(E / ak + 1.0)

    def directions(self, x, phi):
        """(x, phi) -> unit vectors (N,3) and the Jacobian e^x."""
        ex = np.exp(x)
        one_m_c = ex - self.am1
        ct = 1.0 - one_m_c
        ct = np.clip(ct, -1.0, 1.0)
        st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
        n = (ct[:, None] * self.zc[None, :]
             + st[:, None] * (np.cos(phi)[:, None] * self.xc[None, :]
                              + np.sin(phi)[:, None] * self.yc[None, :]))
        return n, ex


class CellGrid:
    """Piecewise-constant majorant over (ln v, x, phi_k) x channels."""

    def __init__(self, geoms, v_min, v_max, n_v, n_x, n_phi,
                 safety=1.5, n_sub=3):
        self.geoms = geoms
        self.n_v, self.n_x, self.n_phi = n_v, n_x, n_phi
        self.n_chan = len(geoms)
        self.safety = float(safety)
        self.n_sub = int(n_sub)

        self.lv_edges = np.linspace(np.log(v_min), np.log(v_max), n_v + 1)
        self.x_edges = [np.linspace(g.x_min, g.x_max, n_x + 1) for g in geoms]
        self.phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)

        self.shape = (self.n_chan, n_v, n_x, n_phi)
        self.n_cells = int(np.prod(self.shape))
        self.I = None
        self.M = None
        self.cdf = None

    # ------------------------------------------------------------------ #
    def cell_coords(self, flat):
        """flat cell index -> (chan, iv, ix, iphi)."""
        return np.unravel_index(flat, self.shape)

    def cell_volume(self, ic, iv, ix):
        dlv = self.lv_edges[iv + 1] - self.lv_edges[iv]
        dx = self.x_edges[ic][ix + 1] - self.x_edges[ic][ix]
        dphi = self.phi_edges[1] - self.phi_edges[0]
        return dlv * dx * dphi

    # ------------------------------------------------------------------ #
    def build(self, density_fn, rng):
        """Estimate per-cell integral and majorant.

        density_fn(v, nhat, chan_index) -> array of the POSITIVE integrand
        already including the channel weight, but NOT the e^x Jacobian nor
        the dv = v dlnv factor (those are applied here).

        All cells and sub-samples are evaluated in one batched call per
        channel -- the whole reason the setup is affordable.
        """
        I = np.zeros(self.shape)
        M = np.zeros(self.shape)

        for ic, g in enumerate(self.geoms):
            iv, ix, ip, isub = np.meshgrid(
                np.arange(self.n_v), np.arange(self.n_x),
                np.arange(self.n_phi), np.arange(self.n_sub),
                indexing="ij")
            iv, ix, ip = iv.ravel(), ix.ravel(), ip.ravel()
            n_pts = iv.size

            # stratified jitter inside each cell
            u = rng.random((n_pts, 3))
            lv = self.lv_edges[iv] + u[:, 0] * np.diff(self.lv_edges)[iv]
            xe = self.x_edges[ic]
            x = xe[ix] + u[:, 1] * np.diff(xe)[ix]
            ph = self.phi_edges[ip] + u[:, 2] * np.diff(self.phi_edges)[ip]

            v = np.exp(lv)
            nhat, ex = g.directions(x, ph)

            f = density_fn(v, nhat, ic)          # positive integrand
            val = f * ex * v                     # * Jacobians dcos, dv

            val = val.reshape(self.n_v, self.n_x, self.n_phi, self.n_sub)
            mean = val.mean(axis=-1)
            mx = val.max(axis=-1)

            dlv = np.diff(self.lv_edges)[:, None, None]
            dx = np.diff(xe)[None, :, None]
            dphi = np.diff(self.phi_edges)[0]
            I[ic] = mean * dlv * dx * dphi
            M[ic] = mx * self.safety

        self.I = np.maximum(I, 0.0)
        self.M = M
        tot = self.I.sum()
        self.cdf = np.cumsum(self.I.ravel()) / tot if tot > 0 else None
        self.total = tot
        return tot

    # ------------------------------------------------------------------ #
    def propose(self, rng, n):
        """Draw n candidate points uniformly inside cells chosen ~ I.

        Returns v (n,), nhat (n,3), chan (n,), majorant (n,), jac (n,)
        where jac = e^x * v is the measure factor the candidate carries.
        """
        flat = np.clip(np.searchsorted(self.cdf, rng.random(n)),
                       0, self.n_cells - 1)
        ic, iv, ix, ip = self.cell_coords(flat)

        u = rng.random((n, 3))
        lv = self.lv_edges[iv] + u[:, 0] * np.diff(self.lv_edges)[iv]
        v = np.exp(lv)

        nhat = np.empty((n, 3))
        ex = np.empty(n)
        for c, g in enumerate(self.geoms):
            sel = np.where(ic == c)[0]
            if not len(sel):
                continue
            xe = self.x_edges[c]
            x = xe[ix[sel]] + u[sel, 1] * np.diff(xe)[ix[sel]]
            ph = (self.phi_edges[ip[sel]]
                  + u[sel, 2] * np.diff(self.phi_edges)[ip[sel]])
            nh, e = g.directions(x, ph)
            nhat[sel] = nh
            ex[sel] = e

        return v, nhat, ic, self.M[ic, iv, ix, ip], ex * v

    def occupancy(self):
        """Diagnostics: how concentrated the grid is."""
        w = self.I.ravel()
        w = w[w > 0]
        if not len(w):
            return dict(nonzero_cells=0, top1_frac=0.0, top10_frac=0.0)
        s = np.sort(w)[::-1] / w.sum()
        return dict(nonzero_cells=int(len(w)),
                    top1_frac=float(s[0]),
                    top10_frac=float(s[:max(1, len(s) // 10)].sum()))
