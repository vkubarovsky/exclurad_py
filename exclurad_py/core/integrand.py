"""One entry point for the radiative integrand LRH = Re(L_R : H).

This is the whole cost of rad generation (docs/cost_model.md): the exact
acceptance and the photon tail sampler between them evaluate it ~232 000
times per written event, and it is the same quantity in both.

Two backends:

  numpy  the default, unchanged, byte-identical to what production ships
  numba  opt-in with RC_NUMBA=1, a fused per-row kernel that never
         materialises the (N,4,4) temporaries

The numba path is NOT bit-identical.  Its disagreement with numpy is
confined to the collinear core and is not a defect of the port: `rad_L`
obtains kappa1 = 2 k.k1 from `dot4`, which cancels catastrophically when
the photon is collinear with the lepton, and the two backends simply round
that ill-conditioned subtraction differently.  Measured on a production
block: relative disagreement 7e-16 away from the collinear region, rising
to ~2e-10 (median) where kappa1/E1om < 1e-5 -- and in that same region the
numpy kappa1 itself differs from the STABLE 2*om*kap_own that
`_channel_nodes` already computes by up to 2.5e-8.  So both backends are
limited by the same conditioning, and neither is the more accurate one.
"""
import os
import warnings

import numpy as np

from .spinors import rad_L, contractLH, uspinor, _ubar, GAMMA
from .hadronic import hadronic_tensor, POL_KEYS

# RC_NUMBA selects the backend.  The numba kernel is the DEFAULT:
#
#   unset   use numba if it is importable, else numpy with a LOUD warning
#   1       REQUIRE numba -- raise rather than fall back
#   0       force numpy deliberately, no warning
#
# Default-on is what production wants, but it makes the silent fallback the
# dangerous case rather than the safe one: a campaign would run 2.3x slower
# than planned and nothing would say so.  An environment variable proves
# what you ASKED FOR; only the artefact proves what you GOT.  So the
# fallback warns here, and `run.py` stamps `rc_backend` into every sample's
# provenance, which is the artefact.
_ENV = os.environ.get("RC_NUMBA", "").strip()
_IMPORT_ERR = None
try:
    from ._kernel_numba import lrh_kernel, HAVE_NUMBA
except Exception as _e:                                # pragma: no cover
    HAVE_NUMBA = False
    _IMPORT_ERR = _e

_why = (f" ({type(_IMPORT_ERR).__name__}: {_IMPORT_ERR})" if _IMPORT_ERR else "")
if _ENV == "0":
    USE_NUMBA = False                                   # deliberate numpy
elif _ENV in ("1", "true", "True", "yes", "on"):
    if not HAVE_NUMBA:
        raise RuntimeError(
            "RC_NUMBA=1 requires the numba kernel and it is unavailable"
            + _why + ".\nRefusing to fall back silently: the run would be "
            "~2.3x slower than planned with nothing to say why.\nUse an "
            "interpreter with numba (on ifarm, a venv built with "
            "--system-site-packages so the system numpy/scipy are kept), "
            "or set RC_NUMBA=0 to choose numpy deliberately.")
    USE_NUMBA = True
else:                                                   # unset: numba by default
    USE_NUMBA = HAVE_NUMBA
    if not HAVE_NUMBA:
        warnings.warn(
            "exclurad_py: the numba integrand kernel is unavailable" + _why
            + "; falling back to the numpy backend, which is correct but "
              "~2.3x slower. Set RC_NUMBA=0 to silence this, or install "
              "numba. The backend actually used is recorded as "
              "`rc_backend` in the run's config.json and .info sidecar.",
            RuntimeWarning, stacklevel=2)

_cache = {}


def _spinor_setup(k1, k2, m, h):
    """u1, ub2, UB2G, GU1 -- photon-independent, so built once per (k1,k2,m,h)."""
    key = (k1.tobytes(), k2.tobytes(), float(m), int(h))
    got = _cache.get(key)
    if got is None:
        u1 = uspinor(k1, m, h)
        ub2 = np.array([_ubar(uspinor(k2, m, l2)) for l2 in (+1, -1)])
        got = (u1, ub2,
               np.einsum('ai,mij->amj', ub2, GAMMA),
               np.einsum('mij,j->mi', GAMMA, u1))
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = got
    return got


def lrh_batch(k1, k2, m, h, k4, qt, p, pv, W2t, sf, M, lam=0.0, H=None):
    """Re(L_R^{mu nu} H_{mu nu}) for a batch of photons, shape (N,)."""
    polarised = bool(lam) and all(k in sf for k in POL_KEYS)
    if USE_NUMBA and not polarised:
        u1, ub2, UB2G, GU1 = _spinor_setup(k1, k2, m, h)
        out = np.empty(len(W2t))
        nbad = np.zeros(1, dtype=np.int64)
        lrh_kernel(np.ascontiguousarray(k1), np.ascontiguousarray(k2), float(m),
                   np.ascontiguousarray(k4), np.ascontiguousarray(qt),
                   np.ascontiguousarray(p), np.ascontiguousarray(pv),
                   np.ascontiguousarray(W2t),
                   np.ascontiguousarray(sf['sT']), np.ascontiguousarray(sf['sL']),
                   np.ascontiguousarray(sf['sTT']), np.ascontiguousarray(sf['sLT']),
                   np.ascontiguousarray(sf['sLTp']),
                   float(M), u1, ub2, UB2G, GU1, out, nbad)
        if nbad[0]:
            # The guard fires only on kinematics that would have divided by
            # zero.  It has never fired in validation; if it does, the rows
            # were dropped and the result is no longer trustworthy, so say
            # so loudly rather than return a quietly biased number.
            raise RuntimeError(
                f"lrh_kernel: {int(nbad[0])} of {len(out)} rows hit a "
                "degenerate denominator and were skipped. The numpy backend "
                "(unset RC_NUMBA) masks these instead. Please report the card.")
        return out
    if H is None:
        H = hadronic_tensor(qt, p, pv, W2t, sf, M=M, lam=lam)
    LR, _, _ = rad_L(k1, k2, m, h, k4)
    return np.real(contractLH(LR, H))
