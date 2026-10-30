"""A majorant cache HIT must leave the normalisation consistent with the cdf.

Regression test for the bug where save_majorant persisted the Mtot-based cdf
but only M, so sigma_total normalised with integral(M*vol) a sample drawn from
M*B and reported sigma_nb low by <B>.
"""
import os, tempfile, numpy as np
from exclurad_py import models
from exclurad_py.generator.events import Generator, Box

def main():
    assert os.environ.get("EXACT_ACCEPT") == "1", "run with EXACT_ACCEPT=1"
    d = tempfile.mkdtemp()
    cache = os.path.join(d, "majorant.npz")
    box = Box((1.5, 3.0), (5.0, 8.0), (-0.8, -0.1))
    m = models.get("pi0.amp2609", t_nucl=-0.30)
    mk = lambda c: Generator(m, 10.6, box, helicity=1, seed=7,
                             majorant_shape=(3, 3, 3, 4),
                             table_shape=(3, 3, 3, 4), maj_cache=c).setup()

    cold = mk(None)
    cold.calibrate(n_probe=400, margin=2.0, verbose=False)
    bbar = cold.maj["Mtot"].sum() / cold.maj["M"].sum()
    cold.save_majorant(cache)

    warm = mk(cache)
    assert warm.maj_from_cache and warm.b_from_cache, "cache was not used"
    assert not warm.maj_calibrated, "calibrate() ran; the bug path needs it skipped"

    assert "Mtot" in warm.maj, "cache hit lost Mtot -- sigma_nb would be low by <B>"
    assert np.allclose(warm.maj["Mtot"], cold.maj["Mtot"]), \
        "restored Mtot differs from the calibrated one"
    assert np.allclose(np.asarray(warm.maj["cdf"]), np.asarray(cold.maj["cdf"])), \
        "cdf and Mtot no longer come from the same array"
    print(f"OK: cache hit restores Mtot; <B> = {bbar:.3f} would have been the bias")

    # and the fallback must refuse rather than guess
    warm.maj.pop("Mtot")
    # n_written too, since 2026-10-30: sigma_total normalises with the number of
    # events that reached the LUND file, and returns nan for an empty sample
    # before it gets as far as the Mtot question.  This is a synthetic state, so
    # it has to satisfy that precondition for the Mtot guard to be what is under
    # test.  The assertion below is unchanged.
    warm.n_accept2, warm.n_trials, warm.n_written = 1, 10, 1
    try:
        warm.sigma_total()
    except RuntimeError as e:
        assert "Mtot" in str(e)
        print("OK: sigma_total raises instead of normalising with M")
    else:
        raise AssertionError("sigma_total silently fell back to M")

if __name__ == "__main__":
    main()
