# Overnight validation, 2026-08-14/15

Each channel: 20k RAD events (capped proposal, ETA_PROPOSAL_CAP=5) validated
against the exact quadrature via production/validate_channel.py -- reweighted
tail fractions vs sigma_obs-weighted theory at v > 0.005/0.01/0.05/0.1/0.3.

Round 1 (<ch>.json):   cell-based soft/hard split (16 tail cells per box).
Round 2 (r2_<ch>.json): EXACT_PER_EVENT=1 -- split and photon density from
                        each event's own kinematics.

    channel   round 1     round 2
    pi0        2/5         5/5 PASS   <- cell approximation closed
    eta        5/5         5/5 PASS
    rho        5/5         5/5 PASS
    phi        3/5         3/5        <- REAL open issue: soft-photon excess
                                         (41.0% vs 33.6% at v>0.005); deep
                                         tail (v>0.05..0.3) agrees
    jpsi       5/5         (stalled)  <- R2 generation did not converge in 9h;
                                         near-threshold efficiency x exact
                                         per-event cost.  R1 verdict stands.

LUND integrity clean on all channels (phi meson mass 1.019412 is the
deliberate diffrad_vm transcription value -- daytime decision pending).

Separately, the eta-Born discrepancy vs exclugen (<Q2> 0.795 vs 1.415) was
localized: our sample follows the exact theory in every Q2 slice (ratios
0.93-1.09); exclugen's own bornin density under-counts low-Q2/large-|t| by
up to x2.7 relative to its own forward-region convention, inflating its
Q2>4 share from 4% to 14%.  Suspect link: the t <-> cos(theta) translation
in exclurad_core at eta masses (pi0 is immune -- small meson mass).  The
sigma_T of the two codes is IDENTICAL pointwise up to Q2=9 (ratio
1.000 +/- 0.001), so this is assembly, not model.
