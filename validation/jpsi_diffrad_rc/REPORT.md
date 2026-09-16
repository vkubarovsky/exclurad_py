# J/psi RC: exclurad_py vs DIFFRAD (same difflt model)

Beam 10.6 GeV, v_cut 0.03 GeV^2, model `jpsi.diffrad` (exact port of difflt, ivec=4) in both codes.

eta = sigma_obs/sigma_Born (DIFFRAD: `de` column).

| W2 | Q2 | t | eta DIFFRAD | eta exclurad_py | diff |
|---:|---:|---:|---:|---:|---:|
| 18.5 | 1.0 | -1.20 | 0.7740 +- 0.0004 | 0.8122 | +0.0382 |
| 18.5 | 1.0 | -1.50 | 0.7750 +- 0.0004 | 0.8120 | +0.0370 |
| 18.5 | 1.0 | -1.80 | 0.7752 +- 0.0006 | 0.8125 | +0.0372 |
| 18.5 | 1.0 | -2.10 | 0.7762 +- 0.0006 | 0.8135 | +0.0372 |
| 18.5 | 1.0 | -2.40 | 0.7772 +- 0.0006 | 0.8148 | +0.0376 |
| 18.5 | 0.5 | -1.80 | 0.7853 +- 0.0003 | 0.8195 | +0.0343 |
| 18.5 | 1.5 | -1.80 | 0.7665 +- 0.0003 | 0.8099 | +0.0434 |
| 17.5 | 1.0 | -1.80 | 0.7720 +- 0.0004 | 0.8127 | +0.0407 |
| 18.0 | 1.0 | -1.80 | 0.7733 +- 0.0005 | 0.8117 | +0.0385 |
| 19.0 | 1.0 | -1.80 | 0.7765 +- 0.0003 | 0.8153 | +0.0388 |

Mean diff (exclurad - DIFFRAD): +0.0383, RMS 0.0384, max |diff| 0.0434.

Both carry the same vacuum polarization (alpha/pi * vacpol = d_vac = 0.033 at Q2=1). The residual sits in the soft+virtual factor: DIFFRAD's deltavr is the inclusive/DIS vertex form with shifted Mandelstams, exclurad_py uses the exclusive vertex.
