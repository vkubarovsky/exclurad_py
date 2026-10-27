# J/psi radiative correction: GEMC rad/Born MC (method 1) against rc_factor (method 2)

F18 inbending and outbending, `prod_jpsi_f18in_20261009` (GEMC 5.14 snapshot of 2026-10-27,
2177 chunks, sha256-checked against ifarm), M. Tenorio's selection, topologies e'e+e- and e'e+e-p'.
Report: `rc_method_jpsi_report.tex` (built by `publish.sh`). Every quoted number is in
`summary.json` and reaches the text through `numbers.tex` / `tab_*.tex`.

## Result in her three W bins (Q2 0.02-0.12)

| topology | W | method 1 inb | method 1 outb | method 2 nominal | method 2 effective |
|---|---|---|---|---|---|
| e'e+e-   | 4.15-4.29 | 0.767 ± 0.017 | 0.784 ± 0.017 | 0.800 | 0.772 |
| e'e+e-   | 4.29-4.40 | 0.774 ± 0.011 | 0.783 ± 0.011 | 0.812 | 0.780 |
| e'e+e-   | 4.40-4.48 | 0.776 ± 0.009 | 0.786 ± 0.009 | 0.816 | 0.783 |
| e'e+e-p' | 4.15-4.29 | 0.771 ± 0.020 | 0.738 ± 0.035 | 0.808 | 0.775 |
| e'e+e-p' | 4.29-4.40 | 0.786 ± 0.012 | 0.787 ± 0.016 | 0.826 | 0.787 |
| e'e+e-p' | 4.40-4.48 | 0.798 ± 0.010 | 0.804 ± 0.013 | 0.834 | 0.793 |

M. Tenorio uses 0.88 x 0.85 = 0.75. The generated 10M without cuts gives 0.902 / 0.925 / 0.933.

## Why the methods differ, in one paragraph

All three arm differences (iborn, itarg, idecay) are in C_rad. With nominal cuts, method 2 is 3-5 % above method 1. The
difference is made up of three parts:
- the pre-vertex radiator's energy shift: -1.24 %. The generator-level closure finds -1.15 %, and the itarg=1 Born run
  gives -1.24 %.
- the decay-photon cut: it acts at k* = 0.111 / 0.123 GeV, not at the M(ee) edge 0.143, because of the M(ee) and MX
  resolution. This is worth -1.8 / -1.1 %.
- the production cut: it acts at v50 = 0.31 / 0.73 GeV^2. This is worth 0 to -1 %.

Migration and acceptance weighting are each below 2 %, except at threshold. With those two effective edges, method 2
matches method 1:
- four-track sample: every binning, chi2/cell about 1;
- three-track sample: W, Q2 and her bins.

It fails in one place: |t| built from e'e+e-. Decay radiation moves t there by up to 15 % at low |t|, and nothing
analytic contains that. Inbending and outbending agree (chi2 51/67 and 32/50).

## Reproduce

```
python3 production/analysis/rc_method_jpsi_combos.py      # both methods, all binnings (~35 min, 10 cores)
python3 production/analysis/rc_method_jpsi_closure.py     # generator level, no cut
python3 production/analysis/rc_method_jpsi_gencuts.py docs/rc_method_jpsi
python3 production/analysis/rc_method_jpsi_cutvars.py docs/rc_method_jpsi
python3 production/analysis/rc_method_jpsi_vdist.py   docs/rc_method_jpsi
python3 production/analysis/rc_method_jpsi_plots.py  /Volumes/wd_14tb/rc_method_jpsi/out docs/rc_method_jpsi
python3 production/analysis/rc_method_jpsi_tables.py docs/rc_method_jpsi
bash docs/rc_method_jpsi/publish.sh
```
Data and caches: `/Volumes/wd_14tb/rc_method_jpsi` (ntuple/, prov/, out/).
