# J/psi radiative correction: GEMC rad/Born MC (method 1) against rc_factor (method 2)

F18 inbending and outbending, `prod_jpsi_f18in_20261009` (GEMC 5.14 snapshot of 2026-10-27,
2177 chunks, sha256-checked against ifarm), M. Tenorio's selection, topologies e'e+e- and e'e+e-p'.
Report: `rc_method_jpsi_report.tex` (built by `publish.sh`). Every quoted number is in
`summary.json` and reaches the text through `numbers.tex` / `tab_*.tex`.

## Result in her three W bins (Q2 0.02-0.12)

| topology | W | method 1 inb | method 1 outb | method 2 nominal | method 2 effective |
|---|---|---|---|---|---|
| e'e+e-   | 4.15-4.29 | 0.757 ± 0.017 | 0.773 ± 0.017 | 0.800 | 0.772 |
| e'e+e-   | 4.29-4.40 | 0.763 ± 0.010 | 0.773 ± 0.010 | 0.812 | 0.780 |
| e'e+e-   | 4.40-4.48 | 0.766 ± 0.009 | 0.776 ± 0.009 | 0.816 | 0.783 |
| e'e+e-p' | 4.15-4.29 | 0.760 ± 0.020 | 0.728 ± 0.034 | 0.808 | 0.775 |
| e'e+e-p' | 4.29-4.40 | 0.776 ± 0.012 | 0.777 ± 0.015 | 0.826 | 0.787 |
| e'e+e-p' | 4.40-4.48 | 0.787 ± 0.010 | 0.794 ± 0.013 | 0.834 | 0.793 |

M. Tenorio uses 0.88 x 0.85 = 0.75. The generated 10M (generation acceptance only), normalisation-corrected: 0.890 / 0.913 / 0.921.

## Normalisation fault found on the way

`sigma_nb` counts every event that passes accept-reject. The iacc=jpsi daughter preset is applied afterwards and drops
events, so in every chunk accepted - cut_acceptance = 5000 written. The preset keeps 0.6168 of rad and 0.6251 of Born.
The rad/Born ratio of the written events is therefore 0.9336, not 0.9462: a factor 0.9867 on every n/N x sigma ratio.
That includes the 10M tables and any iacc != none campaign.
Corrected here by `rc_method_jpsi.sigma()`, chunk by chunk. The production fix is for the Launch generator session.

## Why the methods differ

With nominal cuts, method 2 is 4-6 % above method 1. The pieces:
- the pre-vertex radiator's energy shift: -1.24 %;
- the decay photon cut at k* 0.111 / 0.123 GeV, not at 0.143: -1.8 / -1.1 %;
- the production cut at v50 0.31 / 0.73: 0 to -1 %;
- the detector's response to radiating leptons: -1 to -2 %. Lepton PID alone is 0.4-0.9 %.

Migration and acceptance weighting are each below 2 %, except at threshold.
The generator-level closure, -2.47 % against -2.56 % predicted (radiator + preset), closes to 0.1 %.

The shapes agree everywhere except 3-track |t|. Built from e+e-, t is moved by decay FSR, up to 15 % at low |t|. That is
a t-dependent systematic in M. Tenorio's 3-track t results, which her single factor 0.88 cannot carry.
Inbending = outbending (chi2 51/67, 32/50).

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
