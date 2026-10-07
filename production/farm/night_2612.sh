#!/bin/bash
# Overnight campaign 2026-10-03: amp2612 through LUND -> GEMC -> coatjava.
#
#   8 samples x 100 k events: pi0/eta, proton/neutron, rad/born.
#
# Proton samples are RG-A fall 2018 inbending (gemc 5.13 + coatjava 10.0.7).
# Neutron samples are RG-B spring 2019 (gemc 5.11 + coatjava 11.1.0): the
# neutron only exists in a deuteron target, so "fall 2018" cannot apply to it.
#
# Stage-2 bound pinned at 8,8,8,12 / 5000 probes: measured 0.57 % violations at
# max x2.18 against amp2609's own 0.47 % / x1.60 at the same settings, and the
# rate is 652 ms/event against 448.  The defaults give 0.67 % / x13.84.
set -u
R=${PROD_DIR:-/volatile/clas12/vpk/night_amp2612_20261003}
V=$HOME/exclurad_py_htilde
NEV_TOT=${NEV_TOT:-100000}
NJOB=${NJOB:-40}
NEV=$(( NEV_TOT / NJOB ))

mkdir -p "$R"
for spec in "pi0p:rad_pi0_rga.txt:pi0.amp2612" \
            "etap:rad_eta_rga.txt:eta.amp2612" \
            "pi0n:rad_pi0n_rgb_none.txt:pi0n.amp2612" \
            "etan:rad_etan_rgb_none.txt:etan.amp2612"; do
  ch=${spec%%:*}; rest=${spec#*:}; card=${rest%%:*}; model=${rest#*:}
  for mode in rad born; do
    ib=$([ "$mode" = born ] && echo 1 || echo 0)
    d="$R/$ch/$mode"; mkdir -p "$d"
    sed -e "s|^[a-z0-9]*\.amp[0-9]*|$model|" \
        -e "s|^[0-9]* \( *\)! nev|$NEV  \1! nev|" \
        -e "s|^[01]  \( *\)! iborn|$ib  \1! iborn|" \
        "$V/cards/$card" > "$d/card.txt"
    echo "  $ch/$mode -> $d/card.txt  ($NEV ev x $NJOB jobs = $NEV_TOT)"
  done
done
echo "cards written under $R"
