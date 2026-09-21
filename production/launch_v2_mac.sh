#!/bin/bash
# bash-3 compatible (macOS /bin/bash): 3/3/2/2 parts over the four spin states
R=$1; H=$2; S0=$3
spin() { case $1 in hp_pp) echo beam_helicity_pos/target_pol_pos;; hp_pm) echo beam_helicity_pos/target_pol_neg;; hm_pp) echo beam_helicity_neg/target_pol_pos;; hm_pm) echo beam_helicity_neg/target_pol_neg;; esac; }
nparts() { case $1 in hp_pp|hp_pm) echo 3;; *) echo 2;; esac; }
k=0
for c in hp_pp hp_pm hm_pp hm_pm; do
  s=$(spin $c); tab=$HOME/tables/$(echo $s | tr / _)_etatable.npz; sb=$((S0 + k*3000000)); k=$((k+1))
  tmux -L prod new-session -d -s v2_$c "caffeinate -dimsu env ETA_TABLE=$tab bash $HOME/exclurad_py/production/run_prod_v2.sh $HOME/exclurad_py $R/cards/$c.txt $R/$s $(nparts $c) $sb $H rad"
done
tmux -L prod new-session -d -s v2_watch "bash $HOME/exclurad_py/production/watchdog.sh $R 900"
sleep 25
echo "  $H: workers=$(pgrep -f 'exclurad_py.run [i]mport' | wc -l | tr -d ' ') tmux=$(tmux -L prod ls | wc -l | tr -d ' ')"
for c in hp_pp hp_pm hm_pp hm_pm; do echo "    $c -> $(spin $c): $(grep -o 'seedbase [0-9]*' $R/$(spin $c)/run.log)  parts $(nparts $c)"; done
