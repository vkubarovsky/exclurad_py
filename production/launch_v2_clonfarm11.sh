#!/bin/bash
R=/data/vpk/prod_pi0_p_v2
for c in hp_pp hp_pm hm_pp hm_pm; do tmux kill-session -t v2_$c 2>/dev/null; done; tmux kill-session -t v2_watch 2>/dev/null
pkill -f "exclurad_py.run [i]mport" 2>/dev/null; sleep 3
rm -rf $R/beam_helicity_*
echo "  clean: workers=$(pgrep -fc "exclurad_py.run [i]mport") tmux=$(tmux ls 2>/dev/null | wc -l) dirs=$(ls -d $R/beam_helicity_* 2>/dev/null | wc -l)"
declare -A B=([hp_pp]=beam_helicity_pos/target_pol_pos [hp_pm]=beam_helicity_pos/target_pol_neg [hm_pp]=beam_helicity_neg/target_pol_pos [hm_pm]=beam_helicity_neg/target_pol_neg)
declare -A SB=([hp_pp]=30000000 [hp_pm]=33000000 [hm_pp]=36000000 [hm_pm]=39000000)
for c in hp_pp hp_pm hm_pp hm_pm; do
  s=${B[$c]}; tab=$HOME/tables/$(echo $s | tr / _)_etatable.npz
  tmux new-session -d -s v2_$c "ETA_TABLE=$tab bash $HOME/exclurad_py/production/run_prod_v2.sh $HOME/exclurad_py $R/cards/$c.txt $R/$s 25 ${SB[$c]} clonfarm11 rad"
done
tmux new-session -d -s v2_watch "bash $HOME/exclurad_py/production/watchdog.sh $R 900"
sleep 70
echo "  relaunched: tmux=$(tmux ls | wc -l) workers=$(pgrep -fc "exclurad_py.run [i]mport") load=$(uptime | sed 's/.*load[ a-z]*: //' | cut -d, -f1)"
for c in hp_pp hp_pm hm_pp hm_pm; do echo "    $c: $(grep -o 'seedbase [0-9]*' $R/${B[$c]}/run.log)"; done
