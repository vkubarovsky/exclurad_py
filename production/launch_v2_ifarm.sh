#!/bin/bash
# --- code: both checkouts
for D in $HOME/exclurad_py; do cd $D && git fetch -q /tmp/ep9.bundle "refs/heads/production:refs/remotes/bun/production" && git merge -q --ff-only refs/remotes/bun/production && echo "  $D: $(git describe --tags) dirty=$(git status --short|wc -l)"; done
# --- tables (already parked for slurm; link the same files under the layout names for v2)
T=/volatile/clas12/vpk/tables; mkdir -p $HOME/tables
declare -A B=([hp_pp]=beam_helicity_pos/target_pol_pos [hp_pm]=beam_helicity_pos/target_pol_neg [hm_pp]=beam_helicity_neg/target_pol_pos [hm_pm]=beam_helicity_neg/target_pol_neg)
for c in hp_pp hp_pm hm_pp hm_pm; do cp $T/pi0_amp2609_$c.eta.npz $HOME/tables/$(echo ${B[$c]} | tr / _)_etatable.npz; done
echo "  tables: $(ls $HOME/tables | wc -l)"
# --- interactive: 24 parts = 6 per spin, local root on /volatile
R=/volatile/clas12/vpk/prod_pi0_p_v2; mkdir -p $R
python3 /tmp/make_pol_cards_generic.py pi0.amp2609 $R/cards 5000 50000000 >/dev/null
declare -A SB=([hp_pp]=50000000 [hp_pm]=53000000 [hm_pp]=56000000 [hm_pm]=59000000)
for c in hp_pp hp_pm hm_pp hm_pm; do
  s=${B[$c]}; tab=$HOME/tables/$(echo $s | tr / _)_etatable.npz
  tmux new-session -d -s v2_$c "ETA_TABLE=$tab bash $HOME/exclurad_py/production/run_prod_v2.sh $HOME/exclurad_py $R/cards/$c.txt $R/$s 6 ${SB[$c]} ifarm2402 rad"
done
tmux new-session -d -s v2_watch "bash $HOME/exclurad_py/production/watchdog.sh $R 900"
# --- slurm: keeper2, flat, 25 parts x 5000 events per spin, 12 h limit
K=/volatile/clas12/vpk/pol_pi0_p_slurm_keep2; mkdir -p $K/state
sed -e 's|pol_pi0_p_slurm_keep$|pol_pi0_p_slurm_keep2|' -e 's|^NPART=30; NEV_PART=25000; TLIM=24:00:00|NPART=25; NEV_PART=5000; TLIM=12:00:00|' \
    -e 's|20000000 + r\*1000000|90000000 + r*1000000|' /volatile/clas12/vpk/pol_pi0_p_slurm_keep/slurm_keeper.sh > $K/slurm_keeper.sh
cp /tmp/make_pol_cards_generic.py $K/
grep -nE "^ROOT=|^NPART=|90000000" $K/slurm_keeper.sh | sed 's/^/  keeper2: /'
bash -n $K/slurm_keeper.sh && tmux new-session -d -s slurm_keeper2 "bash $K/slurm_keeper.sh"
sleep 90
echo "  interactive workers: $(pgrep -fc 'exclurad_py.run [i]mport')   tmux: $(tmux ls | wc -l)"
echo "  slurm: R=$(squeue -u vpk -h -t R -r | wc -l) PD=$(squeue -u vpk -h -t PD -r | wc -l)   keeper log: $(tail -2 $K/keeper.log | tr '\n' '|' | cut -c1-200)"
