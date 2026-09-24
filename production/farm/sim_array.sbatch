#!/bin/bash
# Stage B: GEMC, background merge, reconstruction, ntuple and DST, for one chunk.
#
#   sbatch --array=1-200 --job-name=sim_born --time=05:00:00 \
#          --dependency=aftercorr:<gen jobid> sim_array.sbatch born
#   sbatch --array=1-200 --job-name=sim_rad  --time=05:00:00 sim_array.sbatch rad
#
# aftercorr ties task i of this array to task i of the generation array, so
# each simulation starts as soon as its own chunk exists.  afterok on the whole
# array would make every task wait for the slowest one -- that pattern cost
# 116 jobs ten idle hours on 2026-09-20.
#
# Configuration: rga_fall2018, inbending (torus -1.00), beam 10604 MeV,
# gemc 5.13 + coatjava 10.0.7 -- the pass-2 pair from clas12-config setup.json,
# i.e. the one the aao_gen pass2 sample on /cache was made with.  The coatjava
# version must match the data pass being analysed; the gemc minor version
# matters less.
#
# Background: 50 nA, merged over the eleven detectors the official production
# merges.  bg-merger's own default is DC,FTOF only -- which would leave the
# calorimeter clean, i.e. no background under the pi0 photons, exactly where
# this sample needs it.
set -u
MODE=${1:?born or rad}
R=/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_20260924
I=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
W=$R/$MODE/work/w_$I
BGDIR=/cache/clas12/rg-a/production/background/rga_fall2018/tor-1.00_sol-1.00/50nA_10604MeV/10k
# 100 background files for 200 tasks: each is used by two tasks, whose
# generated events are independent, so no correlation is introduced
BG=$(printf "%s/%05d.hipo" "$BGDIR" $(( (SLURM_ARRAY_TASK_ID - 1) % 100 + 1 )))
DET='DC,FTOF,ECAL,HTCC,LTCC,BST,BMT,CND,CTOF,FTCAL,FTHODO'
CFG=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod
GC=$CFG/gemc/5.13/rga_fall2018.gcard
YAML=$CFG/coatjava/10.0.7/rga_fall2018.yaml
OUTTAG=excluradpy_pi0_${MODE}_fa18_inb_bg50nA_$I

export TMPDIR=/tmp                       # before any module: the per-job TMPDIR is not writable
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/geant4/modules
module load gemc/5.13
module load sqlite/5.13
export RCDB_CONNECTION=mysql://null
module load coatjava/10.0.7
which gemc recon-util bg-merger

rm -rf "$W"; mkdir -p "$W"; cd "$W"
cp "$R/$MODE/chunks/lund_$I.dat" lund.dat || { echo "FATAL: no chunk lund_$I.dat"; exit 1; }
[ -f "$BG" ] || { echo "FATAL: no background file $BG"; exit 1; }
echo "chunk $(grep -c '^ *[0-9]* *1 *1 ' lund.dat) events, background $(basename "$BG")"

T0=$(date +%s)
gemc -USE_GUI=0 -NGENP=100 -N=100000 -INPUT_GEN_FILE="lund, lund.dat" "$GC" \
     -SCALE_FIELD="binary_torus, -1.00" -SCALE_FIELD="binary_solenoid, -1.00" \
     -OUTPUT="hipo, gemc.hipo" -INTEGRATEDRAW="*" > gemc.log 2>&1
T1=$(date +%s); echo "GEMC seconds: $((T1-T0))  size: $(du -h gemc.hipo 2>/dev/null | cut -f1)"
# a gemc.hipo of a few kB means gemc read nothing -- almost always a '#' in the LUND
SZ=$(stat -c %s gemc.hipo 2>/dev/null || echo 0)
[ "$SZ" -lt 1000000 ] && { echo "FATAL: gemc.hipo is $SZ bytes, gemc read no events"; tail -20 gemc.log; exit 1; }

rm -f merged.hipo
bg-merger -b "$BG" -i gemc.hipo -o merged.hipo -d "$DET" > bg.log 2>&1
T2=$(date +%s); echo "BGMERGE seconds: $((T2-T1))  size: $(du -h merged.hipo 2>/dev/null | cut -f1)"
[ -s merged.hipo ] || { echo "FATAL: bg-merger produced nothing"; tail -20 bg.log; exit 1; }

rm -f recon.hipo
recon-util -y "$YAML" -i merged.hipo -o recon.hipo > recon.log 2>&1
T3=$(date +%s); echo "RECON seconds: $((T3-T2))  size: $(du -h recon.hipo 2>/dev/null | cut -f1)"
[ -s recon.hipo ] || { echo "FATAL: recon-util produced nothing"; tail -20 recon.log; exit 1; }

module purge                              # java 21 from coatjava breaks run-groovy
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy "$R/hipo_to_ntuple.groovy" \
    recon.hipo 1000000 "$R/$MODE/out/$OUTTAG.f32" 2>&1 | grep -E "DONE|FATAL|Exception" || true
/group/clas12/packages/coatjava/9.0.2/bin/hipo-utils -filter \
    -b "RUN::*,REC::*,RECFT::*,MC::*" -merge \
    -o "$R/$MODE/dst/$OUTTAG.hipo" recon.hipo > /dev/null 2>&1
T4=$(date +%s); echo "NTUPLE+DST seconds: $((T4-T3))"
echo "TOTAL seconds: $((T4-T0))"
ls -la "$R/$MODE/dst/$OUTTAG.hipo" "$R/$MODE/out/$OUTTAG.f32" 2>/dev/null
cd "$R"; rm -rf "$W"
echo "DONE $MODE $I"
