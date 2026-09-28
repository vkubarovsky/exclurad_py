#!/bin/bash
# Stage B: GEMC, background merge, reconstruction, ntuple and DST, for one chunk.
#
#   sbatch --array=1-200 --job-name=sim_rad --time=07:00:00 sim_array.sbatch rad
#
# Configuration: rga_fall2018, inbending (torus -1.00), beam 10604 MeV,
# gemc 5.13 + coatjava 10.0.7 -- the pass-2 pair from clas12-config setup.json.
# Background: 50 nA over the eleven detectors the official production merges;
# bg-merger's own default is DC,FTOF only, which would leave the calorimeter
# clean -- no background under the pi0 photons, exactly where it is needed.
#
# THREE FIXES after the 2026-09-25 born array (104 of 200 jobs lost):
#
# 1. sqlite-jdbc could not load its native library on the compute nodes:
#        No native library found for os.name=Linux, os.arch=x86_64
#    The identical command with the identical modules works on ifarm2402, so
#    it is the node.  The driver extracts its .so into java.io.tmpdir (/tmp)
#    and dlopens it; on the node that fails -- /tmp is either noexec or full
#    after GEMC has spent three hours there.  Both cures are the same: give
#    the JVM a temp directory on /volatile.  bg-merger and recon-util call
#    java directly with fixed flags, so JDK_JAVA_OPTIONS (Java 9+) is the
#    only way in.
#
# 2. The merged.hipo guard was `[ -s merged.hipo ]`.  A header-only hipo is
#    336 kB and passes it, so the job fed garbage to recon-util and reported
#    a reconstruction failure for what was a merge failure.  Both guards now
#    demand tens of MB, the way the gemc.hipo one already did.
#
# 3. gemc.hipo is kept until the DST exists.  Three hours of GEMC must never
#    again be thrown away by a failure in a step that takes minutes; on a
#    failure the work directory survives and post_array.sbatch resumes from it.
set -u
MODE=${1:?born or rad}
# ONE production directory for all three stages.  Until 2026-09-28 this
# script said 20260924 while sim_array.sbatch said t2_20260925: stage A
# wrote where stage B did not look.  Set PROD_DIR once and export it.
R=${PROD_DIR:-/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_t2_20260925}
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

rm -rf "$W"; mkdir -p "$W"; cd "$W"
export TMPDIR=$W                         # not /tmp: see fix 1
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/geant4/modules
module load gemc/5.13
module load sqlite/5.13
export RCDB_CONNECTION=mysql://null
module load coatjava/10.0.7
export JDK_JAVA_OPTIONS="-Djava.io.tmpdir=$W -Dorg.sqlite.tmpdir=$W"
which gemc recon-util bg-merger

cp "$R/$MODE/chunks/lund_$I.dat" lund.dat || { echo "FATAL: no chunk lund_$I.dat"; exit 1; }
[ -f "$BG" ] || { echo "FATAL: no background file $BG"; exit 1; }
echo "chunk $(grep -c '^ *[0-9]* *1 *1 ' lund.dat) events, background $(basename "$BG")"

T0=$(date +%s)
gemc -USE_GUI=0 -NGENP=100 -N=100000 -INPUT_GEN_FILE="lund, lund.dat" "$GC" \
     -SCALE_FIELD="binary_torus, -1.00" -SCALE_FIELD="binary_solenoid, -1.00" \
     -OUTPUT="hipo, gemc.hipo" -INTEGRATEDRAW="*" > gemc.log 2>&1
T1=$(date +%s); GS=$(stat -c %s gemc.hipo 2>/dev/null || echo 0)
echo "GEMC seconds: $((T1-T0))  size: $GS"
# a gemc.hipo of a few kB means gemc read nothing -- almost always a '#' in the LUND
if [ "$GS" -lt 1000000 ]; then echo "FATAL: gemc.hipo is $GS bytes, gemc read no events"; tail -20 gemc.log; exit 1; fi

rm -f merged.hipo
bg-merger -b "$BG" -i gemc.hipo -o merged.hipo -d "$DET" > bg.log 2>&1
T2=$(date +%s); MS=$(stat -c %s merged.hipo 2>/dev/null || echo 0)
echo "BGMERGE seconds: $((T2-T1))  size: $MS"
if [ "$MS" -lt 10000000 ]; then echo "FATAL: merged.hipo is $MS bytes -- keeping gemc.hipo for post_array"; tail -25 bg.log; exit 1; fi

rm -f recon.hipo
recon-util -y "$YAML" -i merged.hipo -o recon.hipo > recon.log 2>&1
T3=$(date +%s); RS=$(stat -c %s recon.hipo 2>/dev/null || echo 0)
echo "RECON seconds: $((T3-T2))  size: $RS"
if [ "$RS" -lt 10000000 ]; then echo "FATAL: recon.hipo is $RS bytes -- keeping gemc.hipo for post_array"; tail -25 recon.log; exit 1; fi

module purge                              # java 21 from coatjava breaks run-groovy
unset JDK_JAVA_OPTIONS
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy "$R/hipo_to_ntuple.groovy" \
    recon.hipo 1000000 "$R/$MODE/out/$OUTTAG.f32" 2>&1 | grep -E "DONE|FATAL|Exception" || true
/group/clas12/packages/coatjava/9.0.2/bin/hipo-utils -filter \
    -b "RUN::*,REC::*,RECFT::*,MC::*" -merge \
    -o "$R/$MODE/dst/$OUTTAG.hipo" recon.hipo > /dev/null 2>&1
T4=$(date +%s); echo "NTUPLE+DST seconds: $((T4-T3))"
echo "TOTAL seconds: $((T4-T0))"
ls -la "$R/$MODE/dst/$OUTTAG.hipo" "$R/$MODE/out/$OUTTAG.f32" 2>/dev/null
# only a finished DST makes three hours of GEMC expendable
if [ -s "$R/$MODE/dst/$OUTTAG.hipo" ]; then cd "$R"; rm -rf "$W"; fi
echo "DONE $MODE $I"
