#!/bin/bash
#SBATCH --job-name=pi0gemc
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=05:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-80
#SBATCH --output=/volatile/clas12/users/USER/gemc_prod/logs/job_%a.log
export TMPDIR=/tmp
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/geant4/modules
module load gemc/5.11
module load sqlite/5.11
export RCDB_CONNECTION=mysql://null
module load coatjava/11.1.0
module switch hipo/4.2.0
I=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
W=/volatile/clas12/users/USER/gemc_prod/work_$I; mkdir -p $W; cd $W
cp /volatile/clas12/users/USER/gemc_prod/chunks/lund_$I.dat lund.dat
GC=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod/gemc/5.11/rgk_fall2018_FTOn.gcard
YAML=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod/coatjava/11.1.0/rgk_fall2018_FTOn.yaml
T0=$(date +%s)
gemc -USE_GUI=0 -NGENP=100 -N=2500 -INPUT_GEN_FILE="lund, lund.dat" $GC -SCALE_FIELD="binary_torus, +1.00" -SCALE_FIELD="binary_solenoid, -1.00" -OUTPUT="hipo, gemc.hipo" -INTEGRATEDRAW="*" 2>&1 > gemc.log 2>&1; grep "Total gemc time" gemc.log
T1=$(date +%s); echo "GEMC seconds: $((T1-T0))"
rm -f recon.hipo; recon-util -y $YAML -i gemc.hipo -o recon.hipo 2>&1 | tail -2
T2=$(date +%s); echo "RECON seconds: $((T2-T1))"
module purge
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy /volatile/clas12/users/USER/gemc_prod/hipo_to_ntuple.groovy recon.hipo 10000 /volatile/clas12/users/USER/gemc_prod/out/pi0gemc_$I.f32 2>&1 | grep "DONE\|FATAL\|Exception"
/group/clas12/packages/coatjava/9.0.2/bin/hipo-utils -filter -b "RUN::*,REC::*,RECFT::*,MC::*" -merge -o /volatile/clas12/users/USER/gemc_prod/dst/pi0gemc_$I.hipo recon.hipo >/dev/null 2>&1
rm -f gemc.hipo recon.hipo lund.dat
