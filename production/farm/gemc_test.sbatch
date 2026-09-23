#!/bin/bash
#SBATCH --job-name=rgk_test
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=01:30:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --output=/volatile/clas12/users/USER/gemc_test/test_%j.log
set -x
cd /volatile/clas12/users/USER/gemc_test
export TMPDIR=/tmp
source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/geant4/modules
module load gemc/5.11
module load sqlite/5.11
export RCDB_CONNECTION=mysql://null
module load coatjava/11.1.0
module switch hipo/4.2.0
which gemc recon-util
GC=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod/gemc/5.11/rgk_fall2018_FTOn.gcard
YAML=/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/prod/coatjava/11.1.0/rgk_fall2018_FTOn.yaml
T0=$(date +%s)
gemc -USE_GUI=0 -NGENP=100 -N=200 -INPUT_GEN_FILE="lund, lund.dat" $GC -SCALE_FIELD="binary_torus, +1.00" -SCALE_FIELD="binary_solenoid, -1.00" -OUTPUT="hipo, gemc.hipo" -INTEGRATEDRAW="*" 2>&1 > gemc.log 2>&1; grep -v "G4Exception|^ *$" gemc.log | tail -8
T1=$(date +%s); echo "GEMC seconds: $((T1-T0))"
recon-util -y $YAML -i gemc.hipo -o recon.hipo 2>&1 | tail -3
T2=$(date +%s); echo "RECON seconds: $((T2-T1))"
ls -la
module purge
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy hipo_to_ntuple.groovy recon.hipo 1000 test.f32 2>&1 | grep "DONE\|FATAL\|Exception"
