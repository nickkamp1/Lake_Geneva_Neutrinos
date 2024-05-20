#!/bin/bash
#SBATCH -J geneva_simulation
#SBATCH -c 4 # Number of cores
#SBATCH -p shared
#SBATCH --mem 30000
#SBATCH -t 0-00:30:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/logging/output/geneva_sim_%A_%a.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/logging/error/geneva_sim_%A_%a.err # Standard error

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

if [ -z ${GEN+x} ]; then
    exit 1
fi

if [ -z ${MESON+x} ]; then
    exit 1
fi

if [ -z ${DetModel+x} ]; then
    DetModel="lake"
fi

if [ -z ${XSModel+x} ]; then
    XSModel="CC"
fi

if [ -z ${Nstart+x} ]; then
    Nstart=0
fi

if [ -z ${N+x} ]; then
    N=50000
fi

CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/MuonSimulation.py -t LHC13 -g ${GEN} -m ${MESON} -ns ${Nstart} -n ${N} -dm ${DetModel} -xm ${XSModel}"
echo $CMD
$CMD
