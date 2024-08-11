#!/bin/bash
#SBATCH -J geneva_simulation
#SBATCH -c 8 # Number of cores
#SBATCH -p shared
#SBATCH --mem 60000
#SBATCH -t 0-02:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/output/geneva_sim_%A_%a.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/error/geneva_sim_%A_%a.err # Standard error

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

if [ -z ${PRIMARY+x} ]; then
    PRIMARY=14
fi

if [ -z ${GEN+x} ]; then
    exit 1
fi

if [ -z ${MESON+x} ]; then
    exit 1
fi

if [ -z ${XSModel+x} ]; then
    XSModel="CC"
fi

if [ -z ${N+x} ]; then
    N=1000000
fi

if [ -z ${OUTPUT+x} ]; then
    OUTPUT="out"
fi

if [ -z ${EXPERIMENT+x} ]; then
    EXPERIMENT="GenevaSurface"
fi

CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/SIREN_Simulation.py --case muon --primary ${PRIMARY} -p LHC13 -g ${GEN} -m ${MESON} -n ${N} -o ${OUTPUT} -e ${EXPERIMENT}"
echo $CMD
$CMD
