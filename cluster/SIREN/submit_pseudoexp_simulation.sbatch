#!/bin/bash
#SBATCH -J geneva_simulation
#SBATCH -c 16 # Number of cores
#SBATCH -p shared
#SBATCH --mem 60000
#SBATCH -t 0-00:30:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/output/geneva_pseudoexp_sim_%A_%a.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/error/geneva_pseudoexp_sim_%A_%a.err # Standard error

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

if [ -z ${LGEN+x} ]; then
    exit 1
fi

if [ -z ${CGEN+x} ]; then
    exit 1
fi

if [ -z ${N+x} ]; then
    N=10000
fi

CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/SIREN_PseudoExperiments.py -lg ${LGEN} -cg ${CGEN} -n ${N}"
echo $CMD
$CMD
