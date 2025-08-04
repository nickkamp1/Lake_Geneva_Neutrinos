#!/bin/bash
#SBATCH -J geneva_simulation
#SBATCH -c 16 # Number of cores
#SBATCH -p shared,sapphire,arguelles_delgado # Partition to submit to
#SBATCH --mem 100000
#SBATCH -t 0-00:20:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/output/geneva_sim_%A_%a.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/error/geneva_sim_%A_%a.err # Standard error

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

if [ -z ${CASE+x} ]; then
    CASE="neutrino"
fi

if [ -z ${PRIMARY+x} ]; then
    PRIMARY=14
fi

if [ -z ${GEN+x} ]; then
    exit 1
fi

if [ -z ${MESON+x} ]; then
    exit 1
fi

if [ -z ${XSMode+x} ]; then
    XSMode="CC"
fi

if [ -z ${N+x} ]; then
    N=1000000
fi

if [ -z ${OUTPUT+x} ]; then
    OUTPUT="out"
fi

if [ -z ${EXPERIMENT+x} ]; then
    EXPERIMENT="SINE_CMS_West"
fi

if [ -z ${M4+x} ]; then
    M4="1000"
fi

if [ -z ${Ue4+x} ]; then
    Ue4="0"
fi

if [ -z ${Um4+x} ]; then
    Um4="0"
fi

if [ -z ${Ut4+x} ]; then
    Ut4="0"
fi

CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/SIREN_Simulation.py --case ${CASE} --primary ${PRIMARY} -p LHC13 -g ${GEN} -m ${MESON} -n ${N} -o ${OUTPUT} -e ${EXPERIMENT} -x ${XSMode} -m4 ${M4} -ue ${Ue4} -um ${Um4} -ut ${Ut4}"
echo $CMD
$CMD
