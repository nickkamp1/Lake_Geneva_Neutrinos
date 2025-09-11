#!/bin/bash
#SBATCH --job-name=hnl_up_sim
#SBATCH -p serial_requeue,shared,sapphire,arguelles_delgado # Partition to submit to
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/output/hnl_array_%A_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/error/hnl_array_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --mem=20000
#SBATCH -c 16 # Number of cores

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

# PARAMS_FILE is passed via --export by the submitter script
if [[ -z "$PARAMS_FILE" ]]; then
  echo "PARAMS_FILE not set"
  exit 1
fi

# read the corresponding line (SLURM_ARRAY_TASK_ID starts at 0)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$PARAMS_FILE")
if [[ -z "$LINE" ]]; then
  echo "No line for task id $SLURM_ARRAY_TASK_ID"
  exit 1
fi

IFS=, read -r PRIMARY GEN MESON N OUTPUT EXPERIMENT XSMode m4 Um4 seed <<< "$LINE"

# export same variables your old submissions used
export CASE=hnl
export PRIMARY
export GEN
export MESON
export N
export OUTPUT
export EXPERIMENT
export XSMode
export m4
export Um4
export seed

# set default values for Ue4 and Ut4 if not provided (change defaults as needed)
Ue4=${Ue4:-0}
Um4=${Um4:-0}
Ut4=${Ut4:-0}
export Ue4 Ut4

echo "Running task $SLURM_ARRAY_TASK_ID with:"
echo "PRIMARY=$PRIMARY GEN=$GEN MESON=$MESON N=$N OUTPUT=$OUTPUT m4=$m4 Ue4=$Ue4 Um4=$Um4 Ut4=$Ut4 seed=$seed"

CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/SIREN_Simulation.py --case ${CASE} --primary ${PRIMARY} -p LHC13 -g ${GEN} -m ${MESON} -n ${N} -o ${OUTPUT} -e ${EXPERIMENT} -x ${XSMode} -m4 ${m4} -ue ${Ue4} -um ${Um4} -ut ${Ut4} --seed ${seed}"
echo $CMD
$CMD
