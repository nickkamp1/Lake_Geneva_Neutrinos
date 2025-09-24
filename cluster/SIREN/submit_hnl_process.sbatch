#!/bin/bash
#SBATCH --job-name=hnl_process
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --output=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/output/hnl_process_array_%A_%a.out
#SBATCH --error=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/cluster/SIREN/logging/error/hnl_process_array_%A_%a.err
#SBATCH --time=00:40:00
#SBATCH --mem=10000
#SBATCH -c 16 # Number of cores

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/spack/share/spack/setup-env.sh
spack env activate lienv

outfile_dir=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/Data/SIREN/Output/SINE_CMS_West/HNLs/

# METAFILELIST is passed via --export by the submitter script
if [[ -z "${METAFILELIST:-}" ]]; then
  echo "METAFILELIST not set"
  exit 1
fi
if [[ ! -f "$METAFILELIST" ]]; then
  echo "METAFILELIST does not exist: $METAFILELIST"
  exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$METAFILELIST")
if [[ -z "$LINE" ]]; then
  echo "No line for task id $SLURM_ARRAY_TASK_ID"
  exit 1
fi

IFS=, read -r OUTPUT_TAG PARAMS_FILE <<< "$LINE"
export OUTPUT_TAG
export PARAMS_FILE


# Skip blanks or comments
[[ -z "${OUTPUT_TAG:-}" || "${OUTPUT_TAG:0:1}" == "#" ]] && continue

if [[ -z "${PARAMS_FILE:-}" ]]; then
  echo "Missing PARAMS_FILE for OUTPUT_TAG=$OUTPUT_TAG in $METAFILELIST"
  return
fi
if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "PARAMS_FILE not found: $PARAMS_FILE (OUTPUT_TAG=$OUTPUT_TAG)"
  return
fi

echo "Processing META entry: OUTPUT_TAG=$OUTPUT_TAG PARAMS_FILE=$PARAMS_FILE"

# Loop over each line in PARAMS_FILE: m4 Um4 <files...>
while IFS= read -r LINE || [[ -n "$LINE" ]]; do
  # Skip blanks or comments
  [[ -z "${LINE//[$'\t\r\n ']}" || "${LINE:0:1}" == "#" ]] && continue

  read -r -a TOKENS <<< "$LINE"
  if (( ${#TOKENS[@]} < 4 )); then
    echo "Malformed line in $PARAMS_FILE: '$LINE'"
    continue
  fi
  m4="${TOKENS[0]}"
  Um4="${TOKENS[1]}"
  OUTPUT_SUBTAG="${TOKENS[2]}"
  WEIGHT_MOD="${TOKENS[3]}"
  FILELIST=("${TOKENS[@]:4}")

  echo "  m4=$m4 Um4=$Um4 (files=${#FILELIST[@]})"
  # Optional: sanity-check files
  for f in "${FILELIST[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "    WARNING: missing file: $f"
    fi
  done

  output_prefix="${outfile_dir}/summary_files/${OUTPUT_TAG}_${OUTPUT_SUBTAG}_m4_${m4}_Um4_${Um4}"

  CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/Geneva/Lake_Geneva_Neutrinos/SIREN_ProcessHNLFile.py \
    --filelist ${FILELIST[@]} -m $m4 -U $Um4 --output-prefix $output_prefix --weight-mod $WEIGHT_MOD"
  echo "  $CMD"
  $CMD
done < "$PARAMS_FILE"
