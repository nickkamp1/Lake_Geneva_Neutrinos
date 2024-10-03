#!/bin/bash

# Define an array of strings
SINE_strings=("LHCb_South" "LHCb_North" "CMS_West" "CMS_East" "ATLAS_West" "ATLAS_East")
UNDINE_strings=("LHCb_North" "CMS_East")
SIREN_folder="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/LIV2/sources/SIREN/resources/Detectors/"

# Loop through each string in the array
# First for SINE
for str in "${SINE_strings[@]}"
do
        mkdir ${SIREN_folder}/densities/SINE_$str
        mkdir ${SIREN_folder}/materials/SINE_$str
        scp SINE_${str}.dat ${SIREN_folder}/densities/SINE_${str}/SINE_${str}-v1.dat
        scp ${SIREN_folder}/materials/GenevaSurface/GenevaSurface-v1.dat ${SIREN_folder}/materials/SINE_${str}/SINE_${str}-v1.dat
done

# Now for UNDINE
for str in "${UNDINE_strings[@]}"
do
        mkdir ${SIREN_folder}/densities/UNDINE_$str
        mkdir ${SIREN_folder}/materials/UNDINE_$str
        scp UNDINE_${str}.dat ${SIREN_folder}/densities/UNDINE_${str}/UNDINE_${str}-v1.dat
        scp ${SIREN_folder}/materials/GenevaLake/GenevaLake-v1.dat ${SIREN_folder}/materials/UNDINE_${str}/UNDINE_${str}-v1.dat
done