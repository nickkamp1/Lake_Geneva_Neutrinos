#!/bin/bash

# Define an array of strings
strings=("LHCb_South" "LHCb_North" "CMS_West" "CMS_East" "ATLAS_West" "ATLAS_East")
SIREN_folder="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/LIV2/sources/SIREN/resources/Detectors/"

# Loop through each string in the array
for str in "${strings[@]}"
do
        mkdir ${SIREN_folder}/densities/SINE_$str
        mkdir ${SIREN_folder}/materials/SINE_$str
        scp SINE_${str}.dat ${SIREN_folder}/densities/SINE_${str}/SINE_${str}-v1.dat 
        scp ${SIREN_folder}/materials/GenevaSurface/GenevaSurface-v1.dat ${SIREN_folder}/materials/SINE_${str}/SINE_${str}-v1.dat
done
