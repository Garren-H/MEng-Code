#!/bin/bash

#PBS -l select=1:ncpus=2:mem=5GB
#PBS -l walltime=01:00:00
#PBS -N get_json_data
#PBS -o get_json_data.out
#PBS -e get_json_data.err

cd Pure\ RK\ PMF\ -\ No\ Temps

module load app/stan/2.34

python3 get_json_data.py "${func_groups_string}"

# usage: qsub -v func_groups_string='"Alkane,Primary alcohol"' get_json_data.sh
