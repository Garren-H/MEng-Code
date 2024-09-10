#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=2:mem=10GB
#PBS -m abe
#PBS -o get_json_data.out
#PBS -e get_json_data.err
#PBS -N get_json_data

# make sure I'm the only one that can read my output
umask 0077

cd Pure\ RK\ PMF

module load app/stan/2.34

python3 get_json_data.py "${func_groups_string}"

# usage for PBS: qsub -v func_groups_string='"Alkane,Primary alcohol"' get_json_data.sh