#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=2:mem=10GB
#PBS -m abe
#PBS -q serial
#PBS -P CSCI1370
#PBS -N get_json_data
#PBS -o get_json_data.out
#PBS -e get_json_data.err

# make sure I'm the only one that can read my output
umask 0077

source /apps/chpc/chem/anaconda3-2021.11/bin/activate /home/ghermanus/cmdstan_condaforge

cd /home/ghermanus/lustre/Hybrid\ PMF

python3 get_json_data.py "${func_groups_string}"

source /apps/chpc/chem/anaconda3-2021.11/bin/deactivate
conda deactivate

#Usage: qsub -v func_groups_string="Alkane.Primary alcohol" get_json_data.sh
