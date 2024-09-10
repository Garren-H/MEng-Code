#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=2:mem=10GB
#PBS -m abe
#PBS -q serial
#PBS -P CSCI1370
#PBS -N compile_stan_models
#PBS -o compile_stan_models.out
#PBS -e compile_stan_models.err

# make sure I'm the only one that can read my output
umask 0077

source /apps/chpc/chem/anaconda3-2021.11/bin/activate /home/ghermanus/cmdstan_condaforge

cd /home/ghermanus/lustre/Hybrid\ PMF\ Adj

python3 compile_stan_models.py

source /apps/chpc/chem/anaconda3-2021.11/bin/deactivate
conda deactivate
