#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=2:mem=10GB
#PBS -m abe
#PBS -o compile_stan_models.out
#PBS -e compile_stan_models.err
#PBS -N compile_stan_models

# make sure I'm the only one that can read my output
umask 0077

cd Pure\ RK\ PMF

module load app/stan/2.34

python3 compile_stan_models.py

# usage for PBS: qsub Pure_PMF.sh