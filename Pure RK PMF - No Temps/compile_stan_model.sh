#!/bin/bash

#PBS -l select=1:ncpus=2:mem=5GB
#PBS -l walltime=01:00:00
#PBS -N compile_stan_models
#PBS -o compile_stan_models.out
#PBS -e compile_stan_models.err

cd Pure\ RK\ PMF\ -\ No\ Temps

module load app/stan/2.34

python3 compile_stan_model.py
