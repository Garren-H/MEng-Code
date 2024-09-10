#!/bin/bash

#PBS -l walltime=336:00:00
#PBS -l select=1:ncpus=1:mem=3GB
#PBS -m abe
#PBS -o Pure_PMF.out
#PBS -e Pure_PMF.err
#PBS -N Pure_PMF

# make sure I'm the only one that can read my output
umask 0077

cd Pure\ RK\ PMF\ -\ No\ Temps

module load app/stan/2.34

python3 Pure_PMF_no_temps.py $include_clusters $add_zeros $ARD "${func_groups_string}" $chain_id

# usage for PBS: qsub -N Pure_PMF_AP -e Pure_PMF_AP.err -o Pure_PMF_AP.out -v include_clusters=0,add_zeros=0,ARD=0,func_groups_string='"Alkane,Primary alcohol"',chain_id=0 Pure_PMF_no_temps.sh