#!/bin/bash

#PBS -l walltime=168:00:00
#PBS -l select=1:ncpus=16:mem=20GB
#PBS -m abe
#PBS -o Pure_PMF.out
#PBS -e Pure_PMF.err
#PBS -N Pure_PMF

# make sure I'm the only one that can read my output
umask 0077

cd Pure\ RK\ PMF

module load app/stan/2.34

python3 Pure_PMF.py ${include_clusters} ${variance_known} ${variance_MC_known}

# usage for PBS: qsub -N Pure_PMF_AP -e Pure_PMF_AP.err -o Pure_PMF_AP.out -v include_clusters=0,variance_known=1,variance_MC_known=1 Pure_PMF.sh