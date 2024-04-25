#!/bin/bash

#PBS -l walltime=168:00:00
#PBS -l select=1:ncpus=32:mem=5GB
#PBS -m abe
#PBS -o Hybrid_PMF.out
#PBS -e Hybrid_PMF.err

# make sure I'm the only one that can read my output
umask 0077

cd Hybrid\ PMF

module load app/stan/2.34

python3 Hybrid_PMF.py ${include_clusters} ${variance_known} 

