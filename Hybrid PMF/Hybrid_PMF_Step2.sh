#!/bin/bash

#PBS -l walltime=96:00:00
#PBS -l select=1:ncpus=24:mem=15GB
#PBS -m abe
#PBS -q smp
#PBS -P CSCI1370
#PBS -o Hybrid_PMF.out
#PBS -e Hybrid_PMF.err
#PBS -N Hybrid_PMF

# make sure I'm the only one that can read my output
umask 0077

source /apps/chpc/chem/anaconda3-2021.11/bin/activate /home/ghermanus/cmdstan_condaforge

cd /home/ghermanus/lustre/Hybrid\ PMF

python3 Hybrid_PMF_Step2.py ${include_clusters} ${variance_known}

source /apps/chpc/chem/anaconda3-2021.11/bin/deactivate
conda deactivate

# usage for PBS: qsub -N S_AP -e S_AP.err -o S_AP.out -v include_clusters=0,variance_known=1 Hybrid_PMF_Step2.sh