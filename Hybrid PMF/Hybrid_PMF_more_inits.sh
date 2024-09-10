#!/bin/bash

#PBS -l walltime=144:00:00
#PBS -l select=1:ncpus=4:mem=15GB
#PBS -m abe
#PBS -M 22796002@sun.ac.za
#PBS -q seriallong
#PBS -P CSCI1370
#PBS -o Hybrid_PMF.out
#PBS -e Hybrid_PMF.err
#PBS -N Hybrid_PMF

# make sure I'm the only one that can read my output
umask 0077

source /apps/chpc/chem/anaconda3-2021.11/bin/activate /home/ghermanus/cmdstan_condaforge

cd /home/ghermanus/lustre/Hybrid\ PMF

python3 Hybrid_PMF_more_inits.py ${include_clusters} ${variance_known} "${func_groups_string}" $chain_id

conda deactivate

# usage for PBS: qsub -N Hybrid_PMF -e Hybrid_PMF.err -o Hybrid_PMF.out -v include_clusters=0,variance_known=1,func_groups_string="Alkane.Primary alcohol",chain_id=0 Hybrid_PMF.sh