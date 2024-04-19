#!/bin/bash

# Select wall time
#PBS -l walltime=420:00:00

# Mail when aborted, begun, executed
#PBS -m abe

# Select number of cores and memory
#PBS -l select=1:ncpus=9:mem=25GB

# Name of program
#PBS -N PMF_RK_MAP

# Error and output files
#PBS -o	RK_MAP.out
#PBS -e	RK_MAP.err

# make sure I'm the only one that can read my output
umask 0077

# Make temp folder
TMP=/scratch-large-network/${PBS_JOBID}
mkdir -p ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/Stan Models to ${TMP}/"
cp -r "${PBS_O_WORKDIR}/Stan Models" ${TMP}/
echo "Copying from .py files from ${PBS_O_WORKDIR} to ${TMP}/"
cp "${PBS_O_WORKDIR}/Pure_RK_PMF_MAP.py" ${TMP}/

cd ${TMP} #navigate to temp file

# load stan library
module load app/stan/2.34

# Run python script
python3 Pure_RK_PMF_MAP.py

# Copy files back to workdir
echo "Copying from ${TMP}/Subsets to ${PBS_O_WORKDIR}/Subsets"
cp -r "${TMP}/Subsets/"* "${PBS_O_WORKDIR}/Subsets/"

# delete my temporary files
[ $? -eq 0 ] && /bin/rm -rf ${TMP}
