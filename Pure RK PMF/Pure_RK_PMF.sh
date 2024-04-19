#!/bin/bash

# Select wall time
#PBS -l walltime=200:00:00

# Mail when aborted, begun, executed
#PBS -m abe

# Select number of cores and memory
#PBS -l select=1:ncpus=8:mem=10GB

# make sure I'm the only one that can read my output
umask 0077

# Make temp folder
TMP=/scratch-large-network/${PBS_JOBID}
mkdir -p ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/Stan Models to ${TMP}/"
cp -r "${PBS_O_WORKDIR}/Stan Models" ${TMP}/
echo "Copying from .py files from ${PBS_O_WORKDIR} to ${TMP}/"
cp "${PBS_O_WORKDIR}/*.py" ${TMP}/

cd ${TMP} #navigate to temp file

# load stan library
module load app/stan/2.34

# Run python script
python3 Pure_RK_PMF.py

# Copy files back to workdir
echo "Copying from ${TMP}/Subsets to ${PBS_O_WORKDIR}/Subsets"
cp -r "${TMP}/Subsets/"* "${PBS_O_WORKDIR}/Subsets/"

# delete my temporary files
[ $? -eq 0 ] && /bin/rm -rf ${TMP}

