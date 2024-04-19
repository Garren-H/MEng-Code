#!/bin/bash

# This script is used for the submission of the Regression.pyscript 
# on a PBS cluster.
# This script is called in the Regression_qsub.sh script which 
# automatically choses which datasets to run based on completed 
# and currently running files.

# The usage for this script is Regression.sh <input_path>
# Where: input_path is an integer referring to the <input_path>.json
#        file to use which has the experimental data for the current
#        dataset considered. This integrer is passed to the Regression.py
#        script which choses the correct data file.

# Select wall time
#PBS -l walltime=24:00:00

# Mail when aborted, begun, executed
#PBS -m abe

# Select number of cores and memory
#PBS -l select=1:ncpus=8:mem=4GB

# make sure I'm the only one that can read my output
umask 0077

# Make temp folder
TMP=/scratch-large-network/${PBS_JOBID}
mkdir -p ${TMP}

# copy the input files (Stan models, Data file and python scripts) to ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/Data/${input_path}.json to ${TMP}/"
cp "${PBS_O_WORKDIR}/Data/${input_path}.json" ${TMP}/
echo "Copying from ${PBS_O_WORKDIR}/Regression.py to ${TMP}/"
cp "${PBS_O_WORKDIR}/Regression.py" ${TMP}/
echo "Copying from ${PBS_O_WORKDIR}/Stan Models to ${TMP}/"
cp -r "${PBS_O_WORKDIR}/Stan Models" ${TMP}/

cd ${TMP} #navigate to temp file

echo "Listing dirs in ${TMP}"
ls "${TMP}/"* #list all files in current directory

# load stan library
module load app/stan/2.34

# Run python script with input
python3 Regression.py ${input_path}

# Copy files back to workdir
echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/Results/${input_path}/"
cp -r "${TMP}/${input_path}/"* "${PBS_O_WORKDIR}/Results/${input_path}"/

# delete my temporary files
[ $? -eq 0 ] && /bin/rm -rf ${TMP}
