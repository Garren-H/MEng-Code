#!/bin/bash

# Similar as the Regression_qsub.sh script, but for
# running the RegressionStep3.py script instead of 
# the Regression.py script

# Select wall time
#PBS -l walltime=24:00:00

# Mail when aborted, begun, executed
#PBS -m abe

# Select number of cores and memory
#PBS -l select=1:ncpus=8:mem=4GB

# make sure I'm the only one that can read my output
umask 0077

cd

cd Regression

# load stan library
module load app/stan/2.34

# Run python script with input
python3 Regression_Step3.py ${input_path}

