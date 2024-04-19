#!/bin/bash

# This script is used for the 'automated' subbing of a number
# of Regression.sh scripts with different datafiles to run
# co-currently on a PBS server as apposed to using mpi-processes
# or for loops.
# The advantage of this approach is that we can handle the load 
# on the cluster better if a shared cluster is used.
# This script calls a bunch of other scripts is updates the running files
# and completed files in the PBS cluster and stores the files ro run in a
# .txt file. 
# These indices of the files to run is then used as input to queue the number
# of scripts to run concurrently <N_bathces>

# Check if the number of batches is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <N_batches>"
    exit 1
fi

# Number of batches to process
N_batches="$1"

cd # navigate home
cd Regression # navigate to Regression

./get_file_to_move.sh # update files first

# Read file paths from Files_to_run.txt into an array
file_paths=($(<Files_to_run.txt))

# Loop through the specified number of batches
for ((i = 0; i < N_batches && i < ${#file_paths[@]}; i++)); do
    path="${file_paths[i]}"
    mkdir -p "Results/${path}" # make subfolder in Results
    # specify the output and error directories to the new dir and run script
    qsub -N "Regression_${path}" -o "Results/${path}/Regression.out" -e "Results/${path}/Regression.err" -v input_path="${path}" Regression.sh
done

./get_file_to_move.sh # update files again

