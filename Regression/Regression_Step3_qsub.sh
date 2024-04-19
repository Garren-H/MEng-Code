#!/bin/bash

# Check if the number of batches is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <N_batches>"
    exit 1
fi

# Number of batches to process
N_batches="$1"

cd # navigate home
cd Regression # navigate to Regression

./get_file_to_move_Step3.sh # update files first

# Read file paths from Files_to_run.txt into an array
file_paths=($(<Files_to_run_Step3.txt))

# Loop through the specified number of batches
for ((i = 0; i < N_batches && i < ${#file_paths[@]}; i++)); do
    path="${file_paths[i]}"
    # Check if inits.json file exists in Results/${path}/Step3 directory
    if [ -f "Results/${path}/Step3/inits.json" ]; then
        # Submit job only if inits.json file exists
        # specify the output and error directories to the new dir and run script
        qsub -N "RegressionStep3_${path}" -o "Results/${path}/RegressionStep3.out" -e "Results/${path}/RegressionStep3.err" -v input_path="${path}" Regression_Step3.sh
    else
        echo "Warning: inits.json file not found for ${path}. Skipping job submission."
    fi
done

./get_file_to_move_Step3.sh # update files again

