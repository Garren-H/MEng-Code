#!/bin/bash

# The file updates the files to run by checking if files are completed or currently running
# The files to run are stored in a .txt file which is used by the Regression_qsub.sh script

cd # navigate to home directory
cd Regression # Navigate to subdirectory Regression
cd Data # Navigate to subdirectory Data

file_paths=($(basename -s .json ./*.json)) # extract all names excluding extension

cd .. # navigate back a level to home/user/Regression

# Remove current .txt files
rm Completed_Files.txt
rm Files_to_run.txt

./get_running_jobs.sh #update running files
file_running_paths=($(<Files_running.txt)) #get the running files

for path in "${file_paths[@]}"; do
    if [ -d "Results/${path}/Step2" ]; then # If directory exists then write path to completed files
        echo "${path}" >> Completed_Files.txt
    elif [[ ! " ${file_running_paths[@]} " =~ " ${path} " ]]; then # Else write files file to run
        echo "${path}" >> Files_to_run.txt
    fi
done
