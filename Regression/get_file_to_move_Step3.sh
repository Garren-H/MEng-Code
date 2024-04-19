#!/bin/bash

# Similar to the get_file_to_move.sh script but slightly different in the way it checks for completed files

cd 
cd Regression
cd Data

file_paths=($(basename -s .json ./*.json)) # extract all names excluding extension

cd ..

rm Completed_Files_Step3.txt
rm Files_to_run_Step3.txt

./get_running_jobs_Step3.sh #update running files
file_running_paths=($(<Files_running_Step3.txt)) #get the running files

for path in "${file_paths[@]}"; do
    if find "Results/${path}/Step3" -maxdepth 1 -type f -name "*.csv" 2>/dev/null | grep -q '.'; then # If directory has .csv files then write path to completed files
        echo "${path}" >> Completed_Files_Step3.txt
    elif [[ ! " ${file_running_paths[@]} " =~ " ${path} " ]]; then # Else write files file to run
        echo "${path}" >> Files_to_run_Step3.txt
    fi
done
