#!/bin/bash
cd 
cd Regression
cd Data

file_paths=($(basename -s .json ./*.json)) # extract all names excluding extension

cd ..

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
