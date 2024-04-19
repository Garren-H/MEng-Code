#!/bin/bash

# This file is used to obatin the currently running files on the server, based on
# the naming convention of the jobs as Regression_<data_file>, where data_file is an integer
# It obtaines the job ids for all the jobs currently running by a user and checks the name of the job
# is 'Regression_<data_file>'. If it is, the <data_file> part is extracted as saved to Files_running.txt 

cd
cd Regression

rm Files_running.txt

qstat_output=$(qstat -u 22796002) # Replace 22796002 with the user name on the cluster

# Extract job names from qstat output
job_names=$(echo "$qstat_output" | awk 'NR > 5 {print $1}')
readarray -t job_names_array <<< "$job_names"
job_ids=($(basename -a -s ".hpc1.hpc" ${job_names_array[@]})) # extracts job id based on the currently server where the job is listed as <job_ID>.hpc1.hpc

for job in "${job_ids[@]}"; do
    names=$(echo $(qstat -fx "${job}") | cut -d ' ' -f6) # Extract the job name from the full by using the qstat prompt (all information is diaplyed about a job, but only the name is extracted)
    job_type=$(echo "${names}" | cut -d '_' -f1) # This extract the first part of the job name before '_' is encountered
    if [ "$job_type" = "Regression" ]; then # If the job name begins with 'Regression'
        real_number=$(echo "${names}" | cut -d '_' -f2) # Extract the second argument <data_file> from the full job name
        echo "${real_number}" >> Files_running.txt # Saves to running files
    fi
done
