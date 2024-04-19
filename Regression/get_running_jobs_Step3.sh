#!/bin/bash
cd
cd Regression

rm Files_running_Step3.txt

qstat_output=$(qstat -u 22796002)

# Extract job names from qstat output
job_names=$(echo "$qstat_output" | awk 'NR > 5 {print $1}')
readarray -t job_names_array <<< "$job_names"
job_ids=($(basename -a -s ".hpc1.hpc" ${job_names_array[@]}))

for job in "${job_ids[@]}"; do
    names=$(echo $(qstat -fx "${job}") | cut -d ' ' -f6)
    job_type=$(echo "${names}" | cut -d '_' -f1)
    if [ "$job_type" = "RegressionStep3" ]; then
        real_number=$(echo "${names}" | cut -d '_' -f2)
        echo "${real_number}" >> Files_running_Step3.txt
    fi
done
