#!/bin/bash

# This script is used to sub jobs to the cluster as bathces
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <include_clusters> <variance_known> <func_groups_string> <total_jobs>"
  exit 1
fi

include_clusters=$1
variance_known=$2
func_groups_string="$3"
total_jobs=$4

if [[ "$func_groups_string" == "Alkane.Primary alcohol" ]]; then
  job_name="Hybrid_PMF_AP"
elif [[ "$func_groups_string" == "all" ]]; then
  job_name="Hybrid_PMF_all"
else
  echo "Functional groups not in training set"
    exit 1
fi

if [[ "$include_clusters" == "1" ]]; then
  job_name="$job_name"_c
fi

for ((i=0; i<total_jobs; i++)); do
  qsub -N "$job_name"_"$i" -e "$job_name"_"$i".err -o "$job_name"_"$i".out -v include_clusters=$include_clusters,variance_known=$variance_known,func_groups_string="$func_groups_string",chain_id=$i Hybrid_PMF.sh
done

# Usage: ./automated_qsub_for_single_jobs.sh 0 1 "Alkane.Primary alcohol" 5