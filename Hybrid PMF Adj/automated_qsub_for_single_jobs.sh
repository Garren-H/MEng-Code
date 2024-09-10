#!/bin/bash

# This script is used to sub jobs to the cluster as bathces
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <include_clusters> <include_zeros> <ARD> <func_groups_string> <total_jobs>"
  exit 1
fi

include_clusters=$1
include_zeros=$2
ARD=$3
func_groups_string="$4"
total_jobs=$5

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

if [[ "$include_zeros" == "1" ]]; then
  job_name="$job_name"_z
fi

if [[ "$ARD" == "1" ]]; then
  job_name="$job_name"_a
fi

for ((i=0; i<total_jobs; i++)); do
  qsub -N "$job_name"_"$i" -e "$job_name"_"$i".err -o "$job_name"_"$i".out -v include_clusters=$include_clusters,include_zeros=$include_zeros,ARD=$ARD,func_groups_string="$func_groups_string",chain_id=$i Hybrid_PMF.sh
done

# Usage: ./automated_qsub_for_single_jobs.sh 0 0 0 "Alkane.Primary alcohol" 6