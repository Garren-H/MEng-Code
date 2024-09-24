#!/bin/bash

# This script is used to sub jobs to the cluster as bathces
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <include_clusters> <include_zeros> <ARD> <func_groups_string> <jobs_ID> <reps_per_job>"
  exit 1
fi

include_clusters=$1
include_zeros=$2
ARD=$3
func_groups_string="$4"
job_ID=$5
reps=$6

if [[ "$func_groups_string" == "Alkane.Primary alcohol" ]]; then
  job_name="AP"
elif [[ "$func_groups_string" == "all" ]]; then
  job_name="all"
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

for ((j=0; j<reps; j++)); do
  qsub -N "$job_name"_"$job_ID"_"$j" -e "$job_name"_"$job_ID"_"$j".err -o "$job_name"_"$job_ID"_"$j".out -v include_clusters=$include_clusters,include_zeros=$include_zeros,ARD=$ARD,func_groups_string="$func_groups_string",chain_id=$job_ID,rep=$j Hybrid_PMF.sh
done

# Usage: ./automated_qsub_for_single_jobs.sh 0 0 0 "Alkane.Primary alcohol" 0 10