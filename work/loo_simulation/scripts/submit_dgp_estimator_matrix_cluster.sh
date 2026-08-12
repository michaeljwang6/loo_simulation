#!/usr/bin/env bash

set -euo pipefail

array_job=$(sbatch --parsable scripts/slurm_dgp_estimator_matrix_cluster.sh)
merge_job=$(
  sbatch \
    --parsable \
    --dependency="afterok:${array_job}" \
    scripts/slurm_merge_dgp_estimator_matrix_cluster.sh
)

echo "Submitted 50-shard array job ${array_job}."
echo "Submitted dependent merge job ${merge_job}."
