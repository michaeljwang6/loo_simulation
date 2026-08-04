#!/usr/bin/env bash
#SBATCH --job-name=loo-matrix
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=loo-matrix-%A_%a.out
#SBATCH --error=loo-matrix-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/dgp_estimator_matrix

# Replace this activation command if the cluster environment is elsewhere.
source .venv/bin/activate

# Each array task is already a separate process. Limit numerical libraries to
# one thread so the array does not oversubscribe the allocated CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

shard_name=$(printf "shard_%04d_of_0020" "${SLURM_ARRAY_TASK_ID}")
python scripts/run_monte_carlo.py \
  --config configs/dgp_estimator_matrix.json \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count 20 \
  --output "results/dgp_estimator_matrix/${shard_name}"

# After all 20 array tasks finish, merge from a login node with:
# python scripts/merge_monte_carlo.py \
#   --inputs results/dgp_estimator_matrix/shard_*_of_0020 \
#   --output results/dgp_estimator_matrix/merged
