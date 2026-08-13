#!/usr/bin/env bash
#SBATCH --job-name=loo-matrix-merge
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=loo-matrix-merge-%j.out
#SBATCH --error=loo-matrix-merge-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python scripts/merge_monte_carlo.py \
  --inputs results/dgp_estimator_matrix_cluster_v2/shard_*_of_0050 \
  --output results/dgp_estimator_matrix_cluster_v2/merged
