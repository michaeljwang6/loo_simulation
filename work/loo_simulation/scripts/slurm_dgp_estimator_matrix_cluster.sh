#!/usr/bin/env bash
#SBATCH --job-name=loo-matrix-25k
#SBATCH --partition=shared
#SBATCH --array=0-49%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=loo-matrix-25k-%A_%a.out
#SBATCH --error=loo-matrix-25k-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/dgp_estimator_matrix_cluster

# Replace this activation command if the cluster environment is elsewhere.
source .venv/bin/activate

# Each array task is one process. Avoid hidden BLAS oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/loo-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

shard_name=$(printf "shard_%04d_of_0050" "${SLURM_ARRAY_TASK_ID}")
python scripts/run_monte_carlo.py \
  --config configs/dgp_estimator_matrix_cluster.json \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count 50 \
  --output "results/dgp_estimator_matrix_cluster/${shard_name}" \
  --resume
