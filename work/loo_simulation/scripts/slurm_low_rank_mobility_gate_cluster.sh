#!/usr/bin/env bash
#SBATCH --job-name=loo-lr-mobility-gate
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=loo-lr-mobility-gate-%j.out
#SBATCH --error=loo-lr-mobility-gate-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/loo-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python scripts/run_monte_carlo.py \
  --config configs/low_rank_mobility_gate_cluster.json \
  --output results/low_rank_mobility_gate_cluster \
  --resume

python scripts/audit_cluster_preflight.py \
  results/low_rank_mobility_gate_cluster
