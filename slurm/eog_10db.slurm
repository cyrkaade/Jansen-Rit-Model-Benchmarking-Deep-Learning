#!/bin/bash -l
#SBATCH --job-name=jr-eog10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=slurm-eog10-%j.out

set -euo pipefail

module load scicomp-python-env
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python -m scripts.run_eog_recovery --config configs/eog_10db.yaml
