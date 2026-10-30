#!/bin/bash
#SBATCH -J tanakah-aie
#SBATCH -n 8 # CPU cores
#SBATCH --mem=16G # avail RAM
#SBATCH -G 1 # N of GPUs, maybe 1 is enough
#SBATCH --output=logsjob/log_%j.log # the log
# Runs a command in the container on the allocated GPU.
# Usage (from aie/, so logsjob/ resolves): sbatch lec02/job.sbatch uv run check_env.py
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
exec ./lec02/sing_run.sh "$@"
