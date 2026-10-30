#!/usr/bin/env bash
# Submits the 発展 runs (用語集 SFT, 4 conditions each): seeds 42 and 1-4.
# Usage (on aic-sv01, from aie/): bash lec02/submit_advanced.sh
set -euo pipefail

for seed in 42 1 2 3 4; do
  sbatch lec02/job.sbatch uv run advanced.py run --seed "$seed"
done
