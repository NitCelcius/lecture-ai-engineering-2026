#!/usr/bin/env bash
# Submits the seed sweeps with sbatch: lora_r 16/4 on the course data, and hitorajio / hitorajio_more at lora_r 16.
# Usage (on aic-sv01, from aie/): bash lec02/submit_sweep.sh
set -euo pipefail

for seed in 1 2 3 4; do
  sbatch lec02/job.sbatch uv run basic.py run --seed "$seed"
  sbatch lec02/job.sbatch uv run basic.py run --lora-r 4 --seed "$seed"
done

for data in hitorajio hitorajio_more; do
  sbatch lec02/job.sbatch uv run basic.py run --data "$data"
  for seed in 1 2 3 4; do
    sbatch lec02/job.sbatch uv run basic.py run --data "$data" --seed "$seed"
  done
done
