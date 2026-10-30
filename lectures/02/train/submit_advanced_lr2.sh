#!/usr/bin/env bash
# Submits the last 発展 tuning round: lr 2e-3 / 4e-3 (seed 42), and lr 1e-3 for seeds 1-4 (epochs 20, lora_r 16).
# Usage (on aic-sv01, from aie/): bash lec02/submit_advanced_lr2.sh
set -euo pipefail

for lr in 2e-3 4e-3; do
  sbatch lec02/job.sbatch uv run advanced.py run --seed 42 --lr "$lr"
done
for seed in 1 2 3 4; do
  sbatch lec02/job.sbatch uv run advanced.py run --seed "$seed" --lr 1e-3
done
