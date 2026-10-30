#!/usr/bin/env bash
# Submits the 発展 learning-rate runs (seed 42, epochs 20, lora_r 16): lr 5e-5 / 1e-4 / 5e-4 / 1e-3.
# Usage (on aic-sv01, from aie/): bash lec02/submit_advanced_lr.sh
set -euo pipefail

for lr in 5e-5 1e-4 5e-4 1e-3; do
  sbatch lec02/job.sbatch uv run advanced.py run --seed 42 --lr "$lr"
done
