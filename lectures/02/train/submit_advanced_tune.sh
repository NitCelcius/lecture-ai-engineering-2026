#!/usr/bin/env bash
# Submits the 発展 tuning runs (seed 42): epochs 60 / 100 x lora_r 16 / 64.
# Usage (on aic-sv01, from aie/): bash lec02/submit_advanced_tune.sh
set -euo pipefail

for epochs in 60 100; do
  for r in 16 64; do
    sbatch lec02/job.sbatch uv run advanced.py run --seed 42 --epochs "$epochs" --lora-r "$r"
  done
done
