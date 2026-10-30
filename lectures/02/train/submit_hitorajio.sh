#!/usr/bin/env bash
# Submits the ひとらじ runs (v2 test): hitorajio, hitorajio_more, and hitorajio_more with 13 epochs (about 40 steps), seeds 42 and 1-4.
# Usage (on aic-sv01, from aie/): bash lec02/submit_hitorajio.sh
set -euo pipefail

for seed in 42 1 2 3 4; do
  sbatch lec02/job.sbatch uv run basic.py run --data hitorajio --seed "$seed"
  sbatch lec02/job.sbatch uv run basic.py run --data hitorajio_more --seed "$seed"
  sbatch lec02/job.sbatch uv run basic.py run --data hitorajio_more --epochs 13 --seed "$seed"
done
