#!/usr/bin/env bash
# Runs a command inside the uv container with only the aie/ dir visible (no home, no other dirs).
# Usage (on aic-sv01, from aie/): CUDA_VISIBLE_DEVICES=0 ./lec02/sing_run.sh uv run check_env.py
# Needs aie/uv.sif (see setup notes); caches and the venv live under aie/.cache and aie/venv-lec02.
set -euo pipefail

AIE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SINGULARITY_CACHEDIR="$AIE/.cache/singularity"
export SINGULARITY_TMPDIR="$AIE/.cache/tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

exec singularity exec --nv --containall --no-home \
  --bind "$AIE:/work" --pwd /work/lec02 \
  --env UV_CACHE_DIR=/work/.cache/uv \
  --env UV_PROJECT_ENVIRONMENT=/work/venv-lec02 \
  --env HF_HOME=/work/.cache/hf \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$AIE/uv.sif" "$@"
