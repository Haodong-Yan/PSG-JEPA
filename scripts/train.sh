#!/bin/bash
# Train PSG-JEPA on OGBench-Cube. Usage: scripts/train.sh [seed]
set -euo pipefail
SEED=${1:-3072}
cd "$(dirname "$0")/.."
python train.py data=ogb_cm seed="$SEED" subdir="psgjepa_cube_seed${SEED}"
# -> checkpoint at $(swm cache dir)/psgjepa_cube_seed${SEED}/psgjepa_epoch_10_object.ckpt
