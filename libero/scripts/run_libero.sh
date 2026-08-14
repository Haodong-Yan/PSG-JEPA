#!/usr/bin/env bash
# Full LIBERO-Goal pipeline for one training seed:
#   encoder pretraining  ->  joint OFT head training  ->  closed-loop evaluation
#
# Usage:
#   libero/scripts/run_libero.sh 3072      # the released seed
#
# Two Python environments are required and cannot be merged: encoder pretraining runs against
# stable-worldmodel (numpy>=2), while the LIBERO simulator stack (robosuite/robomimic) pins
# numpy<2. Point PY_PRETRAIN and PY_LIBERO at the two interpreters; everything else is shared.
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY_PRETRAIN=${PY_PRETRAIN:-python}                       # stable-worldmodel env (numpy>=2)
PY_LIBERO=${PY_LIBERO:-python}                           # LIBERO env (numpy<2)
# LIBERO_ROOT is the benchmark asset tree (task bddl files + init states) passed as
# --libero-root. LIBERO_PKG is the importable `libero` Python package. In a plain clone of
# Lifelong-Robot-Learning/LIBERO these are the same directory, which is the default; they differ
# when the assets are vendored separately from the installed package, so keep them separate.
LIBERO_ROOT=${LIBERO_ROOT:-$REPO/assets/benchmarks/LIBERO}
LIBERO_PKG=${LIBERO_PKG:-$LIBERO_ROOT}
STABLEWM_HOME=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
OUT=${OUT:-$REPO/libero/runs}

PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-40}
HEAD_EPOCH=${HEAD_EPOCH:-30}
EVAL_SEED=${EVAL_SEED:-4242}
N_EVAL=${N_EVAL:-50}
MAX_STEPS=${MAX_STEPS:-600}
GPU=${GPU:-0}

SEED=${1:-3072}

# Mirrors libero/inventory.py; keep the two in sync.
DATA=libero_goal_cm_state
GROUNDING=(loss.grounding.weight=0.1)

RUN=psgjepa_seed${SEED}
RUN_DIR=$OUT/$RUN
ENC_SUBDIR=libero_psgjepa_seed${SEED}_ep${PRETRAIN_EPOCH}
ENC_CKPT=$STABLEWM_HOME/$ENC_SUBDIR/lewm_v2_epoch_${PRETRAIN_EPOCH}_object.ckpt
EVAL_JSON=$OUT/${RUN}_eval_all_n${N_EVAL}_s${EVAL_SEED}_ep${HEAD_EPOCH}.json
mkdir -p "$RUN_DIR" "$STABLEWM_HOME"

export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
log() { echo "[$(date '+%F %T')] [psgjepa seed=$SEED] $*"; }

# ---- 0. preflight: everything the last stage needs, checked before the first one ----
# Pretraining alone takes hours, so a missing simulator or dataset must fail now, not after it.
preflight() {
  local ok=0
  if [[ ! -s "$STABLEWM_HOME/$DATA.h5" ]]; then
    log "MISSING dataset $STABLEWM_HOME/$DATA.h5 -- run libero/convert_libero_to_swm.py first"
    ok=1
  fi
  if ! PYTHONPATH="$REPO:$REPO/libero:$LIBERO_PKG" "$PY_LIBERO" -c \
      'import libero.lifelong.metric' 2>/dev/null; then
    log "MISSING the LIBERO python package: 'import libero.lifelong' failed."
    log "  LIBERO_PKG=$LIBERO_PKG -- set it to the LIBERO repo root, or pip install it."
    ok=1
  fi
  [[ -d "$LIBERO_ROOT/libero/libero/init_files" ]] || {
    log "MISSING benchmark assets under LIBERO_ROOT=$LIBERO_ROOT (expected libero/libero/init_files)"
    ok=1
  }
  return $ok
}
preflight || { log "preflight failed, nothing was run"; exit 1; }

# ---- 1. encoder pretraining ----
if [[ ! -s "$ENC_CKPT" ]]; then
  log "pretraining encoder -> $ENC_CKPT"
  ( cd "$REPO" && STABLEWM_HOME=$STABLEWM_HOME "$PY_PRETRAIN" train.py \
      --config-name=psgjepa_libero \
      data="$DATA" seed="$SEED" \
      trainer.max_epochs="$PRETRAIN_EPOCH" subdir="$ENC_SUBDIR" "${GROUNDING[@]}" ) || exit 1
  [[ -s "$ENC_CKPT" ]] || { log "pretraining produced no checkpoint"; exit 1; }
else
  log "encoder exists, skipping pretraining"
fi

# ---- 2. joint OFT head training (encoder is fine-tuned alongside the head) ----
export PYTHONPATH="$REPO:$REPO/libero:$LIBERO_PKG"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero} MUJOCO_GL=${MUJOCO_GL:-egl}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

HEAD_CKPT=$RUN_DIR/lewm_libero_oft_head_epoch_${HEAD_EPOCH}.ckpt
if [[ ! -s "$HEAD_CKPT" ]]; then
  log "training OFT head -> $HEAD_CKPT"
  "$PY_LIBERO" "$REPO/libero/train_oft_head.py" \
    --libero-root "$LIBERO_ROOT" --init-policy "$ENC_CKPT" --tasks all \
    --seq-len 2 --image-keys agentview_rgb,eye_in_hand_rgb \
    --chunk-len 8 --action-horizon 8 \
    --head-type oft --hidden-dim 1024 --num-layers 4 --num-heads 8 --dropout 0.1 \
    --batch-size 32 --lr 5e-5 --num-workers 4 --max-epochs "$HEAD_EPOCH" --save-every 5 \
    --seed "$SEED" --train-encoder --run-dir "$RUN_DIR" || exit 1
  [[ -s "$HEAD_CKPT" ]] || { log "head training produced no checkpoint"; exit 1; }
else
  log "head checkpoint exists, skipping training"
fi

# ---- 3. closed-loop evaluation: 10 tasks x N_EVAL rollouts ----
if [[ ! -s "$EVAL_JSON" ]]; then
  log "evaluating -> $EVAL_JSON"
  "$PY_LIBERO" "$REPO/libero/eval_oft_head.py" \
    --libero-root "$LIBERO_ROOT" \
    --checkpoint "$HEAD_CKPT" --tasks all --n-eval "$N_EVAL" --max-steps "$MAX_STEPS" \
    --seed "$EVAL_SEED" --output "$EVAL_JSON" || exit 1
fi
log "mean success = $("$PY_LIBERO" -c "import json,sys;print(json.load(open(sys.argv[1]))['mean_success_rate'])" "$EVAL_JSON")"
