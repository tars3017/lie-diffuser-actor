#!/usr/bin/env bash
# Usage: bash scripts/train.sh <config.yaml> [extra args ...]
# Wraps lda_oft.train with torchrun. Set NGPUS / NPROC_PER_NODE / MASTER_PORT
# in the env to override defaults.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [extra finetune.py args]"
    echo
    echo "Available configs:"
    ls lda_oft/configs/*.yaml
    exit 1
fi

CONFIG="$1"
shift 1

if [[ ! -f "$CONFIG" ]]; then echo "Config not found: $CONFIG"; exit 1; fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS:-4}}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM + 29000))}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-data/modified_libero_rlds}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-train_logs}"

torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    -m lda_oft.train \
    --config "$CONFIG" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --run_root_dir "$RUN_ROOT_DIR" \
    "$@"
