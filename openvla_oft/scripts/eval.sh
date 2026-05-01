#!/usr/bin/env bash
# Usage: bash scripts/eval.sh <config.yaml> <pretrained_checkpoint> [extra args ...]
#
# <pretrained_checkpoint> is either a local LoRA-checkpoint dir (e.g.
#   ckpts/openvla-oft-lie-lora-150000) or an HF model ID (e.g.
#   tars3017/lie_diffuser_actor/openvla-oft-lie-lora-150000).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config.yaml> <pretrained_checkpoint> [extra run_libero_eval.py args]"
    echo
    echo "Available configs:"
    ls lda_oft/configs/*.yaml
    exit 1
fi

CONFIG="$1"
CKPT="$2"
shift 2

if [[ ! -f "$CONFIG" ]]; then echo "Config not found: $CONFIG"; exit 1; fi

# By default use whichever `python` is on PATH (e.g. the active `lda-oft`
# conda env). Override via `PYTHON=/path/to/python bash scripts/eval.sh …`.
PYTHON="${PYTHON:-python}"
"$PYTHON" -m lda_oft.eval --config "$CONFIG" --pretrained_checkpoint "$CKPT" "$@"
