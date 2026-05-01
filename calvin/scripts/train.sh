#!/usr/bin/env bash
# Usage: bash scripts/train.sh configs/<config>.yaml [extra args ...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [extra args]"
    exit 1
fi

CONFIG="$1"
shift

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

NGPUS="${NGPUS:-8}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM + 1024))}"

CLI_ARGS=$(python -c "
from lda.utils.config import load_config, to_cli_args
import shlex
print(' '.join(shlex.quote(a) for a in to_cli_args(load_config('$CONFIG'))))
")

echo "Config: $CONFIG"
echo "GPUs: $NGPUS"
echo "Args: $CLI_ARGS $@"

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
    train.py $CLI_ARGS "$@"
