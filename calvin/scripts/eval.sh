#!/usr/bin/env bash
# Usage: bash scripts/eval.sh configs/<config>.yaml ckpts/<ckpt>.pth [extra args ...]
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config.yaml> <ckpt.pth> [extra args]"
    exit 1
fi

CONFIG="$1"
CKPT="$2"
shift 2

if [[ ! -f "$CONFIG" ]]; then echo "Config not found: $CONFIG"; exit 1; fi
if [[ ! -f "$CKPT" ]]; then echo "Checkpoint not found: $CKPT"; exit 1; fi

NGPUS="${NGPUS:-1}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM + 1024))}"

# Pick whichever CALVIN simulator dataset is present — D-task validation data
# is identical across task_ABC_D / task_ABCD_D / task_D_D, so any one works
# for eval. Avoids forcing the user to download a specific split just to run
# scripts/eval.sh after setting up the other split via scripts/setup_calvin.sh.
DATASET_ROOT="${CALVIN_DATASET_ROOT:-../third_party/calvin/dataset}"
for split in task_ABC_D task_ABCD_D task_D_D; do
    if [[ -d "$DATASET_ROOT/$split" ]]; then
        CALVIN_TASK_DIR="$DATASET_ROOT/$split"
        break
    fi
done
if [[ -z "${CALVIN_TASK_DIR:-}" ]]; then
    echo "Error: no task_(ABC|ABCD|D)_D found under $DATASET_ROOT/. Run scripts/setup_calvin.sh first."
    exit 1
fi

CLI_ARGS=$(python -c "
from lda.utils.config import load_config, to_cli_args
import shlex
print(' '.join(shlex.quote(a) for a in to_cli_args(load_config('$CONFIG'))))
")

EXP_DIR="$(python -c "from lda.utils.config import load_config; print(load_config('$CONFIG').get('exp_log_dir', 'eval'))")"
RUN_DIR="$(python -c "from lda.utils.config import load_config; print(load_config('$CONFIG').get('run_log_dir', 'eval'))")"

torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
    -m lda.eval.evaluate_policy \
    --calvin_dataset_path "$CALVIN_TASK_DIR" \
    --calvin_model_path ../third_party/calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --calvin_gripper_loc_bounds "$CALVIN_TASK_DIR/validation/statistics.yaml" \
    --action_dim 7 \
    --base_log_dir "train_logs/${EXP_DIR}/${RUN_DIR}/eval_logs/" \
    --checkpoint "$CKPT" \
    --save_video 0 \
    $CLI_ARGS "$@"
