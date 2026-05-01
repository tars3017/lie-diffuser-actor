#!/usr/bin/env bash
# Set up CALVIN: simulator, raw dataset, preprocessed dataset, and CLIP instruction embeddings.
# Run from the calvin/ directory of this repo.
set -euo pipefail

THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-../third_party}"
SPLIT="${SPLIT:-ABC}"   # ABC for ABC->D row, ABCD for ABCD->D row
NUM_WORKERS="${NUM_WORKERS:-$(python -c 'import os; print(max(1, (os.cpu_count() or 4) // 2))')}"
mkdir -p "$THIRD_PARTY_DIR"

# 1. CALVIN simulator (one-time clone). The actual *Python* package install
#    (calvin_env/tacto, calvin_env, calvin_models) is handled separately in
#    README § 1a — upstream's install.sh pins yanked + unbuildable deps.
if [[ ! -d "$THIRD_PARTY_DIR/calvin" ]]; then
    echo "[1/4] Cloning CALVIN simulator..."
    cd "$THIRD_PARTY_DIR"
    git clone --recurse-submodules https://github.com/mees/calvin.git
    cd calvin
    cd calvin_env && git checkout -b main --track origin/main && cd ..
    cd - >/dev/null
    echo "[1/4] Now run README § 1a to pip-install the CALVIN packages."
else
    echo "[1/4] CALVIN simulator already installed; skipping."
fi

# 2. CALVIN raw dataset (~500GB per split, expensive)
DATASET_DIR="$THIRD_PARTY_DIR/calvin/dataset"
if [[ ! -d "$DATASET_DIR/task_${SPLIT}_D" ]]; then
    echo "[2/4] Downloading CALVIN ${SPLIT} dataset (~500GB)..."
    cd "$DATASET_DIR"
    sh download_data.sh "$SPLIT"
    cd - >/dev/null
else
    echo "[2/4] task_${SPLIT}_D dataset already present; skipping."
fi

# 3. CLIP-encoded instruction embeddings (~few hundred MB)
# Precomputed embeddings hosted by upstream 3D Diffuser Actor. Faster than re-encoding.
if [[ ! -d instructions ]]; then
    echo "[3/4] Downloading precomputed CLIP instruction embeddings..."
    wget -q https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/instructions.zip
    unzip -q instructions.zip
    rm instructions.zip
else
    echo "[3/4] instructions/ already present; skipping."
fi

# 4. Preprocess raw dataset → packaged_<SPLIT>_D_full/
PACKAGED="data/calvin/packaged_${SPLIT}_D_full"
if [[ ! -d "$PACKAGED" ]]; then
    echo "[4/4] Preprocessing raw dataset → $PACKAGED ..."
    mkdir -p "$PACKAGED"
    python scripts/package_calvin.py --split training \
        --root_dir "$DATASET_DIR/task_${SPLIT}_D" \
        --save_path "$PACKAGED" --num_workers "$NUM_WORKERS"
    python scripts/package_calvin.py --split validation \
        --root_dir "$DATASET_DIR/task_${SPLIT}_D" \
        --save_path "$PACKAGED" --num_workers "$NUM_WORKERS"
else
    echo "[4/4] $PACKAGED already present; skipping."
fi

echo "CALVIN setup complete."
echo ""
echo "Next steps:"
echo "  bash scripts/download_ckpts.sh            # ~3GB, paper-table checkpoints"
echo "  bash scripts/eval.sh configs/lda_${SPLIT,,}_d.yaml ckpts/<ckpt>.pth"
