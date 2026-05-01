#!/usr/bin/env bash
# Download all paper-table CALVIN checkpoints from HuggingFace into ckpts/<name>.pth
# so eval.sh / train.sh can use them directly without further moves.
set -euo pipefail

REPO="tars3017/lie_diffuser_actor"
DEST="${DEST:-ckpts}"

mkdir -p "$DEST"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

# The HF repo hosts CALVIN ckpts under lie_diffusion/, alongside other (openvla-oft)
# folders we don't need. `--include` keeps the download narrow.
echo "Downloading $REPO/lie_diffusion/*.pth -> $DEST/ (~3GB)"
huggingface-cli download "$REPO" --include "lie_diffusion/*.pth" --local-dir "$DEST"

# Flatten lie_diffusion/<name>.pth -> ckpts/<name>.pth so README/eval paths
# (`ckpts/<ckpt>.pth`) work without any further file moves on the user side.
if [[ -d "$DEST/lie_diffusion" ]]; then
    mv "$DEST"/lie_diffusion/*.pth "$DEST"/
    rmdir "$DEST/lie_diffusion" 2>/dev/null || true
fi

echo "Done. Files in $DEST/:"
ls -lh "$DEST"/*.pth
