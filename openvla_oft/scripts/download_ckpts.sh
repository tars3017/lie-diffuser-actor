#!/usr/bin/env bash
# Download released openvla-oft checkpoints from huggingface.co/tars3017/lie_diffuser_actor.
# Currently the Lie SM and Euclidean SM ckpts are published; the baseline ckpt
# is not — train it from scratch (see README.md §5) or use upstream
# moojink/openvla-7b-oft-finetuned-libero-10 as a comparable public equivalent.
set -euo pipefail

REPO="tars3017/lie_diffuser_actor"
DEST="${DEST:-ckpts}"

# Folders to download (one per published variant). Each is downloaded under
# $DEST preserving its folder name, so scripts/eval.sh can be pointed at it
# directly:
#   bash scripts/eval.sh lda_oft/configs/oft_lie_sm_libero10.yaml      ckpts/openvla-oft-lie-lora-150000
#   bash scripts/eval.sh lda_oft/configs/oft_euclidean_sm_libero10.yaml ckpts/openvla-oft-euclidean-lora-150000
FOLDERS=(
    "openvla-oft-lie-lora-150000"
    "openvla-oft-euclidean-lora-150000"
)

mkdir -p "$DEST"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

for folder in "${FOLDERS[@]}"; do
    echo "Downloading $REPO/$folder/ -> $DEST/$folder/  (~15 GB each)"
    huggingface-cli download "$REPO" --include "$folder/*" --local-dir "$DEST"
    echo "  -> $DEST/$folder/"
    ls -lh "$DEST/$folder/" | head -3
    echo
done

echo "Done. Downloaded variants:"
for folder in "${FOLDERS[@]}"; do
    echo "  $DEST/$folder/"
done
