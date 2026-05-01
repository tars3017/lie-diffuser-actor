#!/usr/bin/env bash
# Clone the LIBERO simulator + install its Python deps in the lda-oft env.
# Run from the openvla_oft/ directory after `conda activate lda-oft`.
set -euo pipefail

LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_DIR="${LIBERO_DIR:-LIBERO}"

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
    echo "Cloning $LIBERO_REPO -> $LIBERO_DIR"
    git clone "$LIBERO_REPO" "$LIBERO_DIR"
else
    echo "Found existing LIBERO clone at $LIBERO_DIR (skipping git clone)"
fi

# Modern setuptools (find_packages) won't pick up the top-level `libero`
# namespace because upstream ships no __init__.py there. Drop an empty one
# so `pip install -e` registers the package and `import libero.libero` works.
touch "$LIBERO_DIR/libero/__init__.py"

pip install -e "$LIBERO_DIR"
pip install -r experiments/robot/libero/libero_requirements.txt

# libero_requirements.txt pulls opencv-python which transitively bumps numpy>=2.
# torch 2.2.0 wheels were built against numpy 1.x and crash on import with numpy 2.
# Pin back. Do this LAST in setup so nothing else clobbers it.
pip install "numpy<2"

echo
echo "Done. Quick sanity check:"
python -c "import libero, torch, numpy; print('LIBERO ok. torch=', torch.__version__, 'numpy=', numpy.__version__)"
