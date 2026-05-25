# CALVIN — Lie Diffuser Actor

This subdirectory reproduces the CALVIN benchmark rows of paper Table 1.

## 1. Environment setup

```bash
cd calvin
# scikit-sparse pulls SuiteSparse system libs that pip can't get, so install it
# via conda first — theseus-ai's wheel build depends on it.
conda install -n lda-calvin -c conda-forge scikit-sparse suitesparse -y  # or run after `conda env create`
conda env create -f environment.yaml
conda activate lda-calvin
pip install -e .
pip install plotly zarr pytorch3d e3nn theseus-ai
pip install "dgl==2.1.0" "torchdata==0.6.1" -f https://data.dgl.ai/wheels/torch-2.0/cu118/repo.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

> **CALVIN simulator + dataset shortcut.** If a host already has the upstream `calvin` repo (with `dataset/`, `calvin_env/`, `calvin_models/`) checked out somewhere, symlink it under `third_party/` instead of running `setup_calvin.sh`:
> ```bash
> ln -sfn /path/to/existing/calvin third_party/calvin
> ```

### 1a. Install the CALVIN packages

```bash
pip install -e third_party/calvin/calvin_env/tacto
pip install -e third_party/calvin/calvin_env
pip install -e third_party/calvin/calvin_models --no-deps  
pip install pyhash                                         
pip install -U "networkx>=3.0"                             
pip install --force-reinstall --no-deps "numpy==1.23.5"    
```

Verify the env with `pytest tests/ -v` from this dir.

## 2. CALVIN dataset (one-time, ~500GB, hours)

```bash
SPLIT=ABC bash scripts/setup_calvin.sh    # or SPLIT=ABCD for the ABCD->D row
NUM_WORKERS=16 SPLIT=ABC bash scripts/setup_calvin.sh   # tune to your host
```

This script (in order):
1. Clones the CALVIN simulator (vendored under `../third_party/calvin/`). Skips if a clone or symlink is already there. **Does not run upstream's `install.sh`** — that pins yanked deps; do § 1a instead.
2. Downloads the raw `task_${SPLIT}_D` dataset (~500GB).
3. Downloads precomputed CLIP instruction embeddings (`instructions/`) from `huggingface.co/katefgroup/3d_diffuser_actor` 
4. Runs `python scripts/package_calvin.py --num_workers $NUM_WORKERS` to render the raw episodes into `data/calvin/packaged_${SPLIT}_D_full/{training,validation}`. The packager uses a `multiprocessing.Pool` of `NUM_WORKERS` PyBullet sims (default ≈ half your cores) — each annotation is independent, so wallclock scales close to linearly. Expect roughly `(17 870 annotations × ~1-2 s/annotation) / NUM_WORKERS` for ABC training; on a 16-core box that's ~30-60 minutes.

If you need to re-encode instructions from scratch (e.g. the `instructions.zip` cache is unreachable):

```bash
python scripts/preprocess_calvin_instructions.py \
    --output instructions/calvin_task_${SPLIT}_D/training.pkl \
    --annotation_path ../third_party/calvin/dataset/task_${SPLIT}_D/training/lang_annotations/auto_lang_ann.npy \
    --batch_size 64
python scripts/preprocess_calvin_instructions.py \
    --output instructions/calvin_task_${SPLIT}_D/validation.pkl \
    --annotation_path ../third_party/calvin/dataset/task_${SPLIT}_D/validation/lang_annotations/auto_lang_ann.npy \
    --batch_size 64
```

The encoder runs on GPU (`--device cuda` by default) and batches `--batch_size` instructions per CLIP forward, so encoding both splits of ABC takes a couple of minutes total.

**Note for ABCD->D:** the merged `instructions/calvin_task_ABCD_D/` array must be concatenated as `ABC's 17870 + D's 5124` (in that order). The training code shifts D-task indices by `+17870` to point into this merged array (controlled by `--training_split ABCD`). If `instructions.zip` does not include `calvin_task_ABCD_D/`, build it from the ABC and D files.

## 3. Download checkpoints (one-time, ~3GB)

```bash
bash scripts/download_ckpts.sh
```

Pulls all `.pth` files from `huggingface.co/tars3017/lie_diffuser_actor` into `ckpts/`.

## 4. Evaluate a released checkpoint

```bash
# ABC->D, LDA (main)
NGPUS=8 bash scripts/eval.sh configs/lda_abc_d.yaml                  ckpts/abc_gat_score_newloss_300k.pth
# ABC->D, LDA w/o Lie Diffusion
NGPUS=8 bash scripts/eval.sh configs/ablation_no_lie_abc_d.yaml      ckpts/abc_gat_ddpm_300k.pth
# ABC->D, LDA w/o GAT Encoder
NGPUS=8 bash scripts/eval.sh configs/ablation_no_gat_abc_d.yaml      ckpts/abc_nogat_600k.pth

# ABCD->D, LDA (main)
NGPUS=8 bash scripts/eval.sh configs/lda_abcd_d.yaml                 ckpts/abcd_gat_score_newloss_300k.pth
# ABCD->D, LDA w/o Lie Diffusion
NGPUS=8 bash scripts/eval.sh configs/ablation_no_lie_abcd_d.yaml     ckpts/abcd_ddpm_300k.pth
# ABCD->D, LDA w/o GAT Encoder
NGPUS=8 bash scripts/eval.sh configs/ablation_no_gat_abcd_d.yaml     ckpts/abcd_nogat_300k.pth
```

The default 1000-sequence protocol takes a few hours on 8× L40. To smoke-test the pipeline first, set `NUM_SEQUENCES=2`:

```bash
NUM_SEQUENCES=2 NGPUS=1 bash scripts/eval.sh configs/lda_abc_d.yaml ckpts/abc_gat_score_newloss_300k.pth
```

## 5. Train from scratch

```bash
NGPUS=8 bash scripts/train.sh configs/lda_abc_d.yaml
```

## 6. Run the test suite

```bash
pytest tests/ -v
```

- `test_imports.py` — every submodule imports.
- `test_config.py` — YAML loader behaves correctly.
- `test_ckpt_compat.py` — every paper-table checkpoint loads under the corresponding config (only meaningful after `scripts/download_ckpts.sh`).
- `test_forward_smoke.py` — single-GPU end-to-end forward smoke for every released ckpt.