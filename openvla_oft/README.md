# OpenVLA-OFT — Lie Diffuser Actor

LIBERO-10 (LIBERO-Long) experiments for the rebuttal section of *The Lie We Tell* (ICML 2026): isolating Lie-Space Diffusion from the GAT confound by applying SE(3) score matching on top of OpenVLA-OFT (7B, flat MLP — no GAT, no point cloud, no 3D scene encoder).

## Acknowledgements

This sibling builds directly on two prior open-source releases:

- **OpenVLA** — the 7B-parameter vision-language-action backbone (`openvla/openvla-7b`). Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024. [openvla.github.io](https://openvla.github.io/) · [github.com/openvla/openvla](https://github.com/openvla/openvla)
- **OpenVLA-OFT** — the parallel-decoding + continuous-action-head fine-tuning recipe (the `experiments/`, `prismatic/`, and `vla-scripts/` subtrees here are vendored from a fork of this repo). Kim, Finn & Liang, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*, arXiv:2502.19645. [openvla-oft.github.io](https://openvla-oft.github.io/) · [github.com/moojink/openvla-oft](https://github.com/moojink/openvla-oft)
  ```bibtex
  @article{kim2025fine,
    title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
    author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
    journal={arXiv preprint arXiv:2502.19645},
    year={2025}
  }
  ```

The vendored source extends `moojink/openvla-oft` with our SE(3) score-matching action head. What this sibling adds on top:

- A YAML-driven, sibling-specific wrapper package (`lda_oft/`) that lets one source tree run all three variants (baseline, +Euclidean SM, +Lie SM) by toggling a `score_matching_lie_group: bool` flag — so the released checkpoints for both score-matching variants load against the same `SE3ScoreMatchingActionHead` class.
- LIBERO simulator integration (`experiments/robot/libero/`) is unchanged from the OpenVLA-OFT upstream beyond the new flag plumbing. LIBERO itself is the [Lifelong Robot Learning benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO).

| Variant | LIBERO-Long SR (3-seed mean) | YAML config | Action head |
|---|---|---|---|
| OpenVLA-OFT (baseline) | 92.20 | `lda_oft/configs/oft_baseline_libero10.yaml` | `L1RegressionActionHead` |
| OpenVLA-OFT + Euclidean Score Matching | 93.87 | `lda_oft/configs/oft_euclidean_sm_libero10.yaml` | `SE3ScoreMatchingActionHead` (`lie_group=False`) |
| OpenVLA-OFT + Lie Score Matching | 94.13 | `lda_oft/configs/oft_lie_sm_libero10.yaml` | `SE3ScoreMatchingActionHead` (`lie_group=True`) |

The two Score-Matching variants share the same class and parameter set; only the runtime math differs. A single `lie_group: bool` flag on `SE3ScoreMatchingActionHead` selects between SE(3) Riemannian (Lie) and flat ℝ⁶ Euclidean diffusion, so both released ckpts load with `strict=True` against the same code.

## 1. Environment setup

```bash
cd openvla_oft
conda env create -f environment.yaml          # creates `lda-oft` (Python 3.10)
conda activate lda-oft

# FlashAttention-2 prebuilt wheel (compiling from source requires CUDA toolkit + nvcc)
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

`environment.yaml` declares Python 3.10 + cmake + ninja; the rest comes from `pyproject.toml` (torch 2.2.0, transformers fork, peft 0.11.1, draccus 0.8.0, …).

## 2. LIBERO simulator + dataset (one-time)

```bash
bash scripts/setup_libero.sh                  # clones LIBERO + installs libero_requirements.txt
```

To launch training you also need the LIBERO RLDS dataset:
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds data/modified_libero_rlds
```

The eval pipeline does not need the RLDS dataset — it streams trials from the LIBERO simulator directly.

## 3. Download checkpoints

Lie SM and Euclidean SM are on HuggingFace; the baseline ckpt is not published — train it from scratch (§5) or use upstream `moojink/openvla-7b-oft-finetuned-libero-10` as a comparable public OFT baseline.

```bash
bash scripts/download_ckpts.sh                # pulls both SM ckpts (~30 GB total) into ckpts/
```

## 4. Reproduce a row

```bash
# Lie SM
bash scripts/eval.sh lda_oft/configs/oft_lie_sm_libero10.yaml \
    ckpts/openvla-oft-lie-lora-150000

# Euclidean SM
bash scripts/eval.sh lda_oft/configs/oft_euclidean_sm_libero10.yaml \
    ckpts/openvla-oft-euclidean-lora-150000
```

Each invocation runs the standard 500-trial LIBERO eval (10 tasks × 50 episodes); takes a few hours on a single L40. To eval a baseline ckpt you trained yourself, pass its checkpoint directory as the second argument and `lda_oft/configs/oft_baseline_libero10.yaml` as the first.

## 5. Train from scratch

```bash
NPROC_PER_NODE=4 \
DATA_ROOT_DIR=data/modified_libero_rlds \
RUN_ROOT_DIR=train_logs/lie_sm \
bash scripts/train.sh lda_oft/configs/oft_lie_sm_libero10.yaml \
    --wandb_entity <your-entity> --wandb_project openvla-oft
```

The wrapper translates the YAML into `vla-scripts/finetune.py` flags and re-executes via `runpy`. 150 K steps at batch 4 / GPU on 4 GPUs takes about 40 hours on L40. Swap the config path for `oft_baseline_libero10.yaml` or `oft_euclidean_sm_libero10.yaml` to train the other variants.

## 6. Run the test suite

```bash
pytest tests/ -v
```

The CPU action-head ckpt-compat gate (`tests/test_ckpt_compat.py`) skips per-pair when the corresponding local ckpt isn't present, so the passing count depends on which ckpts you've downloaded / trained.

## 7. Configuration reference

| YAML key | Type | Notes |
|---|---|---|
| `variant` | str | One of `baseline` / `euclidean_sm` / `lie_sm`; informational only. |
| `task_suite` | str | `libero_10` everywhere in this release. |
| `unnorm_key` | str | `libero_10_no_noops` everywhere. |
| `vla_path` | str | Base VLA — `openvla/openvla-7b`. |
| `use_l1_regression` | bool | Baseline action head. |
| `use_se3_score_matching` | bool | Score-matching action head (Lie or Euclidean). |
| `score_matching_lie_group` | bool | Only when `use_se3_score_matching`: `true` = SE(3) Lie, `false` = flat ℝ⁶ Euclidean. |
| `num_se3_steps_train` / `num_se3_steps_inference` | int | 100 / 100 for both SM variants. |
| `use_lora` / `lora_rank` / `lora_dropout` | bool / int / float | LoRA r=32, dropout 0.0. |
| `batch_size` / `learning_rate` / `max_steps` | int / float / int | 4 / 5e-4 / 150 005 for all variants. |
| `num_images_in_input` / `use_proprio` | int / bool | 2 + proprio. |
| `center_crop` / `num_trials_per_task` | bool / int | Eval-only — 50 trials × 10 LIBERO tasks. |
