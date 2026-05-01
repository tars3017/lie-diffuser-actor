"""YAML config loader for openvla-oft variants.

Translates a sibling-specific YAML (`configs/oft_*.yaml`) into the flag set
expected by the upstream `FinetuneConfig` (`vla-scripts/finetune.py`) and
`GenerateConfig` (`experiments/robot/libero/run_libero_eval.py`).

The two projection helpers (`train_overrides`, `eval_overrides`) take the
intersection of the YAML keys with the upstream config's known fields, so
unrecognized YAML keys are silently dropped — keeps draccus happy and
lets us add comments / future-only fields without breaking older runners.
"""
from pathlib import Path
from typing import Any, Dict, Union

import yaml

# Keys that map directly to FinetuneConfig fields.
_TRAIN_KEYS = frozenset({
    "vla_path",
    "use_l1_regression",
    "use_diffusion",
    "num_diffusion_steps_train",
    "use_se3_score_matching",
    "num_se3_steps_train",
    "score_matching_lie_group",
    "use_film",
    "num_images_in_input",
    "use_proprio",
    "use_lora",
    "lora_rank",
    "lora_dropout",
    "batch_size",
    "learning_rate",
    "num_steps_before_decay",
    "max_steps",
    "save_freq",
    "image_aug",
})

# Keys that map directly to GenerateConfig fields (eval-side).
_EVAL_KEYS = frozenset({
    "use_l1_regression",
    "use_diffusion",
    "num_diffusion_steps_train",
    "num_diffusion_steps_inference",
    "use_se3_score_matching",
    "num_se3_steps_train",
    "num_se3_steps_inference",
    "score_matching_lie_group",
    "use_film",
    "num_images_in_input",
    "use_proprio",
    "center_crop",
    "num_trials_per_task",
    "unnorm_key",
})


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def train_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Project a parsed YAML to the subset of keys understood by FinetuneConfig."""
    return {k: cfg[k] for k in _TRAIN_KEYS if k in cfg}


def eval_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Project a parsed YAML to the subset of keys understood by GenerateConfig."""
    return {k: cfg[k] for k in _EVAL_KEYS if k in cfg}
