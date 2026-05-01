"""TDD: YAML config loader → train/eval override projections."""
from pathlib import Path

import pytest

from lda_oft.config import (
    eval_overrides,
    load_config,
    train_overrides,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "lda_oft" / "configs"
ALL = [
    "oft_baseline_libero10.yaml",
    "oft_euclidean_sm_libero10.yaml",
    "oft_lie_sm_libero10.yaml",
]


@pytest.mark.parametrize("name", ALL)
def test_yaml_parses_with_expected_shape(name):
    cfg = load_config(CONFIG_DIR / name)
    assert cfg["variant"] in {"baseline", "euclidean_sm", "lie_sm"}
    assert cfg["task_suite"] == "libero_10"
    assert cfg["unnorm_key"] == "libero_10_no_noops"


def test_baseline_train_overrides():
    cfg = load_config(CONFIG_DIR / "oft_baseline_libero10.yaml")
    overrides = train_overrides(cfg)
    assert overrides["use_l1_regression"] is True
    assert overrides["use_diffusion"] is False
    assert overrides["use_se3_score_matching"] is False
    assert "score_matching_lie_group" not in overrides  # only meaningful when SM is on


def test_euclidean_sm_train_overrides():
    cfg = load_config(CONFIG_DIR / "oft_euclidean_sm_libero10.yaml")
    overrides = train_overrides(cfg)
    assert overrides["use_l1_regression"] is False
    assert overrides["use_se3_score_matching"] is True
    assert overrides["score_matching_lie_group"] is False
    assert overrides["num_se3_steps_train"] == 100


def test_lie_sm_train_overrides():
    cfg = load_config(CONFIG_DIR / "oft_lie_sm_libero10.yaml")
    overrides = train_overrides(cfg)
    assert overrides["use_se3_score_matching"] is True
    assert overrides["score_matching_lie_group"] is True
    assert overrides["num_se3_steps_train"] == 100


@pytest.mark.parametrize("name,expected_lie", [
    ("oft_euclidean_sm_libero10.yaml", False),
    ("oft_lie_sm_libero10.yaml", True),
])
def test_eval_overrides_carry_lie_group(name, expected_lie):
    cfg = load_config(CONFIG_DIR / name)
    overrides = eval_overrides(cfg)
    assert overrides["score_matching_lie_group"] is expected_lie
    assert overrides["num_se3_steps_inference"] == 100


def test_unrecognized_keys_are_silently_dropped():
    """Override projection should be a strict subset of supported flags;
    unrecognized YAML keys must not blow up upstream draccus."""
    cfg = load_config(CONFIG_DIR / "oft_lie_sm_libero10.yaml")
    cfg["completely_made_up_key"] = "irrelevant"
    train = train_overrides(cfg)
    eval_ = eval_overrides(cfg)
    assert "completely_made_up_key" not in train
    assert "completely_made_up_key" not in eval_
