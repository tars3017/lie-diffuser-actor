"""TDD: train and eval wrappers produce the expected argv vectors."""
from pathlib import Path

import pytest

from lda_oft import eval as eval_mod
from lda_oft import train as train_mod

CONFIG_DIR = Path(__file__).resolve().parents[1] / "lda_oft" / "configs"


def test_train_argv_baseline_has_l1_flag(tmp_path):
    argv = train_mod.build_argv(
        config=CONFIG_DIR / "oft_baseline_libero10.yaml",
        data_root_dir=tmp_path / "data",
        run_root_dir=tmp_path / "out",
    )
    assert "--use_l1_regression" in argv
    i = argv.index("--use_l1_regression")
    assert argv[i + 1] == "True"
    assert "--use_se3_score_matching" in argv
    j = argv.index("--use_se3_score_matching")
    assert argv[j + 1] == "False"
    assert "--score_matching_lie_group" not in argv  # baseline doesn't carry the flag


def test_train_argv_lie_sets_lie_group_true(tmp_path):
    argv = train_mod.build_argv(
        config=CONFIG_DIR / "oft_lie_sm_libero10.yaml",
        data_root_dir=tmp_path / "data",
        run_root_dir=tmp_path / "out",
    )
    assert "--use_se3_score_matching" in argv
    assert argv[argv.index("--score_matching_lie_group") + 1] == "True"


def test_train_argv_euclidean_sets_lie_group_false(tmp_path):
    argv = train_mod.build_argv(
        config=CONFIG_DIR / "oft_euclidean_sm_libero10.yaml",
        data_root_dir=tmp_path / "data",
        run_root_dir=tmp_path / "out",
    )
    assert argv[argv.index("--score_matching_lie_group") + 1] == "False"


def test_train_extra_args_pass_through(tmp_path):
    argv = train_mod.build_argv(
        config=CONFIG_DIR / "oft_lie_sm_libero10.yaml",
        data_root_dir=tmp_path / "data",
        run_root_dir=tmp_path / "out",
        extra=["--learning_rate", "1e-5"],   # later --learning_rate wins by argparse convention
    )
    # The override appears at the end (after our YAML-derived flags) so the trailing one wins.
    last = max(i for i, a in enumerate(argv) if a == "--learning_rate")
    assert argv[last + 1] == "1e-5"


def test_eval_argv_lie_includes_se3_inference_steps():
    argv = eval_mod.build_argv(
        config=CONFIG_DIR / "oft_lie_sm_libero10.yaml",
        pretrained_checkpoint="/some/local/ckpt",
    )
    assert argv[1] == "--model_family" and argv[2] == "openvla"
    assert "--pretrained_checkpoint" in argv
    assert argv[argv.index("--task_suite_name") + 1] == "libero_10"
    assert argv[argv.index("--num_se3_steps_inference") + 1] == "100"
    assert argv[argv.index("--score_matching_lie_group") + 1] == "True"


def test_eval_argv_baseline_omits_se3_flags():
    argv = eval_mod.build_argv(
        config=CONFIG_DIR / "oft_baseline_libero10.yaml",
        pretrained_checkpoint="/some/local/ckpt",
    )
    assert "--num_se3_steps_inference" not in argv
    assert "--score_matching_lie_group" not in argv
    assert argv[argv.index("--use_l1_regression") + 1] == "True"
