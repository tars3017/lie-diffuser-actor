"""Thin wrapper: YAML config → experiments/robot/libero/run_libero_eval.py argv.

Usage:
    python -m lda_oft.eval \\
        --config lda_oft/configs/oft_lie_sm_libero10.yaml \\
        --pretrained_checkpoint /path/to/...150000_chkpt
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

from lda_oft.config import eval_overrides, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval.py"


def build_argv(
    config: Path,
    pretrained_checkpoint: str,
    *,
    extra: Sequence[str] = (),
) -> list[str]:
    cfg = load_config(config)
    overrides = eval_overrides(cfg)

    argv = [
        str(EVAL_SCRIPT),
        "--model_family", "openvla",
        "--pretrained_checkpoint", pretrained_checkpoint,
        "--task_suite_name", cfg["task_suite"],
    ]
    for k, v in overrides.items():
        argv.extend([f"--{k}", str(v)])
    argv.extend(extra)
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML in lda_oft/configs/")
    parser.add_argument("--pretrained_checkpoint", required=True, type=str,
                        help="Path or HF ID of the checkpoint dir to evaluate.")
    args, extra = parser.parse_known_args()

    argv = build_argv(
        config=args.config,
        pretrained_checkpoint=args.pretrained_checkpoint,
        extra=extra,
    )
    sys.argv = argv
    runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
