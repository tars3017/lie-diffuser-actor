"""Thin wrapper: YAML config → vla-scripts/finetune.py argv.

Usage:
    python -m lda_oft.train \\
        --config lda_oft/configs/oft_lie_sm_libero10.yaml \\
        --data_root_dir /path/to/modified_libero_rlds \\
        --run_root_dir /path/to/output

Pass-through extras (anything else after the recognised flags) are forwarded
verbatim to finetune.py — useful for things like ``--wandb_entity foo``.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

from lda_oft.config import load_config, train_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNE = REPO_ROOT / "vla-scripts" / "finetune.py"


def build_argv(
    config: Path,
    data_root_dir: Path,
    run_root_dir: Path,
    *,
    dataset_name: str | None = None,
    wandb_entity: str = "",
    wandb_project: str = "openvla-oft",
    extra: Sequence[str] = (),
) -> list[str]:
    """Construct the argv that would be passed to finetune.py."""
    cfg = load_config(config)
    overrides = train_overrides(cfg)
    dataset_name = dataset_name or cfg["unnorm_key"]

    argv = [
        str(FINETUNE),
        "--data_root_dir", str(data_root_dir),
        "--dataset_name", dataset_name,
        "--run_root_dir", str(run_root_dir),
        "--wandb_entity", wandb_entity,
        "--wandb_project", wandb_project,
    ]
    for k, v in overrides.items():
        argv.extend([f"--{k}", str(v)])
    argv.extend(extra)
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML in lda_oft/configs/")
    parser.add_argument("--data_root_dir", required=True, type=Path)
    parser.add_argument("--run_root_dir", required=True, type=Path)
    parser.add_argument("--dataset_name", default=None,
                        help="Override the YAML's `unnorm_key` value used as dataset_name.")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_project", default="openvla-oft")
    args, extra = parser.parse_known_args()

    argv = build_argv(
        config=args.config,
        data_root_dir=args.data_root_dir,
        run_root_dir=args.run_root_dir,
        dataset_name=args.dataset_name,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        extra=extra,
    )
    sys.argv = argv
    runpy.run_path(str(FINETUNE), run_name="__main__")


if __name__ == "__main__":
    main()
