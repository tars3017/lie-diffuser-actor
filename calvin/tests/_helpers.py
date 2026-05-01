"""Shared test helpers — instantiate DiffuserActor from a YAML config the
same way ``lda.trainer.get_model`` does, plus the DDP-stripping ckpt-load
that mirrors what online evaluation expects.
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
CKPT_DIR = REPO_ROOT / "ckpts"


def build_namespace_from_config(cfg_path):
    """Load YAML config -> argparse.Namespace consumable by DiffuserActor.__init__.

    Casts integer-valued YAML scalars to int, then backfills the
    ``lda.trainer.Arguments`` defaults that the YAML configs don't override
    (the trainer relies on ``tap.Tap`` to fill these in).
    """
    from lda.utils.config import load_config
    cfg = load_config(cfg_path)
    int_keys = {
        "use_gat", "use_instruction", "diffusion_timesteps",
        "num_history", "embedding_dim", "dense_interpolation",
        "interpolation_length", "fps_subsampling_factor",
        "lang_enhanced", "relative_action", "train_iters",
        "batch_size", "batch_size_val", "num_workers",
        "max_episode_length", "val_freq", "val_iters",
        "cache_size", "cache_size_val", "keypose_only",
    }
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])
    defaults = {
        "num_joints": 7,
        "num_vis_ins_attn_layers": 2,
        "noise_start": 1e-8,
        "noise_end": 1.0,
        "noise_power": 3.0,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return argparse.Namespace(**cfg)


def build_model(args):
    """Mirror ``lda.trainer.get_model`` exactly — resolve gripper_loc_bounds
    from its JSON path and call DiffuserActor with explicit kwargs.
    """
    from lda.model.diffuser_actor import DiffuserActor
    from lda.utils.common import get_gripper_loc_bounds

    bounds_path = args.gripper_loc_bounds
    if not Path(bounds_path).is_absolute():
        bounds_path = (REPO_ROOT / bounds_path).resolve()
    bounds = get_gripper_loc_bounds(
        str(bounds_path),
        task=args.tasks[0] if isinstance(args.tasks, list) else args.tasks,
        buffer=args.gripper_loc_bounds_buffer,
    )
    args.gripper_loc_bounds = bounds

    return DiffuserActor(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        use_instruction=bool(args.use_instruction),
        fps_subsampling_factor=args.fps_subsampling_factor,
        gripper_loc_bounds=args.gripper_loc_bounds,
        rotation_parametrization=args.rotation_parametrization,
        quaternion_format=args.quaternion_format,
        diffusion_timesteps=args.diffusion_timesteps,
        nhist=args.num_history,
        relative=bool(args.relative_action),
        lang_enhanced=bool(args.lang_enhanced),
        args=args,
    )


def load_ckpt_state_dict(ckpt_path):
    """Read a released ckpt's state_dict, peeling the DDP ``module.`` prefix
    that all paper-table-row checkpoints carry."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("weight", ckpt.get("state_dict", ckpt))
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd
