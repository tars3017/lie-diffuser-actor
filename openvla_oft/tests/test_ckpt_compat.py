"""CPU action-head ckpt-compat gate.

Verifies each released action_head checkpoint loads with strict=True against the
class instantiated from the corresponding YAML config. The full 7B base model
load is deferred to a Phase-2 GPU smoke test — this file only catches silent
class/attr renames in the action heads, which are the components we touched.

The vendored ``prismatic`` package's ``__init__.py`` pulls in heavy deps
(draccus, peft, transformers fork) that only the lda-oft env has. We bypass
those by loading ``action_heads.py`` directly via importlib, the same trick used
in tests/test_imports.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Tuple

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
# Override on hosts that store local 150K ckpts elsewhere:
#   LDA_OFT_CKPT_ROOT=/path/to/checkpoints pytest tests/
# Default ("ckpts/") matches scripts/download_ckpts.sh; per-pair tests skip
# automatically if the corresponding ckpt isn't present.
CKPT_ROOT = Path(os.environ.get("LDA_OFT_CKPT_ROOT", REPO / "ckpts"))
LIBERO_10_LORA = "openvla-7b+libero_10_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--150000_chkpt"

# (variant_name, ckpt_dir_name, action-head class name, lie_group flag value)
PAIRS = [
    ("baseline",     "openvla-7b-oft-finetuned-libero-10",                  "L1RegressionActionHead",     None),
    ("euclidean_sm", "openvla-7b-euclidean-finetuned-libero-10-lora",       "SE3ScoreMatchingActionHead", False),
    ("lie_sm",       "openvla-7b-lie-head-finetuned-libero-10-lora",        "SE3ScoreMatchingActionHead", True),
]


def _load_action_heads_module() -> types.ModuleType:
    """Import action_heads.py without triggering prismatic/__init__.py."""
    if "_action_heads_compat" in sys.modules:
        return sys.modules["_action_heads_compat"]

    constants = types.ModuleType("prismatic.vla.constants")
    # ACTION_DIM = 7 matches the LIBERO chunk_len=8 / action_dim=7 setup
    constants.ACTION_DIM = 7
    constants.ACTION_TOKEN_BEGIN_IDX = 0
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = 8
    constants.PROPRIO_DIM = 8
    constants.STOP_INDEX = 2
    sys.modules.setdefault("prismatic", types.ModuleType("prismatic"))
    sys.modules.setdefault("prismatic.vla", types.ModuleType("prismatic.vla"))
    sys.modules["prismatic.vla.constants"] = constants

    spec = importlib.util.spec_from_file_location(
        "_action_heads_compat", REPO / "prismatic" / "models" / "action_heads.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_action_heads_compat"] = m
    spec.loader.exec_module(m)
    return m


def _strip_ddp_prefix(sd: dict) -> dict:
    """Mirror the upstream load_component_state_dict helper:
    strip a leading 'module.' from every key (legacy DDP-saved ckpts)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _make_head(cls_name: str, lie_group: bool | None):
    m = _load_action_heads_module()
    cls = getattr(m, cls_name)
    if cls_name == "L1RegressionActionHead":
        return cls(input_dim=4096, hidden_dim=4096, action_dim=7)
    if cls_name == "SE3ScoreMatchingActionHead":
        kwargs = dict(input_dim=4096, hidden_dim=4096, action_dim=7)
        if lie_group is not None:
            kwargs["lie_group"] = lie_group
        return cls(**kwargs)
    raise NotImplementedError(cls_name)


@pytest.mark.parametrize("variant,ckpt_dir,head_cls,lie_group", PAIRS, ids=[p[0] for p in PAIRS])
def test_action_head_state_dict_loads_strict(variant, ckpt_dir, head_cls, lie_group):
    ckpt_path = CKPT_ROOT / ckpt_dir / LIBERO_10_LORA / "action_head--150000_checkpoint.pt"
    if not ckpt_path.exists():
        pytest.skip(f"ckpt not present on this box: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = _strip_ddp_prefix(raw)

    head = _make_head(head_cls, lie_group)
    missing, unexpected = head.load_state_dict(sd, strict=False)
    assert not missing, f"[{variant}] missing keys when loading {head_cls}: {sorted(missing)[:5]}…"
    assert not unexpected, f"[{variant}] unexpected keys when loading {head_cls}: {sorted(unexpected)[:5]}…"


def test_se3_ckpts_have_identical_key_sets():
    """The Euclidean and Lie ckpts must share an identical key set, so a single
    SE3ScoreMatchingActionHead class loads either one. Catches silent regressions
    where one variant gains/loses a parameter that the other doesn't."""
    eu = CKPT_ROOT / "openvla-7b-euclidean-finetuned-libero-10-lora" / LIBERO_10_LORA / "action_head--150000_checkpoint.pt"
    lie = CKPT_ROOT / "openvla-7b-lie-head-finetuned-libero-10-lora" / LIBERO_10_LORA / "action_head--150000_checkpoint.pt"
    if not (eu.exists() and lie.exists()):
        pytest.skip("local SM ckpts not present on this box")

    keys_eu = set(_strip_ddp_prefix(torch.load(eu, map_location="cpu", weights_only=True)).keys())
    keys_lie = set(_strip_ddp_prefix(torch.load(lie, map_location="cpu", weights_only=True)).keys())
    assert keys_eu == keys_lie, f"key set drift: only-eu={keys_eu - keys_lie}, only-lie={keys_lie - keys_eu}"
