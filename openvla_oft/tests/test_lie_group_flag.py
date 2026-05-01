"""Verify the lie_group flag does NOT change the parameter set of SE3ScoreMatchingActionHead.

Both the local Euclidean ckpt and the released Lie ckpt must load into the merged
class with strict=True. The flag may only branch runtime math, never register or
unregister Parameters/Buffers.

Note: importing ``prismatic.models.action_heads`` triggers ``prismatic/__init__.py``,
which pulls in the full upstream package (draccus / peft / transformers fork). That
chain only works in the lda-oft env. We ``importorskip`` so this file passes in
lighter envs; the always-runnable bypass smoke lives in ``test_imports.py``.
"""
import pytest

pytest.importorskip("draccus", reason="lda-oft env not installed")
pytest.importorskip("peft", reason="lda-oft env not installed")

import torch

from prismatic.models.action_heads import SE3ScoreMatchingActionHead


def _make(lie_group: bool) -> SE3ScoreMatchingActionHead:
    return SE3ScoreMatchingActionHead(
        input_dim=4096, hidden_dim=4096, action_dim=7, lie_group=lie_group
    )


def test_state_dict_keys_invariant_under_lie_group_flag():
    head_lie = _make(lie_group=True)
    head_eu = _make(lie_group=False)
    assert set(head_lie.state_dict().keys()) == set(head_eu.state_dict().keys()), (
        "lie_group flag must not change parameter names — would break ckpt-compat"
    )


def test_lie_group_default_is_true():
    head = SE3ScoreMatchingActionHead(input_dim=4096, hidden_dim=4096, action_dim=7)
    assert head.lie_group is True, "default must be True to match the released lie ckpt"


def test_lie_group_attr_is_python_bool_not_buffer():
    """self.lie_group must NOT be a Buffer/Parameter; it's a config flag, not state."""
    head = _make(lie_group=False)
    assert head.lie_group is False
    sd = head.state_dict()
    assert "lie_group" not in sd, "lie_group leaked into state_dict — would break strict=True load"


def test_sample_noisy_actions_shapes_match_across_flag():
    """Forward pass shapes must be identical regardless of lie_group."""
    actions = torch.randn(2, 8, 7)  # (B=2, chunk=8, action_dim=7)
    out_lie = _make(lie_group=True).sample_noisy_actions(actions)
    out_eu = _make(lie_group=False).sample_noisy_actions(actions)
    for k in ("noisy_actions", "noise_se3", "noise_grip", "diffusion_timestep_embeddings", "timesteps"):
        assert out_lie[k].shape == out_eu[k].shape, f"{k}: lie={out_lie[k].shape} vs eu={out_eu[k].shape}"
