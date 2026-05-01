"""Smoke import test for lda_oft + the vendored prismatic tree.

Some prismatic modules pull in heavy deps (draccus, transformers fork,
flash-attn, peft, ...) that only the lda-oft env has installed. We
``importorskip`` those so this file passes from any working env, while
still failing loudly if the *intended* env ever loses the dep.
"""
import pytest


def test_lda_oft_top_level_imports():
    import lda_oft  # noqa: F401
    from lda_oft.config import (  # noqa: F401
        eval_overrides,
        load_config,
        train_overrides,
    )


def test_prismatic_action_heads_import_in_lda_oft_env():
    """Heavy import — needs lda-oft env (transformers, peft, draccus, ...)."""
    pytest.importorskip("draccus", reason="lda-oft env not installed")
    pytest.importorskip("peft", reason="lda-oft env not installed")
    from prismatic.models.action_heads import (  # noqa: F401
        DiffusionActionHead,
        L1RegressionActionHead,
        SE3ScoreMatchingActionHead,
    )


def test_prismatic_se3_helpers_import_in_lda_oft_env():
    pytest.importorskip("draccus", reason="lda-oft env not installed")
    pytest.importorskip("peft", reason="lda-oft env not installed")
    from prismatic.models.action_heads import (  # noqa: F401
        SE3PowerNoiseSchedule,
        se3_exp_map,
        se3_log_map,
    )


def test_lie_group_flag_smoke():
    """Bypass prismatic.models.__init__ — load action_heads.py directly so the
    smoke test works even before lda-oft env is installed."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[1]

    # Stub the small `prismatic.vla.constants` module that action_heads imports.
    constants = types.ModuleType("prismatic.vla.constants")
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
        "_action_heads_smoke", REPO / "prismatic" / "models" / "action_heads.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    head_lie = m.SE3ScoreMatchingActionHead(input_dim=4096, hidden_dim=4096, action_dim=7, lie_group=True)
    head_eu = m.SE3ScoreMatchingActionHead(input_dim=4096, hidden_dim=4096, action_dim=7, lie_group=False)
    assert set(head_lie.state_dict().keys()) == set(head_eu.state_dict().keys())
    assert head_lie.lie_group is True and head_eu.lie_group is False
