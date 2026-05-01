"""CPU-only checkpoint compatibility test.

For each (config, ckpt) pair in the paper table, instantiate the model
under that config and verify the ckpt's state_dict loads with strict=True.

Run only after `bash scripts/download_ckpts.sh`. Skips automatically if
ckpts are missing.

Note: this test intentionally uses ``strict=True`` to catch the silent
corruption case where a refactor renames a parameter-bearing attribute
or class. Any missing/unexpected keys = ckpt would not load = the cleanup
broke compatibility.
"""
import pytest

from _helpers import (
    CKPT_DIR, CONFIGS_DIR,
    build_namespace_from_config, build_model, load_ckpt_state_dict,
)

# (config_yaml, ckpt_filename) — paper-table rows.
PAIRS = [
    ("lda_abc_d.yaml", "abc_gat_score_newloss_300k.pth"),
    ("lda_abcd_d.yaml", "abcd_gat_score_newloss_300k.pth"),
    ("ablation_no_gat_abc_d.yaml", "abc_nogat_600k.pth"),
    ("ablation_no_gat_abcd_d.yaml", "abcd_nogat_300k.pth"),
    ("ablation_no_lie_abc_d.yaml", "abc_gat_ddpm_300k.pth"),
    ("ablation_no_lie_abcd_d.yaml", "abcd_ddpm_300k.pth"),
]


@pytest.mark.parametrize("cfg_name,ckpt_name", PAIRS)
def test_ckpt_loads_strict(cfg_name, ckpt_name):
    cfg_path = CONFIGS_DIR / cfg_name
    ckpt_path = CKPT_DIR / ckpt_name

    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not downloaded: {ckpt_path}")

    args = build_namespace_from_config(cfg_path)
    model = build_model(args)
    state_dict = load_ckpt_state_dict(ckpt_path)

    # strict=False so we surface a structured diff; strict=True would raise
    # before we get a chance to print which keys disagreed.
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    assert not missing, (
        f"Missing keys when loading {ckpt_name} into {cfg_name}: "
        f"{missing[:10]} (total {len(missing)})"
    )
    assert not unexpected, (
        f"Unexpected keys when loading {ckpt_name} into {cfg_name}: "
        f"{unexpected[:10]} (total {len(unexpected)})"
    )
