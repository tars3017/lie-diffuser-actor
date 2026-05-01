import pytest
from lda.utils.config import load_config, to_cli_args


def test_load_simple(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("lr: 3e-4\nbatch_size: 4\nuse_gat: true\n")
    cfg = load_config(p)
    assert cfg == {"lr": 3e-4, "batch_size": 4, "use_gat": True}


def test_to_cli_args_basic():
    cfg = {"lr": 3e-4, "batch_size": 4, "use_gat": True}
    args = to_cli_args(cfg)
    assert args == ["--lr", "0.0003", "--batch_size", "4", "--use_gat"]


def test_to_cli_args_false_bool_omitted():
    cfg = {"foo": True, "bar": False}
    args = to_cli_args(cfg)
    assert args == ["--foo"]


def test_load_rejects_nested(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  use_gat: true\n")
    with pytest.raises(ValueError, match="must be flat"):
        load_config(p)


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")
