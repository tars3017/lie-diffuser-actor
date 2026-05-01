"""Smoke test: every submodule imports without ImportError."""
import importlib
import pytest

MODULES = [
    "lda",
    "lda.engine",
    "lda.model",
    "lda.model.diffuser_actor",
    "lda.model.liepose_diffusion",
    "lda.diffusion.lie",
    "lda.diffusion.lie.dist.se3",
    "lda.diffusion.lie.dist.so3",
    "lda.diffusion.lie.metrics.se3",
    "lda.diffusion.lie.metrics.so3",
    "lda.diffusion.lie.noise.power",
    "lda.diffusion.lie.utils.ops",
    "lda.encoder",
    "lda.encoder.clip",
    "lda.encoder.encoder",
    "lda.encoder.layers",
    "lda.encoder.multihead_custom_attention",
    "lda.encoder.position_encodings",
    "lda.encoder.resnet",
    "lda.gat",
    "lda.gat.graph_planner",
    "lda.gat.models.gat_encoder",
    "lda.data",
    "lda.data.calvin",
    "lda.data.engine",
    "lda.data.utils",
    "lda.eval.evaluate_model",
    "lda.eval.evaluate_policy",
    "lda.utils.common",
    "lda.utils.calvin",
    "lda.utils.pytorch3d_transforms",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
