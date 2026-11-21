import pytest
import torch
from safetensors.torch import load_file, save_file

from kiln.model.loader import WeightLoadError, load_model
from tests.model.tiny import make_tiny_hf_model


def _rewrite(path, fn):
    f = path / "model.safetensors"
    tensors = load_file(f)
    fn(tensors)
    save_file(tensors, f, metadata={"format": "pt"})


def test_missing_weight_reported(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    gone = ["model.layers.1.mlp.up_proj.weight", "model.norm.weight"]
    _rewrite(path, lambda t: [t.pop(k) for k in gone])
    with pytest.raises(WeightLoadError) as e:
        load_model(path)
    for k in gone:
        assert k in str(e.value)


def test_unexpected_weight_reported(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    _rewrite(path, lambda t: t.__setitem__("model.layers.0.foo", torch.zeros(3)))
    with pytest.raises(WeightLoadError, match="model.layers.0.foo"):
        load_model(path)
