import os
import tempfile

import numpy as np
import pytest
import torch

from twisterl.utils import (
    dynamic_import,
    json_load_tuples,
    load_checkpoint,
    convert_pt_to_safetensors,
)
from twisterl.nn.utils import make_sequential
from twisterl.nn.policy import BasicPolicy
from twisterl.rl.algorithm import timed


def test_dynamic_import():
    sqrt = dynamic_import("math.sqrt")
    assert sqrt(9) == 3


def test_json_load_tuples():
    d = {"__tuple_list__": True, "list": [[1, 2], [3, 4]]}
    assert json_load_tuples(d) == [(1, 2), (3, 4)]


def test_make_sequential():
    seq = make_sequential(3, (2, 1), final_relu=False)
    layers = list(seq)
    assert len(layers) == 3
    assert layers[-1].__class__.__name__ == "Linear"


def test_basic_policy_predict():
    policy = BasicPolicy(
        [3],
        2,
        embedding_size=4,
        common_layers=(),
        policy_layers=(2,),
        value_layers=(),
        device="cpu",
    )
    import torch

    with torch.no_grad():
        actions, value = policy.predict(np.array([0.1, 0.2, 0.3], dtype=float))
    assert actions.shape == (2,)
    assert np.isclose(actions.sum(), 1.0)
    assert value.shape == (1,)


def test_timed_decorator():
    @timed
    def add(x, y):
        return x + y

    result, elapsed = add(1, 2)
    assert result == 3
    assert elapsed >= 0


class TestCheckpointFormats:
    """Tests for safetensors and pt checkpoint loading/saving."""

    @pytest.fixture
    def sample_state_dict(self):
        """Create a simple state dict for testing."""
        return {
            "layer1.weight": torch.randn(4, 3),
            "layer1.bias": torch.randn(4),
            "layer2.weight": torch.randn(2, 4),
            "layer2.bias": torch.randn(2),
        }

    def test_load_checkpoint_safetensors(self, sample_state_dict):
        """Test loading a safetensors checkpoint."""
        from safetensors.torch import save_file

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file(sample_state_dict, f.name)
            temp_path = f.name

        try:
            loaded = load_checkpoint(temp_path)
            for key in sample_state_dict:
                assert key in loaded
                assert torch.allclose(sample_state_dict[key], loaded[key])
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_pt(self, sample_state_dict):
        """Test loading a legacy pt checkpoint (with warning)."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(sample_state_dict, f)
            temp_path = f.name

        try:
            loaded = load_checkpoint(temp_path)
            for key in sample_state_dict:
                assert key in loaded
                assert torch.allclose(sample_state_dict[key], loaded[key])
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_unknown_format(self):
        """Test that unknown formats raise ValueError."""
        with pytest.raises(ValueError, match="Unknown checkpoint format"):
            load_checkpoint("model.unknown")

    def test_convert_pt_to_safetensors(self, sample_state_dict):
        """Test converting a pt file to safetensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "model.pt")
            torch.save(sample_state_dict, open(pt_path, "wb"))

            output_path = convert_pt_to_safetensors(pt_path)

            assert output_path == os.path.join(tmpdir, "model.safetensors")
            assert os.path.exists(output_path)

            loaded = load_checkpoint(output_path)
            for key in sample_state_dict:
                assert key in loaded
                assert torch.allclose(sample_state_dict[key], loaded[key])

    def test_convert_pt_to_safetensors_custom_output(self, sample_state_dict):
        """Test converting with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "model.pt")
            output_path = os.path.join(tmpdir, "custom_name.safetensors")
            torch.save(sample_state_dict, open(pt_path, "wb"))

            result = convert_pt_to_safetensors(pt_path, output_path)

            assert result == output_path
            assert os.path.exists(output_path)
