import json
import torch
import numpy as np
from pathlib import Path

from twisterl.utils import load_config, prepare_algorithm, pull_hub_algorithm
from twisterl.defaults import make_config
from twisterl.nn.utils import sequential_to_rust, embeddingbag_to_rust
from twisterl.nn.policy import BasicPolicy, Conv1dPolicy, Transpose
from twisterl.rl.ppo import PPO
from twisterl.rl.az import AZ
from twisterl.defaults import PPO_CONFIG, AZ_CONFIG


class DummyEnv:
    def __init__(self, size=3):
        self.size = size
        self.difficulty = 0

    def twists(self):
        return [], []

    def obs_shape(self):
        return [self.size]

    def num_actions(self):
        return self.size

    def set_state(self, state):
        self.state = state


def test_load_config(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"t": {"__tuple_list__": True, "list": [[1, 2]]}}))
    cfg = load_config(p)
    assert cfg["t"] == [(1, 2)]


def test_make_config():
    cfg = make_config("PPO", {"policy": {"embedding_size": 128}})
    assert cfg["policy"]["embedding_size"] == 128
    assert cfg["optimizer"]["lr"] == 0.0003


def test_prepare_algorithm():
    config = {
        "env_cls": f"{__name__}.DummyEnv",
        "policy_cls": "twisterl.nn.policy.BasicPolicy",
        "algorithm_cls": "twisterl.rl.ppo.PPO",
        "env": {"size": 3},
        "policy": {
            "embedding_size": 4,
            "common_layers": [],
            "policy_layers": [],
            "value_layers": [],
            "device": "cpu",
        },
        "algorithm": {},
    }
    algo = prepare_algorithm(config)
    assert isinstance(algo.env, DummyEnv)
    assert isinstance(algo.policy, BasicPolicy)
    assert isinstance(algo, PPO)


def test_sequential_and_embeddingbag_to_rust():
    seq = torch.nn.Sequential(
        torch.nn.Linear(3, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
    )
    rs_seq = sequential_to_rust(seq)
    assert rs_seq.__class__.__name__ == "Sequential"

    linear = torch.nn.Linear(3, 2)
    rs_eb = embeddingbag_to_rust(linear, [3], 0)
    assert rs_eb.__class__.__name__ == "EmbeddingBag"


def _make_policy():
    return BasicPolicy(
        [3],
        2,
        embedding_size=4,
        common_layers=(),
        policy_layers=(2,),
        value_layers=(),
        device="cpu",
    )


def test_basic_policy_forward_and_to_rust():
    pol = _make_policy()
    x = torch.randn(1, 3)
    logits, value = pol(x)
    assert logits.shape == (1, 2)
    assert value.shape == (1, 1)
    rs_pol = pol.to_rust()
    assert rs_pol.__class__.__name__ == "Policy"


def test_transpose_module():
    t = Transpose()
    x = torch.randn(1, 2, 3)
    y = t(x)
    assert y.shape == (1, 3, 2)


def test_conv1d_policy_forward_to_rust():
    pol = Conv1dPolicy(
        [2, 3],
        4,
        embedding_size=6,
        conv_dim=0,
        common_layers=(),
        policy_layers=(4,),
        value_layers=(),
        obs_perms=(),
        act_perms=(),
    )
    x = torch.randn(1, 2, 3)
    logits, val = pol(x)
    assert logits.shape == (1, 4)
    assert val.shape == (1, 1)
    rs_pol = pol.to_rust()
    assert rs_pol.__class__.__name__ == "Policy"


class DummyPPOData:
    def __init__(self):
        self.obs = [[0, 1]]
        self.logits = [[0.0, 0.0]]
        self.values = [0.0]
        self.rewards = [0.0]
        self.actions = [0]
        self.additional_data = {"rets": [0.0], "advs": [0.0]}


class DummyAZData:
    def __init__(self):
        self.obs = [[0, 1]]
        self.logits = [[0.5, 0.5]]
        self.additional_data = {"remaining_values": [0.0]}


def _make_ppo():
    env = DummyEnv()
    pol = _make_policy()
    cfg = {
        "device": "cpu",
        "collecting": PPO_CONFIG["collecting"],
        "training": {**PPO_CONFIG["training"], "num_epochs": 1},
        "optimizer": PPO_CONFIG["optimizer"],
    }
    return PPO(env, pol, cfg)


def _make_az():
    env = DummyEnv()
    pol = _make_policy()
    collecting = AZ_CONFIG["collecting"].copy()
    collecting.pop("seed", None)
    cfg = {
        "device": "cpu",
        "collecting": collecting,
        "training": {**AZ_CONFIG["training"], "num_epochs": 1},
        "optimizer": AZ_CONFIG["optimizer"],
    }
    return AZ(env, pol, cfg)


def test_ppo_data_to_torch_and_train_step():
    algo = _make_ppo()
    data = DummyPPOData()
    torch_data, _ = algo.data_to_torch(data)
    metrics, _ = algo.train_step(torch_data)
    assert "total" in metrics


def test_az_data_to_torch_and_train_step():
    algo = _make_az()
    data = DummyAZData()
    torch_data, _ = algo.data_to_torch(data)
    metrics, _ = algo.train_step(torch_data)
    assert "total" in metrics


def _reference_one_hot_encoding(obs, obs_size):
    """Reference implementation of the original one-hot encoding (for testing)."""
    np_obs = np.zeros((len(obs), obs_size), dtype=float)
    for i, obs_i in enumerate(obs):
        np_obs[i, obs_i] = 1.0
    return np_obs


def _vectorized_one_hot_encoding(obs, obs_size):
    """Vectorized one-hot encoding (optimized implementation)."""
    n_samples = len(obs)
    obs_lengths = [len(o) for o in obs]
    row_indices = np.repeat(np.arange(n_samples), obs_lengths)
    col_indices = np.concatenate(obs).astype(int) if sum(obs_lengths) > 0 else np.array([], dtype=int)
    np_obs = np.zeros((n_samples, obs_size), dtype=np.float32)
    if len(col_indices) > 0:
        np_obs[row_indices, col_indices] = 1.0
    return np_obs


def test_one_hot_encoding_single_observation():
    """Test one-hot encoding with a single observation."""
    obs = [[0, 2]]
    obs_size = 4
    reference = _reference_one_hot_encoding(obs, obs_size)
    optimized = _vectorized_one_hot_encoding(obs, obs_size)
    np.testing.assert_array_equal(reference, optimized)
    expected = np.array([[1.0, 0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(optimized, expected)


def test_one_hot_encoding_multiple_observations():
    """Test one-hot encoding with multiple observations of varying lengths."""
    obs = [[0, 1], [2], [0, 1, 2, 3]]
    obs_size = 5
    reference = _reference_one_hot_encoding(obs, obs_size)
    optimized = _vectorized_one_hot_encoding(obs, obs_size)
    np.testing.assert_array_equal(reference, optimized)
    expected = np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ])
    np.testing.assert_array_equal(optimized, expected)


def test_one_hot_encoding_empty_observations():
    """Test one-hot encoding with empty observation list."""
    obs = []
    obs_size = 4
    reference = _reference_one_hot_encoding(obs, obs_size)
    optimized = _vectorized_one_hot_encoding(obs, obs_size)
    assert reference.shape == (0, 4)
    assert optimized.shape == (0, 4)


def test_one_hot_encoding_empty_single_obs():
    """Test one-hot encoding with an observation containing no indices."""
    obs = [[], [1, 2]]
    obs_size = 4
    reference = _reference_one_hot_encoding(obs, obs_size)
    optimized = _vectorized_one_hot_encoding(obs, obs_size)
    np.testing.assert_array_equal(reference, optimized)
    expected = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    np.testing.assert_array_equal(optimized, expected)


def test_one_hot_encoding_dtype():
    """Test that optimized version produces float32 dtype."""
    obs = [[0, 1]]
    obs_size = 3
    optimized = _vectorized_one_hot_encoding(obs, obs_size)
    assert optimized.dtype == np.float32


class LargeDummyPPOData:
    """Larger test data for PPO to test vectorized implementation."""
    def __init__(self, n_samples=100, obs_size=10):
        self.obs = [list(np.random.choice(obs_size, size=np.random.randint(1, obs_size), replace=False))
                    for _ in range(n_samples)]
        self.logits = np.random.randn(n_samples, 2).tolist()
        self.values = np.random.randn(n_samples).tolist()
        self.rewards = np.random.randn(n_samples).tolist()
        self.actions = np.random.randint(0, 2, n_samples).tolist()
        self.additional_data = {
            "rets": np.random.randn(n_samples).tolist(),
            "advs": np.random.randn(n_samples).tolist(),
        }


class LargeDummyAZData:
    """Larger test data for AZ to test vectorized implementation."""
    def __init__(self, n_samples=100, obs_size=10):
        self.obs = [list(np.random.choice(obs_size, size=np.random.randint(1, obs_size), replace=False))
                    for _ in range(n_samples)]
        self.logits = np.random.randn(n_samples, 2).tolist()
        self.additional_data = {
            "remaining_values": np.random.randn(n_samples).tolist(),
        }


def test_ppo_data_to_torch_large_batch():
    """Test PPO data_to_torch with larger batch to verify vectorized implementation."""
    algo = _make_ppo()
    data = LargeDummyPPOData(n_samples=50, obs_size=3)
    torch_data, _ = algo.data_to_torch(data)
    pt_obs, pt_log_probs, pt_acts, pt_advs, pt_rets, pt_perm_idx = torch_data

    # Verify shapes
    assert pt_obs.shape == (50, 3)
    assert pt_log_probs.shape == (50,)
    assert pt_acts.shape == (50,)
    assert pt_advs.shape == (50,)
    assert pt_rets.shape == (50,)

    # Verify obs is proper one-hot (each row sums to number of active indices)
    obs_sums = pt_obs.sum(dim=1).cpu().numpy()
    expected_sums = [len(o) for o in data.obs]
    np.testing.assert_array_equal(obs_sums, expected_sums)

    # Verify dtype
    assert pt_obs.dtype == torch.float32


def test_az_data_to_torch_large_batch():
    """Test AZ data_to_torch with larger batch to verify vectorized implementation."""
    algo = _make_az()
    data = LargeDummyAZData(n_samples=50, obs_size=3)
    torch_data, _ = algo.data_to_torch(data)
    pt_obs, pt_probs, pt_vals = torch_data

    # Verify shapes
    assert pt_obs.shape == (50, 3)
    assert pt_probs.shape == (50, 2)
    assert pt_vals.shape == (50, 1)

    # Verify obs is proper one-hot (each row sums to number of active indices)
    obs_sums = pt_obs.sum(dim=1).cpu().numpy()
    expected_sums = [len(o) for o in data.obs]
    np.testing.assert_array_equal(obs_sums, expected_sums)

    # Verify dtype
    assert pt_obs.dtype == torch.float32


class DummyHubModelHandler:
    def __init__(
        self,
        repo_id="Qiskit/example",
        model_path="../models/",
        revision="main",
        validate=False,
    ):
        self.repo_id = repo_id
        self.model_path = model_path
        self.revision = revision
        self.validate = validate


def fake_snapshot_download(
    repo_id, cache_dir, allow_patterns, revision, force_download
):
    snapshot_folder = (
        Path(cache_dir)
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    snapshot_folder.mkdir(parents=True, exist_ok=True)
    return str(snapshot_folder)


def test_pull_new_hub_model(mocker, tmp_path):
    dummy_hub = DummyHubModelHandler()
    cache_dir = tmp_path / "snapshot_cache"
    cache_dir.mkdir()
    mocker.patch(
        "twisterl.utils.snapshot_download",
        autospec=True,
        side_effect=lambda repo_id=dummy_hub.repo_id, cache_dir=None, allow_patterns=None, revision=None, force_download=False: fake_snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            revision=revision,
            force_download=force_download,
        ),
    )
    result = pull_hub_algorithm(
        repo_id=dummy_hub.repo_id,
        model_path=dummy_hub.model_path,
        revision=dummy_hub.revision,
        validate=dummy_hub.validate,
    )
    assert result is not False
