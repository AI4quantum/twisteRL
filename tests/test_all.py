import json
import torch
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
        return [self.size, self.size]

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
    assert cfg["observation_encoder"]["type"] == "multi_hot"


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
        [3, 3],
        2,
        embedding_size=4,
        common_layers=(),
        policy_layers=(2,),
        value_layers=(),
        device="cpu",
    )


def test_basic_policy_forward_and_to_rust():
    pol = _make_policy()
    x = torch.randn(1, 9)
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

class DummyHubModelHandler:
    def __init__(self, repo_id="Qiskit/example", model_path="../models/", revision="main", validate=False):
        self.repo_id = repo_id
        self.model_path = model_path
        self.revision = revision
        self.validate = validate
        
def fake_snapshot_download(repo_id, cache_dir, allow_patterns, revision, force_download):
    snapshot_folder = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    snapshot_folder.mkdir(parents=True, exist_ok=True)
    return str(snapshot_folder)

def test_pull_new_hub_model(mocker, tmp_path):
    dummy_hub = DummyHubModelHandler()
    cache_dir = tmp_path / "snapshot_cache"
    cache_dir.mkdir()
    mocker.patch('twisterl.utils.snapshot_download', autospec=True, side_effect=lambda repo_id = dummy_hub.repo_id, cache_dir=None, allow_patterns=None, revision=None, force_download=False: fake_snapshot_download(repo_id, cache_dir=cache_dir, allow_patterns=allow_patterns, revision=revision, force_download=force_download))
    result = pull_hub_algorithm(
        repo_id=dummy_hub.repo_id,
        model_path=dummy_hub.model_path,
        revision=dummy_hub.revision,
        validate=dummy_hub.validate
    )
    assert result != False
