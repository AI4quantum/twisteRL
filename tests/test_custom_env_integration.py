import functools
import importlib
import os
import subprocess
import sys
from pathlib import Path

from twisterl.utils import load_config, prepare_algorithm


@functools.lru_cache(maxsize=1)
def _ensure_grid_world_available():
    """Install the external grid_world example if it is not already importable."""
    module_name = "grid_world"
    try:
        importlib.import_module(module_name)
        return module_name
    except ModuleNotFoundError:
        pass

    # pytest.importorskip(
    #     "maturin", reason="Grid World example needs maturin to build the extension"
    # )
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "grid_world"
    env = os.environ.copy()
    venv_bin = Path(sys.executable).resolve().parent
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=str(example_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    importlib.invalidate_caches()
    importlib.import_module(module_name)
    return module_name


def test_grid_world_external_env_works_with_twisterl():
    module_name = _ensure_grid_world_available()

    config_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "grid_world"
        / "ppo_grid_world_5x5_v1.json"
    )
    algo_config = load_config(config_path)
    algo_cfg = algo_config["algorithm"]
    algo_cfg["collecting"].update({"num_cores": 1, "num_episodes": 1})

    grid_world_module = importlib.import_module(module_name)
    GridWorld = grid_world_module.GridWorld

    env_args = algo_config["env"].copy()
    env = GridWorld(**env_args)
    env.reset()
    # Observations & states should match the board size
    state = env.get_state()
    expected_cells = env_args["width"] * env_args["height"]
    assert len(state) == expected_cells
    assert {0, 1, 2, 3}.issuperset(set(state))

    algo = prepare_algorithm(algo_config)

    # Ensure the basic interactions work as expected
    algo.env.reset()
    observed = algo.env.observe()
    assert len(observed) == algo.env.obs_shape()[0]

    collected, _ = algo.collect()
    assert collected.obs
    assert collected.additional_data["rets"]
