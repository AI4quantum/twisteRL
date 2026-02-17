# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import importlib
import inspect
import json
import torch
import numpy as np
from huggingface_hub import HfApi, snapshot_download
import fnmatch
from loguru import logger
from safetensors.torch import load_file, save_file


def dynamic_import(path):
    try:
        module_name, attr_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{path}'. Error: {e}")


def json_load_tuples(dct):
    if "__tuple_list__" in dct:
        return [tuple(item) for item in dct["list"]]
    return dct


def validate_algorithm_from_hub(repo_id: str, revision: str = "main"):
    """Validate that a Hugging Face repository contains required model files.

        Parameters
    ----------
    repo_id : str
        Hugging Face repository identifier (e.g., "username/model-name" or an organization repo).

    revision : str, optional
        Git revision, branch, tag, or commit hash to inspect. Defaults to "main".

    Returns
    -------
    `dict`
        A dictionary with keys:
        - "is_valid" (bool): **True** if all required file patterns are present, **False** otherwise.
        - "missing" (list): list of required patterns that were not found; if the repo cannot be reached
          returns ["<repo not found>"].
    """

    # List of required file patterns to validate model
    # Accept either safetensors (preferred) or pt (legacy)
    REQUIRED_JSON = "*.json"
    CHECKPOINT_PATTERNS = ["*.safetensors", "*.pt"]
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, revision=revision)
    except Exception:
        return {"is_valid": False, "missing": ["<repo not found>"]}
    files_set = set(files)
    missing = []

    # Check for JSON config files
    if not any(fnmatch.fnmatch(file, REQUIRED_JSON) for file in files_set):
        missing.append(REQUIRED_JSON)

    # Check for checkpoint files (either safetensors or pt)
    has_checkpoint = any(
        fnmatch.fnmatch(file, pattern)
        for file in files_set
        for pattern in CHECKPOINT_PATTERNS
    )
    if not has_checkpoint:
        missing.append("*.safetensors or *.pt")

    is_valid = len(missing) == 0
    return {"is_valid": is_valid, "missing": missing}


def pull_hub_algorithm(
    repo_id, model_path="../models/", revision="main", validate=False
):
    """ ""Download a model/algorithm snapshot from a Hugging Face repo into a local cache.

    Parameters
    ----------
    repo_id : str
        Hugging Face repository identifier (e.g., "username/model-name" or an organization repo).

    model_path : str optional
        Local cache directory to store the snapshot. Defaults to "../models/".

    revision : str optional
        Git revision, branch, tag, or commit hash to download. Defaults to "main".

    validate : bool, optional
        If True, run validate_algorithm_from_hub(repo_id) before downloading; abort and return False if validation fails.

    Returns
    -------
    `str` or `bool`
        On success returns the local filesystem path to the downloaded snapshot directory.
        On validation or download failure returns False.
    """
    if validate:
        validate_algorithm = validate_algorithm_from_hub(repo_id)
        if not validate_algorithm["is_valid"]:
            logger.warning(f"Missing model files: {validate_algorithm['missing']}")
            return False
    try:
        local_repo_path = snapshot_download(
            repo_id,
            cache_dir=model_path,
            allow_patterns=["*.json", "*.safetensors", "*.pt"],
            revision=revision,
            force_download=False,
        )
        logger.info(f"Model files are now in: {local_repo_path}")
        return local_repo_path
    except Exception as e:
        logger.warning(f"Error: {e}")
        return False


def load_checkpoint(checkpoint_path):
    """Load a checkpoint from either safetensors or pt format.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.safetensors or .pt).

    Returns
    -------
    dict
        The state dict loaded from the checkpoint.
    """
    if checkpoint_path.endswith(".safetensors"):
        return load_file(checkpoint_path, device="cpu")
    elif checkpoint_path.endswith(".pt"):
        logger.warning(
            f"Loading legacy .pt checkpoint: {checkpoint_path}. "
            "Consider converting to .safetensors using convert_pt_to_safetensors()."
        )
        return torch.load(open(checkpoint_path, "rb"), map_location=torch.device("cpu"))
    else:
        raise ValueError(
            f"Unknown checkpoint format: {checkpoint_path}. "
            "Expected .safetensors or .pt extension."
        )


def convert_pt_to_safetensors(pt_path, output_path=None):
    """Convert a PyTorch .pt checkpoint to safetensors format.

    Parameters
    ----------
    pt_path : str
        Path to the .pt checkpoint file.
    output_path : str, optional
        Path for the output .safetensors file.
        If not provided, replaces .pt extension with .safetensors.

    Returns
    -------
    str
        Path to the saved .safetensors file.
    """
    if output_path is None:
        if pt_path.endswith(".pt"):
            output_path = pt_path[:-3] + ".safetensors"
        else:
            output_path = pt_path + ".safetensors"

    state_dict = torch.load(open(pt_path, "rb"), map_location=torch.device("cpu"))
    save_file(state_dict, output_path)
    logger.info(f"Converted {pt_path} -> {output_path}")
    return output_path


def prepare_algorithm(config, run_path=None, load_checkpoint_path=None):
    # Import env class and make env
    env_cls = dynamic_import(config["env_cls"])
    env = env_cls(**config["env"])

    # Import policy class and make policy
    policy_cls = dynamic_import(config["policy_cls"])
    policy_kwargs = dict(config["policy"])
    action_space = getattr(env, "action_space", None)
    action_mode = str(policy_kwargs.get("action_mode", "categorical")).strip().lower()
    num_action_factors = policy_kwargs.get("num_action_factors", None)
    should_auto_factorize = (
        action_mode == "categorical"
        and (num_action_factors is None or int(num_action_factors) <= 0)
    )
    if (
        should_auto_factorize
        and action_space is not None
        and action_space.__class__.__name__ == "MultiBinary"
    ):
        n_bits = getattr(action_space, "n", None)
        if n_bits is None:
            shape = getattr(action_space, "shape", None)
            n_bits = int(np.prod(shape)) if shape is not None else 0
        else:
            n_bits = int(np.prod(np.asarray(n_bits)))
        if n_bits > 0:
            policy_kwargs["action_mode"] = "factorized_bernoulli"
            policy_kwargs["num_action_factors"] = n_bits

    # Backward compatibility: older policy classes may not accept the newest
    # config kwargs (e.g., action_mode / num_action_factors).
    sig = inspect.signature(policy_cls.__init__)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_kwargs:
        accepted = {name for name in sig.parameters if name != "self"}
        dropped = [k for k in list(policy_kwargs.keys()) if k not in accepted]
        for key in dropped:
            policy_kwargs.pop(key, None)
        if dropped:
            logger.warning(
                f"Dropping unsupported policy kwargs for {policy_cls.__name__}: {dropped}"
            )

    obs_perms, act_perms = env.twists()
    policy = policy_cls(
        env.obs_shape(),
        env.num_actions(),
        **policy_kwargs,
        obs_perms=obs_perms,
        act_perms=act_perms,
    )
    if load_checkpoint_path is not None:
        policy.load_state_dict(load_checkpoint(load_checkpoint_path))

    # Import algo class and make algorithm
    algo_cls = dynamic_import(config["algorithm_cls"])
    return algo_cls(env, policy, config["algorithm"], run_path)


def load_config(config_path):
    return json.load(open(config_path), object_hook=json_load_tuples)
