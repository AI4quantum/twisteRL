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

import torch
import numpy as np

from twisterl.rl.algorithm import Algorithm, timed
from twisterl import twisterl


class AZ(Algorithm):
    def __init__(self, env, policy, config, run_path=None):
        super().__init__(env, policy, config, run_path)
        self.collector = twisterl.collector.AZCollector(**self.config["collecting"])
        if hasattr(self.policy, "set_permutations"):
            self.policy.set_permutations([], [])
            self.sync_rs_policy()

    @timed
    def data_to_torch(self, data):
        obs, probs, vals = (
            data.obs,
            data.logits,
            data.additional_data["remaining_values"],
        )

        # Vectorized one-hot encoding (Optimization 1)
        n_samples = len(obs)
        obs_lengths = [len(o) for o in obs]
        row_indices = np.repeat(np.arange(n_samples), obs_lengths)
        col_indices = np.concatenate(obs).astype(int) if sum(obs_lengths) > 0 else np.array([], dtype=int)
        np_obs = np.zeros((n_samples, self.obs_size), dtype=np.float32)  # Optimization 2: float32
        if len(col_indices) > 0:
            np_obs[row_indices, col_indices] = 1.0

        pt_obs = torch.from_numpy(np_obs).to(self.config["device"])  # Optimization 3: from_numpy
        pt_probs = torch.tensor(probs, dtype=torch.float, device=self.config["device"])
        pt_vals = torch.tensor(
            vals, dtype=torch.float, device=self.config["device"]
        ).unsqueeze(1)

        return pt_obs, pt_probs, pt_vals

    @timed
    def train_step(self, torch_data):
        pt_obs, pt_probs, pt_vals = torch_data

        pred_probs, pred_vals = self.policy(pt_obs)

        policy_loss = torch.nn.functional.cross_entropy(pred_probs, pt_probs)
        value_loss = torch.nn.functional.mse_loss(pred_vals, pt_vals)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "total": loss.item(),
        }
