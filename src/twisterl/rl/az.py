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
        device = self.config["device"]
        N = len(obs)

        # Sparse observation format: flat indices + offsets
        all_indices = []
        offsets = []
        for obs_i in obs:
            offsets.append(len(all_indices))
            all_indices.extend(obs_i)
        pt_indices = torch.tensor(all_indices, dtype=torch.long, device=device)
        pt_offsets = torch.tensor(offsets, dtype=torch.long, device=device)

        pt_probs = torch.tensor(probs, dtype=torch.float, device=device)
        pt_vals = torch.tensor(vals, dtype=torch.float, device=device).unsqueeze(1)

        return pt_indices, pt_offsets, N, pt_probs, pt_vals

    @timed
    def train_step(self, torch_data):
        pt_indices, pt_offsets, N, pt_probs, pt_vals = torch_data

        pred_probs, pred_vals = self.policy.forward_sparse(pt_indices, pt_offsets, N)

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
