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


class PPO(Algorithm):
    def __init__(self, env, policy, config, run_path=None):
        super().__init__(env, policy, config, run_path)
        self.collector = twisterl.collector.PPOCollector(**self.config["collecting"])

    @timed
    def data_to_torch(self, data):
        obs, logits, _, _, acts, rets, advs, perms = (
            data.obs,
            data.logits,
            data.values,
            data.rewards,
            data.actions,
            data.additional_data["rets"],
            data.additional_data["advs"],
            getattr(data, "perms", [-1] * len(data.obs)),
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

        pt_logits = torch.tensor(logits, dtype=torch.float, device=device)
        pt_acts = torch.tensor(acts, dtype=torch.long, device=device)
        pt_rets = torch.tensor(rets, dtype=torch.float, device=device)
        pt_advs = torch.tensor(advs, dtype=torch.float, device=device)
        pt_perm_idx = torch.tensor(perms, dtype=torch.long, device=device)

        with torch.no_grad():
            if self.config["training"].get("normalize_advantage", False):
                pt_advs = (pt_advs - pt_advs.mean()) / (pt_advs.std() + 1e-8)
            pt_log_probs = torch.distributions.Categorical(logits=pt_logits).log_prob(
                pt_acts
            )

        return pt_indices, pt_offsets, N, pt_log_probs, pt_acts, pt_advs, pt_rets, pt_perm_idx

    @timed
    def train_step(self, torch_data):
        pt_indices, pt_offsets, N, pt_log_probs, pt_acts, pt_advs, pt_rets, pt_perm_idx = torch_data

        # Forward pass using sparse observations
        pred_logits, pred_vals = self.policy.forward_sparse(
            pt_indices, pt_offsets, N, perm_indices=pt_perm_idx
        )

        # Get log-probabilities of the actions actually taken
        dist = torch.distributions.Categorical(logits=pred_logits)
        pred_log_probs = dist.log_prob(pt_acts)

        # Entropy of the action distribution
        entropy = dist.entropy()
        entropy_loss = entropy.mean()

        # Compute the ratio between new and old policy probabilities
        ratios = torch.exp(pred_log_probs - pt_log_probs)

        # Compute the PPO clipped objective
        surr1 = ratios * pt_advs
        surr2 = (
            torch.clamp(
                ratios,
                1 - self.config["training"]["clip_ratio"],
                1 + self.config["training"]["clip_ratio"],
            )
            * pt_advs
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = torch.nn.functional.mse_loss(pred_vals, pt_rets.unsqueeze(1))

        # Combined loss
        loss = (
            policy_loss
            + self.config["training"]["vf_coef"] * value_loss
            - self.config["training"]["ent_coef"] * entropy_loss
        )

        # Update the policy via gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "entropy": entropy_loss.item(),
            "total": loss.item(),
        }
