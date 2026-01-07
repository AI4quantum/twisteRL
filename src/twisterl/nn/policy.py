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

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from twisterl import twisterl
from twisterl.nn.utils import (
    embeddingbag_to_rust,
    make_sequential,
    sequential_to_rust,
)


class BasicPolicy(torch.nn.Module):
    def __init__(
        self,
        obs_shape: list[int],
        num_actions: int,
        embedding_size: int,
        common_layers=(256,),
        policy_layers=tuple(),
        value_layers=tuple(),
        obs_perms=tuple(),
        act_perms=tuple(),
        device="cuda",
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_size = np.prod(obs_shape)
        self.num_actions = num_actions
        self.embeddings = torch.nn.Linear(self.obs_size, embedding_size)
        self.device = device
        self._expects_conv_input = False

        in_size = embedding_size
        if len(common_layers) > 0:
            self.common = make_sequential(in_size, common_layers)
            in_size = common_layers[-1]
        else:
            self.common = torch.nn.Sequential()

        self.action = make_sequential(
            in_size, tuple(policy_layers) + (num_actions,), final_relu=False
        )
        self.value = make_sequential(
            in_size, tuple(value_layers) + (1,), final_relu=False
        )
        self.register_buffer(
            "_obs_perm_tensor", torch.empty((0, 0), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_obs_perm_inv_tensor",
            torch.empty((0, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_act_perm_tensor", torch.empty((0, 0), dtype=torch.long), persistent=False
        )

        self.set_permutations(obs_perms, act_perms)

    def set_permutations(
        self,
        obs_perms: Sequence[Sequence[int]],
        act_perms: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.obs_perms = [list(p) for p in obs_perms]
        if act_perms is None:
            self.act_perms = []
        else:
            self.act_perms = [list(p) for p in act_perms]

        if self.obs_perms and not self.act_perms:
            self.act_perms = [list(range(self.num_actions)) for _ in self.obs_perms]
        if self.act_perms and len(self.act_perms) != len(self.obs_perms):
            raise ValueError("obs_perms and act_perms must have the same length.")

        if self.obs_perms:
            obs_perm_array = np.array(self.obs_perms, dtype=np.int64)
            obs_perm_inv_array = np.argsort(obs_perm_array, axis=1)
            act_perm_array = np.array(self.act_perms, dtype=np.int64)
            self._obs_perm_tensor = torch.tensor(
                obs_perm_array, dtype=torch.long, device=self._obs_perm_tensor.device
            )
            self._obs_perm_inv_tensor = torch.tensor(
                obs_perm_inv_array,
                dtype=torch.long,
                device=self._obs_perm_inv_tensor.device,
            )
            self._act_perm_tensor = torch.tensor(
                act_perm_array, dtype=torch.long, device=self._act_perm_tensor.device
            )
        else:
            self._obs_perm_tensor = torch.empty(
                (0, 0), dtype=torch.long, device=self._obs_perm_tensor.device
            )
            self._obs_perm_inv_tensor = torch.empty(
                (0, 0), dtype=torch.long, device=self._obs_perm_inv_tensor.device
            )
            self._act_perm_tensor = torch.empty(
                (0, 0), dtype=torch.long, device=self._act_perm_tensor.device
            )

    def _forward_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._expects_conv_input:
            x = x.reshape((-1, *self.obs_shape))
        common_in = torch.nn.functional.relu(self.embeddings(x))
        common = self.common(common_in)
        return self.action(common), self.value(common)

    def _forward_with_indices(
        self, x: torch.Tensor, perm_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._obs_perm_tensor.numel() == 0:
            return self._forward_core(x)

        perm_indices = perm_indices.to(device=x.device, dtype=torch.long)
        logits = torch.empty(
            x.shape[0], self.num_actions, device=x.device, dtype=x.dtype
        )
        values = torch.empty(x.shape[0], 1, device=x.device, dtype=x.dtype)

        unique_perms = torch.unique(perm_indices)
        for perm in unique_perms:
            perm_val = int(perm.item())
            mask = perm_indices == perm
            if not torch.any(mask):
                continue
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            x_sel = x.index_select(0, idxs)

            if perm_val >= 0:
                perm_inv = self._obs_perm_inv_tensor[perm_val].to(device=x.device)
                gather_idx = perm_inv.unsqueeze(0).expand(x_sel.shape[0], -1)
                x_perm = torch.gather(x_sel, 1, gather_idx)
            else:
                x_perm = x_sel

            logits_sel, values_sel = self._forward_core(x_perm)

            if perm_val >= 0 and self._act_perm_tensor.numel() > 0:
                act_perm = self._act_perm_tensor[perm_val].to(device=logits_sel.device)
                logits_sel = logits_sel.index_select(1, act_perm)

            logits.index_copy_(0, idxs, logits_sel)
            values.index_copy_(0, idxs, values_sel)

        return logits, values

    def forward(
        self,
        x: torch.Tensor,
        perm_indices: Optional[torch.Tensor] = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape((-1, self.obs_size)).contiguous()

        if self._obs_perm_tensor.numel() == 0:
            return self._forward_core(x)

        if perm_indices is not None:
            return self._forward_with_indices(x, perm_indices)

        random_indices = torch.randint(
            self._obs_perm_tensor.shape[0], (x.shape[0],), device=x.device
        )
        return self._forward_with_indices(x, random_indices)

    def predict(self, obs):
        torch_obs = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(
            0
        )
        actions, value = self.forward(torch_obs)
        actions_np = torch.softmax(actions, axis=1).squeeze(0).cpu().numpy()
        value_np = value.squeeze(0).cpu().numpy()

        return actions_np, value_np

    def to_rust(self):
        return twisterl.nn.Policy(
            embeddingbag_to_rust(self.embeddings, [self.obs_size], 0),
            sequential_to_rust(self.common),
            sequential_to_rust(self.action),
            sequential_to_rust(self.value),
            self.obs_perms,
            self.act_perms,
        )


class Transpose(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.permute((0, 2, 1))


class Conv1dPolicy(BasicPolicy):
    def __init__(
        self,
        obs_shape: list[int],
        num_actions: int,
        embedding_size: int,
        conv_dim: int = 0,
        common_layers=(256,),
        policy_layers=tuple(),
        value_layers=tuple(),
        obs_perms=tuple(),
        act_perms=tuple(),
    ):
        super().__init__(
            obs_shape,
            num_actions,
            embedding_size,
            common_layers,
            policy_layers,
            value_layers,
            obs_perms,
            act_perms,
        )
        self.conv_dim = conv_dim
        self._expects_conv_input = True

        layers = []
        if conv_dim == 1:
            layers.append(Transpose())

        self.conv_layer = torch.nn.Conv1d(
            obs_shape[conv_dim],
            embedding_size // obs_shape[1 - conv_dim],
            kernel_size=1,
            bias=False,
        )
        layers.append(self.conv_layer)
        layers.append(Transpose())
        layers.append(torch.nn.Flatten())

        self.embeddings = torch.nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        perm_indices: Optional[torch.Tensor] = None,
    ):
        if x.shape[1:] != self.obs_shape:
            x = x.reshape((-1, *self.obs_shape))
        return super().forward(x, perm_indices=perm_indices)

    def to_rust(self):
        return twisterl.nn.Policy(
            embeddingbag_to_rust(self.conv_layer, self.obs_shape, self.conv_dim),
            sequential_to_rust(self.common),
            sequential_to_rust(self.action),
            sequential_to_rust(self.value),
            self.obs_perms,
            self.act_perms,
        )
