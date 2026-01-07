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

from random import random_float64, seed
from math import exp, log
from collections import List
from memory import memcpy


fn argmax(values: List[Float32]) -> Int:
    """Returns the index of the maximum value in a list."""
    if len(values) == 0:
        return 0

    var max_idx: Int = 0
    var max_val = values[0]

    for i in range(1, len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i

    return max_idx


fn sample(probs: List[Float32]) -> Int:
    """Sample an index from a probability distribution."""
    if len(probs) == 0:
        return 0

    var rand_val = random_float64().cast[DType.float32]()
    var cumsum: Float32 = 0.0

    for i in range(len(probs)):
        cumsum += probs[i]
        if rand_val < cumsum:
            return i

    return len(probs) - 1


fn sample_from_logits(logits: List[Float32]) -> Int:
    """Sample from logits using Gumbel-max trick."""
    var perturbed = List[Float32]()

    for i in range(len(logits)):
        # Gumbel noise: -log(-log(u)) where u is uniform(0,1)
        var u = random_float64().cast[DType.float32]()
        # Clamp to avoid log(0)
        if u < 1e-10:
            u = 1e-10
        if u > 1.0 - 1e-10:
            u = 1.0 - 1e-10
        var gumbel = -log(-log(u))
        perturbed.append(logits[i] + gumbel)

    return argmax(perturbed)


fn softmax(logits: List[Float32]) -> List[Float32]:
    """Compute softmax probabilities from logits."""
    var result = List[Float32]()

    if len(logits) == 0:
        return result

    # Find max for numerical stability
    var max_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_val:
            max_val = logits[i]

    # Compute exp and sum
    var sum_exp: Float32 = 0.0
    for i in range(len(logits)):
        var exp_val = exp(logits[i] - max_val)
        result.append(exp_val)
        sum_exp += exp_val

    # Normalize
    for i in range(len(result)):
        result[i] = result[i] / (sum_exp + 1e-10)

    return result


fn log_softmax(logits: List[Float32]) -> List[Float32]:
    """Compute log softmax from logits."""
    var result = List[Float32]()

    if len(logits) == 0:
        return result

    # Find max for numerical stability
    var max_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_val:
            max_val = logits[i]

    # Compute log sum exp
    var sum_exp: Float32 = 0.0
    for i in range(len(logits)):
        sum_exp += exp(logits[i] - max_val)
    var log_sum_exp = log(sum_exp) + max_val

    # Compute log softmax
    for i in range(len(logits)):
        result.append(logits[i] - log_sum_exp)

    return result


struct Policy(Copyable, Movable):
    """Neural network policy for reinforcement learning."""
    var embedding_weights: List[List[Float32]]
    var embedding_bias: List[Float32]
    var embedding_apply_relu: Bool
    var embedding_obs_shape: List[Int]
    var embedding_conv_dim: Int

    var common_weights: List[List[Float32]]
    var common_biases: List[List[Float32]]
    var common_apply_relus: List[Bool]

    var action_weights: List[List[Float32]]
    var action_biases: List[List[Float32]]
    var action_apply_relus: List[Bool]

    var value_weights: List[List[Float32]]
    var value_biases: List[List[Float32]]
    var value_apply_relus: List[Bool]

    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    fn __init__(
        out self,
        embedding_weights: List[List[Float32]],
        embedding_bias: List[Float32],
        embedding_apply_relu: Bool,
        embedding_obs_shape: List[Int],
        embedding_conv_dim: Int,
        common_weights: List[List[Float32]],
        common_biases: List[List[Float32]],
        common_apply_relus: List[Bool],
        action_weights: List[List[Float32]],
        action_biases: List[List[Float32]],
        action_apply_relus: List[Bool],
        value_weights: List[List[Float32]],
        value_biases: List[List[Float32]],
        value_apply_relus: List[Bool],
        obs_perms: List[List[Int]],
        act_perms: List[List[Int]],
    ):
        self.embedding_weights = embedding_weights
        self.embedding_bias = embedding_bias
        self.embedding_apply_relu = embedding_apply_relu
        self.embedding_obs_shape = embedding_obs_shape
        self.embedding_conv_dim = embedding_conv_dim
        self.common_weights = common_weights
        self.common_biases = common_biases
        self.common_apply_relus = common_apply_relus
        self.action_weights = action_weights
        self.action_biases = action_biases
        self.action_apply_relus = action_apply_relus
        self.value_weights = value_weights
        self.value_biases = value_biases
        self.value_apply_relus = value_apply_relus
        self.obs_perms = obs_perms
        self.act_perms = act_perms

    fn _embedding_forward(self, obs: List[Int]) -> List[Float32]:
        """Forward pass through embedding layer."""
        var out = List[Float32]()
        for i in range(len(self.embedding_bias)):
            out.append(self.embedding_bias[i])

        if len(self.embedding_obs_shape) == 1:
            # 1D observation
            for idx in obs:
                if idx < len(self.embedding_weights):
                    var vec = self.embedding_weights[idx]
                    for j in range(len(vec)):
                        if j < len(out):
                            out[j] += vec[j]
        elif len(self.embedding_obs_shape) == 2:
            # 2D observation
            var v_size = len(self.embedding_weights[0]) if len(self.embedding_weights) > 0 else 0
            for idx in obs:
                var row = idx // self.embedding_obs_shape[1]
                var col = idx % self.embedding_obs_shape[1]

                if self.embedding_conv_dim == 1:
                    var temp = row
                    row = col
                    col = temp

                if row < len(self.embedding_weights):
                    var vec = self.embedding_weights[row]
                    for j in range(len(vec)):
                        var out_idx = col * v_size + j
                        if out_idx < len(out):
                            out[out_idx] += vec[j]

        if self.embedding_apply_relu:
            for i in range(len(out)):
                if out[i] < 0:
                    out[i] = 0

        return out

    fn _linear_forward(
        self,
        input: List[Float32],
        weights: List[Float32],
        bias: List[Float32],
        apply_relu: Bool,
    ) -> List[Float32]:
        """Forward pass through a linear layer."""
        var n_out = len(bias)
        var n_in = len(weights) // n_out if n_out > 0 else 0

        var out = List[Float32]()
        for i in range(n_out):
            var sum_val = bias[i]
            for j in range(n_in):
                if j < len(input):
                    sum_val += weights[i * n_in + j] * input[j]
            if apply_relu and sum_val < 0:
                sum_val = 0
            out.append(sum_val)

        return out

    fn _sequential_forward(
        self,
        input: List[Float32],
        weights: List[List[Float32]],
        biases: List[List[Float32]],
        apply_relus: List[Bool],
    ) -> List[Float32]:
        """Forward pass through sequential layers."""
        var x = input
        for i in range(len(weights)):
            x = self._linear_forward(x, weights[i], biases[i], apply_relus[i])
        return x

    fn _get_perm_id(self) -> Int:
        """Get a random permutation index, or -1 if no permutations."""
        if len(self.obs_perms) == 0:
            return -1
        var rand_val = random_float64()
        return Int(rand_val * len(self.obs_perms)) % len(self.obs_perms)

    fn _raw_predict(self, owned obs: List[Int], n_perm: Int) -> Tuple[List[Float32], Float32]:
        """Raw forward pass with optional permutation."""
        # Apply observation permutation if needed
        if n_perm >= 0 and n_perm < len(self.obs_perms):
            var perm = self.obs_perms[n_perm]
            var permuted_obs = List[Int]()
            for i in range(len(obs)):
                if obs[i] < len(perm):
                    permuted_obs.append(perm[obs[i]])
                else:
                    permuted_obs.append(obs[i])
            obs = permuted_obs

        # Embedding forward
        var emb_out = self._embedding_forward(obs)

        # Common network forward
        var common_out = self._sequential_forward(
            emb_out, self.common_weights, self.common_biases, self.common_apply_relus
        )

        # Value network forward
        var value_out = self._sequential_forward(
            common_out, self.value_weights, self.value_biases, self.value_apply_relus
        )
        var value: Float32 = 0.0
        for i in range(len(value_out)):
            value += value_out[i]

        # Action network forward
        var action_logits = self._sequential_forward(
            common_out, self.action_weights, self.action_biases, self.action_apply_relus
        )

        # Apply action permutation if needed
        if n_perm >= 0 and n_perm < len(self.act_perms):
            var perm = self.act_perms[n_perm]
            var permuted_logits = List[Float32]()
            for i in range(len(perm)):
                if perm[i] < len(action_logits):
                    permuted_logits.append(action_logits[perm[i]])
                else:
                    permuted_logits.append(0.0)
            action_logits = permuted_logits

        return (action_logits, value)

    fn predict(self, obs: List[Int], masks: List[Bool]) -> Tuple[List[Float32], Float32]:
        """Forward pass returning normalized action probabilities and value."""
        var result = self._raw_predict(obs, self._get_perm_id())
        var action_logits = result[0]
        var value = result[1]

        # Apply masks and compute exp
        var exp_masked_probs = List[Float32]()
        for i in range(len(action_logits)):
            if i < len(masks) and masks[i]:
                exp_masked_probs.append(exp(action_logits[i]))
            else:
                exp_masked_probs.append(0.0)

        # Normalize
        var sum_probs: Float32 = 0.0
        for i in range(len(exp_masked_probs)):
            sum_probs += exp_masked_probs[i]

        for i in range(len(exp_masked_probs)):
            exp_masked_probs[i] = exp_masked_probs[i] / (sum_probs + 1e-6)

        return (exp_masked_probs, value)

    fn forward(self, obs: List[Int], masks: List[Bool]) -> Tuple[List[Float32], Float32]:
        """Forward pass returning masked logits and value."""
        var result = self._raw_predict(obs, self._get_perm_id())
        var action_logits = result[0]
        var value = result[1]

        # Apply masks (set masked actions to very negative value)
        var masked_logits = List[Float32]()
        for i in range(len(action_logits)):
            if i < len(masks) and masks[i]:
                masked_logits.append(action_logits[i])
            else:
                masked_logits.append(-1e10)

        return (masked_logits, value)

    fn full_predict(self, obs: List[Int], masks: List[Bool]) -> Tuple[List[Float32], Float32]:
        """Forward pass averaging over all permutations."""
        if len(self.obs_perms) == 0:
            return self.predict(obs, masks)

        var n_perms = len(self.obs_perms)
        var n_actions = len(self.act_perms[0]) if len(self.act_perms) > 0 else 0

        # Initialize accumulators
        var action_logits = List[Float32]()
        for _ in range(n_actions):
            action_logits.append(0.0)
        var value: Float32 = 0.0

        # Average over all permutations
        for pi in range(n_perms):
            var result = self._raw_predict(obs, pi)
            var logits_pi = result[0]
            var value_pi = result[1]

            value += value_pi / Float32(n_perms)
            for i in range(len(logits_pi)):
                if i < len(action_logits):
                    action_logits[i] += logits_pi[i] / Float32(n_perms)

        # Apply masks and compute exp
        var exp_masked_probs = List[Float32]()
        for i in range(len(action_logits)):
            if i < len(masks) and masks[i]:
                exp_masked_probs.append(exp(action_logits[i]))
            else:
                exp_masked_probs.append(0.0)

        # Normalize
        var sum_probs: Float32 = 0.0
        for i in range(len(exp_masked_probs)):
            sum_probs += exp_masked_probs[i]

        for i in range(len(exp_masked_probs)):
            exp_masked_probs[i] = exp_masked_probs[i] / (sum_probs + 1e-6)

        return (exp_masked_probs, value)
