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

from collections import List
from ..nn.policy import Policy, sample_from_logits
from .collector import CollectedData, merge


struct PPOCollector(Copyable, Movable):
    """Proximal Policy Optimization data collector."""
    var num_episodes: Int
    var gamma: Float32
    var lambda_: Float32  # lambda is a reserved word in some contexts
    var num_cores: Int

    fn __init__(
        out self,
        num_episodes: Int,
        gamma: Float32,
        lambda_: Float32,
        num_cores: Int = 1,
    ):
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_cores = num_cores

    fn get_step_data[
        E: Movable
    ](
        self,
        env: E,
        policy: Policy,
        env_observe_fn: fn (E) -> List[Int],
        env_masks_fn: fn (E) -> List[Bool],
        env_reward_fn: fn (E) -> Float32,
    ) -> Tuple[List[Int], List[Float32], Int, Float32, Float32]:
        """Get data for a single step."""
        var obs = env_observe_fn(env)
        var masks = env_masks_fn(env)
        var reward = env_reward_fn(env)
        var result = policy.forward(obs, masks)
        var logits = result[0]
        var value = result[1]
        var action = sample_from_logits(logits)
        return (obs, logits, action, value, reward)

    fn single_collect[
        E: Movable
    ](
        self,
        owned env: E,
        policy: Policy,
        env_reset_fn: fn (mut E) -> None,
        env_observe_fn: fn (E) -> List[Int],
        env_masks_fn: fn (E) -> List[Bool],
        env_reward_fn: fn (E) -> Float32,
        env_is_final_fn: fn (E) -> Bool,
        env_step_fn: fn (mut E, Int) -> None,
        env_clone_fn: fn (E) -> E,
    ) -> CollectedData:
        """Runs one episode, returns its CollectedData."""
        env_reset_fn(env)

        var obss = List[List[Int]]()
        var log_probs = List[List[Float32]]()
        var vals = List[Float32]()
        var rews = List[Float32]()
        var acts = List[Int]()

        while True:
            var step_data = self.get_step_data[E](
                env, policy, env_observe_fn, env_masks_fn, env_reward_fn
            )
            var obs = step_data[0]
            var log_prob = step_data[1]
            var act = step_data[2]
            var val = step_data[3]
            var rew = step_data[4]

            obss.append(obs)
            log_probs.append(log_prob)
            vals.append(val)
            rews.append(rew)
            acts.append(act)

            if env_is_final_fn(env):
                break
            env_step_fn(env, act)

        # Compute GAE advantages and returns
        var n = len(rews)
        var advs = List[Float32]()
        var rets = List[Float32]()

        # Initialize with zeros
        for _ in range(n):
            advs.append(0.0)
            rets.append(0.0)

        if n > 0:
            advs[n - 1] = rews[n - 1] - vals[n - 1]
            rets[n - 1] = rews[n - 1]

            # Backward pass to compute GAE
            for t_rev in range(n - 1):
                var t = n - 2 - t_rev
                rets[t] = rews[t] + self.gamma * (vals[t + 1] + self.lambda_ * advs[t + 1])
                advs[t] = rets[t] - vals[t]

        var data = CollectedData(
            obss,
            log_probs,
            vals,
            rews,
            acts,
        )
        data.advs = advs
        data.rets = rets

        return data

    fn collect[
        E: Movable
    ](
        self,
        env: E,
        policy: Policy,
        env_reset_fn: fn (mut E) -> None,
        env_observe_fn: fn (E) -> List[Int],
        env_masks_fn: fn (E) -> List[Bool],
        env_reward_fn: fn (E) -> Float32,
        env_is_final_fn: fn (E) -> Bool,
        env_step_fn: fn (mut E, Int) -> None,
        env_clone_fn: fn (E) -> E,
    ) -> CollectedData:
        """Runs the collection process and returns accumulated data."""
        # Note: Mojo doesn't have rayon-style parallelism yet,
        # so we run sequentially for now
        var chunks = List[CollectedData]()

        for _ in range(self.num_episodes):
            var env_copy = env_clone_fn(env)
            var episode_data = self.single_collect[E](
                env_copy,
                policy,
                env_reset_fn,
                env_observe_fn,
                env_masks_fn,
                env_reward_fn,
                env_is_final_fn,
                env_step_fn,
                env_clone_fn,
            )
            chunks.append(episode_data)

        return merge(chunks)


# Simplified version for easier usage
struct PPOCollectorSimple(Copyable, Movable):
    """Simplified PPO data collector."""
    var num_episodes: Int
    var gamma: Float32
    var lambda_: Float32
    var num_cores: Int

    fn __init__(
        out self,
        num_episodes: Int,
        gamma: Float32 = 0.99,
        lambda_: Float32 = 0.95,
        num_cores: Int = 1,
    ):
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_cores = num_cores
