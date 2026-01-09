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
from ..nn.policy import Policy, sample
from .collector import CollectedData, merge


# Forward declaration - Env will be imported from rl module
# In Mojo, we need to define the interface here or use a trait


struct AZCollector(Copyable, Movable):
    """AlphaZero-style data collector using MCTS."""
    var num_episodes: Int
    var num_mcts_searches: Int
    var C: Float32
    var max_expand_depth: Int
    var num_cores: Int

    fn __init__(
        out self,
        num_episodes: Int,
        num_mcts_searches: Int,
        C: Float32,
        max_expand_depth: Int,
        num_cores: Int,
    ):
        self.num_episodes = num_episodes
        self.num_mcts_searches = num_mcts_searches
        self.C = C
        self.max_expand_depth = max_expand_depth
        self.num_cores = num_cores

    fn single_collect[
        E: Movable
    ](
        self,
        owned env: E,
        policy: Policy,
        predict_probs_mcts_fn: fn (E, Policy, Int, Float32, Int) -> List[Float32],
        env_reset_fn: fn (mut E) -> None,
        env_observe_fn: fn (E) -> List[Int],
        env_reward_fn: fn (E) -> Float32,
        env_is_final_fn: fn (E) -> Bool,
        env_step_fn: fn (mut E, Int) -> None,
        env_clone_fn: fn (E) -> E,
    ) -> CollectedData:
        """Runs one episode, returns its CollectedData."""
        env_reset_fn(env)

        # Init data lists
        var obs = List[List[Int]]()
        var probs = List[List[Float32]]()
        var vals = List[Float32]()
        var total_vals = List[Float32]()

        var total_val: Float32 = 0.0

        # Loop until a final state
        while True:
            # Calculate MCTS probs for current state
            var env_clone = env_clone_fn(env)
            var mcts_probs = predict_probs_mcts_fn(
                env_clone, policy, self.num_mcts_searches, self.C, self.max_expand_depth
            )

            # Select next action and get current value
            var action = sample(mcts_probs)
            var val = env_reward_fn(env)
            total_vals.append(total_val)

            total_val += val

            # Store data
            obs.append(env_observe_fn(env))
            probs.append(mcts_probs)
            vals.append(val)

            # Break if we are in a final state
            if env_is_final_fn(env):
                break

            # Move to next state
            env_step_fn(env, action)

        # Post process rewards - compute remaining values
        var remaining_vals = List[Float32]()
        for i in range(len(total_vals)):
            remaining_vals.append(total_val - total_vals[i])

        var data = CollectedData(
            obs,
            probs,
            List[Float32](),  # values not used in AZ
            List[Float32](),  # rewards not used in AZ
            List[Int](),  # actions not stored in AZ
        )
        data.remaining_values = remaining_vals

        return data

    fn collect[
        E: Movable
    ](
        self,
        env: E,
        policy: Policy,
        predict_probs_mcts_fn: fn (E, Policy, Int, Float32, Int) -> List[Float32],
        env_reset_fn: fn (mut E) -> None,
        env_observe_fn: fn (E) -> List[Int],
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
                predict_probs_mcts_fn,
                env_reset_fn,
                env_observe_fn,
                env_reward_fn,
                env_is_final_fn,
                env_step_fn,
                env_clone_fn,
            )
            chunks.append(episode_data)

        return merge(chunks)


# Simplified version that works with a concrete Env type
struct AZCollectorSimple(Copyable, Movable):
    """Simplified AlphaZero-style data collector."""
    var num_episodes: Int
    var num_mcts_searches: Int
    var C: Float32
    var max_expand_depth: Int
    var num_cores: Int

    fn __init__(
        out self,
        num_episodes: Int,
        num_mcts_searches: Int,
        C: Float32,
        max_expand_depth: Int,
        num_cores: Int = 1,
    ):
        self.num_episodes = num_episodes
        self.num_mcts_searches = num_mcts_searches
        self.C = C
        self.max_expand_depth = max_expand_depth
        self.num_cores = num_cores
