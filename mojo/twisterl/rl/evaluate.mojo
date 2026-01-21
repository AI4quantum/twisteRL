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
from ..nn.policy import Policy
from .solve import solve, solve_simple


fn evaluate[
    E: Movable
](
    env: E,
    policy: Policy,
    num_episodes: Int,
    deterministic: Bool,
    num_searches: Int,
    num_mcts_searches: Int,
    seed: Int,  # unused for now
    C: Float32,
    max_expand_depth: Int,
    num_cores: Int,
    predict_probs_mcts_fn: fn (E, Policy, Int, Float32, Int) -> List[Float32],
    env_reset_fn: fn (mut E) -> None,
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
) -> Tuple[Float32, Float32]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic action selection
        num_searches: Number of solve attempts per episode
        num_mcts_searches: Number of MCTS searches per step
        seed: Random seed (unused for now)
        C: UCB exploration constant for MCTS
        max_expand_depth: Maximum depth for MCTS expansion
        num_cores: Number of cores for parallel evaluation (unused in Mojo currently)
        *_fn: Environment and MCTS interface functions

    Returns:
        (average_success_rate, average_reward)
    """
    var env_copy = env_clone_fn(env)

    # Note: Mojo doesn't have rayon-style parallelism yet,
    # so we run sequentially regardless of num_cores

    var successes: Float32 = 0.0
    var rewards: Float32 = 0.0

    for _ in range(num_episodes):
        env_reset_fn(env_copy)

        var result = solve[E](
            env_copy,
            policy,
            deterministic,
            num_searches,
            num_mcts_searches,
            C,
            max_expand_depth,
            predict_probs_mcts_fn,
            env_observe_fn,
            env_masks_fn,
            env_reward_fn,
            env_is_final_fn,
            env_step_fn,
            env_clone_fn,
        )

        var result_tuple = result[0]
        var success = result_tuple[0]
        var reward = result_tuple[1]

        successes += success
        rewards += reward

    var avg_success = successes / Float32(num_episodes)
    var avg_reward = rewards / Float32(num_episodes)

    return (avg_success, avg_reward)


fn evaluate_simple[
    E: Movable
](
    env: E,
    policy: Policy,
    num_episodes: Int,
    deterministic: Bool,
    num_searches: Int,
    env_reset_fn: fn (mut E) -> None,
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
) -> Tuple[Float32, Float32]:
    """
    Simplified evaluation without MCTS - uses policy directly.

    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic action selection
        num_searches: Number of solve attempts per episode
        env_*_fn: Environment interface functions

    Returns:
        (average_success_rate, average_reward)
    """
    var env_copy = env_clone_fn(env)

    var successes: Float32 = 0.0
    var rewards: Float32 = 0.0

    for _ in range(num_episodes):
        env_reset_fn(env_copy)

        var result = solve_simple[E](
            env_copy,
            policy,
            deterministic,
            num_searches,
            env_observe_fn,
            env_masks_fn,
            env_reward_fn,
            env_is_final_fn,
            env_step_fn,
            env_clone_fn,
        )

        var result_tuple = result[0]
        var success = result_tuple[0]
        var reward = result_tuple[1]

        successes += success
        rewards += reward

    var avg_success = successes / Float32(num_episodes)
    var avg_reward = rewards / Float32(num_episodes)

    return (avg_success, avg_reward)
