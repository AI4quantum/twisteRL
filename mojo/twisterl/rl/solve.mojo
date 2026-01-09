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
from ..nn.policy import Policy, sample, argmax


fn single_solve[
    E: Movable
](
    owned env: E,
    policy: Policy,
    deterministic: Bool,
    num_mcts_searches: Int,
    C: Float32,
    max_expand_depth: Int,
    predict_probs_mcts_fn: fn (E, Policy, Int, Float32, Int) -> List[Float32],
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
) -> Tuple[Tuple[Float32, Float32], List[Int]]:
    """
    Run a single solve attempt on the environment.

    Returns:
        ((success, total_reward), solution_path)
    """
    var total_val: Float32 = 0.0
    var solution = List[Int]()

    # Step until final
    while not env_is_final_fn(env):
        var val = env_reward_fn(env)
        var obs = env_observe_fn(env)
        var masks = env_masks_fn(env)
        total_val += val

        # Choose probs via either policy or MCTS
        var probs: List[Float32]
        if num_mcts_searches == 0:
            var result = policy.predict(obs, masks)
            probs = result[0]
        else:
            # Use MCTS to get probabilities
            var env_clone = env_clone_fn(env)
            probs = predict_probs_mcts_fn(
                env_clone, policy, num_mcts_searches, C, max_expand_depth
            )

        var action: Int
        if deterministic:
            action = argmax(probs)
        else:
            action = sample(probs)

        env_step_fn(env, action)
        solution.append(action)

    var val = env_reward_fn(env)
    total_val += val

    # Success if final reward is 1.0
    var success: Float32 = 1.0 if val == 1.0 else 0.0

    return ((success, total_val), solution)


fn solve[
    E: Movable
](
    env: E,
    policy: Policy,
    deterministic: Bool,
    num_searches: Int,
    num_mcts_searches: Int,
    C: Float32,
    max_expand_depth: Int,
    predict_probs_mcts_fn: fn (E, Policy, Int, Float32, Int) -> List[Float32],
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
) -> Tuple[Tuple[Float32, Float32], List[Int]]:
    """
    Run multiple solve attempts and return the best result.

    Args:
        env: The environment to solve
        policy: The policy to use for action selection
        deterministic: Whether to use deterministic action selection
        num_searches: Number of solve attempts
        num_mcts_searches: Number of MCTS searches per step (0 for policy-only)
        C: UCB exploration constant for MCTS
        max_expand_depth: Maximum depth for MCTS expansion
        predict_probs_mcts_fn: Function to predict probabilities using MCTS
        env_*_fn: Environment interface functions

    Returns:
        ((success, total_reward), solution_path)
    """
    var best_result: Tuple[Float32, Float32] = (0.0, Float32.MIN)
    var best_path = List[Int]()

    for _ in range(num_searches):
        var cloned_env = env_clone_fn(env)
        var result = single_solve[E](
            cloned_env,
            policy,
            deterministic,
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
        var path = result[1]

        # Compare results: first by success, then by reward
        if result_tuple[0] > best_result[0] or (
            result_tuple[0] == best_result[0] and result_tuple[1] > best_result[1]
        ):
            best_result = result_tuple
            best_path = path

    return (best_result, best_path)


# Simplified version for direct policy usage (no MCTS)
fn solve_simple[
    E: Movable
](
    env: E,
    policy: Policy,
    deterministic: Bool,
    num_searches: Int,
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
) -> Tuple[Tuple[Float32, Float32], List[Int]]:
    """
    Simplified solve without MCTS - uses policy directly.

    Returns:
        ((success, total_reward), solution_path)
    """
    var best_result: Tuple[Float32, Float32] = (0.0, Float32.MIN)
    var best_path = List[Int]()

    for _ in range(num_searches):
        var cloned_env = env_clone_fn(env)
        var total_val: Float32 = 0.0
        var solution = List[Int]()

        # Step until final
        while not env_is_final_fn(cloned_env):
            var val = env_reward_fn(cloned_env)
            var obs = env_observe_fn(cloned_env)
            var masks = env_masks_fn(cloned_env)
            total_val += val

            var result = policy.predict(obs, masks)
            var probs = result[0]

            var action: Int
            if deterministic:
                action = argmax(probs)
            else:
                action = sample(probs)

            env_step_fn(cloned_env, action)
            solution.append(action)

        var val = env_reward_fn(cloned_env)
        total_val += val

        # Success if final reward is 1.0
        var success: Float32 = 1.0 if val == 1.0 else 0.0
        var result_tuple: Tuple[Float32, Float32] = (success, total_val)

        # Compare results
        if result_tuple[0] > best_result[0] or (
            result_tuple[0] == best_result[0] and result_tuple[1] > best_result[1]
        ):
            best_result = result_tuple
            best_path = solution

    return (best_result, best_path)
