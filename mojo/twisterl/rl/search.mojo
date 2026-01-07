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
from math import sqrt

from ..nn.policy import Policy, sample, softmax
from .tree import MCTSNode, MCTSTree, SimpleTree, SimpleNode


fn ucb_score(
    parent_visits: Int,
    child_visits: Int,
    child_value: Float32,
    prior: Float32,
    c_puct: Float32,
) -> Float32:
    """
    Calculate UCB (Upper Confidence Bound) score for MCTS node selection.

    Uses the PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    """
    var q: Float32 = 0.0
    if child_visits > 0:
        q = child_value / Float32(child_visits)
    var exploration = c_puct * prior * sqrt(Float32(parent_visits)) / (Float32(child_visits) + 1.0)
    return q + exploration


struct MCTS(Copyable, Movable):
    """Monte Carlo Tree Search configuration."""
    var num_simulations: Int
    var c_puct: Float32
    var max_expand_depth: Int

    fn __init__(
        out self,
        num_simulations: Int,
        c_puct: Float32 = 1.0,
        max_expand_depth: Int = 100,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_expand_depth = max_expand_depth


fn predict_probs_mcts[
    E: Movable
](
    owned env: E,
    policy: Policy,
    num_simulations: Int,
    c_puct: Float32,
    max_expand_depth: Int,
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
    env_num_actions_fn: fn (E) -> Int,
) -> List[Float32]:
    """
    Run MCTS from the current environment state and return action probabilities.

    This implementation matches the Rust MCTS algorithm:
    1. Creates tree and root node
    2. Expands root with all valid actions
    3. For each simulation:
       - Traverse tree using UCB until leaf
       - Expand and sample until end state
       - Backpropagate value
    4. Returns visit count proportions as action probabilities
    """
    var tree = MCTSTree()
    var num_actions = env_num_actions_fn(env)
    var root_masks = env_masks_fn(env)
    var root_obs = env_observe_fn(env)

    # Get initial policy output for the root
    var result = policy.full_predict(root_obs, root_masks)
    var action_probs = result[0]

    # Create root node
    var root_node = MCTSNode(root_obs, -1, 0.0)
    root_node.visit_count = 1
    var root_idx = tree.new_node(root_node)

    # Expand root: add child for each valid action
    for action in range(num_actions):
        if action < len(root_masks) and root_masks[action]:
            if action < len(action_probs) and action_probs[action] > 0.0:
                var env_copy = env_clone_fn(env)
                env_step_fn(env_copy, action)
                var child_obs = env_observe_fn(env_copy)
                var child_node = MCTSNode(child_obs, action, action_probs[action])
                _ = tree.add_child_to_node(child_node, root_idx)

    # Run simulations
    for _ in range(num_simulations):
        var node_idx = root_idx
        var env_copy = env_clone_fn(env)

        # Selection: traverse tree using UCB until leaf node
        while len(tree.nodes[node_idx].children) > 0:
            node_idx = tree.next(node_idx, c_puct)
            var action = tree.nodes[node_idx].val.action_taken
            if action >= 0:
                env_step_fn(env_copy, action)

        var value: Float32 = 0.0
        var expanded_depth = 0

        # Expansion and simulation: expand until end state or max depth
        while expanded_depth < max_expand_depth:
            # Get value
            value = env_reward_fn(env_copy)

            # Break if is_final
            if env_is_final_fn(env_copy):
                break

            # Get policy predictions
            var obs = env_observe_fn(env_copy)
            var masks = env_masks_fn(env_copy)
            var pred_result = policy.full_predict(obs, masks)
            var probs = pred_result[0]
            var new_value = pred_result[1]

            # Expand tree: add children for valid actions
            for action in range(len(probs)):
                if action < len(masks) and masks[action] and probs[action] > 0.0:
                    var next_env = env_clone_fn(env_copy)
                    env_step_fn(next_env, action)
                    var child_obs = env_observe_fn(next_env)
                    var child_node = MCTSNode(child_obs, action, probs[action])
                    _ = tree.add_child_to_node(child_node, node_idx)

            # Select child by sampling
            if len(tree.nodes[node_idx].children) > 0:
                node_idx = tree.next_sample(node_idx, sample)
                var action = tree.nodes[node_idx].val.action_taken
                if action >= 0:
                    env_step_fn(env_copy, action)
                value = new_value
            else:
                break

            expanded_depth += 1

        # Backpropagation
        tree.backpropagate(node_idx, value)

    # Calculate action probabilities from root's children visit counts
    var mcts_action_probs = List[Float32]()
    for _ in range(num_actions):
        mcts_action_probs.append(0.0)

    for i in range(len(tree.nodes[root_idx].children)):
        var child_idx = tree.nodes[root_idx].children[i]
        var action = tree.nodes[child_idx].val.action_taken
        if action >= 0 and action < len(mcts_action_probs):
            mcts_action_probs[action] = Float32(tree.nodes[child_idx].val.visit_count)

    # Normalize probabilities
    var sum_probs: Float32 = 0.0
    for i in range(len(mcts_action_probs)):
        sum_probs += mcts_action_probs[i]

    if sum_probs > 0.0:
        for i in range(len(mcts_action_probs)):
            mcts_action_probs[i] = mcts_action_probs[i] / sum_probs
    else:
        # Uniform distribution if no visits
        for i in range(num_actions):
            if i < len(root_masks) and root_masks[i]:
                mcts_action_probs[i] = 1.0 / Float32(num_actions)

    return mcts_action_probs


fn mcts_search[
    E: Movable
](
    env: E,
    policy: Policy,
    num_simulations: Int,
    c_puct: Float32,
    max_expand_depth: Int,
    env_observe_fn: fn (E) -> List[Int],
    env_masks_fn: fn (E) -> List[Bool],
    env_reward_fn: fn (E) -> Float32,
    env_is_final_fn: fn (E) -> Bool,
    env_step_fn: fn (mut E, Int) -> None,
    env_clone_fn: fn (E) -> E,
    env_num_actions_fn: fn (E) -> Int,
) -> List[Float32]:
    """
    Convenience wrapper for MCTS search.
    """
    var env_copy = env_clone_fn(env)
    return predict_probs_mcts[E](
        env_copy,
        policy,
        num_simulations,
        c_puct,
        max_expand_depth,
        env_observe_fn,
        env_masks_fn,
        env_reward_fn,
        env_is_final_fn,
        env_step_fn,
        env_clone_fn,
        env_num_actions_fn,
    )


# Simplified MCTS for direct use with PuzzleEnv
fn predict_probs_mcts_simple(
    owned env: PuzzleEnv,
    policy: Policy,
    num_simulations: Int,
    c_puct: Float32,
    max_expand_depth: Int,
) -> List[Float32]:
    """
    Simplified MCTS for PuzzleEnv that doesn't require function pointers.
    """
    var tree = MCTSTree()
    var num_actions = env.num_actions()
    var root_masks = env.masks()
    var root_obs = env.observe()

    # Get initial policy output for the root
    var result = policy.full_predict(root_obs, root_masks)
    var action_probs = result[0]

    # Create root node
    var root_node = MCTSNode(root_obs, -1, 0.0)
    root_node.visit_count = 1
    var root_idx = tree.new_node(root_node)

    # Expand root: add child for each valid action
    for action in range(num_actions):
        if action < len(root_masks) and root_masks[action]:
            if action < len(action_probs) and action_probs[action] > 0.0:
                var env_copy = env.clone()
                env_copy.step(action)
                var child_obs = env_copy.observe()
                var child_node = MCTSNode(child_obs, action, action_probs[action])
                _ = tree.add_child_to_node(child_node, root_idx)

    # Run simulations
    for _ in range(num_simulations):
        var node_idx = root_idx
        var env_copy = env.clone()

        # Selection: traverse tree using UCB until leaf node
        while len(tree.nodes[node_idx].children) > 0:
            node_idx = tree.next(node_idx, c_puct)
            var action = tree.nodes[node_idx].val.action_taken
            if action >= 0:
                env_copy.step(action)

        var value: Float32 = 0.0
        var expanded_depth = 0

        # Expansion and simulation
        while expanded_depth < max_expand_depth:
            value = env_copy.reward()

            if env_copy.is_final():
                break

            var obs = env_copy.observe()
            var masks = env_copy.masks()
            var pred_result = policy.full_predict(obs, masks)
            var probs = pred_result[0]
            var new_value = pred_result[1]

            # Expand tree
            for action in range(len(probs)):
                if action < len(masks) and masks[action] and probs[action] > 0.0:
                    var next_env = env_copy.clone()
                    next_env.step(action)
                    var child_obs = next_env.observe()
                    var child_node = MCTSNode(child_obs, action, probs[action])
                    _ = tree.add_child_to_node(child_node, node_idx)

            # Select child by sampling
            if len(tree.nodes[node_idx].children) > 0:
                node_idx = tree.next_sample(node_idx, sample)
                var action = tree.nodes[node_idx].val.action_taken
                if action >= 0:
                    env_copy.step(action)
                value = new_value
            else:
                break

            expanded_depth += 1

        tree.backpropagate(node_idx, value)

    # Calculate action probabilities
    var mcts_action_probs = List[Float32]()
    for _ in range(num_actions):
        mcts_action_probs.append(0.0)

    for i in range(len(tree.nodes[root_idx].children)):
        var child_idx = tree.nodes[root_idx].children[i]
        var action = tree.nodes[child_idx].val.action_taken
        if action >= 0 and action < len(mcts_action_probs):
            mcts_action_probs[action] = Float32(tree.nodes[child_idx].val.visit_count)

    var sum_probs: Float32 = 0.0
    for i in range(len(mcts_action_probs)):
        sum_probs += mcts_action_probs[i]

    if sum_probs > 0.0:
        for i in range(len(mcts_action_probs)):
            mcts_action_probs[i] = mcts_action_probs[i] / sum_probs
    else:
        for i in range(num_actions):
            if i < len(root_masks) and root_masks[i]:
                mcts_action_probs[i] = 1.0 / Float32(num_actions)

    return mcts_action_probs


# Import PuzzleEnv for the simplified version
from ..envs.puzzle import PuzzleEnv
