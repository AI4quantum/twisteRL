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

"""
Test file for TwisterL Mojo implementation.

This file tests the basic functionality of all implemented modules.
"""

from collections import List

# Import from submodules
from twisterl.nn.layers import Linear, EmbeddingBag, relu
from twisterl.nn.modules import Sequential
from twisterl.nn.policy import Policy, argmax, sample, softmax, sample_from_logits
from twisterl.rl.tree import SimpleNode, SimpleTree, MCTSNode, MCTSTree
from twisterl.envs.puzzle import PuzzleEnv
from twisterl.collector.collector import CollectedData, merge


fn test_relu():
    """Test ReLU activation function."""
    print("Testing relu...")
    var pos = relu(5.0)
    var neg = relu(-5.0)
    var zero = relu(0.0)

    if pos == 5.0 and neg == 0.0 and zero == 0.0:
        print("  relu: PASSED")
    else:
        print("  relu: FAILED")


fn test_argmax():
    """Test argmax function."""
    print("Testing argmax...")
    var values = List[Float32]()
    values.append(1.0)
    values.append(5.0)
    values.append(3.0)

    var idx = argmax(values)
    if idx == 1:
        print("  argmax: PASSED")
    else:
        print("  argmax: FAILED (expected 1, got", idx, ")")


fn test_softmax():
    """Test softmax function."""
    print("Testing softmax...")
    var logits = List[Float32]()
    logits.append(1.0)
    logits.append(2.0)
    logits.append(3.0)

    var probs = softmax(logits)

    # Sum should be approximately 1.0
    var sum_probs: Float32 = 0.0
    for i in range(len(probs)):
        sum_probs += probs[i]

    if sum_probs > 0.99 and sum_probs < 1.01:
        print("  softmax: PASSED (sum =", sum_probs, ")")
    else:
        print("  softmax: FAILED (sum =", sum_probs, ")")


fn test_sample():
    """Test sample function."""
    print("Testing sample...")
    var probs = List[Float32]()
    probs.append(0.0)
    probs.append(1.0)  # Should always select index 1
    probs.append(0.0)

    var idx = sample(probs)
    if idx == 1:
        print("  sample: PASSED")
    else:
        print("  sample: FAILED (expected 1, got", idx, ")")


fn test_sample_from_logits():
    """Test sample_from_logits function."""
    print("Testing sample_from_logits...")
    var logits = List[Float32]()
    logits.append(-100.0)
    logits.append(100.0)  # Should almost always select this
    logits.append(-100.0)

    # Run multiple times to check it tends to select index 1
    var count_1 = 0
    for _ in range(100):
        var idx = sample_from_logits(logits)
        if idx == 1:
            count_1 += 1

    if count_1 > 90:  # Should be almost always
        print("  sample_from_logits: PASSED (selected 1:", count_1, "/100 times)")
    else:
        print("  sample_from_logits: FAILED (selected 1:", count_1, "/100 times)")


fn test_linear():
    """Test Linear layer."""
    print("Testing Linear layer...")
    var weights = List[Float32]()
    weights.append(1.0)
    weights.append(0.0)
    weights.append(0.0)
    weights.append(1.0)

    var bias = List[Float32]()
    bias.append(0.0)
    bias.append(0.0)

    var layer = Linear(weights, bias, False)

    var input = List[Float32]()
    input.append(2.0)
    input.append(3.0)

    var output = layer.forward(input)

    if len(output) == 2 and output[0] == 2.0 and output[1] == 3.0:
        print("  Linear: PASSED")
    else:
        print("  Linear: FAILED")


fn test_linear_with_relu():
    """Test Linear layer with ReLU activation."""
    print("Testing Linear with ReLU...")
    var weights = List[Float32]()
    weights.append(-1.0)
    weights.append(0.0)
    weights.append(0.0)
    weights.append(1.0)

    var bias = List[Float32]()
    bias.append(0.0)
    bias.append(0.0)

    var layer = Linear(weights, bias, True)

    var input = List[Float32]()
    input.append(2.0)
    input.append(3.0)

    var output = layer.forward(input)

    # First output should be 0 (ReLU of -2)
    # Second output should be 3
    if len(output) == 2 and output[0] == 0.0 and output[1] == 3.0:
        print("  Linear with ReLU: PASSED")
    else:
        print("  Linear with ReLU: FAILED (got", output[0], ",", output[1], ")")


fn test_sequential():
    """Test Sequential module."""
    print("Testing Sequential...")
    var seq = Sequential()

    # Add identity layer
    var w1 = List[Float32]()
    w1.append(2.0)  # Scale by 2
    var b1 = List[Float32]()
    b1.append(0.0)
    var layer1 = Linear(w1, b1, False)
    seq.add_layer(layer1)

    var input = List[Float32]()
    input.append(5.0)

    var output = seq.forward(input)

    if len(output) == 1 and output[0] == 10.0:
        print("  Sequential: PASSED")
    else:
        print("  Sequential: FAILED (got", output[0], ")")


fn test_simple_tree():
    """Test SimpleTree structure."""
    print("Testing SimpleTree...")
    var tree = SimpleTree()

    var root = tree.new_node(0.0)
    var child1 = tree.add_child_to_node(1.0, root)
    _ = tree.add_child_to_node(2.0, root)

    if tree.size() == 3:
        print("  SimpleTree size: PASSED")
    else:
        print("  SimpleTree size: FAILED (size =", tree.size(), ")")

    # Test backpropagation
    tree.backpropagate(child1, 1.0)
    var root_node = tree.get_node(root)
    if root_node.visit_count == 1 and root_node.total_value == 1.0:
        print("  SimpleTree backpropagate: PASSED")
    else:
        print("  SimpleTree backpropagate: FAILED")


fn test_mcts_node():
    """Test MCTSNode structure."""
    print("Testing MCTSNode...")
    var obs = List[Int]()
    obs.append(0)
    obs.append(1)
    obs.append(2)

    var node = MCTSNode(obs, 0, 0.5)

    if node.action_taken == 0 and node.prior == 0.5 and len(node.state_repr) == 3:
        print("  MCTSNode: PASSED")
    else:
        print("  MCTSNode: FAILED")


fn test_mcts_tree():
    """Test MCTSTree structure."""
    print("Testing MCTSTree...")
    var tree = MCTSTree()

    var root_obs = List[Int]()
    root_obs.append(0)
    var root_node = MCTSNode(root_obs, -1, 0.0)
    root_node.visit_count = 1
    var root_idx = tree.new_node(root_node)

    var child_obs = List[Int]()
    child_obs.append(1)
    var child_node = MCTSNode(child_obs, 0, 0.5)
    var child_idx = tree.add_child_to_node(child_node, root_idx)

    if tree.size() == 2:
        print("  MCTSTree size: PASSED")
    else:
        print("  MCTSTree size: FAILED")

    # Test backpropagation
    tree.backpropagate(child_idx, 1.0)
    if tree.nodes[root_idx].val.visit_count == 2:
        print("  MCTSTree backpropagate: PASSED")
    else:
        print("  MCTSTree backpropagate: FAILED (visit_count =", tree.nodes[root_idx].val.visit_count, ")")


fn test_ucb():
    """Test UCB calculation in MCTSNode."""
    print("Testing UCB calculation...")
    var parent_obs = List[Int]()
    var parent = MCTSNode(parent_obs, -1, 0.0)
    parent.visit_count = 10

    var child_obs = List[Int]()
    var child = MCTSNode(child_obs, 0, 0.5)
    child.visit_count = 2
    child.value_sum = 1.0

    var ucb = parent.ucb(child, 1.0)

    # UCB = Q + C * sqrt(N_parent) / (1 + N_child) * prior
    # = 0.5 + 1.0 * sqrt(10) / 3 * 0.5
    # = 0.5 + 3.16 / 3 * 0.5
    # ~= 0.5 + 0.53 = 1.03
    if ucb > 1.0 and ucb < 1.1:
        print("  UCB calculation: PASSED (ucb =", ucb, ")")
    else:
        print("  UCB calculation: FAILED (ucb =", ucb, ")")


fn test_puzzle_env():
    """Test PuzzleEnv."""
    print("Testing PuzzleEnv...")
    var env = PuzzleEnv(3, 3, 0, 2, 20)

    # Should be solved initially
    if env.solved():
        print("  PuzzleEnv initial state: PASSED")
    else:
        print("  PuzzleEnv initial state: FAILED")

    # Test actions
    if env.num_actions() == 4:
        print("  PuzzleEnv num_actions: PASSED")
    else:
        print("  PuzzleEnv num_actions: FAILED")

    # Test masks
    var masks = env.masks()
    if len(masks) == 4:
        print("  PuzzleEnv masks: PASSED")
    else:
        print("  PuzzleEnv masks: FAILED")

    # Test observe
    var obs = env.observe()
    if len(obs) == 9:  # 3x3 puzzle
        print("  PuzzleEnv observe: PASSED")
    else:
        print("  PuzzleEnv observe: FAILED")

    # Test step and clone
    var env2 = env.clone()
    env2.step(2)  # Move right
    if env2.zero_x == 1 and env.zero_x == 0:
        print("  PuzzleEnv clone and step: PASSED")
    else:
        print("  PuzzleEnv clone and step: FAILED")


fn test_puzzle_env_reset():
    """Test PuzzleEnv reset with difficulty."""
    print("Testing PuzzleEnv reset...")
    var env = PuzzleEnv(3, 3, 5, 2, 20)

    # Reset should scramble the puzzle
    env.reset()

    # After reset with difficulty > 0, it might not be solved
    # (though there's a chance random moves cancel out)
    if env.depth == 10:  # depth_slope * difficulty = 2 * 5 = 10
        print("  PuzzleEnv reset depth: PASSED")
    else:
        print("  PuzzleEnv reset depth: FAILED (depth =", env.depth, ")")


fn test_puzzle_env_reward():
    """Test PuzzleEnv reward function."""
    print("Testing PuzzleEnv reward...")
    var env = PuzzleEnv(2, 2, 0, 2, 20)

    # Initially solved, should get reward 1.0
    var reward = env.reward()
    if reward == 1.0:
        print("  PuzzleEnv solved reward: PASSED")
    else:
        print("  PuzzleEnv solved reward: FAILED (reward =", reward, ")")


fn test_collected_data():
    """Test CollectedData struct."""
    print("Testing CollectedData...")

    var data = CollectedData()

    var obs1 = List[Int]()
    obs1.append(0)
    obs1.append(1)
    data.obs.append(obs1)

    var logits1 = List[Float32]()
    logits1.append(0.5)
    logits1.append(0.5)
    data.logits.append(logits1)

    data.values.append(0.5)
    data.rewards.append(1.0)
    data.actions.append(0)

    if data.len() == 1:
        print("  CollectedData: PASSED")
    else:
        print("  CollectedData: FAILED")


fn test_merge():
    """Test merge function for CollectedData."""
    print("Testing merge...")

    var data1 = CollectedData()
    var obs1 = List[Int]()
    obs1.append(0)
    data1.obs.append(obs1)
    data1.values.append(0.5)

    var data2 = CollectedData()
    var obs2 = List[Int]()
    obs2.append(1)
    data2.obs.append(obs2)
    data2.values.append(0.7)

    var chunks = List[CollectedData]()
    chunks.append(data1)
    chunks.append(data2)

    var merged = merge(chunks)

    if len(merged.obs) == 2 and len(merged.values) == 2:
        print("  merge: PASSED")
    else:
        print("  merge: FAILED")


fn test_embedding_bag():
    """Test EmbeddingBag layer."""
    print("Testing EmbeddingBag...")

    var vectors = List[List[Float32]]()
    var v1 = List[Float32]()
    v1.append(1.0)
    v1.append(2.0)
    var v2 = List[Float32]()
    v2.append(3.0)
    v2.append(4.0)
    vectors.append(v1)
    vectors.append(v2)

    var bias = List[Float32]()
    bias.append(0.0)
    bias.append(0.0)

    var obs_shape = List[Int]()
    obs_shape.append(2)

    var emb = EmbeddingBag(vectors, bias, False, obs_shape, 0)

    var input = List[Int]()
    input.append(0)
    input.append(1)

    var output = emb.forward(input)

    # Should sum vectors: [1+3, 2+4] = [4, 6]
    if len(output) == 2 and output[0] == 4.0 and output[1] == 6.0:
        print("  EmbeddingBag: PASSED")
    else:
        print("  EmbeddingBag: FAILED (got", output[0], ",", output[1], ")")


fn main():
    """Run all tests."""
    print("=" * 60)
    print("TwisterL Mojo Implementation Tests")
    print("=" * 60)
    print("")

    # Basic function tests
    print("--- Basic Functions ---")
    test_relu()
    test_argmax()
    test_softmax()
    test_sample()
    test_sample_from_logits()
    print("")

    # Neural network layer tests
    print("--- Neural Network Layers ---")
    test_linear()
    test_linear_with_relu()
    test_sequential()
    test_embedding_bag()
    print("")

    # Tree structure tests
    print("--- Tree Structures ---")
    test_simple_tree()
    test_mcts_node()
    test_mcts_tree()
    test_ucb()
    print("")

    # Environment tests
    print("--- Environment ---")
    test_puzzle_env()
    test_puzzle_env_reset()
    test_puzzle_env_reward()
    print("")

    # Data collection tests
    print("--- Data Collection ---")
    test_collected_data()
    test_merge()
    print("")

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
