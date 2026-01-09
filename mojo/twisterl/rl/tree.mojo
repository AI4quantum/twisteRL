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


struct Node[T: Copyable & Movable](Copyable, Movable):
    """A generic node in a tree structure."""
    var idx: Int
    var val: T
    var parent: Int  # Using -1 to indicate no parent
    var children: List[Int]

    fn __init__(out self, idx: Int, val: T):
        self.idx = idx
        self.val = val
        self.parent = -1
        self.children = List[Int]()

    fn is_root(self) -> Bool:
        """Check if this node is the root."""
        return self.parent == -1

    fn is_leaf(self) -> Bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0


struct Tree[T: Copyable & Movable](Copyable, Movable):
    """Generic tree structure."""
    var nodes: List[Node[T]]

    fn __init__(out self):
        self.nodes = List[Node[T]]()

    fn new_node(mut self, val: T) -> Int:
        """Create a new node and return its index."""
        var idx = len(self.nodes)
        self.nodes.append(Node[T](idx, val))
        return idx

    fn add_child_to_node(mut self, val: T, parent_idx: Int) -> Int:
        """Add a child node to an existing node."""
        var child_idx = self.new_node(val)
        self.nodes[parent_idx].children.append(child_idx)
        self.nodes[child_idx].parent = parent_idx
        return child_idx

    fn get_node(self, idx: Int) -> Node[T]:
        """Get a node by index."""
        return self.nodes[idx]

    fn size(self) -> Int:
        """Return the number of nodes in the tree."""
        return len(self.nodes)


# MCTS-specific node that stores state representation and MCTS statistics
struct MCTSNode(Copyable, Movable):
    """Node for Monte Carlo Tree Search with state tracking."""
    var state_repr: List[Int]  # State representation (observation encoding)
    var action_taken: Int  # Action that led to this state (-1 for root)
    var prior: Float32  # Prior probability from policy network
    var visit_count: Int
    var value_sum: Float32

    fn __init__(
        out self,
        state_repr: List[Int],
        action_taken: Int,
        prior: Float32,
    ):
        self.state_repr = state_repr
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0

    fn __init__(out self):
        """Create an empty MCTSNode."""
        self.state_repr = List[Int]()
        self.action_taken = -1
        self.prior = 0.0
        self.visit_count = 0
        self.value_sum = 0.0

    fn ucb(self, child: MCTSNode, C: Float32) -> Float32:
        """Calculate UCB (Upper Confidence Bound) score."""
        var q: Float32 = 0.0
        if child.visit_count > 0:
            q = child.value_sum / Float32(child.visit_count)
        return q + C * (sqrt(Float32(self.visit_count)) / (Float32(child.visit_count) + 1.0)) * child.prior

    fn average_value(self) -> Float32:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / Float32(self.visit_count)


struct MCTSTree(Copyable, Movable):
    """MCTS tree structure with specialized methods for MCTS algorithm."""
    var nodes: List[Node[MCTSNode]]

    fn __init__(out self):
        self.nodes = List[Node[MCTSNode]]()

    fn new_node(mut self, val: MCTSNode) -> Int:
        """Create a new node and return its index."""
        var idx = len(self.nodes)
        self.nodes.append(Node[MCTSNode](idx, val))
        return idx

    fn add_child_to_node(mut self, val: MCTSNode, parent_idx: Int) -> Int:
        """Add a child node to an existing node."""
        var child_idx = self.new_node(val)
        self.nodes[parent_idx].children.append(child_idx)
        self.nodes[child_idx].parent = parent_idx
        return child_idx

    fn get_node(self, idx: Int) -> Node[MCTSNode]:
        """Get a node by index."""
        return self.nodes[idx]

    fn size(self) -> Int:
        """Return the number of nodes in the tree."""
        return len(self.nodes)

    fn backpropagate(mut self, node_idx: Int, value: Float32):
        """Backpropagate value from a node to the root."""
        var current_idx = node_idx
        while current_idx >= 0:
            self.nodes[current_idx].val.value_sum += value
            self.nodes[current_idx].val.visit_count += 1
            current_idx = self.nodes[current_idx].parent

    fn next(self, node_idx: Int, C: Float32) -> Int:
        """Select the best child node based on UCB score."""
        var best_child: Int = -1
        var best_ucb: Float32 = -1e10

        var parent_node = self.nodes[node_idx].val

        for i in range(len(self.nodes[node_idx].children)):
            var child_idx = self.nodes[node_idx].children[i]
            var child_node = self.nodes[child_idx].val
            var ucb = parent_node.ucb(child_node, C)

            if ucb > best_ucb:
                best_child = child_idx
                best_ucb = ucb

        return best_child

    fn next_sample(self, node_idx: Int, sample_fn: fn (List[Float32]) -> Int) -> Int:
        """Select a child node by sampling based on priors."""
        var priors = List[Float32]()
        for i in range(len(self.nodes[node_idx].children)):
            var child_idx = self.nodes[node_idx].children[i]
            priors.append(self.nodes[child_idx].val.prior)

        var child_num = sample_fn(priors)
        return self.nodes[node_idx].children[child_num]


# Simple tree for basic use cases (non-MCTS)
struct SimpleNode(Copyable, Movable):
    """A simple node in the tree."""
    var idx: Int
    var val: Float32
    var parent: Int  # Using -1 to indicate no parent
    var children: List[Int]
    var visit_count: Int
    var total_value: Float32

    fn __init__(out self, idx: Int, val: Float32):
        self.idx = idx
        self.val = val
        self.parent = -1
        self.children = List[Int]()
        self.visit_count = 0
        self.total_value = 0.0

    fn is_root(self) -> Bool:
        """Check if this node is the root."""
        return self.parent == -1

    fn is_leaf(self) -> Bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    fn average_value(self) -> Float32:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / Float32(self.visit_count)


struct SimpleTree(Copyable, Movable):
    """Simple tree structure for storing search results."""
    var nodes: List[SimpleNode]

    fn __init__(out self):
        self.nodes = List[SimpleNode]()

    fn new_node(mut self, val: Float32) -> Int:
        """Create a new node and return its index."""
        var idx = len(self.nodes)
        self.nodes.append(SimpleNode(idx, val))
        return idx

    fn add_child_to_node(mut self, val: Float32, parent_idx: Int) -> Int:
        """Add a child node to an existing node."""
        var child_idx = self.new_node(val)
        self.nodes[parent_idx].children.append(child_idx)
        self.nodes[child_idx].parent = parent_idx
        return child_idx

    fn get_node(self, idx: Int) -> SimpleNode:
        """Get a node by index."""
        return self.nodes[idx]

    fn update_node(mut self, idx: Int, value: Float32):
        """Update a node's statistics after a simulation."""
        self.nodes[idx].visit_count += 1
        self.nodes[idx].total_value += value

    fn backpropagate(mut self, leaf_idx: Int, value: Float32):
        """Backpropagate value from leaf to root."""
        var current_idx = leaf_idx
        while current_idx >= 0:
            self.update_node(current_idx, value)
            current_idx = self.nodes[current_idx].parent

    fn size(self) -> Int:
        """Return the number of nodes in the tree."""
        return len(self.nodes)
