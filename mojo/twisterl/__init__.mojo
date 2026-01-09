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
TwisterL - Reinforcement Learning Library in Mojo

A Mojo reimplementation of the TwisterL Rust library for
reinforcement learning with neural networks and MCTS.

Modules:
- nn: Neural network components (Policy, Linear, EmbeddingBag, Sequential)
- rl: Reinforcement learning components (Env trait, MCTS, solve, evaluate)
- collector: Data collection (CollectedData, AZCollector, PPOCollector)
- envs: Environment implementations (PuzzleEnv)
"""

# Re-export main components for convenience
# Users can import directly: from twisterl import PuzzleEnv, Policy
# Or import from submodules: from twisterl.nn import Policy

# Neural network components
from .nn import Policy, Linear, EmbeddingBag, Sequential
from .nn import argmax, sample, sample_from_logits, softmax, log_softmax, relu

# RL components
from .rl import Env, MCTSNode, MCTSTree, SimpleTree
from .rl import MCTS, predict_probs_mcts_simple
from .rl import solve, solve_simple, single_solve
from .rl import evaluate, evaluate_simple

# Environment implementations
from .envs import PuzzleEnv, Puzzle

# Data collection
from .collector import CollectedData, merge
from .collector import AZCollector, PPOCollector
