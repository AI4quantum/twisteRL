# -*- coding: utf-8 -*-
# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0.

"""
Run the trained puzzle solver.

This script loads a trained model and uses it to solve puzzles.
Optimized version using InlineArray for stack allocation.

Usage:
    mojo run run_puzzle.mojo

Make sure to run train_puzzle.mojo first to create the model file!
"""

from collections import List, InlineArray
from random import random_float64, seed
from math import exp, sqrt

from twisterl.envs.puzzle import PuzzleEnv, PUZZLE_SIZE, NUM_ACTIONS
from twisterl.nn.policy import argmax, sample, softmax
from twisterl.nn.layers import relu


# ============================================
# Simple Policy Network (optimized version)
# ============================================

struct SimplePolicy:
    """Simple 2-layer policy network with optimized InlineArray support."""
    var obs_size: Int
    var hidden_size: Int
    var num_actions: Int
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]

    fn __init__(out self, obs_size: Int, hidden_size: Int, num_actions: Int):
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.w1 = List[Float32](capacity=obs_size * hidden_size)
        for _ in range(obs_size * hidden_size):
            self.w1.append(0.0)
        self.b1 = List[Float32](capacity=hidden_size)
        for _ in range(hidden_size):
            self.b1.append(0.0)
        self.w2 = List[Float32](capacity=hidden_size * num_actions)
        for _ in range(hidden_size * num_actions):
            self.w2.append(0.0)
        self.b2 = List[Float32](capacity=num_actions)
        for _ in range(num_actions):
            self.b2.append(0.0)

    fn forward_opt(self, obs: InlineArray[Int, PUZZLE_SIZE], masks: InlineArray[Bool, NUM_ACTIONS]) -> List[Float32]:
        """Optimized forward pass using InlineArray inputs."""
        var x = List[Float32](capacity=self.obs_size)
        for _ in range(self.obs_size):
            x.append(0.0)
        for i in range(PUZZLE_SIZE):
            if obs[i] < self.obs_size:
                x[obs[i]] = 1.0

        var h = List[Float32](capacity=self.hidden_size)
        for i in range(self.hidden_size):
            var sum_val = self.b1[i]
            for j in range(self.obs_size):
                sum_val += x[j] * self.w1[j * self.hidden_size + i]
            h.append(relu(sum_val))

        var logits = List[Float32](capacity=self.num_actions)
        for i in range(self.num_actions):
            var sum_val = self.b2[i]
            for j in range(self.hidden_size):
                sum_val += h[j] * self.w2[j * self.num_actions + i]
            logits.append(sum_val)

        for i in range(NUM_ACTIONS):
            if not masks[i]:
                logits[i] = -1e10

        return softmax(logits)

    fn set_params(mut self, params: List[Float32]):
        """Set all parameters from a flat list."""
        var idx = 0
        for i in range(len(self.w1)):
            self.w1[i] = params[idx]
            idx += 1
        for i in range(len(self.b1)):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(len(self.w2)):
            self.w2[i] = params[idx]
            idx += 1
        for i in range(len(self.b2)):
            self.b2[i] = params[idx]
            idx += 1

    fn load(mut self, path: String) raises:
        """Load model weights from a file."""
        with open(path, "r") as f:
            var content = f.read()
            var lines = content.split("\n")
            var header = lines[0].split(",")
            var loaded_obs_size = Int(header[0])
            var loaded_hidden_size = Int(header[1])
            var loaded_num_actions = Int(header[2])

            if loaded_obs_size != self.obs_size or loaded_hidden_size != self.hidden_size or loaded_num_actions != self.num_actions:
                print("Warning: Architecture mismatch!")
                return

            var param_strs = lines[1].split(",")
            var params = List[Float32]()
            for i in range(len(param_strs)):
                params.append(Float32(Float64(param_strs[i])))
            self.set_params(params)
        print("Model loaded from:", path)


# ============================================
# Puzzle Solver (optimized with InlineArray)
# ============================================

fn solve_puzzle(env: PuzzleEnv, policy: SimplePolicy, deterministic: Bool = True) -> Tuple[Bool, List[Int]]:
    """
    Solve a puzzle using the trained policy.
    Uses optimized InlineArray-based methods.
    Returns (success, list_of_actions).
    """
    var env_copy = env.clone()
    var actions = List[Int]()
    var max_steps = env_copy.max_depth * 2

    for _ in range(max_steps):
        if env_copy.solved():
            return (True, actions)

        if env_copy.is_final():
            break

        # Use optimized InlineArray methods
        var obs = env_copy.observe()
        var masks = env_copy.masks()
        var probs = policy.forward_opt(obs, masks)

        var action: Int
        if deterministic:
            action = argmax(probs)
        else:
            action = sample(probs)

        actions.append(action)
        env_copy.step(action)

    return (env_copy.solved(), actions)


fn action_name(action: Int) -> String:
    """Convert action number to readable name."""
    if action == 0:
        return "LEFT"
    elif action == 1:
        return "UP"
    elif action == 2:
        return "RIGHT"
    elif action == 3:
        return "DOWN"
    return "UNKNOWN"


# ============================================
# Main
# ============================================

fn main() raises:
    print("=" * 60)
    print("TwisterL Mojo Puzzle Solver")
    print("=" * 60)
    print()

    # Create environment and policy
    var env = PuzzleEnv(3, 3, 5, 2, 20)  # difficulty 5
    var policy = SimplePolicy(81, 64, 4)

    # Load trained model
    print("Loading model...")
    try:
        policy.load("puzzle_model.weights")
    except e:
        print("Error: Could not load model file!")
        print("Please run 'mojo run train_puzzle.mojo' first to train a model.")
        return

    print()

    # Solve multiple puzzles
    var num_puzzles = 10
    var solved_count = 0

    print("Solving", num_puzzles, "puzzles at difficulty", env.difficulty, "...")
    print()

    for puzzle_num in range(num_puzzles):
        seed(puzzle_num * 12345)  # Different seed for each puzzle
        env.reset()

        print("Puzzle", puzzle_num + 1, ":")
        print("Initial state:")
        env.display()

        var result = solve_puzzle(env, policy, True)
        var success = result[0]
        var actions = result[1]

        if success:
            solved_count += 1
            print("SOLVED in", len(actions), "moves!")
            print("Actions:", end=" ")
            for i in range(len(actions)):
                print(action_name(actions[i]), end=" ")
            print()
        else:
            print("FAILED to solve")

        print()

    print("=" * 60)
    print("Results:", solved_count, "/", num_puzzles, "puzzles solved")
    print("Success rate:", Int(Float32(solved_count) / Float32(num_puzzles) * 100), "%")
    print("=" * 60)
