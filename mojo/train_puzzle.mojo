# -*- coding: utf-8 -*-
# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0.

"""
Training example for TwisterL Mojo implementation.

This demonstrates training a simple policy to solve the 8-puzzle
using an evolutionary strategy (ES) approach.

Since Mojo doesn't have automatic differentiation like PyTorch,
we use Evolution Strategies which only requires forward passes
and reward signals - no backpropagation needed!

Usage:
    mojo run train_puzzle.mojo
"""

from collections import List, InlineArray
from random import random_float64, seed
from math import exp, sqrt, log, cos
from time import perf_counter_ns

from twisterl.envs.puzzle import PuzzleEnv, PUZZLE_SIZE, NUM_ACTIONS
from twisterl.nn.policy import argmax, sample, softmax
from twisterl.nn.layers import Linear, relu


# ============================================
# Simple Policy Network (weights as flat list)
# ============================================

struct SimplePolicy:
    """
    A simple 2-layer policy network for the puzzle.

    Architecture:
    - Input: one-hot encoded observation (obs_size)
    - Hidden: hidden_size neurons with ReLU
    - Output: num_actions logits
    """
    var obs_size: Int
    var hidden_size: Int
    var num_actions: Int

    # Weights stored as flat lists for easy manipulation
    var w1: List[Float32]  # obs_size x hidden_size
    var b1: List[Float32]  # hidden_size
    var w2: List[Float32]  # hidden_size x num_actions
    var b2: List[Float32]  # num_actions

    fn __init__(out self, obs_size: Int, hidden_size: Int, num_actions: Int):
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Initialize weights with small random values
        self.w1 = List[Float32]()
        for _ in range(obs_size * hidden_size):
            self.w1.append((random_float64().cast[DType.float32]() - 0.5) * 0.1)

        self.b1 = List[Float32]()
        for _ in range(hidden_size):
            self.b1.append(0.0)

        self.w2 = List[Float32]()
        for _ in range(hidden_size * num_actions):
            self.w2.append((random_float64().cast[DType.float32]() - 0.5) * 0.1)

        self.b2 = List[Float32]()
        for _ in range(num_actions):
            self.b2.append(0.0)

    fn forward(self, obs: List[Int], masks: List[Bool]) -> List[Float32]:
        """Forward pass returning action probabilities (List-based for compatibility)."""
        # Convert sparse obs to dense one-hot
        var x = List[Float32](capacity=self.obs_size)
        for _ in range(self.obs_size):
            x.append(0.0)
        for i in range(len(obs)):
            if obs[i] < self.obs_size:
                x[obs[i]] = 1.0

        # Hidden layer: h = ReLU(x @ W1 + b1)
        var h = List[Float32](capacity=self.hidden_size)
        for i in range(self.hidden_size):
            var sum_val = self.b1[i]
            for j in range(self.obs_size):
                sum_val += x[j] * self.w1[j * self.hidden_size + i]
            h.append(relu(sum_val))

        # Output layer: logits = h @ W2 + b2
        var logits = List[Float32](capacity=self.num_actions)
        for i in range(self.num_actions):
            var sum_val = self.b2[i]
            for j in range(self.hidden_size):
                sum_val += h[j] * self.w2[j * self.num_actions + i]
            logits.append(sum_val)

        # Apply mask (set invalid actions to very negative)
        for i in range(self.num_actions):
            if i < len(masks) and not masks[i]:
                logits[i] = -1e10

        # Softmax to get probabilities
        return softmax(logits)

    fn forward_opt(self, obs: InlineArray[Int, PUZZLE_SIZE], masks: InlineArray[Bool, NUM_ACTIONS]) -> List[Float32]:
        """Optimized forward pass using InlineArray inputs to avoid heap allocation."""
        # Convert sparse obs to dense one-hot
        var x = List[Float32](capacity=self.obs_size)
        for _ in range(self.obs_size):
            x.append(0.0)
        for i in range(PUZZLE_SIZE):
            if obs[i] < self.obs_size:
                x[obs[i]] = 1.0

        # Hidden layer: h = ReLU(x @ W1 + b1)
        var h = List[Float32](capacity=self.hidden_size)
        for i in range(self.hidden_size):
            var sum_val = self.b1[i]
            for j in range(self.obs_size):
                sum_val += x[j] * self.w1[j * self.hidden_size + i]
            h.append(relu(sum_val))

        # Output layer: logits = h @ W2 + b2
        var logits = List[Float32](capacity=self.num_actions)
        for i in range(self.num_actions):
            var sum_val = self.b2[i]
            for j in range(self.hidden_size):
                sum_val += h[j] * self.w2[j * self.num_actions + i]
            logits.append(sum_val)

        # Apply mask (set invalid actions to very negative)
        for i in range(NUM_ACTIONS):
            if not masks[i]:
                logits[i] = -1e10

        # Softmax to get probabilities
        return softmax(logits)

    fn num_params(self) -> Int:
        """Return total number of parameters."""
        return len(self.w1) + len(self.b1) + len(self.w2) + len(self.b2)

    fn get_params(self) -> List[Float32]:
        """Get all parameters as a flat list."""
        var params = List[Float32]()
        for i in range(len(self.w1)):
            params.append(self.w1[i])
        for i in range(len(self.b1)):
            params.append(self.b1[i])
        for i in range(len(self.w2)):
            params.append(self.w2[i])
        for i in range(len(self.b2)):
            params.append(self.b2[i])
        return params

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

    fn save(self, path: String) raises:
        """Save model weights to a file."""
        with open(path, "w") as f:
            # Write header with architecture info
            f.write(String(self.obs_size) + "," + String(self.hidden_size) + "," + String(self.num_actions) + "\n")
            # Write all parameters
            var params = self.get_params()
            for i in range(len(params)):
                f.write(String(params[i]))
                if i < len(params) - 1:
                    f.write(",")
            f.write("\n")
        print("Model saved to:", path)

    fn load(mut self, path: String) raises:
        """Load model weights from a file."""
        with open(path, "r") as f:
            var content = f.read()
            var lines = content.split("\n")

            # Parse header
            var header = lines[0].split(",")
            var loaded_obs_size = Int(header[0])
            var loaded_hidden_size = Int(header[1])
            var loaded_num_actions = Int(header[2])

            # Verify architecture matches
            if loaded_obs_size != self.obs_size or loaded_hidden_size != self.hidden_size or loaded_num_actions != self.num_actions:
                print("Warning: Architecture mismatch!")
                return

            # Parse parameters
            var param_strs = lines[1].split(",")
            var params = List[Float32]()
            for i in range(len(param_strs)):
                params.append(Float32(Float64(param_strs[i])))

            self.set_params(params)
        print("Model loaded from:", path)


# ============================================
# Episode Runner
# ============================================

fn run_episode(env: PuzzleEnv, policy: SimplePolicy, deterministic: Bool) -> Tuple[Bool, Float32]:
    """
    Run a single episode and return (solved, total_reward).

    Uses optimized InlineArray-based observe() and masks() methods.
    """
    var env_copy = env.clone()
    var total_reward: Float32 = 0.0
    var max_steps = env_copy.max_depth * 2  # Safety limit

    for _ in range(max_steps):
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

        env_copy.step(action)
        total_reward += env_copy.reward()

    return (env_copy.solved(), total_reward)


fn evaluate_policy(env: PuzzleEnv, policy: SimplePolicy, num_episodes: Int, deterministic: Bool) -> Tuple[Float32, Float32]:
    """
    Evaluate policy over multiple episodes.
    Returns (success_rate, average_reward).
    """
    var successes: Float32 = 0.0
    var total_rewards: Float32 = 0.0

    for _ in range(num_episodes):
        var env_copy = env.clone()
        env_copy.reset()
        var result = run_episode(env_copy, policy, deterministic)
        if result[0]:
            successes += 1.0
        total_rewards += result[1]

    return (successes / Float32(num_episodes), total_rewards / Float32(num_episodes))


# ============================================
# Evolution Strategies Training
# ============================================

fn train_es(
    mut env: PuzzleEnv,
    mut policy: SimplePolicy,
    num_iterations: Int,
    population_size: Int,
    sigma: Float32,
    learning_rate: Float32,
    eval_episodes: Int,
    save_path: String = "model.weights",
    checkpoint_freq: Int = 20,
) raises:
    """Train using Evolution Strategies (ES).

    ES works by:
    1. Sample perturbations to policy parameters.
    2. Evaluate each perturbed policy.
    3. Update parameters using reward-weighted perturbations.

    This is a derivative-free optimization method that works well for RL.

    Args:
        env: The puzzle environment to train on.
        policy: The policy network to train.
        num_iterations: Number of training iterations.
        population_size: Number of perturbed policies per iteration.
        sigma: Standard deviation of parameter perturbations.
        learning_rate: Learning rate for parameter updates.
        eval_episodes: Number of episodes for policy evaluation.
        save_path: Path to save the final model weights.
        checkpoint_freq: Save checkpoint every N iterations (0 to disable).
    """
    print("Starting Evolution Strategies training...")
    print("  Population size:", population_size)
    print("  Sigma (noise):", sigma)
    print("  Learning rate:", learning_rate)
    print("  Save path:", save_path)
    print()

    var base_params = policy.get_params()
    var num_params = len(base_params)

    for iteration in range(num_iterations):
        var start_time = perf_counter_ns()

        # Generate perturbations and evaluate
        var perturbations = List[List[Float32]]()
        var rewards = List[Float32]()

        for _ in range(population_size):
            # Generate noise
            var noise = List[Float32]()
            for _ in range(num_params):
                # Simple Gaussian approximation using Box-Muller
                var u1 = random_float64().cast[DType.float32]()
                var u2 = random_float64().cast[DType.float32]()
                if u1 < 1e-10:
                    u1 = 1e-10
                var z = sqrt(-2.0 * Float32(log(Float64(u1)))) * Float32(cos(6.28318 * Float64(u2)))
                noise.append(z * sigma)
            perturbations.append(noise)

            # Create perturbed policy
            var perturbed_params = List[Float32]()
            for i in range(num_params):
                perturbed_params.append(base_params[i] + noise[i])
            policy.set_params(perturbed_params)

            # Evaluate
            var env_copy = env.clone()
            env_copy.reset()
            var result = run_episode(env_copy, policy, False)
            var reward: Float32 = 0.0
            if result[0]:
                reward = 1.0  # Bonus for solving
            reward += result[1]
            rewards.append(reward)

        # Compute reward statistics for normalization
        var mean_reward: Float32 = 0.0
        for i in range(len(rewards)):
            mean_reward += rewards[i]
        mean_reward /= Float32(len(rewards))

        var std_reward: Float32 = 0.0
        for i in range(len(rewards)):
            std_reward += (rewards[i] - mean_reward) * (rewards[i] - mean_reward)
        std_reward = sqrt(std_reward / Float32(len(rewards)) + 1e-8)

        # Update parameters using reward-weighted perturbations
        var new_params = List[Float32]()
        for i in range(num_params):
            var grad: Float32 = 0.0
            for j in range(population_size):
                var normalized_reward = (rewards[j] - mean_reward) / std_reward
                grad += perturbations[j][i] * normalized_reward
            grad /= Float32(population_size) * sigma
            new_params.append(base_params[i] + learning_rate * grad)

        base_params = new_params
        policy.set_params(base_params)

        # Evaluate current policy
        var eval_result = evaluate_policy(env, policy, eval_episodes, True)
        var success_rate = eval_result[0]
        var avg_reward = eval_result[1]

        var elapsed_ms = (perf_counter_ns() - start_time) / 1_000_000

        # Print progress
        if iteration % 5 == 0 or success_rate > 0.5:
            print(
                "Iter", iteration,
                "| Success:", Int(success_rate * 100), "%",
                "| Reward:", avg_reward,
                "| Time:", elapsed_ms, "ms"
            )

        # Increase difficulty if doing well
        if success_rate >= 0.8 and env.difficulty < 10:
            env.set_difficulty(env.difficulty + 1)
            print("  -> Increased difficulty to", env.difficulty)

        # Save checkpoint periodically
        if checkpoint_freq > 0 and iteration > 0 and iteration % checkpoint_freq == 0:
            var checkpoint_path = save_path + ".checkpoint_" + String(iteration)
            policy.save(checkpoint_path)

        # Early stopping if solved consistently at high difficulty
        if success_rate >= 0.9 and env.difficulty >= 8:
            print("Training converged!")
            policy.save(save_path)
            break

    # Save final model
    print()
    policy.save(save_path)
    print("Training complete!")


# ============================================
# Main
# ============================================

fn main():
    print("=" * 60)
    print("TwisterL Mojo Training Example")
    print("Training on 8-puzzle (3x3) using Evolution Strategies")
    print("=" * 60)
    print()

    # Seed random for reproducibility
    seed(42)

    # Create environment
    # Parameters: width, height, difficulty, depth_slope, max_depth
    var env = PuzzleEnv(3, 3, 1, 2, 20)

    print("Environment:")
    print("  Size: 3x3 (8-puzzle)")
    print("  Initial difficulty:", env.difficulty)
    print("  Max depth:", env.max_depth)
    print()

    # Create policy
    # obs_size = width * height * width * height (for one-hot encoding)
    var obs_size = 9 * 9  # 81 for 3x3 puzzle
    var hidden_size = 64
    var num_actions = 4

    var policy = SimplePolicy(obs_size, hidden_size, num_actions)
    print("Policy:")
    print("  Input size:", obs_size)
    print("  Hidden size:", hidden_size)
    print("  Output size:", num_actions)
    print("  Total parameters:", policy.num_params())
    print()

    # Initial evaluation
    print("Initial evaluation (before training):")
    var initial_eval = evaluate_policy(env, policy, 100, True)
    print("  Success rate:", Int(initial_eval[0] * 100), "%")
    print("  Average reward:", initial_eval[1])
    print()

    # Train!
    try:
        train_es(
            env,
            policy,
            num_iterations=100,
            population_size=32,
            sigma=0.1,
            learning_rate=0.03,
            eval_episodes=50,
            save_path="puzzle_model.weights",
            checkpoint_freq=20,
        )
    except e:
        print("Error during training:", e)

    # Final evaluation
    print()
    print("Final evaluation (after training):")
    env.set_difficulty(5)  # Test at medium difficulty
    var final_eval = evaluate_policy(env, policy, 100, True)
    print("  Difficulty:", env.difficulty)
    print("  Success rate:", Int(final_eval[0] * 100), "%")
    print("  Average reward:", final_eval[1])

    print()
    print("=" * 60)
    print("Training complete!")
    print("Model saved to: puzzle_model.weights")
    print("=" * 60)
